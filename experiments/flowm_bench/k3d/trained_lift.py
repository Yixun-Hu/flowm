"""CPU lift, direct patch motion and mask estimator gate on a kinetic checkpoint.

Separate checkpoint-hash caches/results; never updates released-weight reports.
"""
import argparse
import csv
import json
import time
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from algorithms.mem_wm.backbones.kwm3d.checkpoints import checkpoint_hash, load_checkpoint_model
from .common import SCRATCH, ROOT, RESULTS, clips, clip_id, truth, ReadOnlyClips, log
from .lift import projected_box, tokens_to_box, observation_lift
from .geometry import BOX, rounded_cells, visible, camera_action, velocity_to_map
from .estimator_gate import intrinsic_incoming
from .measurement import NodeTracker
from .matching import match_velocities, summary, SPEED_BINS

PATCH_KS = (8, 32, 256)
STAGE1_PAIRS = RESULTS / "followup_patch_tokens_k8_pairs.csv"
C_GATE_ERROR = "Arm C uses per-node writes and has no trained pooled observation write; use a_oracle, a, b or c_pw"


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def pooled_tokens(model, frames):
    # Exactly the same zero-map pooled write that observation_imprint uses.
    written = model.observation_write(frames)
    return written[:, model._get_fov_mask(frames.device)].contiguous()


def lift_tokens(model, frames, arm):
    if arm == "c":
        # Keep all stage-1 assertions, including equality across nodes and g*h_new.
        return observation_lift(model, frames)["tokens"]
    return pooled_tokens(model, frames)


def patch_pairs(path, tokens, actions, labels, k, source):
    """Use stage 1's exact batching, projection, search and pair-row convention."""
    from .patch_motion import single_patch_search
    boxes = tokens_to_box(tokens.float())[0] if k == 256 else projected_box(tokens, k)
    pos = labels["block_pos_agent"][:140]
    vis = visible(pos)
    rows = []
    for lo in range(1, 140, 16):
        ts = torch.arange(lo, min(lo+16, 140))
        ts = ts[(actions[ts] != 0) & (actions[ts] != 1)]
        pair, block = (vis[ts] & vis[ts-1]).nonzero(as_tuple=True)
        if not len(pair):
            continue
        t = ts[pair]
        centers = rounded_cells(pos[t-1, block])-torch.tensor(BOX)
        result = single_patch_search(boxes[ts-1], boxes[ts], centers, pair,
                                     camera_action(actions[t]))
        targets = torch.stack([intrinsic_incoming(labels, int(tt))[bb] for tt, bb in zip(t, block)])
        error = (result["velocity"]-targets).norm(dim=-1)
        error[~result["usable"]] = float("inf")
        for i in range(len(t)):
            rows.append(dict(source=source, k=k, clip=clip_id(path), frame=int(t[i]), block=int(block[i]),
                speed=float(labels["block_speed"][t[i]-1, block[i]]),
                distance=float(pos[t[i]-1, block[i], 0]), cell_x=int(centers[i, 1]),
                vx=float(result["velocity"][i, 0]), vy=float(result["velocity"][i, 1]),
                true_vx=float(targets[i, 0]), true_vy=float(targets[i, 1]), error=float(error[i]),
                usable=bool(result["usable"][i]), boundary=bool(result["boundary"][i])))
    return rows


def check_pair_set(rows, paths):
    selected = {clip_id(path) for path in paths}
    key = lambda row: (row["clip"], int(row["frame"]), int(row["block"]))
    with STAGE1_PAIRS.open(newline="") as f:
        reference = list(csv.DictReader(f))
    absent = selected - {r["clip"] for r in reference}
    expected = {key(r) for r in reference if r["clip"] in selected}
    actual = {key(r) for r in rows}
    if absent or actual != expected or len(actual) != len(rows):
        raise ValueError(f"Patch pair set differs from stage 1: missing clips={sorted(absent)}, "
                         f"missing pairs={len(expected-actual)}, extra pairs={len(actual-expected)}, "
                         f"duplicate pairs={len(rows)-len(actual)}")
    return len(actual)


@torch.inference_mode()
def run(checkpoint, arm, n=200, stage="all", batch_size=None, output_dir=None):
    if arm == "c" and stage not in ("lift", "patch"):
        raise ValueError(C_GATE_ERROR)
    from .patch_motion import summarize
    # C follows the stage-1 lift batch size; pooled arms retain their default.
    batch_size = batch_size if batch_size is not None else (32 if arm == "c" else 8)
    digest = checkpoint_hash(checkpoint)
    cache = SCRATCH / "trained_lifts" / digest / arm
    output = Path(output_dir) if output_dir is not None else ROOT / "outputs/k3d_trained_lifts" / digest / arm
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    paths = clips("k3d_v_validation", n)
    dataset = ReadOnlyClips(paths)
    model = load_checkpoint_model(checkpoint, arm)
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.perf_counter()
    patch_rows, gate_rows = {k: [] for k in PATCH_KS}, []
    source = "observation-only tokens" if arm == "c" else "trained pooled write"
    lift_definition = ("lift.observation_lift(model, frames)['tokens']: per-node write on a zero map, node 0"
                       if arm == "c" else "model.observation_write(frames): V-pooled write on a zero map")
    lift_seconds = patch_seconds = gate_seconds = 0.
    lifted_clips = 0
    for i, path in enumerate(paths):
        dest = cache / (clip_id(path).replace("/", "_") + ".pt")
        tick = time.perf_counter()
        if not dest.exists():
            item = dataset[i]
            try:
                tokens = torch.cat([lift_tokens(model, x, arm) for x in item["videos"].split(batch_size)])
            except AssertionError as error:
                failure = dict(checkpoint=str(Path(checkpoint).resolve()), sha256=digest, arm=arm,
                               lift_definition=lift_definition, status="failed", clip=clip_id(path),
                               error=f"Observation lift assertion failed: {error}")
                (output / f"{stage}_run.json").write_text(json.dumps(failure, indent=2)+"\n")
                raise
            torch.save(dict(tokens=tokens.half(), actions=item["conds"].long(), checkpoint_sha256=digest,
                            lift_definition=lift_definition), dest.with_suffix(".tmp"))
            dest.with_suffix(".tmp").replace(dest)
            lifted_clips += 1
        data = torch.load(dest, map_location="cpu", weights_only=True)
        assert data["checkpoint_sha256"] == digest and data["tokens"].shape == (140, 177, 256)
        assert data["tokens"].dtype == torch.float16
        assert data.get("lift_definition", lift_definition) == lift_definition
        lift_seconds += time.perf_counter()-tick
        if stage == "lift":
            log(f"trained lift {stage} {i+1}/{n}")
            continue
        labels = truth(path)
        if stage in ("all", "patch"):
            tick = time.perf_counter()
            for k in PATCH_KS:
                patch_rows[k].extend(patch_pairs(path, data["tokens"], data["actions"], labels, k, source))
            patch_seconds += time.perf_counter()-tick
        if stage == "patch":
            log(f"trained lift {stage} {i+1}/{n}")
            continue
        tick = time.perf_counter()
        boxes = projected_box(data["tokens"], 8)
        pos, actions = labels["block_pos_agent"][:140], data["actions"]
        vis = visible(pos)
        tracker = NodeTracker(1, 1., static_residual="mask")
        history = defaultdict(list)
        for t in range(1, 140):
            turning = actions[t] in (0, 1)
            if stage in ("all", "gate"):
                result = tracker.step(boxes[t-1:t], boxes[t:t+1], actions[t:t+1])
                for key, value in result.items():
                    history[key].append(value[0].clone())
                if not turning:
                    take = vis[t] & vis[t-1]
                    target = velocity_to_map(intrinsic_incoming(labels, t)[take])
                    match = match_velocities(result["velocity"][0][result["active"][0]].numpy(), target.numpy())
                    for j, speed in enumerate(labels["block_speed"][t-1][take]):
                        gate_rows.append(dict(clip=clip_id(path), frame=t, speed=float(speed),
                                              error=float(match["error"][j]), miss=bool(match["miss"][j])))
        if history:
            torch.save({k: torch.stack(v) for k, v in history.items()}, cache / (dest.stem + "_gate_mask_k8_s1.pt"))
        gate_seconds += time.perf_counter()-tick
        log(f"trained lift {stage} {i+1}/{n}")
    pairs = None
    if stage in ("all", "patch"):
        # Validate every K before publishing any CSVs.
        for rows in patch_rows.values():
            pairs = check_pair_set(rows, paths)
        for k, rows in patch_rows.items():
            write_rows(output / f"patch_k{k}_pairs.csv", rows)
            write_rows(output / f"patch_k{k}_summary.csv", summarize(rows, source, k))
    if gate_rows:
        write_rows(output / "gate_pairs.csv", gate_rows)
        rows = []
        for name, lo, hi in [("all", -np.inf, np.inf), ("0.25-1", .25-1e-6, 1+1e-6),
                             *[(f"{a:g}-{min(b,1):g}", a, b) for a, b in SPEED_BINS]]:
            selected = [r for r in gate_rows if lo <= r["speed"] < hi]
            values = summary([r["error"] for r in selected], [r["miss"] for r in selected])
            rows.append(dict(truth="intrinsic_incoming", residual_mode="mask", k=8, sigma=1,
                speed_bin=name, **values, verdict="PASS" if values["median"] <= .1 and values["miss_rate"] <= .3 else "FAIL"))
        write_rows(output / "gate_summary.csv", rows)
    info = dict(checkpoint=str(Path(checkpoint).resolve()), sha256=digest, arm=arm, clips=len(paths),
                frames=140, k=8, sigma=1, residual_mode="mask", truth="intrinsic_incoming",
                seconds=time.perf_counter()-started, cache=str(cache), output=str(output), stage=stage)
    if stage != "gate":
        info.update(lift_definition=lift_definition, batch_size=batch_size, lifted_clips=lifted_clips,
                    lift_seconds=lift_seconds, patch_seconds=patch_seconds, gate_seconds=gate_seconds)
    if pairs is not None:
        info.update(Ks=list(PATCH_KS), pairs=pairs, pair_set_matches_stage1=True)
        if stage != "patch":
            (output / "patch_run.json").write_text(json.dumps(info, indent=2)+"\n")
    (output / f"{stage}_run.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info), flush=True)
    return output


def main(argv=None, *, stage=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arm", required=True, choices=["c", "c_pw", "a_oracle", "a", "b"])
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--batch-size", type=int, help="lift batch size (C: stage-1 default 32; pooled arms: 8)")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--stage", choices=["all", "lift", "patch", "gate"],
                   help="default: all, or the dispatching stage-1 entry point")
    p.add_argument("--split", default="k3d_v_validation",
                   help="trained-lift checks use k3d_v_validation only")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        p.error("Stage-1 options are not supported with --checkpoint: " + " ".join(unknown) +
                ". Trained-lift checks fix patch K=8/32/256; the pooled gate uses mask residuals, K=8, sigma=1 and intrinsic_incoming truth; "
                "use --clips/--batch-size to size the CPU run.")
    if args.split != "k3d_v_validation":
        p.error("--checkpoint trained-lift checks require --split k3d_v_validation; other stage-1 splits are unsupported")
    selected_stage = stage or args.stage or "all"
    if args.arm == "c" and (args.stage in ("gate", "all") or selected_stage not in ("lift", "patch")):
        p.error(C_GATE_ERROR)
    if stage is not None and args.stage not in (None, "all", stage):
        p.error(f"This entry point dispatches trained-lift --stage {stage}")
    if args.clips < 1 or (args.batch_size is not None and args.batch_size < 1):
        p.error("--clips and --batch-size must be positive")
    batch_size = args.batch_size if args.batch_size is not None else (32 if args.arm == "c" else 8)
    run(args.checkpoint, args.arm, args.clips, selected_stage, batch_size, args.output_dir)


def dispatch_checkpoint(stage):
    """Add the trained path to the existing stage-1 CLIs without changing defaults."""
    argv = sys.argv[1:]
    if not any(arg == "--checkpoint" or arg.startswith("--checkpoint=") for arg in argv):
        return False
    main(argv, stage=stage)
    return True


if __name__ == "__main__":
    main()
