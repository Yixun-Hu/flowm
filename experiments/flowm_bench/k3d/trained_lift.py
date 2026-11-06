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
from .common import SCRATCH, ROOT, clips, clip_id, truth, ReadOnlyClips, log
from .lift import projected_box
from .geometry import BOX, rounded_cells, visible, camera_action, velocity_to_map
from .estimator_gate import intrinsic_incoming
from .measurement import NodeTracker
from .matching import match_velocities, summary, SPEED_BINS


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


@torch.inference_mode()
def run(checkpoint, arm, n=200, stage="all", batch_size=8):
    from .patch_motion import single_patch_search, summarize
    digest = checkpoint_hash(checkpoint)
    cache = SCRATCH / "trained_lifts" / digest / arm
    output = ROOT / "outputs/k3d_trained_lifts" / digest / arm
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    paths = clips("k3d_v_validation", n)
    dataset = ReadOnlyClips(paths)
    model = load_checkpoint_model(checkpoint, arm)
    torch.backends.mha.set_fastpath_enabled(False)
    started = time.perf_counter()
    patch_rows, gate_rows = [], []
    for i, path in enumerate(paths):
        dest = cache / (clip_id(path).replace("/", "_") + ".pt")
        if not dest.exists():
            item = dataset[i]
            tokens = torch.cat([pooled_tokens(model, x) for x in item["videos"].split(batch_size)])
            torch.save(dict(tokens=tokens.half(), actions=item["conds"].long(), checkpoint_sha256=digest), dest)
        data = torch.load(dest, weights_only=True)
        assert data["checkpoint_sha256"] == digest and data["tokens"].shape == (140, 177, 256)
        if stage == "lift":
            continue
        boxes = projected_box(data["tokens"], 8)
        labels = truth(path)
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
            if stage in ("all", "patch") and not turning:
                ids = (vis[t] & vis[t-1]).nonzero().flatten()
                if not len(ids):
                    continue
                centers = rounded_cells(pos[t-1, ids])-torch.tensor(BOX)
                result = single_patch_search(boxes[t-1:t], boxes[t:t+1], centers,
                    torch.zeros(len(ids), dtype=torch.long), camera_action(actions[t].expand(len(ids))))
                target = intrinsic_incoming(labels, t)[ids]
                errors = (result["velocity"] - target).norm(dim=-1)
                errors[~result["usable"]] = torch.inf
                for j, block in enumerate(ids):
                    patch_rows.append(dict(clip=clip_id(path), frame=t, block=int(block),
                        speed=float(labels["block_speed"][t-1, block]), distance=float(pos[t-1, block, 0]),
                        cell_x=int(centers[j, 1]), error=float(errors[j]), usable=bool(result["usable"][j]),
                        boundary=bool(result["boundary"][j])))
        if history:
            torch.save({k: torch.stack(v) for k, v in history.items()}, cache / (dest.stem + "_gate_mask_k8_s1.pt"))
        log(f"trained lift {stage} {i+1}/{n}")
    if patch_rows:
        write_rows(output / "patch_pairs.csv", patch_rows)
        write_rows(output / "patch_summary.csv", summarize(patch_rows, "trained pooled write", 8))
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
    info = dict(checkpoint=str(Path(checkpoint).resolve()), sha256=digest, arm=arm, clips=n,
                frames=140, k=8, sigma=1, residual_mode="mask", truth="intrinsic_incoming",
                seconds=time.perf_counter()-started, cache=str(cache), output=str(output), stage=stage)
    (output / f"{stage}_run.json").write_text(json.dumps(info, indent=2)+"\n")
    print(json.dumps(info), flush=True)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--arm", required=True, choices=["c", "c_pw", "a_oracle", "a", "b"])
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stage", choices=["all", "lift", "patch", "gate"], default="all")
    args = p.parse_args()
    run(args.checkpoint, args.arm, args.clips, args.stage, args.batch_size)


def dispatch_checkpoint(stage):
    """Add the trained path to the existing stage-1 CLIs without changing defaults."""
    if "--checkpoint" not in sys.argv:
        return False
    sys.argv.extend(["--stage", stage])
    main()
    return True


if __name__ == "__main__":
    main()
