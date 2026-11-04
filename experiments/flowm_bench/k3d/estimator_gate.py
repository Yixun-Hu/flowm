"""Stateful v17 gate on cached lifts; all widths, K, alignments and lift variants."""
import argparse
from collections import defaultdict
import time
import numpy as np
import torch
from .common import clips, clip_id, truth, SCRATCH, write_csv, log
from .geometry import visible, velocity_to_map
from .lift import read_cache, projected_box
from .measurement import NodeTracker
from .matching import match_velocities, nearest_integer, summary, SPEED_BINS


def trajectory_path(path, variant, k, sigma):
    return SCRATCH / "trajectories" / path.parent.parent.name / f"{clip_id(path).replace('/', '_')}_{variant}_k{k}_s{sigma:g}.pt"


@torch.inference_mode()
def run_trajectories(paths, variant, k, sigma, batch_size=8):
    """Reset at clip start; 139 incoming pairs, retaining 140 source frames."""
    for lo in range(0, len(paths), batch_size):
        batch = [p for p in paths[lo:lo+batch_size] if not trajectory_path(p, variant, k, sigma).exists()]
        if not batch:
            continue
        start = time.perf_counter()
        cached = [read_cache(p) for p in batch]
        boxes = torch.stack([projected_box(c[variant], k) for c in cached])
        actions = torch.stack([c["actions"] for c in cached])
        del cached
        tracker = NodeTracker(len(batch), sigma)
        history = defaultdict(list)
        for t in range(1, 140):
            result = tracker.step(boxes[:, t-1], boxes[:, t], actions[:, t])
            for name, value in result.items():
                history[name].append(value.clone())
        history = {name: torch.stack(values, 1) for name, values in history.items()}
        elapsed = time.perf_counter()-start
        for i, path in enumerate(batch):
            dest = trajectory_path(path, variant, k, sigma)
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = {name: value[i] for name, value in history.items()}
            data.update(seconds=elapsed/len(batch), variant=variant, k=k, sigma=sigma)
            torch.save(data, dest.with_suffix(".tmp"))
            dest.with_suffix(".tmp").replace(dest)
        log(f"estimator {paths[0].parent.parent.name} {variant} K={k} sigma={sigma} "
            f"{lo+len(batch)}/{len(paths)} {elapsed:.2f}s ({elapsed/len(batch)/139:.4f}s/step)")


def intrinsic_incoming(labels, t):
    """Actual incoming world displacement, expressed in the CURRENT agent frame."""
    v = labels["block_vel_world"][t-1][:, [0, 2]]
    angle = labels["agent_dir"][t]
    c, s = angle.cos(), angle.sin()
    return torch.stack((v[:, 0]*c-v[:, 1]*s, v[:, 0]*s+v[:, 1]*c), -1)


def score(paths, variants, ks, sigmas):
    pooled = defaultdict(lambda: dict(error=[], miss=[], false=[], steps=0, turns=0))
    clip_rows = []
    for variant in variants:
        for k in ks:
            for sigma in sigmas:
                for path in paths:
                    data = torch.load(trajectory_path(path, variant, k, sigma), weights_only=True)
                    labels = truth(path)
                    visibility = visible(labels["block_pos_agent"][:141])
                    local = defaultdict(lambda: dict(error=[], miss=[], false=[], steps=0, turns=0))
                    for t in range(1, 140):
                        turns = bool(data["turns"][t-1])
                        nodes = data["velocity"][t-1][data["active"][t-1]].numpy()
                        for mode in ("literal", "intrinsic_incoming"):
                            for delay in (0, 1):
                                tt = t + delay
                                # Restrict label use to the first 140 frames, including delay.
                                if tt >= 140:
                                    continue
                                pair_turn = turns or (delay == 1 and bool(data["turns"][t]))
                                category = "turn" if pair_turn else "nonturn"
                                take = visibility[tt] & visibility[tt-1]
                                target = (labels["block_vel_agent"][tt] if mode == "literal"
                                          else intrinsic_incoming(labels, tt))
                                speed = (target.norm(dim=-1) if mode == "literal"
                                         else labels.get("block_speed", labels["block_vel_world"].norm(dim=-1))[tt-1])
                                target = velocity_to_map(target[take]).numpy()
                                speed = speed[take].numpy()
                                result = match_velocities(nodes, target)
                                reference = nearest_integer(target)
                                bins = [("all", np.ones(len(speed), dtype=bool)),
                                        ("0.25-1.0", (speed >= .25-1e-6) & (speed <= 1+1e-6))]
                                bins += [(f"{a:g}-{min(b,1):g}", (speed >= a) & (speed < b)) for a,b in SPEED_BINS]
                                for bin_name, mask in bins:
                                    key = (variant, k, sigma, mode, delay, category, bin_name)
                                    values = local[key]
                                    values["error"].extend(result["error"][mask].tolist())
                                    values["miss"].extend(result["miss"][mask].tolist())
                                    # Node speed defines the FA denominator for speed-bin rows;
                                    # truth filtering never changes the Hungarian assignment.
                                    ns = np.linalg.norm(nodes, axis=-1)
                                    if bin_name in ("all", "0.25-1.0"):
                                        nm = np.ones(len(nodes), dtype=bool)
                                    else:
                                        a, b = map(float, bin_name.split("-"))
                                        nm = (ns >= a) & (ns < b + (1e-6 if b == 1 else 0))
                                    values["false"].extend(result["false"][nm].tolist())
                                    values["steps"] += 1
                                    values["turns"] += pair_turn
                                    if variant == variants[0] and k == ks[0] and sigma == sigmas[0]:
                                        rkey = ("nearest_integer", 0, 0, mode, delay, category, bin_name)
                                        ref = local[rkey]
                                        ref["error"].extend(reference[mask].tolist())
                                        ref["miss"].extend((reference[mask] > .5).tolist())
                                        ref["steps"] += 1
                    for key, values in local.items():
                        for name in ("error", "miss", "false"):
                            pooled[key][name].extend(values[name])
                        pooled[key]["steps"] += values["steps"]
                        pooled[key]["turns"] += values["turns"]
                        clip_rows.append(format_row(key, values, clip=clip_id(path),
                                                    seconds=0 if key[0] == "nearest_integer" else data["seconds"]))
    rows = [format_row(key, values, clip="pooled", seconds=float("nan")) for key, values in pooled.items()]
    write_csv("gate_summary.csv", rows)
    write_csv("gate_clips.csv", clip_rows)


def format_row(key, values, clip, seconds):
    variant, k, sigma, mode, delay, category, speed_bin = key
    row = dict(variant=variant, k=k, sigma=sigma, truth_mode=mode, delay=delay,
               step_type=category, speed_bin=speed_bin, clip=clip, steps=values["steps"])
    row.update(summary(values["error"], values["miss"]))
    row.update(nodes=len(values["false"]), false_alarm_rate=float(np.mean(values["false"])) if values["false"] else float("nan"),
               seconds=seconds)
    row["passed"] = row["median"] <= .1 and row["miss_rate"] <= .3
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=["tokens", "h_new"])
    p.add_argument("--ks", type=int, nargs="+", default=[8, 32])
    p.add_argument("--sigmas", type=float, nargs="+", default=[.5, 1., 2.])
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    paths = clips("k3d_v_validation", args.clips)
    if not args.score_only:
        for variant in args.variants:
            for k in args.ks:
                for sigma in args.sigmas:
                    run_trajectories(paths, variant, k, sigma, args.batch_size)
    score(paths, args.variants, args.ks, args.sigmas)


if __name__ == "__main__":
    main()
