"""Action, velocity and support ablations for null alarms and K3D-v locks."""
from collections import defaultdict
import time
import numpy as np
import torch
from .common import clips, clip_id, write_csv, log
from .lift import read_cache, projected_box
from .estimator_gate import trajectory_path
from .followup import (OBS, run, load_trajectory, support_for, append_result, table,
                       initialize_report)


def action_categories(actions):
    """139 pair categories; turn adjacency includes either neighboring pair.

    Frame-zero placeholder is excluded. Current turns are always 'turn'.
    Nonturn adjacent pairs form one exclusive group; forward/no-op are the rest.
    """
    a = torch.as_tensor(actions)[1:]
    turn = (a == 0) | (a == 1)
    adjacent = torch.zeros_like(turn)
    adjacent[1:] |= turn[:-1]
    adjacent[:-1] |= turn[1:]
    labels = np.where(a.numpy() == 2, "forward", "no-op").astype(object)
    labels[adjacent.numpy()] = "turn-adjacent"
    labels[turn.numpy()] = "turn"
    return labels


def breakdown_counts(data, kind):
    """Return masks and denominators without counting carried turns as detections."""
    categories = action_categories(data["actions"])
    nonturn = ~data["turns"]
    velocity = data["measured"] if kind == "alarm" else data["velocity"]
    selected = ((data["agreement"] >= .1) & (velocity.norm(dim=-1) > .15)
                if kind == "alarm" else data["active"].clone())
    selected &= nonturn[:, None]
    selected[:, 0] = False
    rows = []
    for category in ("all", "forward", "no-op", "turn-adjacent", "turn"):
        keep = nonturn if category == "all" else torch.from_numpy(categories == category)
        valid = keep & nonturn
        v = velocity[keep][selected[keep]]
        near_forward = ((v-v.new_tensor([1., 0.])).norm(dim=-1) <= .25)
        near_zero = v.norm(dim=-1) <= .25
        rows.append(dict(action=category, steps=int(keep.sum()), measured_steps=int(valid.sum()),
                         node_steps=17*int(valid.sum()), detections=len(v),
                         steps_with_detection=int(selected[keep].any(-1).sum()),
                         evidenced_detections=int((selected[keep] & (data["evidence"][keep] > 1e-6)).sum()),
                         locked_detections=int((selected[keep] & data["active"][keep]).sum()),
                         near_forward=int(near_forward.sum()), near_zero=int(near_zero.sum()),
                         vx_sum=float(v[:, 0].sum()), vy_sum=float(v[:, 1].sum())))
    return rows, velocity, selected, categories


def get_data(path, mode, region):
    if mode == "subtract" and region == "all":
        d = torch.load(trajectory_path(path, "tokens", 8, 1.), weights_only=True)
        d["actions"] = read_cache(path)["actions"]
        return d
    return load_trajectory(path, mode=mode, region=region)


def score(paths, mode, region):
    split = paths[0].parent.parent.name
    kind = "alarm" if "null" in split else "locked"
    rows, hist = [], defaultdict(int)
    edges = np.array([-np.inf, -1.5, -1., -.5, 0., .5, 1., 1.5, np.inf])
    for path in paths:
        d = get_data(path, mode, region)
        cr, velocity, selected, category = breakdown_counts(d, kind)
        for r in cr:
            rows.append(dict(source=OBS, split=split, kind=kind, mode=mode, region=region,
                             clip=clip_id(path), **r))
        for name in ("all", "forward", "no-op", "turn-adjacent"):
            keep = np.ones(139, bool) if name == "all" else category == name
            v = velocity[keep][selected[keep]].numpy()
            counts, _, _ = np.histogram2d(v[:, 0], v[:, 1], bins=[edges, edges])
            for ix in range(8):
                for iy in range(8):
                    hist[name, ix, iy] += int(counts[ix, iy])
    pooled = []
    for action in ("all", "forward", "no-op", "turn-adjacent", "turn"):
        rr = [r for r in rows if r["action"] == action]
        r = dict(source=OBS, split=split, kind=kind, mode=mode, region=region, action=action)
        for k in ("steps", "measured_steps", "node_steps", "detections", "steps_with_detection", "evidenced_detections", "locked_detections", "near_forward", "near_zero", "vx_sum", "vy_sum"):
            r[k] = sum(x[k] for x in rr)
        r["node_rate"] = r["detections"]/r["node_steps"] if r["node_steps"] else float("nan")
        r["step_rate"] = r["steps_with_detection"]/r["measured_steps"] if r["measured_steps"] else float("nan")
        r["evidenced_fraction"] = r["evidenced_detections"]/r["detections"] if r["detections"] else float("nan")
        r["locked_fraction"] = r["locked_detections"]/r["detections"] if r["detections"] else float("nan")
        for k in ("near_forward", "near_zero"):
            r[k+"_fraction"] = r[k]/r["detections"] if r["detections"] else float("nan")
        for axis in ("vx", "vy"):
            r["mean_"+axis] = r[axis+"_sum"]/r["detections"] if r["detections"] else float("nan")
        pooled.append(r)
    hr = [dict(source=OBS, split=split, kind=kind, mode=mode, region=region, action=action,
               vx_lo=edges[ix], vx_hi=edges[ix+1], vy_lo=edges[iy], vy_hi=edges[iy+1], count=count)
          for (action, ix, iy), count in hist.items()]
    prefix = f"followup_breakdown_{split}_{mode}_{region}"
    write_csv(prefix+"_clips.csv", rows)
    write_csv(prefix+"_summary.csv", pooled)
    write_csv(prefix+"_histogram.csv", hr)
    # Marginals make the full 2D histogram readable in the report.
    marginal = []
    for axis in ("vx", "vy"):
        for i in range(8):
            count = sum(r["count"] for r in hr if r["action"] == "all" and r[axis+"_lo"] == edges[i])
            marginal.append(dict(source=OBS, axis=axis, lo=edges[i], hi=edges[i+1], count=count,
                                 fraction=count/max(1,pooled[0]["detections"])))
    append_result(f"breakdown-{split}-{mode}-{region}",
        f"### 2. {kind.capitalize()} breakdown — {OBS}, {split}, {mode}, {region} support\n\n" +
        table(pooled, ["source", "action", "steps", "node_steps", "detections", "node_rate", "step_rate", "evidenced_fraction", "locked_fraction", "near_forward_fraction", "near_zero_fraction", "mean_vx", "mean_vy"]) +
        "\n\nNonturn groups are exclusive: turn-adjacent means the preceding or following incoming pair is a turn; forward/no-op exclude those pairs. Actual turn pairs carry state but have no new measurement and are excluded from detection denominators. Near (+1,0) and near zero mean Euclidean radius .25. Nulls use immediate measured velocities and the original alarm definition; K3D-v uses locked/evidenced recurrent velocities, as in the gate. `evidenced_fraction` counts selected evidence >1e-6; `locked_fraction` counts active locks among nominal alarms. Pre-search agreement alarms need not survive mask-mode search confirmation or mode selection. Near support is cone∩{rel_x≤6}, far is cone∩{rel_x≥7}; each support reruns the complete recurrence. No new acceptance criterion.\n\n" +
        table(marginal, ["source", "axis", "lo", "hi", "count", "fraction"]) +
        f"\n\nFull action-conditioned 2D velocity histogram: `results/{prefix}_histogram.csv`; half-open bins include under/overflow.")
    log(str(pooled[0]))


@torch.inference_mode()
def stationary_content(paths):
    """Compare same-cell and ego-transported imprint SSD on common valid support."""
    rows = []
    for path in paths:
        cache = read_cache(path)
        x = projected_box(cache["tokens"], 8)
        forward = cache["actions"][1:] == 2
        prev, cur = x[:-1][forward], x[1:][forward]
        for region in ("all", "near", "far"):
            support = support_for(region)
            overlap = support.clone()
            overlap[:, :-1] &= support[:, 1:]
            overlap[:, -1] = False
            same = (cur-prev).square().sum(1)[:, overlap].mean(1)
            ego = (cur-prev.roll(-1, -1)).square().sum(1)[:, overlap].mean(1)
            for i in range(len(same)):
                rows.append(dict(source=OBS, split=path.parent.parent.name, clip=clip_id(path),
                                 forward_pair=i, region=region, same_ssd=float(same[i]), ego_ssd=float(ego[i])))
    write_csv("followup_cone_stationary_steps.csv", rows)
    pooled = []
    for region in ("all", "near", "far"):
        rr = [r for r in rows if r["region"] == region]
        same = np.array([r["same_ssd"] for r in rr])
        ego = np.array([r["ego_ssd"] for r in rr])
        pooled.append(dict(source=OBS, region=region, pairs=len(rr), same_cell_ssd=float(same.mean()),
                           ego_ssd=float(ego.mean()), ratio_of_means=float(same.mean()/ego.mean()),
                           same_cell_better=float((same < ego).mean())))
    write_csv("followup_cone_stationary_summary.csv", pooled)
    append_result("stationary-content", "### 2. Direct cone-stationarity check — observation-only tokens\n\n" +
                  table(pooled, ["source", "region", "pairs", "same_cell_ssd", "ego_ssd", "ratio_of_means", "same_cell_better"]) +
                  "\n\nK=8, empty-room forward pairs only. SSD sums channels then averages shared valid cells. Both hypotheses use identical overlap, excluding newly visible edge cells. Same-cell matches imply apparent intrinsic (+1,0) after adding forward ego motion. This identifies cone-stationary imprint structure; the test does not semantically separate walls from floor texture.")


def main():
    initialize_report()
    start = time.perf_counter()
    for split, n in (("k3d_null_noblocks", 50), ("k3d_v_validation", 200)):
        paths = clips(split, n)
        for mode in ("subtract", "mask"):
            for region in ("all", "near", "far"):
                if not (mode == "subtract" and region == "all"):
                    run(paths, mode=mode, region=region)
                score(paths, mode, region)
    stationary_content(clips("k3d_null_noblocks", 50))
    seconds = time.perf_counter()-start
    write_csv("followup_breakdown_time.csv", [dict(seconds=seconds)])
    log(f"DONE breakdown {seconds:.2f}s")


if __name__ == "__main__":
    main()
