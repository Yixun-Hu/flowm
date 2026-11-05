"""Resumable stage-one follow-up, with isolated artifacts and incremental report."""
import argparse
from collections import defaultdict
from datetime import datetime
import time

import numpy as np
import torch
from kwm.collision import named_nodes

from .common import ROOT, SCRATCH, RESULTS, clips, clip_id, truth, write_csv, log
from .geometry import cone_mask, visible, velocity_to_map
from .lift import read_cache, projected_box
from .measurement import NodeTracker, LOCK_AGREEMENT
from .estimator_gate import intrinsic_incoming
from .matching import match_velocities, summary, SPEED_BINS

DOC = ROOT / "experiments/flowm_bench/docs/K3D_stage1_results.md"
VERSION = 2
OBS = "observation-only tokens"
MEM = "memory post-gate max-V tokens"


def source_name(source):
    return OBS if source == "obs" else MEM


def append_result(key, content):
    """Append only; never rewrite the original stage-one report."""
    marker = f"<!-- followup:{key} -->"
    if marker not in DOC.read_text():
        with DOC.open("a") as f:
            f.write(f"\n\n{marker}\n{content.rstrip()}\n")


def table(rows, fields):
    def cell(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.5f}" if np.isfinite(x) else str(x)
        return str(x)
    return "\n".join(["| " + " | ".join(fields) + " |",
                      "| " + " | ".join(["---"] * len(fields)) + " |"] +
                     ["| " + " | ".join(cell(r.get(k, "—")) for k in fields) + " |" for r in rows])


def initialize_report():
    append_result("header", """## Follow-up — 2026-11-04

CPU only, eight threads; all clips retain frames 0–139. Original sections above are unchanged.
Follow-up uses KWM `2e73903`, fixed K=8 unless stated, and cached fp16 lifts with fp32 measurements.
Primary gate truth is now **intrinsic_incoming**; literal outgoing camera-relative metadata is secondary.
Null alarm remains normalized LK agreement ≥.1 and immediate measured speed >.15, divided by all 17 node-steps.
Null PASS requires alarm ≤2% and evidence-weighted speed ≤.05 on both nulls at the gate-selected sigma.
Gate PASS requires Hungarian matched lower median ≤.1 and misses ≤30% in speeds .25–1 on nonturn pairs.
Evidence diagnostics retain every node, before and after mode selection, irrespective of locking or assignment.
Follow-up artifacts have the `followup_` prefix; trajectories are under `.scratch/followup_trajectories/`.
Results below are appended as each measurement completes. Fractions, including alarm/miss rates, are in [0,1].""")


def support_for(region):
    support = cone_mask()[7:26, 16:33].clone()
    x = torch.arange(17)[None]
    if region == "near":
        support &= x <= 6
    elif region == "far":
        support &= x >= 7
    elif region != "all":
        raise ValueError(region)
    return support


def memory_cache_path(path):
    return SCRATCH / "lifts_mem" / path.parent.parent.name / (clip_id(path).replace("/", "_") + ".pt")


def get_lift(path, source):
    if source == "obs":
        return read_cache(path)
    data = torch.load(memory_cache_path(path), map_location="cpu", weights_only=True)
    assert data["version"] == 1 and data["tokens"].shape == (140, 177, 256)
    return data


def trajectory(path, source="obs", mode="mask", sigma=1., region="all"):
    return (SCRATCH / "followup_trajectories" / path.parent.parent.name /
            f"{clip_id(path).replace('/', '_')}_{source}_{mode}_s{sigma:g}_{region}_v{VERSION}.pt")


@torch.inference_mode()
def run(paths, source="obs", mode="mask", sigma=1., region="all", batch_size=8):
    for lo in range(0, len(paths), batch_size):
        batch = [p for p in paths[lo:lo+batch_size] if not trajectory(p, source, mode, sigma, region).exists()]
        if not batch:
            continue
        start = time.perf_counter()
        caches = [get_lift(p, source) for p in batch]
        boxes = torch.stack([projected_box(c["tokens"], 8) for c in caches])
        actions = torch.stack([c["actions"] for c in caches])
        del caches
        tracker = NodeTracker(len(batch), sigma, static_residual=mode,
                              support_mask=support_for(region), return_diagnostics=True)
        history = defaultdict(list)
        for t in range(1, 140):
            for name, value in tracker.step(boxes[:, t-1], boxes[:, t], actions[:, t]).items():
                history[name].append(value.clone())
        history = {k: torch.stack(v, 1) for k, v in history.items()}
        elapsed = time.perf_counter() - start
        for i, path in enumerate(batch):
            dest = trajectory(path, source, mode, sigma, region)
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = {name: value[i] for name, value in history.items()}
            data.update(actions=actions[i], seconds=elapsed/len(batch), source=source,
                        mode=mode, sigma=sigma, region=region, version=VERSION)
            torch.save(data, dest.with_suffix(".tmp"))
            dest.with_suffix(".tmp").replace(dest)
        log(f"followup {source} {mode} s={sigma:g} {region} {paths[0].parent.parent.name} "
            f"{lo+len(batch)}/{len(paths)} {elapsed:.2f}s")


def load_trajectory(path, **config):
    return torch.load(trajectory(path, **config), map_location="cpu", weights_only=True)


def null_score(paths, source, sigma):
    rows = []
    for path in paths:
        d = load_trajectory(path, source=source, sigma=sigma)
        keep = ~d["turns"]
        speed = d["measured"][keep].norm(dim=-1)
        e = d["evidence"][keep]
        alarm = (d["agreement"][keep] >= LOCK_AGREEMENT) & (speed > .15)
        rows.append(dict(source=source_name(source), split=path.parent.parent.name, sigma=sigma,
                         clip=clip_id(path), steps=int(keep.sum()), turns=int(d["turns"].sum()),
                         alarms=int(alarm.sum()), node_steps=alarm.numel(), evidence=float(e.sum()),
                         weighted_speed=float((e*speed).sum()), seconds=d["seconds"]))
    split = paths[0].parent.parent.name
    write_csv(f"followup_null_{source}_{split}_s{sigma:g}_clips.csv", rows)
    r = dict(source=source_name(source), split=split, sigma=sigma, clips=len(paths))
    for k in ("steps", "turns", "alarms", "node_steps", "evidence", "weighted_speed", "seconds"):
        r[k] = sum(x[k] for x in rows)
    r["alarm_rate"] = r["alarms"] / r["node_steps"]
    r["moving_node_alarm_rate"] = r["alarms"] / (16*r["steps"])
    r["evidence_weighted_speed"] = r["weighted_speed"] / max(r["evidence"], 1e-6)
    r["verdict"] = ("contrast" if "int_" in split else
                    "PASS" if r["alarm_rate"] <= .02 and r["evidence_weighted_speed"] <= .05 else "FAIL")
    write_csv(f"followup_null_{source}_{split}_s{sigma:g}_summary.csv", [r])
    append_result(f"null-{source}-{split}-{sigma:g}",
                  f"### {'1' if source == 'obs' else '4'}. Mask null / contrast — {source_name(source)}\n\n" +
                  table([r], ["source", "split", "clips", "sigma", "steps", "alarms", "alarm_rate",
                              "evidence_weighted_speed", "verdict"]) +
                  "\n\nCriterion: alarm ≤.02 and weighted speed ≤.05; K3D-int is a moving contrast, with no null criterion.")
    log(str(r))
    return r


def evidence_step(evidence, nodes, target):
    """All-node selectivity; wrong means >.5 from EVERY visible true velocity."""
    distance = torch.cdist(target, nodes)
    nearest = distance.argmin(-1)
    maximum = evidence.max()
    wins = (evidence[nearest] == maximum) & (maximum > 0)
    right = evidence[nearest].max()
    wrong = evidence[distance.min(0).values > .5].max() if (distance.min(0).values > .5).any() else evidence.new_tensor(0.)
    ratio = (float(wrong/right) if right > 0 else float("inf") if wrong > 0 else float("nan"))
    return dict(objects=len(target), wins=int(wins.sum()), any_win=bool(wins.any()),
                all_win=bool(wins.all()), nearest_evidence=float(right), wrong_evidence=float(wrong),
                best_wrong_ratio=ratio)


def gate_score(paths, source, sigma):
    pooled = defaultdict(lambda: dict(error=[], miss=[], false=[]))
    clip_rows, diagnostics = [], []
    node_totals = {name: torch.zeros(17) for name in ("evidence", "evidence_before_modes", "locked")}
    nsteps = 0
    lattice = named_nodes("v17")
    for path in paths:
        d = load_trajectory(path, source=source, sigma=sigma)
        labels = truth(path)
        vis = visible(labels["block_pos_agent"][:140])
        local = defaultdict(lambda: dict(error=[], miss=[], false=[]))
        for ti in range(139):
            if d["turns"][ti]:
                continue
            t = ti+1
            nsteps += 1
            for name in node_totals:
                node_totals[name] += d[name][ti].float()
            take = vis[t] & vis[t-1]
            nodes = d["velocity"][ti][d["active"][ti]].numpy()
            for mode in ("intrinsic_incoming", "literal"):
                target = velocity_to_map((intrinsic_incoming(labels, t) if mode == "intrinsic_incoming"
                                          else labels["block_vel_agent"][t])[take])
                speeds = (labels["block_speed"][t-1][take] if mode == "intrinsic_incoming"
                          else target.norm(dim=-1)).numpy()
                result = match_velocities(nodes, target.numpy())
                bins = [("all", np.ones(len(speeds), bool)), ("0.25-1", (speeds >= .25-1e-6) & (speeds <= 1+1e-6))]
                bins += [(f"{a:g}-{min(b,1):g}", (speeds >= a) & (speeds < b)) for a,b in SPEED_BINS]
                for name, mask in bins:
                    value = local[mode, name]
                    value["error"].extend(result["error"][mask])
                    value["miss"].extend(result["miss"][mask])
                    ns = np.linalg.norm(nodes, axis=-1)
                    if name in ("all", "0.25-1"):
                        nm = np.ones(len(nodes), bool)
                    else:
                        a,b = map(float, name.split("-"))
                        nm = (ns >= a) & (ns < b + (1e-6 if b == 1 else 0))
                    value["false"].extend(result["false"][nm])
                if len(target):
                    for basis, candidates in (("lattice", lattice), ("prior_adaptive", d["prior"][ti])):
                        for field in ("evidence", "evidence_before_modes"):
                            diagnostics.append(dict(source=source_name(source), sigma=sigma, clip=clip_id(path),
                                                    frame=t, truth_mode=mode, node_basis=basis, field=field,
                                                    **evidence_step(d[field][ti], candidates, target)))
        for (mode, name), values in local.items():
            for k, v in values.items():
                pooled[mode, name][k].extend(v)
            clip_rows.append(dict(source=source_name(source), sigma=sigma, clip=clip_id(path), truth_mode=mode,
                                  speed_bin=name, **summary(values["error"], values["miss"]),
                                  false_alarm_rate=float(np.mean(values["false"])) if values["false"] else float("nan")))
    rows = []
    for (mode, name), values in pooled.items():
        r = dict(source=source_name(source), sigma=sigma, truth_mode=mode, speed_bin=name,
                 **summary(values["error"], values["miss"]),
                 false_alarm_rate=float(np.mean(values["false"])) if values["false"] else float("nan"))
        r["verdict"] = "PASS" if r["median"] <= .1 and r["miss_rate"] <= .3 else "FAIL"
        rows.append(r)
    prefix = f"followup_gate_{source}_s{sigma:g}"
    write_csv(prefix+"_summary.csv", rows)
    write_csv(prefix+"_clips.csv", clip_rows)
    write_csv(prefix+"_evidence_steps.csv", diagnostics)
    evidence_rows = []
    for mode in ("intrinsic_incoming", "literal"):
        for basis in ("lattice", "prior_adaptive"):
            for field in ("evidence", "evidence_before_modes"):
                rr = [r for r in diagnostics if r["truth_mode"] == mode and r["node_basis"] == basis and r["field"] == field]
                ratios = np.array([r["best_wrong_ratio"] for r in rr])
                valid = ratios[~np.isnan(ratios)]
                finite = ratios[np.isfinite(ratios)]
                evidence_rows.append(dict(source=source_name(source), sigma=sigma, truth_mode=mode,
                    node_basis=basis, field=field, steps=len(rr),
                    any_nearest_top=float(np.mean([r["any_win"] for r in rr])),
                    all_nearest_top=float(np.mean([r["all_win"] for r in rr])),
                    object_nearest_top=sum(r["wins"] for r in rr)/max(1,sum(r["objects"] for r in rr)),
                    mean_best_wrong_ratio=float(valid.mean()) if len(valid) else float("nan"),
                    finite_mean_ratio=float(finite.mean()) if len(finite) else float("nan"),
                    infinite_ratios=int(np.isinf(ratios).sum()), undefined_0_0=int(np.isnan(ratios).sum())))
    write_csv(prefix+"_evidence_summary.csv", evidence_rows)
    node_rows = [dict(source=source_name(source), sigma=sigma, node=i, vx=float(lattice[i,0]), vy=float(lattice[i,1]),
                      **{name: float(total[i]/nsteps) for name,total in node_totals.items()}) for i in range(17)]
    write_csv(prefix+"_nodes.csv", node_rows)
    append_result(f"gate-{source}-{sigma:g}",
        f"### {'1' if source == 'obs' else '4'}. Mask estimator gate — {source_name(source)}, σ={sigma:g}\n\n" +
        table(rows, ["source", "truth_mode", "speed_bin", "truths", "assigned", "median", "p90", "miss_rate", "false_alarm_rate", "verdict"]) +
        "\n\nCriterion applies to intrinsic_incoming, speed .25–1: median ≤.1 and miss ≤.30. Other rows are diagnostics.\n\n" +
        "All-node selectivity on nonturn steps with jointly visible blocks. `any_nearest_top` is the fraction of steps where at least one truth's nearest node has positive maximum evidence; `all_nearest_top` requires all truths' nearest nodes to tie that maximum. `object_nearest_top` counts object-step pairs. Nearest nodes use the fixed lattice or pre-update adaptive velocities as labelled. Wrong nodes lie >.5 from **every** visible truth; ratio is their largest evidence divided by the largest nearest-true-node evidence. Zero denominator with positive wrong mass is infinity; 0/0 is undefined and excluded. Both counts are explicit.\n\n" +
        table(evidence_rows, ["source", "truth_mode", "node_basis", "field", "steps", "any_nearest_top", "all_nearest_top", "object_nearest_top", "mean_best_wrong_ratio", "finite_mean_ratio", "infinite_ratios", "undefined_0_0"]) +
        "\n\nMean evidence and lock fraction on **all** nonturn node-steps (no truth selection):\n\n" +
        table(node_rows, ["source", "node", "vx", "vy", "evidence", "evidence_before_modes", "locked"]) +
        f"\n\nPer-step evidence comparisons: `results/{prefix}_evidence_steps.csv`. All 17-node arrays and estimates at every step are retained in the versioned trajectory tensors.")
    log(str([r for r in rows if r["speed_bin"] == "0.25-1"]))
    return rows


def mask_campaign(source="obs", vclips=200):
    initialize_report()
    sigmas = [.5, 1.] if source == "obs" else [1.]
    splits = [("k3d_null_noblocks", 50), ("k3d_null_static", 50)]
    if source == "obs":
        splits.append(("k3d_int_validation", 20))
    for split, n in splits:
        paths = clips(split, n)
        for sigma in sigmas:
            run(paths, source=source, sigma=sigma)
            null_score(paths, source, sigma)
    paths = clips("k3d_v_validation", vclips)
    for sigma in sigmas:
        run(paths, source=source, sigma=sigma)
        gate_score(paths, source, sigma)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("item", choices=["mask", "memory-score"])
    args = p.parse_args()
    start = time.perf_counter()
    mask_campaign("obs" if args.item == "mask" else "mem", 200 if args.item == "mask" else 100)
    elapsed = time.perf_counter()-start
    write_csv(f"followup_{args.item}_time.csv", [dict(item=args.item, seconds=elapsed,
              finished=datetime.now().astimezone().isoformat())])
    log(f"DONE {args.item} {elapsed:.2f}s")


if __name__ == "__main__":
    main()
