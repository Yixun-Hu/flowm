"""Background-null alarms, raw static-block velocities, and integer-motion contrast."""
import argparse
import numpy as np
import torch
from .common import clips, clip_id, write_csv
from .measurement import LOCK_AGREEMENT
from .estimator_gate import run_trajectories, trajectory_path


def score(paths, variants, sigmas):
    rows, distributions, raw = [], [], []
    for variant in variants:
        for sigma in sigmas:
            pooled = []
            for path in paths:
                data = torch.load(trajectory_path(path, variant, 8, sigma), weights_only=True)
                keep = ~data["turns"]
                # Report both all-v17 and the stricter moving-v16 denominator.
                v = data["measured"][keep, 1:]
                evidence = data["evidence"][keep, 1:]
                agreement = data["agreement"][keep, 1:]
                speed = v.norm(dim=-1)
                alarm = (agreement >= LOCK_AGREEMENT) & (speed > .15)
                row = dict(split=path.parent.parent.name, variant=variant, sigma=sigma, clip=clip_id(path),
                           node_steps=int(keep.sum())*17, moving_node_steps=alarm.numel(),
                           alarms=int(alarm.sum()), alarm_rate=float(alarm.sum())/(int(keep.sum())*17),
                           moving_node_alarm_rate=float(alarm.float().mean()),
                           evidence_sum=float(evidence.sum()), weighted_speed_sum=float((evidence*speed).sum()),
                           evidence_weighted_speed=float((evidence*speed).sum()/evidence.sum().clamp_min(1e-6)),
                           turn_steps=int(data["turns"].sum()), nonturn_steps=int(keep.sum()), seconds=data["seconds"])
                rows.append(row)
                pooled.append((v.flatten(0, 1), evidence.flatten(), agreement.flatten()))
                if path.parent.parent.name == "k3d_null_static":
                    times = keep.nonzero().flatten()+1
                    for ti, t in enumerate(times.tolist()):
                        for ni in range(16):
                            raw.append(dict(variant=variant, sigma=sigma, clip=clip_id(path), frame=t, node=ni+1,
                                            vx=float(v[ti,ni,0]), vy=float(v[ti,ni,1]), speed=float(speed[ti,ni]),
                                            evidence=float(evidence[ti,ni]), agreement=float(agreement[ti,ni]),
                                            alarm=bool(alarm[ti,ni])))
            vel = torch.cat([z[0] for z in pooled]).numpy()
            weight = torch.cat([z[1] for z in pooled]).numpy()
            agreement = torch.cat([z[2] for z in pooled]).numpy()
            for field, values in (("vx", vel[:,0]), ("vy", vel[:,1]), ("speed", np.linalg.norm(vel, axis=1))):
                for subset, mask in (("all", np.ones(len(vel), bool)), ("lock_evidence", agreement >= LOCK_AGREEMENT)):
                    x = values[mask]
                    for q in [0, .01, .05, .25, .5, .75, .9, .95, .99, 1]:
                        distributions.append(dict(split=paths[0].parent.parent.name, variant=variant, sigma=sigma,
                                                  field=field, subset=subset, quantile=q, n=len(x),
                                                  value=float(np.quantile(x,q)) if len(x) else float("nan")))
    split = paths[0].parent.parent.name
    write_csv(f"null_{split}_clips.csv", rows)
    write_csv(f"null_{split}_distribution.csv", distributions)
    if raw:
        write_csv("null_static_raw_velocities.csv", raw)
    summary = []
    for variant in variants:
        for sigma in sigmas:
            rr = [r for r in rows if r["variant"] == variant and r["sigma"] == sigma]
            row = dict(split=split, variant=variant, sigma=sigma, clips=len(rr))
            for field in ("node_steps", "moving_node_steps", "alarms", "evidence_sum", "weighted_speed_sum", "turn_steps", "nonturn_steps", "seconds"):
                row[field] = sum(r[field] for r in rr)
            row["alarm_rate"] = row["alarms"] / row["node_steps"]
            row["moving_node_alarm_rate"] = row["alarms"] / row["moving_node_steps"]
            row["evidence_weighted_speed"] = row["weighted_speed_sum"] / max(row["evidence_sum"],1e-6)
            row["passed"] = row["alarm_rate"] <= .02 and row["evidence_weighted_speed"] <= .05
            summary.append(row)
    write_csv(f"null_{split}_summary.csv", summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="k3d_null_noblocks")
    p.add_argument("--clips", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--variants", nargs="+", default=["tokens", "h_new"])
    p.add_argument("--sigmas", type=float, nargs="+", default=[.5,1.,2.])
    p.add_argument("--score-only", action="store_true")
    args = p.parse_args()
    paths = clips(args.split, args.clips)
    if not args.score_only:
        for variant in args.variants:
            for sigma in args.sigmas:
                run_trajectories(paths, variant, 8, sigma, args.batch_size)
    score(paths, args.variants, args.sigmas)


if __name__ == "__main__":
    main()
