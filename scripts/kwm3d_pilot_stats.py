#!/usr/bin/env python3
"""Addendum 2 paired clip bootstrap and pre-registered pilot decision."""
import argparse
from pathlib import Path
import numpy as np

SEED = 20261105
RESAMPLES = 10000


def read_dump(path):
    import torch
    path = Path(path)
    return torch.load(path / "per_clip.pt" if path.is_dir() else path, map_location="cpu", weights_only=True)


def validate(dump):
    names = dump["metric_names"]
    e, n = [np.asarray(dump[k], dtype=np.float64) for k in ("squared_error", "element_count")]
    assert e.shape == n.shape == (len(dump["paths"]), len(names)) and len(e) > 0
    assert len(set(names)) == len(names) and "block_mse" in names and "mse" in names
    assert np.isfinite(e).all() and np.isfinite(n).all() and (e >= 0).all() and (n >= 0).all()
    assert (n == np.floor(n)).all() and (e[n == 0] == 0).all()
    assert len(set(zip(dump["paths"], map(int, dump["dataset_indices"])))) == len(e)
    return e, n


def bootstrap(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    means = np.concatenate([values[rng.integers(0, len(values), (min(250, RESAMPLES-i), len(values)))].mean(1)
                            for i in range(0, RESAMPLES, 250)])
    return tuple(np.percentile(means, [2.5, 97.5]))


def compare(c, a, baseline, horizon=140):
    ce, cn = validate(c)
    ae, an = validate(a)
    be, bn = validate(baseline)
    assert c["paths"] == a["paths"], "Arm clips/order differ"
    assert np.array_equal(c["dataset_indices"], a["dataset_indices"]), "Arm dataset indices differ"
    assert c["metric_names"] == a["metric_names"] == baseline["metric_names"]
    assert np.array_equal(cn, an), "Arm region counts differ"
    for dump in (c, a, baseline):
        assert dump["n_frames"] == horizon and dump["context_frames"] == 70, "Horizon/context mismatch"
    # The full baseline may be supplied for a smoke test on fewer arm clips.
    # Report this explicitly; real pilot commands must use the matched 1,000.
    rows = []
    for j, name in enumerate(c["metric_names"]):
        pool = lambda e, n: float(e[:, j].sum() / n[:, j].sum()) if n[:, j].sum() else float("nan")
        cp, ap, bp = pool(ce, cn), pool(ae, an), pool(be, bn)
        valid = (cn[:, j] > 0) & (an[:, j] > 0)
        difference = ae[valid, j]/an[valid, j] - ce[valid, j]/cn[valid, j]
        lower, upper = bootstrap(difference)
        rows.append(dict(metric=name, c=cp, a_oracle=ap, baseline=bp,
                         reduction=(cp-ap)/cp if cp else float("nan"),
                         paired_mean=float(difference.mean()) if len(difference) else float("nan"),
                         ci_low=lower, ci_high=upper, paired=len(difference), excluded=int((~valid).sum())))
    primary = next(row for row in rows if row["metric"] == "block_mse")
    if not all(np.isfinite(primary[k]) for k in ("c", "a_oracle", "baseline")):
        raise ValueError("Primary pooled block_mse is undefined (no block pixels)")
    if primary["a_oracle"] >= primary["baseline"] and primary["c"] >= primary["baseline"]:
        verdict = "INCONCLUSIVE"
    elif primary["a_oracle"] <= .90 * primary["c"] and primary["ci_high"] < 0:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return rows, verdict


def markdown(rows, verdict, horizon, c, baseline):
    lines = [f"Pilot {horizon}: **{verdict}**" + (" (secondary; primary decision requires horizon 140)" if horizon != 140 else ""), "",
             f"Arm clips: {len(c['paths'])}; baseline clips: {len(baseline['paths'])}. "
             f"10,000 paired clip resamples, numpy Generator seed {SEED}, percentile 95% CI.", "",
             "| Metric | C pooled | A-oracle pooled | Repeat pooled | Reduction | Mean A−C | 95% CI | Paired | Excluded |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['c']:.9g} | {r['a_oracle']:.9g} | {r['baseline']:.9g} | "
                     f"{r['reduction']:.3%} | {r['paired_mean']:.9g} | [{r['ci_low']:.9g}, {r['ci_high']:.9g}] | {r['paired']} | {r['excluded']} |")
    lines += ["", "One seed per arm: this interval measures clip sampling uncertainty, not training seed variance."]
    return "\n".join(lines) + "\n"


def sign_disagreement(c, a, other_c, other_a):
    """Reported diagnostic only; the primary decision always uses 140 frames."""
    assert {c["n_frames"], other_c["n_frames"]} == {140, 280}
    signs = []
    for cd, ad in ((c, a), (other_c, other_a)):
        ce, cn = validate(cd)
        ae, an = validate(ad)
        assert cd["paths"] == ad["paths"]
        assert np.array_equal(cd["dataset_indices"], ad["dataset_indices"])
        assert np.array_equal(cn, an)
        assert cd["n_frames"] == ad["n_frames"]
        assert cd["context_frames"] == ad["context_frames"] == 70
        j = cd["metric_names"].index("block_mse")
        assert cn[:, j].sum() > 0
        signs.append(np.sign(ce[:, j].sum()/cn[:, j].sum() - ae[:, j].sum()/an[:, j].sum()))
    return bool(signs[0] != signs[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c", required=True)
    p.add_argument("--a_oracle", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--horizon", type=int, choices=(140, 280), default=140)
    p.add_argument("--output", type=Path)
    p.add_argument("--other-c", help="optional opposite-horizon C dump for sign diagnostic")
    p.add_argument("--other-a-oracle", help="optional opposite-horizon A-oracle dump")
    args = p.parse_args()
    c, a, baseline = [read_dump(x) for x in (args.c, args.a_oracle, args.baseline)]
    rows, verdict = compare(c, a, baseline, args.horizon)
    result = markdown(rows, verdict, args.horizon, c, baseline)
    if bool(args.other_c) != bool(args.other_a_oracle):
        p.error("provide both opposite-horizon dumps")
    if args.other_c:
        disagree = sign_disagreement(c, a, read_dump(args.other_c), read_dump(args.other_a_oracle))
        result += f"\nHorizon sign disagreement (pooled block_mse reduction): **{disagree}**.\n"
    dest = args.output or (Path(args.c) if Path(args.c).is_dir() else Path(args.c).parent) / f"pilot_stats_{args.horizon}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result)
    print(result)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
