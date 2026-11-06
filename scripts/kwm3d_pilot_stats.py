#!/usr/bin/env python3
"""Addendum 2 paired clip bootstrap and pre-registered pilot decision."""
import argparse
from pathlib import Path
import numpy as np

SEED = 20261105
RESAMPLES = 10000
CLIPS = 1000


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
    assert np.asarray(dump["dataset_indices"]).shape == (len(e),), "Dataset index count differs"
    assert len(set(dump["paths"])) == len(e), "Duplicate clip paths"
    assert len(set(zip(dump["paths"], map(int, dump["dataset_indices"])))) == len(e)
    blocks = [j for j, name in enumerate(names) if name != "mse"]
    assert np.array_equal(np.asarray(dump["block_pixel_count"]) * 3, n[:, blocks]), "Block pixel counts differ"
    return e, n


def validate_protocol(dump, horizon, smoke=False):
    stats = validate(dump)
    assert dump["n_frames"] == horizon and dump["context_frames"] == 70, "Horizon/context mismatch"
    assert smoke or len(dump["paths"]) == CLIPS, "Pilot dumps must contain exactly 1,000 clips (use --smoke for plumbing checks)"
    return stats


def matched(c, other, label, horizon, smoke=False, subset=False):
    """Validate identities/counts, subsetting only comparison predictors in smoke mode."""
    validate_protocol(other, horizon, smoke)
    if smoke and subset:
        lookup = {path: i for i, path in enumerate(other["paths"])}
        missing = set(c["paths"]) - lookup.keys()
        assert not missing, f"{label} missing arm paths: {sorted(missing)}"
        ids = [lookup[path] for path in c["paths"]]
        other = dict(other, paths=[other["paths"][i] for i in ids],
                     **{k: np.asarray(other[k])[ids] for k in
                        ("dataset_indices", "squared_error", "element_count", "block_pixel_count")})
    assert c["paths"] == other["paths"], f"{label} clips/order differ"
    assert np.array_equal(c["dataset_indices"], other["dataset_indices"]), f"{label} dataset indices differ"
    assert c["metric_names"] == other["metric_names"], f"{label} metric names differ"
    assert np.array_equal(c["element_count"], other["element_count"]), f"{label} region/full-frame counts differ"
    assert np.array_equal(c["block_pixel_count"], other["block_pixel_count"]), f"{label} block pixel counts differ"
    return validate(other)


def bootstrap(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    means = np.concatenate([values[rng.integers(0, len(values), (min(250, RESAMPLES-i), len(values)))].mean(1)
                            for i in range(0, RESAMPLES, 250)])
    return tuple(np.percentile(means, [2.5, 97.5]))


def compare(c, a, baseline, horizon=140, *, smoke=False):
    ce, cn = validate_protocol(c, horizon, smoke)
    ae, an = matched(c, a, "Arm", horizon, smoke)
    be, bn = matched(c, baseline, "Baseline", horizon, smoke, subset=True)
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
    if smoke:
        return rows, None
    if primary["a_oracle"] >= primary["baseline"] and primary["c"] >= primary["baseline"]:
        verdict = "INCONCLUSIVE"
    elif primary["a_oracle"] <= .90 * primary["c"] and primary["ci_high"] < 0:
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return rows, verdict


def reference_rows(c, a, references, horizon=140, *, smoke=False):
    """Reported-only values; never passed to the registered decision function."""
    ce, cn = validate_protocol(c, horizon, smoke)
    ae, an = matched(c, a, "Arm", horizon, smoke)
    rows = []
    for name, dump in references.items():
        re, rn = matched(c, dump, f"Reference {name}", horizon, smoke, subset=True)
        for metric in ("block_mse", "mse"):
            j = c["metric_names"].index(metric)
            pool = lambda e, n: float(e[:, j].sum()/n[:, j].sum()) if n[:, j].sum() else float("nan")
            cp, ap, rp = pool(ce, cn), pool(ae, an), pool(re, rn)
            rows.append(dict(reference=name, metric=metric, c=cp, a_oracle=ap, value=rp,
                             c_below=cp < rp, a_oracle_below=ap < rp))
    return rows


def markdown(rows, verdict, horizon, c, baseline, references=(), *, smoke=False):
    heading = (f"SMOKE {horizon}: plumbing check only; no pilot decision" if smoke else
               f"Pilot {horizon}: **{verdict}**" + (" (secondary; primary decision requires horizon 140)" if horizon != 140 else ""))
    baseline_count = len(c['paths']) if smoke else len(baseline['paths'])
    lines = [heading, "",
             f"Arm clips: {len(c['paths'])}; baseline clips used: {baseline_count} (source: {len(baseline['paths'])}). "
             f"10,000 paired clip resamples, numpy Generator seed {SEED}, percentile 95% CI.", "",
             "| Metric | C pooled | A-oracle pooled | Repeat pooled | Reduction | Mean A−C | 95% CI | Paired | Excluded |",
             "|---|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['c']:.9g} | {r['a_oracle']:.9g} | {r['baseline']:.9g} | "
                     f"{r['reduction']:.3%} | {r['paired_mean']:.9g} | [{r['ci_low']:.9g}, {r['ci_high']:.9g}] | {r['paired']} | {r['excluded']} |")
    lines += ["", "One seed per arm: this interval measures clip sampling uncertainty, not training seed variance."]
    if references:
        lines += ["", "Reported-only reference, not part of the criterion (at either horizon).", "",
                  "| Reference | Metric | C pooled | A-oracle pooled | Reference pooled | C below reference | A-oracle below reference |",
                  "|---|---|---:|---:|---:|---|---|"]
        for r in references:
            lines.append(f"| {r['reference']} | {r['metric']} | {r['c']:.9g} | {r['a_oracle']:.9g} | "
                         f"{r['value']:.9g} | {r['c_below']} | {r['a_oracle_below']} |")
    return "\n".join(lines) + "\n"


def sign_disagreement(c, a, other_c, other_a, *, smoke=False):
    """Reported diagnostic only; the primary decision always uses 140 frames."""
    assert {c["n_frames"], other_c["n_frames"]} == {140, 280}
    signs = []
    for cd, ad in ((c, a), (other_c, other_a)):
        ce, cn = validate_protocol(cd, cd["n_frames"], smoke)
        ae, an = matched(cd, ad, "Arm", cd["n_frames"], smoke)
        j = cd["metric_names"].index("block_mse")
        assert cn[:, j].sum() > 0
        signs.append(np.sign(ce[:, j].sum()/cn[:, j].sum() - ae[:, j].sum()/an[:, j].sum()))
    return bool(signs[0] != signs[1])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c", required=True)
    p.add_argument("--a_oracle", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--horizon", type=int, choices=(140, 280), default=140)
    p.add_argument("--output", type=Path)
    p.add_argument("--smoke", action="store_true", help="allow small arm samples and subset predictors by path; never emit a verdict")
    p.add_argument("--reference", action="append", default=[], metavar="NAME=DUMP",
                   help="reported-only predictor (repeatable); never used in the criterion")
    p.add_argument("--other-c", help="optional opposite-horizon C dump for sign diagnostic")
    p.add_argument("--other-a-oracle", help="optional opposite-horizon A-oracle dump")
    args = p.parse_args(argv)
    references = {}
    for spec in args.reference:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path or name in references:
            p.error("--reference requires unique NAME=DUMP entries")
        references[name] = read_dump(path)
    c, a, baseline = [read_dump(x) for x in (args.c, args.a_oracle, args.baseline)]
    rows, verdict = compare(c, a, baseline, args.horizon, smoke=args.smoke)
    reported = reference_rows(c, a, references, args.horizon, smoke=args.smoke)
    result = markdown(rows, verdict, args.horizon, c, baseline, reported, smoke=args.smoke)
    if bool(args.other_c) != bool(args.other_a_oracle):
        p.error("provide both opposite-horizon dumps")
    if args.other_c:
        disagree = sign_disagreement(c, a, read_dump(args.other_c), read_dump(args.other_a_oracle), smoke=args.smoke)
        result += f"\nHorizon sign disagreement (pooled block_mse reduction): **{disagree}**.\n"
    dest = args.output or (Path(args.c) if Path(args.c).is_dir() else Path(args.c).parent) / f"pilot_stats_{args.horizon}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(result)
    print(result)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
