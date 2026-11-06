#!/usr/bin/env python3
"""Render the separate, reported-only patch-motion diagnostic (CPU, no torch)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

KS = (8, 32, 256)
NOTICE = "Reported only: not part of the pilot criterion; the verdict does not read this file"
GROUPS = (("all", ("all",)),
          ("speed", (".25-.5", ".5-.75", ".75-1")),
          ("distance", ("[1,5)", "[5,9)", "[9,13)", "[13,17)")))


def number(value):
    value = float(value)
    if np.isnan(value):
        return "n/a"
    if np.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.6f}"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def named(value):
    name, sep, text = value.partition("=")
    if not sep or not name.strip() or not text.strip():
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    return name, text


def read_summaries(directory, prefix):
    rows = {}
    for k in KS:
        path = directory / f"{prefix}_k{k}_summary.csv"
        with path.open(newline="") as f:
            records = list(csv.DictReader(f))
        for row in records:
            key = (k, row["group"], row["bin"])
            if int(row["k"]) != k or key in rows:
                raise ValueError(f"Invalid or duplicate summary row in {path}: {key}")
            rows[key] = row
        for group, bins in GROUPS:
            for name in bins:
                if (k, group, name) not in rows:
                    raise ValueError(f"Missing {group}/{name} summary in {path}")
    return rows


def render(released_results, runs, missing, failed=(), *, allow_partial=False):
    released = read_summaries(released_results, "followup_patch_tokens")
    sources = [("frozen released lift (stage 1)", released)]
    provenance = []
    partial = False
    for name, directory in runs:
        directory = Path(directory)
        info = json.loads((directory / "patch_run.json").read_text())
        rows = read_summaries(directory, "patch")
        if info["pair_set_matches_stage1"] is not True:
            raise ValueError(f"{name}: pair-set check did not succeed")
        if info["Ks"] != list(KS):
            raise ValueError(f"{name}: expected Ks {KS}")
        if any(int(rows[k, "all", "all"]["pairs"]) != info["pairs"] for k in KS):
            raise ValueError(f"{name}: summary pair count disagrees with patch_run.json")
        for k in KS:
            pairs = int(rows[k, "all", "all"]["pairs"])
            expected = int(released[k, "all", "all"]["pairs"])
            if pairs != expected:
                if not allow_partial:
                    raise ValueError(f"{name}: K={k} pair count {pairs} differs from stage 1 ({expected}); "
                                     "use --allow-partial only for rehearsals")
                partial = True
        sources.append((name, rows))
        provenance.append((name, info))
    lines = [NOTICE]
    if partial:
        lines.append("PARTIAL RUN: pair counts differ from stage 1; not comparable")
    lines += ["",
              "The C row and the stage 1 row use the per-node write (attention over V identical copies of the zero map); "
              "the A-oracle row uses its V-pooled single-copy write, which differs from the per-node write even at identical weights, "
              "so compare A-oracle with the released pooled-write row and not only with the stage 1 row.",
              "The A-oracle write was trained only through reconstruction with oracle velocities, never through an estimator.",
              "", "Fractions are in [0, 1]; errors are in intrinsic cells. Unusable pairs count as infinite error."]
    for group, bins in GROUPS:
        title = {"all": "All pairs", "speed": "Speed bins", "distance": "Distance bins"}[group]
        lines += ["", title, ""]
        bin_header = " Bin |" if group != "all" else ""
        lines += [f"| Source | K |{bin_header} Pairs | Usable | Median error (cells) | p90 | Fraction within 0.25 | Boundary fraction |",
                  "|---|---:|" + ("---|" if group != "all" else "") + "---:|---:|---:|---:|---:|---:|"]
        for source, rows in sources:
            for k in KS:
                for name in bins:
                    r = rows[k, group, name]
                    values = [cell(source), str(k)] + ([cell(name)] if group != "all" else [])
                    values += [str(int(r["pairs"])), str(int(r["usable"]))]
                    values += [number(r[key]) for key in ("median", "p90", "within_025", "boundary_fraction")]
                    lines.append("| " + " | ".join(values) + " |")
    for name, info in provenance:
        lines += ["", f"{cell(name)}: checkpoint `{cell(info['checkpoint'])}`; sha256 `{cell(info['sha256'])}`; "
                  f"pair_set_matches_stage1: true; pairs: {info['pairs']}; "
                  f"clips: {info['clips']}; frames: {info['frames']}.",
                  f"Lift: {cell(info['lift_definition'])}."]
    for name, reason in missing:
        lines += ["", f"{cell(name)}: test not run — {cell(reason)}"]
    for name, directory in failed:
        path = Path(directory) / "patch_run.json"
        info = json.loads(path.read_text()) if path.exists() else {}
        reason = info.get("error") or "no failure record; see the log"
        lines += ["", f"{cell(name)}: test failed — {cell(reason)}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--released-results", type=Path, default=Path("experiments/flowm_bench/k3d/results"))
    p.add_argument("--run", type=named, action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--missing", type=named, action="append", default=[], metavar="NAME=TEXT")
    p.add_argument("--failed", type=named, action="append", default=[], metavar="NAME=DIR")
    p.add_argument("--allow-partial", action="store_true", help="allow mismatched pair counts for rehearsals only")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    # Check both the requested name and symlink target before reading or writing.
    if any(path.name.startswith("pilot_stats") for path in (args.output, args.output.resolve())):
        p.error("--output must not name a pilot_stats file")
    names = [name for name, _ in args.run + args.missing + args.failed]
    if len(set(names)) != len(names):
        p.error("--run, --missing and --failed names must be unique")
    result = render(args.released_results, args.run, args.missing, args.failed, allow_partial=args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)


if __name__ == "__main__":
    main()
