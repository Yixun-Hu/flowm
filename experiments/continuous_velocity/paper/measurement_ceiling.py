"""CPU-only perfect-prediction ceilings for every evaluation suite and horizon.

The generated target frames themselves are evaluated as predictions. This tests
the evaluator, not a model. Cases, seed counts, renderers, and contexts exactly
reuse the formal suite generator. Detector thresholds are not tuned.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.evaluate import SUITES, cases_for_seed
from paper.metrics import ROOT, METRICS_VERSION, evaluate_case, summarize


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_csv(path, rows):
    if rows:
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/continuous_velocity/paper_v1')
    parser.add_argument('--suites', nargs='+', default=list(SUITES))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    source_paths = [Path(__file__), Path(__file__).parent/'metrics.py', Path(__file__).parent/'evaluate.py',
                    Path(__file__).parent/'data.py', ROOT/'experiments/fractional_velocity/measurement.py']
    manifest = dict(complete=False, metrics_version=METRICS_VERSION, suites=args.suites,
                    predictions='exact ground-truth target frames; no neural model used', device='cpu',
                    detector_thresholds_changed=False, context_anchor='last present and non-ghost context detection; initial winding chosen once',
                    continuity_speed_bound_per_axis=2.5, continuity_localization_slack=1.,
                    continuity_policy='Permanent unknown winding when a gap permits half-world traversal or no unique bounded increment exists',
                    suite_specifications={name: SUITES[name] for name in args.suites},
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    manifest_path = args.out/'measurement_ceiling.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    rows = []; began = time.monotonic()
    for suite in args.suites:
        for seed in range(SUITES[suite]['seeds']):
            for case in cases_for_seed(seed, suite):
                prediction = case['frames'][case['target_indices']]
                measured, _ = evaluate_case(case, prediction)
                for row in measured:
                    row.update(suite=suite, direction_split=case['direction_split'])
                rows.extend(measured)
            if (seed+1) % 4 == 0:
                print(json.dumps(dict(suite=suite, seed=seed, rows=len(rows), seconds=round(time.monotonic()-began, 1))), flush=True)
        write_csv(args.out/'measurement_ceiling.csv', rows)
    grouped = {}
    for row in rows:
        key = (row['suite'], row['mode'], row['scenario'], row['direction_split'], row['horizon'])
        grouped.setdefault(key, []).append(row)
    summaries = [dict(suite=key[0], mode=key[1], scenario=key[2], direction_split=key[3], horizon=key[4],
                      **summarize(group)) for key, group in sorted(grouped.items())]
    manifest.update(complete=True, elapsed_seconds=time.monotonic()-began, rows=len(rows),
                    model_sequences=len({(r['suite'], r['case_id']) for r in rows}), summaries=summaries,
                    interpretation='Perfect-frame prediction can fail CV certification or conservative continuity identification. Frame-localization success and MSE are primary; ceiling differences quantify evaluator limitations.')
    manifest_path.write_text(json.dumps(clean(manifest), indent=2, allow_nan=False))
    print(json.dumps(dict(complete=True, rows=len(rows), groups=len(summaries), seconds=manifest['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
