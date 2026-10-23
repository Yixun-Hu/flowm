"""Resumable paired evaluation for the kinetic study (fork of paper/evaluate.py).

Suites add constant-acceleration scenes. For every case the transport truth is
the unwrapped displacement from the last context frame, so transport errors are
valid under any motion law; velocity diagnostics compare against the true mean
velocity over each scored horizon. Only ``oracle=True`` jobs receive the true
velocity and acceleration at the last context time. Test scene seeds may be
offset (``seed_start``) for confirmatory evaluation on unused scenes.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from paper.evaluate import _atomic_csv, _atomic_json, _read_csv, _save_example, _sha  # noqa: E402
from paper.metrics import evaluate_case  # noqa: E402
from kinetic.memory import build_kinetic_model  # noqa: E402
from kinetic.scenes import ROOT, kinetic_model_batch, make_kinetic_case  # noqa: E402
from kinetic.training import SPEC_KEYS  # noqa: E402

SOURCE_PATHS = [Path(__file__), PARENT/'kinetic'/'scenes.py', PARENT/'kinetic'/'memory.py',
                PARENT/'kinetic'/'transport.py', PARENT/'kinetic'/'training.py', PARENT/'paper'/'data.py',
                PARENT/'paper'/'metrics.py', PARENT/'paper'/'mechanism.py', PARENT/'paper'/'train.py',
                PARENT/'paper'/'evaluate.py', PARENT/'model.py', PARENT/'data.py',
                ROOT/'experiments/fractional_velocity/measurement.py']
LOADED_SOURCE_HASHES = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in SOURCE_PATHS}
HELDOUT_SPEEDS = [.37, .63, 1.23]
SUITES = {
    'core': dict(speeds=[0, .37, .5, .63, 1, 1.23, 1.5, 2], accelerations=[0.], horizon=40,
                 contexts=['regular'], seeds=32),
    'time': dict(speeds=HELDOUT_SPEEDS, accelerations=[0.], horizon=160,
                 contexts=['regular'], seeds=16),
    'accel': dict(speeds=HELDOUT_SPEEDS,
                  accelerations=[0., 5e-4, -5e-4, 1e-3, -1e-3, 2e-3, -2e-3, 4e-3, -4e-3, 1.2e-2, -1.2e-2],
                  horizon=40, contexts=['regular'], seeds=16),
    'accel_long': dict(speeds=HELDOUT_SPEEDS, accelerations=[0., 1e-3, -1e-3, 4e-3, -4e-3],
                       horizon=80, contexts=['regular'], seeds=16),
}
MODES = ('bilinear', 'nearest')


def cases_for_seed(seed, suite):
    spec = SUITES[suite]
    cases = []
    for context in spec['contexts']:
        for mode in MODES:
            for speed in spec['speeds']:
                for acceleration in spec['accelerations']:
                    case = make_kinetic_case(seed, [speed, 0.], [acceleration, 0.], mode=mode,
                                             horizon=spec['horizon'], context=context)
                    case['case_id'] = (f'seed{seed:03d}__{mode}__v0{speed:+.6g}_a{acceleration:+.6g}'
                                       f'__{case["scenario"]}')
                    cases.append(case)
    return cases


def setup_model(job, tag, device):
    train_seed = int(job.get('train_seed', 2026))
    torch.manual_seed(train_seed)
    saved, checkpoint = None, None
    metadata = dict(train_seed=train_seed, train_steps=0, checkpoint=None, checkpoint_sha256=None,
                    training_config=None)
    if job.get('run_id'):
        checkpoint = (ROOT/'downloaded_checkpoints/continuous_velocity'/tag/job['run_id']
                      / job.get('checkpoint', 'budget_5000.pt'))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        spec = saved['spec']
        for key in SPEC_KEYS:
            if key in job.get('spec', {}) and job['spec'][key] != spec[key]:
                raise ValueError(f'Job spec differs from checkpoint spec for {job["name"]}: {key}')
        if int(saved['config']['seed']) != train_seed:
            raise ValueError('Job training seed differs from checkpoint training seed')
        metadata.update(train_steps=int(saved['step']), checkpoint=str(checkpoint.resolve()),
                        checkpoint_sha256=_sha(checkpoint), training_config=saved['config'])
    else:
        spec = job['spec']
    model = build_kinetic_model(spec)
    if saved is not None:
        model.load_state_dict(saved['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    zero = job.get('zero_acceleration')
    if zero not in (None, False, 'both', 'transport'):
        raise ValueError(f'Unknown zero_acceleration mode: {zero}')
    metadata.update(spec=model.spec, kind=model.spec['kind'],
                    parameters=sum(p.numel() for p in model.parameters()),
                    oracle=bool(job.get('oracle', False)), oracle_kind='velocity and acceleration at t0',
                    zero_acceleration=zero or '')
    return model, metadata


@torch.inference_mode()
def infer(model, job, cases, device):
    inputs = kinetic_model_batch(cases, device)
    if job.get('oracle', False):
        inputs['velocity_override'] = torch.as_tensor(
            np.stack([case['velocity'] for case in cases]), device=device, dtype=torch.float32)
        inputs['acceleration_override'] = torch.as_tensor(
            np.stack([case['acceleration'] for case in cases]), device=device, dtype=torch.float32)
    if job.get('zero_acceleration'):
        inputs['zero_acceleration'] = job['zero_acceleration']
    prediction, auxiliary = model(**inputs)
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Nonfinite predictions during evaluation')
    keys = ('velocity', 'acceleration', 'estimated_velocity', 'estimated_acceleration', 'offsets',
            'memory', 'acceleration_gate', 'acceleration_se', 'acceleration_raw', 'refined', 'weights')
    arrays = {key: auxiliary[key].detach().cpu().numpy() for key in keys
              if key in auxiliary and isinstance(auxiliary[key], torch.Tensor)}
    return prediction.detach().cpu().numpy(), arrays


def evaluate_job(job, suite, tag='kinetic_v1', device='cuda:1', seeds=None, seed_start=0):
    if suite not in SUITES:
        raise ValueError(f'Unknown suite: {suite}')
    job = dict(job)
    name = job['name']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('Job name must be a simple path component')
    seeds = SUITES[suite]['seeds'] if seeds is None else int(seeds)
    seed_start = int(seed_start)
    if not (0 <= seed_start and 1 <= seeds and seed_start+seeds <= 64):
        raise ValueError('Evaluation scene seeds must lie within 0:64')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_metadata = setup_model(job, tag, device)
    source_hashes = {str(path.relative_to(ROOT)): _sha(path) for path in SOURCE_PATHS}
    if source_hashes != LOADED_SOURCE_HASHES:
        raise RuntimeError('Evaluation sources changed after import; restart the Python worker')
    specification = dict(job=job, suite=suite, suite_specification=SUITES[suite],
                         metric_schema_version=2, seeds=seeds, seed_start=seed_start,
                         device=str(device), source_sha256=source_hashes, model=model_metadata,
                         tolerance=2., batch_size=8, modes=list(MODES), metrics_horizons=[20, 40, 80, 160])
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    suffix = suite if seed_start == 0 else f'{suite}_from{seed_start}'
    out = ROOT/f'outputs/continuous_velocity/{tag}/evaluation/{name}/{suffix}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'examples').mkdir(exist_ok=True)
    manifest_path = out/'manifest.json'
    rows, completed_seeds, previous_seconds = [], [], 0.
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous['signature'] != signature:
            raise ValueError(f'Evaluation signature changed for {out}; use a new job name/tag')
        if previous.get('complete'):
            del model
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
            return previous
        completed_seeds = previous.get('completed_seeds', [])
        previous_seconds = previous.get('elapsed_seconds', 0.)
        if completed_seeds:
            rows = [row for row in _read_csv(out/'metrics.csv') if row['seed'] in completed_seeds]
    manifest = dict(**specification, signature=signature, complete=False, completed_seeds=completed_seeds,
                    output_directory=str(out), started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    selected_examples='all cases for the first evaluated scene seed only',
                    metric_weighting='pooled ratios of sums; training seeds remain separate',
                    transport_truth='unwrapped displacement from the last context frame (any motion law)',
                    velocity_truth='mean true velocity over each scored horizon')
    _atomic_json(manifest_path, manifest)
    began = time.monotonic()
    for seed in range(seed_start, seed_start+seeds):
        if seed in completed_seeds:
            continue
        cases = cases_for_seed(seed, suite)
        groups = defaultdict(list)
        for case in cases:
            groups[(len(case['context_indices']), len(case['target_indices']))].append(case)
        for group in groups.values():
            for start in range(0, len(group), 8):
                batch = group[start:start+8]
                predictions, auxiliary = infer(model, job, batch, device)
                for i, (case, prediction) in enumerate(zip(batch, predictions)):
                    offsets = auxiliary['offsets'][i]
                    transport_errors = np.linalg.norm(offsets-case['true_offsets'], axis=-1)
                    correspondence = auxiliary['estimated_velocity'][i]
                    acceleration = auxiliary['acceleration'][i]
                    estimated_acceleration = auxiliary['estimated_acceleration'][i]
                    diagnostics = dict(correspondence_velocity=correspondence.tolist(),
                                       acceleration=acceleration.tolist())
                    measured, arrays = evaluate_case(case, prediction, diagnostics=diagnostics)
                    # Frames the image tracker could use (visible and detected); the
                    # returned arrays cover the longest requested horizon, and shorter
                    # horizons are prefixes.
                    usable = np.asarray(arrays['visible'], bool) & np.asarray(arrays['present'], bool) & ~np.asarray(arrays['ghost'], bool)
                    for row in measured:
                        h = row['horizon']
                        span = case['target_times'][h-1]-case['target_times'][0]
                        # Window-matched truths: the same first-to-last target span for
                        # the latent velocity, and a least-squares slope of the true
                        # positions over the frames the image tracker could use.
                        latent = (offsets[h-1]-offsets[0])/span
                        true_mean = (case['true_offsets'][h-1]-case['true_offsets'][0])/span
                        target_positions = case['positions'][case['target_indices'][:h]]
                        times_h = case['target_times'][:h]
                        keep = usable[:h] if len(usable) >= h else np.ones(h, bool)
                        if keep.sum() >= 2 and np.ptp(times_h[keep]) > 0:
                            true_slope = np.array([np.polyfit(times_h[keep], target_positions[keep, k], 1)[0] for k in range(2)])
                        else:
                            true_slope = np.full(2, np.nan)
                        predicted = np.array([row['predicted_vx'], row['predicted_vy']])
                        row.update(
                            initial_vx=float(case['initial_velocity'][0]), initial_vy=float(case['initial_velocity'][1]),
                            acceleration_x=float(case['acceleration'][0]), acceleration_y=float(case['acceleration'][1]),
                            true_mean_vx=float(true_mean[0]), true_mean_vy=float(true_mean[1]),
                            latent_vx=float(latent[0]), latent_vy=float(latent[1]),
                            latent_velocity_error=float(np.linalg.norm(latent-true_mean)),
                            image_velocity_error=float(np.linalg.norm(predicted-true_slope)),
                            true_slope_vx=float(true_slope[0]), true_slope_vy=float(true_slope[1]),
                            correspondence_vx=float(correspondence[0]), correspondence_vy=float(correspondence[1]),
                            correspondence_velocity_error=float(np.linalg.norm(correspondence-case['velocity'])),
                            estimated_ax=float(estimated_acceleration[0]), estimated_ay=float(estimated_acceleration[1]),
                            used_ax=float(acceleration[0]), used_ay=float(acceleration[1]),
                            acceleration_error=float(np.linalg.norm(acceleration-case['acceleration'])),
                            acceleration_gate=float(auxiliary['acceleration_gate'][i][0]),
                            acceleration_gate_y=float(auxiliary['acceleration_gate'][i][1]),
                            acceleration_se=float(auxiliary['acceleration_se'][i][0]),
                            acceleration_se_y=float(auxiliary['acceleration_se'][i][1]),
                            refined_fraction=float(auxiliary['refined'][i].mean()),
                            weight_scale=float(auxiliary['weights'][i].mean()),
                            transport_error_sum=float(transport_errors[:h].sum()), transport_count=h,
                            transport_endpoint_error=float(transport_errors[h-1]),
                            grid_distance=float('nan') if case['acceleration'][0] != 0 else row['grid_distance'],
                            direction_split=case['direction_split'], model=name,
                            kind=model_metadata['kind'], model_family=job.get('family', name),
                            run_id=job.get('run_id', ''), train_seed=model_metadata['train_seed'],
                            train_steps=model_metadata['train_steps'], oracle=bool(job.get('oracle', False)),
                            intervention=(f"zero_acceleration_{job['zero_acceleration']}" if job.get('zero_acceleration') else ''),
                            **{f'spec_{k}': v for k, v in model_metadata['spec'].items()})
                    rows.extend(measured)
                    if seed == seed_start:
                        arrays['offsets'] = offsets
                        arrays['true_offsets'] = case['true_offsets']
                        if 'memory' in auxiliary:
                            arrays['canonical_memory'] = auxiliary['memory'][i]
                        _save_example(out/'examples'/f'{case["case_id"]}.npz', case, prediction, arrays, diagnostics)
        completed_seeds.append(seed)
        _atomic_csv(out/'metrics.csv', rows)
        manifest.update(completed_seeds=completed_seeds, elapsed_seconds=previous_seconds+time.monotonic()-began,
                        metric_rows=len(rows), sequences=len({r['case_id'] for r in rows}))
        _atomic_json(manifest_path, manifest)
        print(json.dumps(dict(evaluation=name, suite=suffix, test_seed=seed, seeds_complete=len(completed_seeds),
                              seeds=seeds, seconds=round(manifest['elapsed_seconds'], 1))), flush=True)
    expected = seeds*len(cases_for_seed(seed_start, suite))
    if len({r['case_id'] for r in rows}) != expected:
        raise RuntimeError('Completed evaluation sequence count differs from protocol')
    if len({(r['case_id'], r['horizon']) for r in rows}) != len(rows):
        raise RuntimeError('Duplicate sequence/horizon measurements')
    manifest.update(complete=True, expected_sequences=expected, completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    _atomic_json(manifest_path, manifest)
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--suite', choices=list(SUITES), required=True)
    parser.add_argument('--tag', default='kinetic_v1')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seeds', type=int)
    parser.add_argument('--seed-start', type=int, default=0)
    args = parser.parse_args()
    evaluate_job(json.loads(args.job.read_text()), args.suite, args.tag, args.device, args.seeds, args.seed_start)


if __name__ == '__main__':
    main()
