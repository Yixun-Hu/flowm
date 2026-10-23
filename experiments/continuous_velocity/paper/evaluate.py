"""Resumable paired evaluation of the paper study's saved models/interventions.

Only explicit ``oracle=True`` jobs receive true velocity. Digit templates and
true positions are used exclusively by the common post-prediction evaluator.
Each completed test seed is an atomic resume unit; incomplete seeds rerun.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from paper.data import ROOT, TWO_D_VELOCITIES, make_case, model_batch
from paper.mechanism import (build_model, mean_memory_copy, raw_matching_copy,
                             VelocityIntervention)
from paper.metrics import evaluate_case

# Modules above are imported once. Refuse to keep running if their source files
# change, rather than hashing new disk text for old in-memory implementations.
SOURCE_PATHS = [Path(__file__), Path(__file__).parent/'data.py',
                Path(__file__).parent/'metrics.py', Path(__file__).parent/'mechanism.py',
                PARENT/'model.py', PARENT/'data.py',
                ROOT/'experiments/fractional_velocity/measurement.py']
LOADED_SOURCE_HASHES = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in SOURCE_PATHS}


SUITES = {
    'core': dict(speeds=[0, .37, .5, .63, 1, 1.23, 1.5, 2], horizon=40,
                 contexts=['regular'], appearance='mnist', seeds=32),
    'intervention': dict(speeds=[.37, .5, .63, 1, 1.23, 1.5], horizon=40,
                         contexts=['regular'], appearance='mnist', seeds=32),
    'time': dict(speeds=[.37, .63, 1.23], horizon=160,
                 contexts=['regular', 'irregular', 'sparse25'], appearance='mnist', seeds=16),
    'gap': dict(speeds=[.37, .63, 1.23], horizon=40,
                contexts=['gap10', 'gap20', 'gap30'], appearance='mnist', seeds=16),
    'appearance': dict(speeds=[.37, .63, 1.23], horizon=80,
                       contexts=['regular'], appearance='shapes', seeds=16),
    '2d': dict(velocities=[list(v) for v in TWO_D_VELOCITIES], horizon=80,
               contexts=['regular'], appearance='mnist', seeds=16),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    temporary.replace(path)


def _atomic_csv(path, rows):
    if not rows:
        return
    temporary = path.with_name(path.name + '.tmp')
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = int(value)
            except ValueError:
                try:
                    row[key] = float(value)
                except ValueError:
                    pass
    return rows


def cases_for_seed(seed, suite):
    """Order-independent paired scenarios, identical across model/optimization seeds."""
    spec = SUITES[suite]
    velocities = spec.get('velocities', [[v, 0] for v in spec.get('speeds', [])])
    cases = []
    for context in spec['contexts']:
        for mode in ('bilinear', 'nearest'):
            for velocity in velocities:
                absence = int(context[3:]) if context.startswith('gap') else 0
                actual_context = 'gap' if absence else ('regular' if context == 'sparse25' else context)
                case = make_case(seed, velocity, mode=mode, horizon=spec['horizon'],
                                 context=actual_context, appearance=spec['appearance'],
                                 absence=absence)
                if context == 'sparse25':
                    # Equal observation count and elapsed span to irregular25.
                    # Most gaps are two frames, with one three-frame interval.
                    indices = np.rint(np.linspace(0, 49, 25)).astype(int)
                    case.update(context='sparse25', context_indices=indices,
                                context_times=case['times'][indices],
                                scenario=f'{spec["appearance"]}_sparse25_released')
                vx, vy = case['velocity']
                case['case_id'] = (f'seed{seed:03d}__{mode}__vx{vx:+.6g}_vy{vy:+.6g}'
                                   f'__{case["scenario"]}')
                cases.append(case)
    return cases


def setup_model(job, tag, device):
    """Load the requested budget artifact before any inference-only intervention."""
    train_seed = int(job.get('train_seed', 2026))
    torch.manual_seed(train_seed)
    kind = job.get('kind', 'physical')
    if kind == 'physical_untrained':
        kind = 'physical'
    checkpoint = None
    saved = None
    metadata = dict(train_seed=train_seed, kind=kind, train_steps=0,
                    checkpoint=None, checkpoint_sha256=None, training_config=None)
    if job.get('run_id'):
        checkpoint = (ROOT / 'downloaded_checkpoints/continuous_velocity' / tag
                      / job['run_id'] / job.get('checkpoint', 'budget_5000.pt'))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = saved['config']
        if config['kind'] != kind:
            raise ValueError(f'Job kind {kind} differs from checkpoint kind {config["kind"]}')
        if int(config['seed']) != train_seed:
            raise ValueError('Job training seed differs from checkpoint training seed')
        metadata.update(train_steps=int(saved['step']), checkpoint=str(checkpoint.resolve()),
                        checkpoint_sha256=_sha(checkpoint), training_config=config)
    model = build_model(kind)
    if saved is not None:
        model.load_state_dict(saved['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    intervention = job.get('intervention')
    if intervention == 'raw_matching':
        model = raw_matching_copy(model)
    elif intervention == 'mean_memory':
        model = mean_memory_copy(model)
    elif intervention not in (None, 'quantize'):
        raise ValueError(f'Unknown inference intervention: {intervention}')
    if intervention == 'quantize' or job.get('oracle', False):
        model = VelocityIntervention(model, grid_spacing=job.get('spacing'),
                                     scope=job.get('scope', 'both'))
    elif job.get('spacing') is not None:
        raise ValueError('A grid spacing requires the quantize intervention')
    model.eval().requires_grad_(False)
    metadata.update(parameters=sum(p.numel() for p in model.parameters()),
                    oracle=bool(job.get('oracle', False)), intervention=intervention,
                    grid_spacing=job.get('spacing'), scope=job.get('scope', 'both'))
    return model, metadata


@torch.inference_mode()
def infer(model, job, cases, device):
    inputs = model_batch(cases, device)
    if job.get('oracle', False):
        inputs['velocity_override'] = torch.as_tensor(
            np.stack([case['velocity'] for case in cases]), device=device, dtype=torch.float32)
    prediction, auxiliary = model(**inputs)
    if not torch.isfinite(prediction).all():
        raise RuntimeError('Nonfinite predictions during evaluation')
    arrays = {key: auxiliary[key].detach().cpu().numpy()
              for key in ('velocity', 'estimated_velocity', 'alignment_velocity', 'offsets', 'memory')
              if key in auxiliary and isinstance(auxiliary[key], torch.Tensor)}
    return prediction.detach().cpu().numpy(), arrays


def _save_example(path, case, pred, arrays, diagnostics):
    content = dict(predictions=pred, **arrays,
                   **{key: case[key] for key in ('frames', 'template', 'positions', 'cameras',
                                                'actions', 'visibility', 'times', 'context_indices',
                                                'target_indices', 'velocity')})
    content['model_diagnostics'] = np.array(json.dumps(diagnostics))
    content['case_id'] = np.array(case['case_id'])
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **content)
    temporary.replace(path)


def evaluate_job(job, suite, tag='paper_v1', device='cuda:1', seeds=None):
    """Evaluate one saved model, preserving complete seeds across interruptions.

    The return value is the completed manifest. A matching complete result is
    reused; any changed model/source/protocol signature raises instead of silently
    mixing results. Raw and neural untrained jobs omit run_id explicitly.
    """
    if suite not in SUITES:
        raise ValueError(f'Unknown suite: {suite}')
    job = dict(job)
    name = job['name']
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        raise ValueError('Job name must be a simple path component')
    seeds = SUITES[suite]['seeds'] if seeds is None else int(seeds)
    if not 1 <= seeds <= 64:
        raise ValueError('Evaluation uses 1 to 64 heldout camera episodes')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, model_metadata = setup_model(job, tag, device)
    source_hashes = {str(path.relative_to(ROOT)): _sha(path) for path in SOURCE_PATHS}
    if source_hashes != LOADED_SOURCE_HASHES:
        raise RuntimeError('Evaluation sources changed after import; restart the Python worker')
    specification = dict(job=job, suite=suite, suite_specification=SUITES[suite],
                         metric_schema_version=2,
                         seeds=seeds, device=str(device), source_sha256=source_hashes,
                         model=model_metadata, tolerance=2., batch_size=8,
                         modes=['bilinear', 'nearest'], metrics_horizons=[20, 40, 80, 160])
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    out = ROOT / f'outputs/continuous_velocity/{tag}/evaluation/{name}/{suite}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'examples').mkdir(exist_ok=True)
    manifest_path = out/'manifest.json'
    rows = []
    completed_seeds = []
    previous_seconds = 0.
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
    manifest = dict(**specification, signature=signature, complete=False,
                    completed_seeds=completed_seeds, output_directory=str(out),
                    started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    selected_examples='all cases for test seed 0 only',
                    metric_weighting='pooled ratios of sums; training seeds remain separate')
    _atomic_json(manifest_path, manifest)
    began = time.monotonic()
    for seed in range(seeds):
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
                    elapsed = case['target_times']-case['context_times'][-1]
                    transport_errors = np.linalg.norm(
                        offsets-elapsed[:, None]*case['velocity'][None], axis=-1)
                    correspondence = auxiliary.get('estimated_velocity', auxiliary['velocity'])[i]
                    effective = (offsets[-1]-offsets[0]) / (case['target_times'][-1]-case['target_times'][0])
                    diagnostics = dict(latent_velocity=effective.tolist(),
                                       correspondence_velocity=correspondence.tolist())
                    measured, arrays = evaluate_case(case, prediction, diagnostics=diagnostics)
                    for row in measured:
                        h = row['horizon']
                        span = case['target_times'][h-1]-case['target_times'][0]
                        velocity = (offsets[h-1]-offsets[0]) / span
                        row.update(latent_vx=float(velocity[0]), latent_vy=float(velocity[1]),
                                   latent_velocity_error=float(np.linalg.norm(velocity-case['velocity'])),
                                   correspondence_vx=float(correspondence[0]),
                                   correspondence_vy=float(correspondence[1]),
                                   correspondence_velocity_error=float(np.linalg.norm(correspondence-case['velocity'])),
                                   transport_error_sum=float(transport_errors[:h].sum()),
                                   transport_count=h,
                                   transport_endpoint_error=float(transport_errors[h-1]),
                                   direction_split=case['direction_split'],
                                   model=name, kind=model_metadata['kind'],
                                   model_family=job.get('family', model_metadata['kind']),
                                   run_id=job.get('run_id', ''), train_seed=model_metadata['train_seed'],
                                   train_steps=model_metadata['train_steps'],
                                   oracle=bool(job.get('oracle', False)),
                                   intervention=job.get('intervention', ''),
                                   intervention_scope=job.get('scope', 'both'))
                    rows.extend(measured)
                    if seed == 0:
                        arrays['offsets'] = offsets
                        if 'memory' in auxiliary:
                            arrays['canonical_memory'] = auxiliary['memory'][i]
                        _save_example(out/'examples'/f'{case["case_id"]}.npz', case,
                                      prediction, arrays, diagnostics)
        completed_seeds.append(seed)
        _atomic_csv(out/'metrics.csv', rows)
        manifest.update(completed_seeds=completed_seeds,
                        elapsed_seconds=previous_seconds+time.monotonic()-began,
                        metric_rows=len(rows), sequences=len({r['case_id'] for r in rows}))
        _atomic_json(manifest_path, manifest)
        print(json.dumps(dict(evaluation=name, suite=suite, test_seed=seed,
                              seeds_complete=len(completed_seeds), seeds=seeds,
                              seconds=round(manifest['elapsed_seconds'], 1))), flush=True)
    expected = seeds*len(cases_for_seed(0, suite))
    if len({r['case_id'] for r in rows}) != expected:
        raise RuntimeError('Completed evaluation sequence count differs from protocol')
    pairs = {(r['case_id'], r['horizon']) for r in rows}
    if len(pairs) != len(rows):
        raise RuntimeError('Duplicate sequence/horizon measurements')
    manifest.update(complete=True, expected_sequences=expected,
                    completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    _atomic_json(manifest_path, manifest)
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True, help='JSON job specification')
    parser.add_argument('--suite', choices=list(SUITES), required=True)
    parser.add_argument('--tag', default='paper_v1')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seeds', type=int)
    args = parser.parse_args()
    evaluate_job(json.loads(args.job.read_text()), args.suite, args.tag, args.device, args.seeds)


if __name__ == '__main__':
    main()
