"""Exploratory temporal-baseline diagnostic for raw correspondence fitting.

This is a post-hoc diagnostic, not validation-based model selection. It does not
change any trained checkpoint, formal evaluation, or campaign hyperparameter.
Only raw context images, known integer camera positions, and timestamps enter
matching/fitting; true velocity is used after fitting to score estimation errors.

Longer pair lags are safe here for the tested regular timestamps and velocities.
They are not a general solution to temporal aliasing on the 50-pixel torus.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from paper.data import ROOT, make_case, model_batch
from paper.mechanism import RawPixelCorrespondence, _physical_fit, world_embed, fit_velocity


LAG_SETS = ((1, 2, 4), (1, 4, 8), (2, 4, 8), (1, 4, 12))
FITS = ('weighted_ls', 'robust_weighted', 'unweighted_ls', 'robust_unweighted')


def cached_fields(context, cameras):
    fields, mask = world_embed(context[:, :, None], cameras)
    fft = torch.fft.rfft2(fields)
    energy = torch.fft.rfft2(fields.square().sum(2))
    maskfft = torch.fft.rfft2(mask[:, :, 0])
    return dict(fft=fft, energy=energy, maskfft=maskfft, length=context.shape[1])


def observations(cache, times, lags, shifts):
    """Exact raw-matcher calculation with only pair-lag selection changed."""
    first, second = [], []
    for lag in lags:
        starts = torch.linspace(0, cache['length']-lag-1, 12, device=times.device).long()
        first.append(starts)
        second.append(starts+lag)
    i, j = torch.cat(first), torch.cat(second)
    dt = times[:, j]-times[:, i]
    fft, energy, maskfft = (cache[key] for key in ('fft', 'energy', 'maskfft'))
    numerator = torch.fft.irfft2((fft[:, i].conj()*fft[:, j]).sum(2), s=(50, 50)).flatten(2)
    denominator = torch.fft.irfft2(
        energy[:, i].conj()*maskfft[:, j]+maskfft[:, i].conj()*energy[:, j],
        s=(50, 50)).flatten(2)
    raw_overlap = denominator.clamp_min(0)
    similarity = (2*numerator/denominator.clamp_min(1e-5)).clamp(-1, 1)
    allowed = (shifts.abs()[None, None] <= (2.5*dt[:, :, None, None]+1)).all(-1)
    allowed = allowed & (raw_overlap > 2)
    attention = (similarity*math.exp(3.7)).masked_fill(~allowed, -10000).softmax(-1)
    displacement = attention@shifts
    match = (attention*similarity).sum(-1)
    support = (attention*raw_overlap).sum(-1)
    weights = .5*match.clamp_min(0).pow(4)*(support/(support+10))*allowed.any(-1)
    return dict(displacements=displacement, dt=dt, weights=weights)


def estimate(observation, method):
    if method not in FITS:
        raise ValueError(method)
    chosen = dict(observation)
    if method in ('unweighted_ls', 'robust_unweighted'):
        chosen['weights'] = (chosen['weights'] > 0).to(chosen['weights'].dtype)
    if method.startswith('robust'):
        return _physical_fit(chosen)
    return fit_velocity(chosen['displacements'], chosen['dt'], chosen['weights'])


def _summary(rows):
    by_group = {}
    for mode in ('bilinear', 'nearest'):
        for lags in LAG_SETS:
            label = '-'.join(map(str, lags))
            for method in FITS:
                selected = [r for r in rows if (r['mode'], r['lags'], r['fit']) == (mode, label, method)]
                key = f'{mode}/{label}/{method}'
                by_group[key] = dict(
                    sequences=len(selected),
                    velocity_l2_mean=float(np.mean([r['velocity_l2'] for r in selected])),
                    vx_mae=float(np.mean([r['vx_abs_error'] for r in selected])),
                    vx_bias=float(np.mean([r['vx_error'] for r in selected])),
                    vy_mae=float(np.mean([r['vy_abs_error'] for r in selected])),
                    by_speed={str(speed): dict(
                        velocity_l2_mean=float(np.mean([r['velocity_l2'] for r in selected if r['true_vx'] == speed])),
                        vx_bias=float(np.mean([r['vx_error'] for r in selected if r['true_vx'] == speed])),
                    ) for speed in (.37, .63, 1.23)},
                )
    # Cluster by test identity/camera seed, keeping its three speeds together.
    paired = {}
    rng = np.random.default_rng(2026)
    seeds = sorted({r['seed'] for r in rows})
    selections = rng.integers(len(seeds), size=(2000, len(seeds)))
    for mode in ('bilinear', 'nearest'):
        base = {(r['seed'], r['true_vx']): r for r in rows
                if (r['mode'], r['lags'], r['fit']) == (mode, '1-2-4', 'robust_weighted')}
        for lags in LAG_SETS:
            label = '-'.join(map(str, lags))
            for method in FITS:
                selected = [r for r in rows if (r['mode'], r['lags'], r['fit']) == (mode, label, method)]
                per_seed = np.array([np.mean([
                    r['velocity_l2']-base[(r['seed'], r['true_vx'])]['velocity_l2']
                    for r in selected if r['seed'] == seed
                ]) for seed in seeds])
                samples = per_seed[selections].mean(1)
                paired[f'{mode}/{label}/{method}'] = dict(
                    difference_vs_original=float(per_seed.mean()),
                    paired_seed_ci95=np.quantile(samples, [.025, .975]).tolist(),
                )
    return by_group, paired


@torch.inference_mode()
def run(seeds=16, tag='paper_v1'):
    torch.set_num_threads(2)
    matcher = RawPixelCorrespondence().eval()
    rows, equivalence = [], []
    began = time.monotonic()
    for seed in range(seeds):
        for mode in ('bilinear', 'nearest'):
            for speed in (.37, .63, 1.23):
                case = make_case(seed, speed, mode=mode, horizon=1)
                inputs = model_batch([case])
                cache = cached_fields(inputs['context'], inputs['cameras'])
                for lags in LAG_SETS:
                    observation = observations(cache, inputs['context_times'], lags, matcher.shifts)
                    if seed == 0 and lags == (1, 2, 4):
                        _, _, reference, _, reference_obs = matcher(
                            inputs['context'], inputs['cameras'], inputs['context_times'])
                        maximum = max((observation[key]-reference_obs[key]).abs().max().item()
                                      for key in ('displacements', 'dt', 'weights'))
                        maximum = max(maximum, (estimate(observation, 'robust_weighted')-reference).abs().max().item())
                        equivalence.append(maximum)
                        if maximum > 1e-6:
                            raise AssertionError(f'Original raw-matcher behavior differs: {maximum}')
                    for method in FITS:
                        velocity = estimate(observation, method)[0].numpy()
                        error = velocity-case['velocity']
                        rows.append(dict(seed=seed, mode=mode, true_vx=speed, true_vy=0.,
                                         lags='-'.join(map(str, lags)), fit=method,
                                         estimated_vx=float(velocity[0]), estimated_vy=float(velocity[1]),
                                         vx_error=float(error[0]), vx_abs_error=float(abs(error[0])),
                                         vy_abs_error=float(abs(error[1])), velocity_l2=float(np.linalg.norm(error)),
                                         valid_pairs=int((observation['weights'] > 0).sum()),
                                         context_visible=int((case['visibility'][:50] >= .35).sum())))
        print(json.dumps(dict(diagnostic='raw_pair_lags', test_seed=seed,
                              seconds=round(time.monotonic()-began, 1))), flush=True)
    grouped, paired = _summary(rows)
    result = dict(exploratory=True, trained_models_changed=False, formal_evaluation_changed=False,
                  test_seeds=seeds, speeds=[.37, .63, 1.23], rendering=['bilinear', 'nearest'],
                  context='50 regularly spaced observations at t=0:49',
                  features='raw pixels only; no digit template or true motion supplied',
                  matching_temperature=math.exp(3.7), pair_count=36, lag_sets=LAG_SETS,
                  methods=FITS, robust_rule='Two Cauchy reweighting steps on pair velocity residuals; scale0.15',
                  unweighted_rule='Uniform weight over pairs with positive original matching weight',
                  original_matcher_max_abs_difference=max(equivalence),
                  warning='Post-hoc test-set exploration. No setting is selected for the formal campaign. '
                          'Longer lags need alias handling for larger velocities or irregular timestamps. '
                          'This diagnostic scores velocity only and does not establish image-prediction improvements.',
                  summary=grouped, paired_vs_original=paired,
                  seconds=time.monotonic()-began)
    out = ROOT/f'outputs/continuous_velocity/{tag}'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'lag_diagnostic.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out/'lag_diagnostic.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=16)
    parser.add_argument('--tag', default='paper_v1')
    args = parser.parse_args()
    result = run(args.seeds, args.tag)
    for name, row in result['summary'].items():
        print(name, f"velocity L2={row['velocity_l2_mean']:.6f}", f"vx bias={row['vx_bias']:+.6f}")


if __name__ == '__main__':
    main()
