"""Kinetic study campaign: training matrix, validation-only selection, evaluation.

Stages (run in order; each is resumable and may be sharded across workers with
``--shard i --shards n``; one worker per GPU process):

  select    learning-rate selection {3e-4, 1e-3} on seed 2026 by validation loss, every variant
  train     five fresh seeds 2027-2031 per variant at the selected rate
  combine   choose the Group A transport/refinement variant with the lowest mean validation
            loss (validation only) and add its order-2 version to both groups (shard 0 trains
            the selection runs itself; other shards wait for the published selection)
  evaluate  all fresh-seed runs on the four suites; oracle and zero-acceleration diagnostics
  extended  extend the seed-2026 selection runs of the baseline, the selected variant and
            order2 to 20k updates in place; evaluate their 5k and 20k budget checkpoints
  confirm   confirmatory evaluation on unused scenes 32-63 for the prespecified pairs

No test result influences any training or selection decision. The protocol,
including the primary contrasts and their test statistics, is written before
any evaluation output exists.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from paper.train import atomic_json  # noqa: E402
from kinetic.training import DEFAULTS, paths, run_training  # noqa: E402
from kinetic.evaluation import SUITES, evaluate_job  # noqa: E402
from kinetic.scenes import ROOT  # noqa: E402

PROTOCOL_VERSION = 2
RATES = [.0003, .001]
HPO_SEED = 2026
SEEDS = [2027, 2028, 2029, 2030, 2031]
BASE = dict(align='bilinear', render='bilinear', refine='none', order=1, kappa=9., kind='physical')
GROUP_A = {
    'base': dict(),
    'alignS': dict(align='spectral'),
    'renderS': dict(render='spectral'),
    'spectral': dict(align='spectral', render='spectral'),
    'newton': dict(refine='newton'),
    'spectral_newton': dict(align='spectral', render='spectral', refine='newton'),
    'order2': dict(order=2, kappa=9.),
    'order2_k4': dict(order=2, kappa=4.),
    'order2_k0': dict(order=2, kappa=0.),
}
GROUP_B = {
    'base': dict(),
    'order2': dict(order=2, kappa=9.),
    'order2_k4': dict(order=2, kappa=4.),
    'order2_k0': dict(order=2, kappa=0.),
    'relaxed': dict(kind='relaxed'),
    'offset': dict(kind='offset'),
}
TRANSPORT_CANDIDATES = ['base', 'alignS', 'renderS', 'spectral', 'newton', 'spectral_newton']
ORACLE_VARIANTS = ['base', 'alignS', 'renderS', 'spectral', 'newton', 'order2']
MOTION = dict(A='horizontal', B='accelerated')
CONFIRM_SUITES = ['core', 'accel']
WAIT_TIMEOUT = 4*3600
PRIMARY = dict(
    statistic='paired t-test across the five fresh training seeds on per-model pooled scores (ratios of sums)',
    family='Holm correction over the four contrasts P1-P4',
    P1=dict(a='A_spectral', b='A_base', suite='core', subset='bilinear, initial speeds .37/.63/1.23, a=0, horizon 40',
            metric='mse', test='two-sided superiority'),
    P2=dict(a='A_newton', b='A_base', suite='core', subset='bilinear, initial speeds .37/.63/1.23, a=0, horizon 40',
            metric='mse', test='two-sided superiority; correspondence velocity error is secondary'),
    P3=dict(a='A_order2', b='A_base', suite='core', subset='bilinear, initial speeds .37/.63/1.23, a=0, horizon 40',
            metric='mse', test='one-sided non-inferiority: one-sided 95% upper bound of the mean paired difference < 1e-4'),
    P4=dict(a='B_order2', b='B_base', suite='accel', subset='bilinear, all accelerations, horizon 40',
            metric='mse', test='two-sided superiority; relaxed/offset/zero-acceleration are labeled controls'),
    secondary='all other contrasts, renderers, horizons and metrics are descriptive; conditional scene-cluster bootstrap intervals are reported alongside',
    confirmatory='scenes 32-63 (unused by the paper campaign) for the primary pairs; reported separately')


def spec(group, variant):
    table = GROUP_A if group == 'A' else GROUP_B
    return {**BASE, **table[variant]}


def run_id(group, variant, seed, lr):
    return f'{group}_{variant}_bilinear_{MOTION[group]}_s{seed}_lr{lr:g}'


def load_state(out):
    path = out/'selected_runs.json'
    return json.loads(path.read_text()) if path.exists() else {}


def complete(cfg_base, rid, steps):
    """Manifest-level completion check that does not re-hash sources (see training.py)."""
    out, _ = paths({**cfg_base, 'run_id': rid})
    manifest = out/'manifest.json'
    if not manifest.exists():
        return False
    old = json.loads(manifest.read_text())
    return bool(old.get('complete')) and old['config']['steps'] >= steps


def _train(cfg_base, group, variant, seed, lr, steps=5000):
    rid = run_id(group, variant, seed, lr)
    if complete(cfg_base, rid, steps):
        return
    cfg = {**cfg_base, **spec(group, variant), 'seed': seed, 'lr': lr, 'steps': steps,
           'motion': MOTION[group], 'run_id': rid}
    return run_training(cfg)


def selected_rate(out, group, variant):
    state = load_state(out)
    key = f'{group}_{variant}'
    if key not in state:
        raise RuntimeError(f'Learning rate not yet selected for {key}; run the select stage')
    return state[key]['lr']


def publish_selection(out, cfg_base, group, variant):
    options = []
    for lr in RATES:
        _, ckpt = paths({**cfg_base, 'run_id': run_id(group, variant, HPO_SEED, lr)})
        if not (ckpt/'budget_5000.pt').exists():
            return False
        fixed = torch.load(ckpt/'budget_5000.pt', map_location='cpu', weights_only=False)
        options.append((float(fixed['best']), lr, run_id(group, variant, HPO_SEED, lr)))
    state = load_state(out)
    best, lr, chosen = min(options)
    state[f'{group}_{variant}'] = dict(lr=lr, tuning=options, hpo_run=chosen, best_validation=best)
    atomic_json(out/'selected_runs.json', state)
    return True


def variants(group):
    return GROUP_A if group == 'A' else GROUP_B


def stage_select(out, cfg_base, shard, shards, groups):
    jobs = [(g, v, lr) for g in groups for v in variants(g) for lr in RATES]
    for index, (g, v, lr) in enumerate(jobs):
        if index % shards == shard:
            _train(cfg_base, g, v, HPO_SEED, lr)
    if shard == 0:
        # Publication happens once every rate has finished (the driver's single-shard pass).
        for g in groups:
            for v in variants(g):
                publish_selection(out, cfg_base, g, v)


def stage_train(out, cfg_base, shard, shards, groups):
    jobs = [(g, v, s) for g in groups for v in variants(g) for s in SEEDS]
    for index, (g, v, s) in enumerate(jobs):
        if index % shards == shard:
            _train(cfg_base, g, v, s, selected_rate(out, g, v))


def validation_mean(out, cfg_base, group, variant):
    lr = selected_rate(out, group, variant)
    values = []
    for seed in SEEDS:
        _, ckpt = paths({**cfg_base, 'run_id': run_id(group, variant, seed, lr)})
        fixed = torch.load(ckpt/'budget_5000.pt', map_location='cpu', weights_only=False)
        values.append(float(fixed['best']))
    return sum(values)/len(values), values


def register_combined(out):
    state = load_state(out)
    best = state.get('combined', {}).get('selected_transport')
    if best and best != 'base':
        GROUP_A['order2_'+best] = {**GROUP_A[best], 'order': 2, 'kappa': 9.}
        GROUP_B['order2_'+best] = {**GROUP_A[best], 'order': 2, 'kappa': 9.}
    return best


def stage_combine(out, cfg_base, shard, shards):
    """Validation-only choice of the transport/refinement package; adds order-2 copies."""
    if shard == 0:
        state = load_state(out)
        if 'combined' not in state:
            scores = {v: validation_mean(out, cfg_base, 'A', v) for v in TRANSPORT_CANDIDATES}
            best = min(scores, key=lambda v: scores[v][0])
            state['combined'] = dict(selected_transport=best, validation_means={k: s[0] for k, s in scores.items()},
                                     validation_values={k: s[1] for k, s in scores.items()},
                                     rule='lowest mean best-validation loss across the five fresh seeds; no test data')
            atomic_json(out/'selected_runs.json', state)
        best = register_combined(out)
        if best is None or best == 'base':
            return
        for g in ('A', 'B'):
            for lr in RATES:
                _train(cfg_base, g, 'order2_'+best, HPO_SEED, lr)
            publish_selection(out, cfg_base, g, 'order2_'+best)
    else:
        started = time.monotonic()
        while 'combined' not in load_state(out):
            if time.monotonic()-started > WAIT_TIMEOUT:
                raise RuntimeError('Timed out waiting for the combine selection')
            time.sleep(20)
        best = register_combined(out)
        if best is None or best == 'base':
            return
        while not all(f'{g}_order2_{best}' in load_state(out) for g in ('A', 'B')):
            if time.monotonic()-started > WAIT_TIMEOUT:
                raise RuntimeError('Timed out waiting for the combined learning-rate selection')
            time.sleep(20)
    jobs = [(g, 'order2_'+best, s) for g in ('A', 'B') for s in SEEDS]
    for index, (g, v, s) in enumerate(jobs):
        if index % shards == shard:
            _train(cfg_base, g, v, s, selected_rate(out, g, v))


def evaluation_jobs(out):
    jobs = []
    for g in ('A', 'B'):
        for v in variants(g):
            lr = selected_rate(out, g, v)
            for s in SEEDS:
                base = dict(name=f'{g}_{v}_s{s}', family=f'{g}_{v}', run_id=run_id(g, v, s, lr), train_seed=s)
                for suite in SUITES:
                    jobs.append((base, suite))
                if v.startswith('order2'):
                    for mode in ('both', 'transport'):
                        for suite in ('accel', 'accel_long', 'core'):
                            jobs.append(({**base, 'name': f'{g}_{v}_s{s}_zeroacc_{mode}',
                                          'family': f'{g}_{v}_zeroacc_{mode}', 'zero_acceleration': mode}, suite))
                if s == SEEDS[0] and v in ORACLE_VARIANTS:
                    for suite in SUITES:
                        jobs.append(({**base, 'name': f'{g}_{v}_s{s}_oracle', 'family': f'{g}_{v}_oracle',
                                      'oracle': True}, suite))
    return jobs


def stage_evaluate(out, device, shard, shards, tag):
    register_combined(out)
    jobs = evaluation_jobs(out)
    if shard == 0:
        atomic_json(out/'evaluation_jobs.json', [dict(job=j, suite=s) for j, s in jobs])
    for index, (job, suite) in enumerate(jobs):
        if index % shards == shard:
            evaluate_job(job, suite, tag=tag, device=device)


def stage_extended(out, cfg_base, shard, shards, tag, device):
    best = register_combined(out) or 'base'
    names = ['base'] + ([best] if best != 'base' else []) + ['order2']
    jobs = [(g, v) for g in ('A', 'B') for v in names if v in variants(g)]
    for index, (g, v) in enumerate(jobs):
        if index % shards != shard:
            continue
        lr = selected_rate(out, g, v)
        _train(cfg_base, g, v, HPO_SEED, lr, steps=20000)   # extends the selection run in place
        for checkpoint, label in (('budget_5000.pt', 'hpo5k'), ('budget_20000.pt', '20k')):
            for suite in SUITES:
                evaluate_job(dict(name=f'{g}_{v}_s{HPO_SEED}_{label}', family=f'{g}_{v}_{label}',
                                  run_id=run_id(g, v, HPO_SEED, lr), checkpoint=checkpoint,
                                  train_seed=HPO_SEED), suite, tag=tag, device=device)


def stage_confirm(out, device, shard, shards, tag):
    best = register_combined(out) or 'base'
    pairs = [('A', 'base'), ('A', 'spectral'), ('A', 'newton'), ('A', 'order2'), ('B', 'base'), ('B', 'order2')]
    if best != 'base':
        pairs += [('A', best), ('A', 'order2_'+best), ('B', 'order2_'+best)]
    jobs = []
    for g, v in dict.fromkeys(pairs):
        lr = selected_rate(out, g, v)
        for s in SEEDS:
            for suite in CONFIRM_SUITES:
                jobs.append((dict(name=f'{g}_{v}_s{s}', family=f'{g}_{v}', run_id=run_id(g, v, s, lr), train_seed=s), suite))
    for index, (job, suite) in enumerate(jobs):
        if index % shards == shard:
            evaluate_job(job, suite, tag=tag, device=device, seeds=32, seed_start=32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='kinetic_v1')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--stage', choices=['select', 'train', 'combine', 'evaluate', 'extended', 'confirm'], required=True)
    parser.add_argument('--groups', default='AB')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    out = ROOT/f'outputs/continuous_velocity/{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    cfg_base = {**DEFAULTS, 'tag': args.tag, 'device': args.device, 'steps': 5000}
    protocol = dict(version=PROTOCOL_VERSION, training_seeds=SEEDS, hpo_seed=HPO_SEED, learning_rates=RATES,
                    fixed_budget_updates=5000, extended_updates=20000,
                    group_a=dict(motion='horizontal', variants={k: {**BASE, **v} for k, v in GROUP_A.items()}),
                    group_b=dict(motion='accelerated', variants={k: {**BASE, **v} for k, v in GROUP_B.items()}),
                    transport_candidates=TRANSPORT_CANDIDATES, oracle_variants=ORACLE_VARIANTS,
                    selection='minimum weighted validation pixel loss on 64 validation sequences; no test data',
                    primary=PRIMARY, suites=SUITES, confirmatory_scenes='32-63',
                    group_b_note='Group B trains on accelerated streams; its speeds are not held out',
                    started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    if not (out/'protocol.json').exists():
        atomic_json(out/'protocol.json', protocol)
    groups = list(args.groups)
    if args.stage == 'select':
        stage_select(out, cfg_base, args.shard, args.shards, groups)
    elif args.stage == 'train':
        stage_train(out, cfg_base, args.shard, args.shards, groups)
    elif args.stage == 'combine':
        stage_combine(out, cfg_base, args.shard, args.shards)
    elif args.stage == 'evaluate':
        stage_evaluate(out, args.device, args.shard, args.shards, args.tag)
    elif args.stage == 'extended':
        stage_extended(out, cfg_base, args.shard, args.shards, args.tag, args.device)
    elif args.stage == 'confirm':
        stage_confirm(out, args.device, args.shard, args.shards, args.tag)
    atomic_json(out/f'stage_{args.stage}_shard{args.shard}of{args.shards}.json',
                dict(complete=True, completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


if __name__ == '__main__':
    main()
