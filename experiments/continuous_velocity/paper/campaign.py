"""Sequential GPU-1 campaign; all decisions depend on validation data only."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.train import ROOT, paths, run_training, atomic_json

KINDS = ['physical', 'relaxed', 'offset', 'mlp_velocity', 'mean_memory', 'raw_correlation']
SEEDS = [2026, 2027, 2028]
RATES = [.0003, .001]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='paper_v1')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--stage', choices=['core', 'generalization', 'extended', 'all'], default='all')
    args = parser.parse_args()
    out = ROOT/f'outputs/continuous_velocity/{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    base = dict(tag=args.tag, device=args.device, steps=5000, batch_size=8,
                validation_size=64, render_mode='bilinear', motion='horizontal', train_size=55000)
    protocol = dict(training_seeds=SEEDS, learning_rates=RATES, fixed_budget_updates=5000,
                    hpo_seed=2026, selection='minimum weighted validation pixel loss; no test selection',
                    core_models=KINDS, validation_sequences=64,
                    extended_models=['offset', 'mlp_velocity', 'physical', 'relaxed'],
                    extended_updates=20000, extended_training_seeds=[2026],
                    note='Extended training diagnoses optimization for one seed; it does not establish convergence.',
                    rendering_train=['bilinear', 'nearest', 'mixed'], rendering_test=['bilinear', 'nearest'],
                    generalization_training_seeds=SEEDS,
                    train_context_frames=50, train_max_future=40, test_max_future=160,
                    train_supervision='future pixels only', device=args.device,
                    started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    if not (out/'protocol.json').exists():
        (out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    selected_path = out/'selected_runs.json'
    selected = json.loads(selected_path.read_text()) if selected_path.exists() else {}

    def save():
        atomic_json(selected_path, selected)

    def execute(**updates):
        cfg = {**base, **updates}
        return run_training(cfg)

    if args.stage in ('core', 'all'):
        for kind in KINDS:
            options = []
            for lr in RATES:
                run_id = f'{kind}_bilinear_horizontal_s2026_lr{lr:g}'
                execute(kind=kind, seed=2026, lr=lr, run_id=run_id)
                # An extended run's best.pt must never enter fixed-budget HPO.
                _, checkpoint_dir = paths({**base, 'run_id': run_id})
                fixed = torch.load(checkpoint_dir/'budget_5000.pt', map_location='cpu', weights_only=False)
                options.append((fixed['best'], lr, run_id))
            _, lr, chosen = min(options)
            selected[kind] = dict(lr=lr, tuning=options, runs=[chosen])
            save()
            for seed in SEEDS[1:]:
                run_id = f'{kind}_bilinear_horizontal_s{seed}_lr{lr:g}'
                execute(kind=kind, seed=seed, lr=lr, run_id=run_id)
                selected[kind]['runs'].append(run_id)
                save()
        # Five independent fits for the main physical-vs-relaxed comparison.
        for kind in ('physical', 'relaxed'):
            lr = selected[kind]['lr']
            for seed in (2029, 2030):
                run_id = f'{kind}_bilinear_horizontal_s{seed}_lr{lr:g}'
                execute(kind=kind, seed=seed, lr=lr, run_id=run_id)
                if run_id not in selected[kind]['runs']:
                    selected[kind]['runs'].append(run_id)
                save()
    if args.stage in ('generalization', 'all'):
        for rendering, motion in [('nearest', 'horizontal'), ('mixed', 'horizontal'), ('mixed', '2d')]:
            key = f'physical_{rendering}_{motion}'
            selected[key] = dict(lr=selected['physical']['lr'], runs=[])
            for seed in SEEDS:
                lr = selected['physical']['lr']
                run_id = f'physical_{rendering}_{motion}_s{seed}_lr{lr:g}'
                execute(kind='physical', seed=seed, lr=lr, render_mode=rendering, motion=motion, run_id=run_id)
                selected[key]['runs'].append(run_id)
                save()
        # Matched 2-D learned velocity control; same data streams/rendering/budget.
        key = 'mlp_velocity_mixed_2d'
        selected[key] = dict(lr=selected['mlp_velocity']['lr'], runs=[])
        for seed in SEEDS:
            lr = selected['mlp_velocity']['lr']
            run_id = f'mlp_velocity_mixed_2d_s{seed}_lr{lr:g}'
            execute(kind='mlp_velocity', seed=seed, lr=lr, render_mode='mixed', motion='2d', run_id=run_id)
            selected[key]['runs'].append(run_id)
            save()
    if args.stage in ('extended', 'all'):
        for kind in ('physical', 'relaxed', 'offset', 'mlp_velocity'):
            run_id = selected[kind]['runs'][0]
            execute(kind=kind, seed=2026, lr=selected[kind]['lr'], run_id=run_id, steps=20000)
            selected[kind]['extended_run'] = run_id
            save()
    (out/f'training_{args.stage}_complete.json').write_text(json.dumps(dict(
        complete=True, stage=args.stage, completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')), indent=2))


if __name__ == '__main__':
    main()
