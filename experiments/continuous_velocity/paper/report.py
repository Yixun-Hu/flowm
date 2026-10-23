"""Regenerate a factual report from completed paper-campaign evaluations.

Run: python experiments/continuous_velocity/paper/report.py --root
     outputs/continuous_velocity/paper_v1
Only complete manifests enter tables. Model-seed SD and conditional test-set
confidence intervals remain separate. Intermediate reports disclose missing jobs.
"""
import argparse
import html
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('MPLCONFIGDIR', '/tmp/flowm-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.statistics import training_seed_summary, paired_seed_comparison, paired_test_bootstrap
from paper.metrics import summarize, METRICS_VERSION

CORE = ['physical', 'relaxed', 'offset', 'mlp_velocity', 'mean_memory', 'raw_correlation']
LABELS = dict(physical='Physical fit', relaxed='Physical + corrections', offset='Free offsets',
              mlp_velocity='MLP velocity', mean_memory='Mean memory', raw_correlation='Raw matching, trained memory',
              raw_untrained='Raw geometric baseline', physical_untrained='Untrained physical',
              physical_nearest_horizontal='Nearest training', physical_mixed_horizontal='Mixed training',
              physical_mixed_2d='Physical 2D', mlp_velocity_mixed_2d='MLP 2D',
              est_q1_transport='Estimated, grid 1', est_q05_transport='Estimated, grid 0.5',
              oracle_cont='Oracle continuous', oracle_q1_transport='Oracle, grid 1',
              oracle_q05_transport='Oracle, grid 0.5')
COLORS = dict(zip(CORE, ['#087f8c', '#7549a1', '#bb6044', '#3d72b4', '#679735', '#ad822f']))
METRICS = ['mse', 'ade', 'joint_success', 'coverage', 'missing_rate', 'ghost_rate', 'latent_velocity_error',
           'frame_joint_success', 'frame_localization_error', 'frame_coverage',
           'certified_joint_success', 'certified_ade', 'certified_coverage',
           'continuity_ade', 'continuity_coverage', 'continuity_ambiguity',
           'transport_ade', 'transport_endpoint_error']
HELDOUT = [.37, .63, 1.23]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False))


def save(fig, root, name, figures):
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(root/'figures'/f'{name}.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig); figures.append(name)


def heldout(data):
    mask = np.isclose(data.true_vy, 0)
    mask &= np.logical_or.reduce([np.isclose(data.true_vx, v) for v in HELDOUT])
    return data[mask]


def prefer_postselection(data):
    post = data[data.train_seed > 2026]
    return (post, 'post-HPO') if len(post) else (data, 'HPO/reference')


def load_complete(root):
    frames, manifests, incomplete = [], [], []
    metric_versions, source_signatures = set(), set()
    for path in sorted((root/'evaluation').glob('*/*/manifest.json')):
        manifest = json.loads(path.read_text())
        entry = dict(job=manifest['job']['name'], suite=manifest['suite'],
                     family=manifest['job'].get('family', manifest['model']['kind']),
                     complete=bool(manifest.get('complete')), path=str(path.relative_to(root)),
                     sequences=int(manifest.get('sequences', 0)),
                     test_seeds=len(manifest.get('completed_seeds', [])))
        manifests.append(entry)
        if not entry['complete']:
            incomplete.append(entry); continue
        csv = path.parent/'metrics.csv'
        if not csv.exists():
            raise ValueError(f'Complete evaluation missing metrics: {csv}')
        frame = pd.read_csv(csv)
        if 'metrics_version' not in frame or not (frame.metrics_version == METRICS_VERSION).all():
            raise ValueError(f'Incompatible metric schema in {csv}; archive old evaluations before reporting v2')
        metric_versions.add(int(manifest.get('metric_schema_version', METRICS_VERSION)))
        sources = manifest.get('source_sha256', {})
        if not sources:
            raise ValueError(f'Complete evaluation lacks source hashes: {path}')
        source_signatures.add(json.dumps(sources, sort_keys=True))
        if len(frame) != manifest['metric_rows']:
            raise ValueError(f'Manifest row-count mismatch: {csv}')
        if frame.duplicated(['case_id', 'horizon']).any():
            raise ValueError(f'Duplicate evaluation rows: {csv}')
        frame['suite'] = entry['suite']
        frame['evaluation_path'] = str(path.parent.relative_to(root))
        frames.append(frame)
    if len(source_signatures) > 1 or metric_versions.difference({METRICS_VERSION}):
        raise ValueError('Completed evaluations have incompatible source hashes or metric schemas; refusing numerical pooling')
    planned_path = root/'evaluation_jobs.json'
    planned = json.loads(planned_path.read_text()) if planned_path.exists() else []
    if isinstance(planned, dict):
        planned = planned.get('jobs', [])
    finished = {(x['job'], x['suite']) for x in manifests if x['complete']}
    missing = [dict(job=x['job']['name'], suite=x['suite']) for x in planned
               if (x['job']['name'], x['suite']) not in finished]
    plan_final = (root/'training_all_complete.json').exists() and (root/'additional_training_complete.json').exists()
    uniform = metric_versions == {METRICS_VERSION} and len(source_signatures) == 1 and 'unavailable' not in source_signatures
    complete = plan_final and len(planned) > 0 and not missing and uniform
    state = dict(complete=complete, plan_final=plan_final,
                 metrics_version=METRICS_VERSION, uniform_evaluation_source_hashes=uniform,
                 planned_jobs=len(planned), completed_jobs=len(finished),
                 completed_model_sequences=sum(x['sequences'] for x in manifests if x['complete']),
                 discovered_incomplete=incomplete, missing_jobs=missing, manifests=manifests)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), state


def group_tables(data, root):
    groups, per_seed = [], []
    if data.empty:
        pd.DataFrame().to_csv(root/'group_summary.csv', index=False)
        pd.DataFrame().to_csv(root/'per_training_seed.csv', index=False)
        return groups
    keys = ['model_family', 'suite', 'mode', 'scenario', 'horizon']
    data = data.copy()
    if 'direction_split' not in data:
        data['direction_split'] = 'not_applicable'
    keys += ['direction_split']
    for key, original in data.groupby(keys, sort=True):
        metadata = dict(zip(keys, key))
        for split, subset in [('all', original), ('heldout3', heldout(original))]:
            if subset.empty:
                continue
            for cohort, selected in [('all', subset), ('post_hpo', subset[subset.train_seed > 2026]),
                                     ('hpo_2026', subset[subset.train_seed == 2026])]:
                if selected.empty:
                    continue
                stats = training_seed_summary(selected)
                row = dict(**metadata, split=split, cohort=cohort,
                           n_training_seeds=stats['n_training_seeds'],
                           test_seeds=selected.seed.nunique(), rows=len(selected))
                for metric in METRICS:
                    row[metric+'_mean'] = stats['metrics'][metric]['mean']
                    row[metric+'_sd'] = stats['metrics'][metric]['sd']
                groups.append(row)
                if cohort == 'all':
                    for seed, measured in stats['per_seed'].items():
                        per_seed.append(dict(**metadata, split=split, train_seed=int(seed), **measured))
    pd.DataFrame(groups).to_csv(root/'group_summary.csv', index=False)
    pd.DataFrame(per_seed).to_csv(root/'per_training_seed.csv', index=False)
    return groups


def common_rows(a, b):
    key = ['train_seed', 'case_id', 'horizon']
    common = a[key].merge(b[key], on=key)
    return a.merge(common, on=key), b.merge(common, on=key)


def paired_comparisons(data, repeats):
    if data.empty:
        return []
    source = heldout(data[(data.horizon == 40) & (data['mode'] == 'bilinear')])
    comparisons = []
    pairs = [('physical', name, 'core') for name in CORE+['physical_untrained'] if name != 'physical']
    pairs += [('physical', name, 'core') for name in ['raw_matching_intervention', 'mean_memory_intervention']]
    pairs += [('physical', name, 'intervention') for name in [
        'est_q1_both', 'est_q05_both', 'est_q1_transport', 'est_q05_transport']]
    pairs += [('oracle_cont', name, 'intervention') for name in [
        'oracle_q1_both', 'oracle_q05_both', 'oracle_q1_transport', 'oracle_q05_transport']]
    for family_a, family_b, suite in pairs:
        a = source[(source.model_family == family_a) & (source.suite == ('core' if family_a == 'physical' else suite))]
        b = source[(source.model_family == family_b) & (source.suite == suite)]
        if a.empty or b.empty:
            continue
        available_a = sorted(map(int, a.train_seed.unique()))
        available_b = sorted(map(int, b.train_seed.unique()))
        a, b = common_rows(a, b)
        if a.empty:
            continue
        # Exact same paired cases and paired model seeds within each cohort.
        for cohort in ['post_hpo', 'all', 'hpo_2026']:
            aa = a[a.train_seed > 2026] if cohort == 'post_hpo' else a[a.train_seed == 2026] if cohort == 'hpo_2026' else a
            bb = b[b.train_seed.isin(aa.train_seed.unique())]
            if aa.empty:
                continue
            result = paired_seed_comparison(aa, bb, repeats=repeats)
            comparisons.append(dict(a=family_a, b=family_b, suite=suite, cohort=cohort,
                                    available_training_seeds_a=available_a, available_training_seeds_b=available_b,
                                    matched_training_seeds=sorted(map(int, aa.train_seed.unique())),
                                    split='heldout3', mode='bilinear', horizon=40, **result))
    # Single deterministic/nontrained reference: conditional test uncertainty for
    # the HPO-seed model only; never duplicate it into independent training runs.
    for baseline in ['raw_untrained']:
        a = source[(source.model_family == 'physical') & (source.suite == 'core') & (source.train_seed == 2026)]
        b = source[(source.model_family == baseline) & (source.suite == 'core')]
        if not a.empty and not b.empty:
            baseline_seed = sorted(b.train_seed.unique())[0]
            b = b[b.train_seed == baseline_seed]
            comparisons.append(dict(a='physical', b=baseline, suite='core', cohort='hpo_2026_vs_reference',
                                    split='heldout3', mode='bilinear', horizon=40,
                                    result=paired_test_bootstrap(a, b, repeats=repeats)))
    # Generalization comparisons preserve each renderer and context scenario.
    generalization = [('core', 40, 'physical', other) for other in
                      ['physical_nearest_horizontal', 'physical_mixed_horizontal']]
    generalization += [('time', 160, 'physical', 'relaxed')]
    generalization += [('2d', h, 'physical_mixed_2d', 'mlp_velocity_mixed_2d') for h in [20, 40, 80]]
    for suite, horizon, family_a, family_b in generalization:
        source = data[(data.suite == suite) & (data.horizon == horizon)].copy()
        if suite != '2d':
            source = heldout(source)
        if 'direction_split' not in source:
            source['direction_split'] = 'not_applicable'
        for (mode, scenario, direction_split), group in source.groupby(['mode', 'scenario', 'direction_split']):
            a = group[group.model_family == family_a]
            b = group[group.model_family == family_b]
            if a.empty or b.empty:
                continue
            available_a = sorted(map(int, a.train_seed.unique()))
            available_b = sorted(map(int, b.train_seed.unique()))
            a, b = common_rows(a, b)
            for cohort in ['post_hpo', 'all', 'hpo_2026']:
                aa = a[a.train_seed > 2026] if cohort == 'post_hpo' else a[a.train_seed == 2026] if cohort == 'hpo_2026' else a
                bb = b[b.train_seed.isin(aa.train_seed.unique())]
                if aa.empty:
                    continue
                comparisons.append(dict(a=family_a, b=family_b, suite=suite, cohort=cohort,
                                        available_training_seeds_a=available_a, available_training_seeds_b=available_b,
                                        matched_training_seeds=sorted(map(int, aa.train_seed.unique())),
                                        split='all' if suite == '2d' else 'heldout3',
                                        direction_split=direction_split, mode=mode, scenario=scenario, horizon=horizon,
                                        **paired_seed_comparison(aa, bb, repeats=repeats)))
    return comparisons


def mechanism_plot(data, root, figures):
    subset = heldout(data[(data.suite == 'core') & (data.horizon == 40) & (data['mode'] == 'bilinear')])
    entries = []
    for family in CORE+['raw_untrained', 'physical_untrained']:
        selected = subset[subset.model_family == family]
        selected, cohort = (selected, 'reference') if family.endswith('untrained') else prefer_postselection(selected)
        if not selected.empty:
            entries.append((family, cohort, training_seed_summary(selected)))
    if not entries:
        return
    fig, axes = plt.subplots(1, 3, figsize=(max(13, len(entries)*1.9), 5.8))
    for ax, metric, title in zip(axes, ['mse', 'frame_joint_success', 'transport_ade'],
                                  ['Image MSE', 'Frame localization success (periodic ≤2 px)', 'Internal transport ADE (px)']):
        for i, (family, cohort, stats) in enumerate(entries):
            value = stats['metrics'][metric]
            ax.bar(i, value['mean'], color=COLORS.get(family, '#899598'), alpha=.85)
            if value['n'] > 1 and np.isfinite(value['sd']):
                ax.errorbar(i, value['mean'], yerr=value['sd'], color='black', capsize=4, fmt='none')
        ax.set_xticks(range(len(entries)), [LABELS.get(f, f)+f'\nn={s["n_training_seeds"]}, {c}' for f, c, s in entries], rotation=55, ha='right', fontsize=8)
        ax.set_title(title); ax.grid(axis='y', alpha=.2)
        if metric == 'frame_joint_success':
            ax.set_ylim(0, 1.05)
    fig.suptitle('Held-out speeds 0.37, 0.63, 1.23; horizon 40; bilinear test. Error bars = model-seed SD.\nFor untrained neural references, SD describes initialization variability.')
    save(fig, root, 'mechanism_controls', figures)


def image_velocity_plot(data, root, figures):
    """Velocity measured from generated images, with all missing-fit denominators."""
    subset = data[(data.suite == 'core') & (data.horizon == 40) & (data['mode'] == 'bilinear')]
    families = ['physical', 'relaxed', 'offset', 'mlp_velocity', 'raw_untrained']
    rows = []
    for family in families:
        selected = subset[subset.model_family == family]
        selected, cohort = (selected, 'reference') if family == 'raw_untrained' else prefer_postselection(selected)
        for speed, group in selected.groupby('true_vx'):
            by_seed = []
            fitted_means = []
            for _, seedgroup in group.groupby('train_seed'):
                measured = summarize(seedgroup)
                finite = np.isfinite(seedgroup.predicted_vx) & np.isfinite(seedgroup.predicted_vy)
                measured['fit_fraction'] = float(finite.mean())
                by_seed.append(measured)
                fitted_means.append(float(seedgroup.loc[finite, 'predicted_vx'].mean()) if finite.any() else np.nan)
            fitted_means = np.asarray(fitted_means)
            finite_seeds = fitted_means[np.isfinite(fitted_means)]
            finite_rows = np.isfinite(group.predicted_vx) & np.isfinite(group.predicted_vy)
            rows.append(dict(model_family=family, cohort=cohort, speed=float(speed),
                             n_model_seeds=len(by_seed), n_model_seeds_with_velocity=len(finite_seeds),
                             valid_image_fits=int(finite_rows.sum()), total_sequences=len(group),
                             mean_image_vx=float(finite_seeds.mean()) if len(finite_seeds) else np.nan,
                             model_seed_sd=float(finite_seeds.std(ddof=1)) if len(finite_seeds) > 1 else np.nan,
                             fit_fraction=float(np.mean([r['fit_fraction'] for r in by_seed])),
                             coverage=float(np.mean([r['coverage'] for r in by_seed])),
                             frame_joint_success=float(np.mean([r['frame_joint_success'] for r in by_seed])),
                             continuity_coverage=float(np.mean([r['continuity_coverage'] for r in by_seed]))))
    if not rows:
        return
    table = pd.DataFrame(rows); table.to_csv(root/'image_velocity_by_speed.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for family, group in table.groupby('model_family', sort=False):
        label = LABELS.get(family, family)+f' (n={int(group.n_model_seeds.max())}, {group.cohort.iloc[0]})'
        color = COLORS.get(family, '#787878')
        # NaN velocities remain gaps; unavailable fits are never assigned zero.
        axes[0, 0].plot(group.speed, group.mean_image_vx, marker='o', color=color, label=label)
        for ax, metric in [(axes[0, 1], 'fit_fraction'), (axes[1, 0], 'coverage'), (axes[1, 1], 'frame_joint_success')]:
            ax.plot(group.speed, group[metric], marker='o', color=color, label=label)
    axes[0, 0].plot([0, 2], [0, 2], 'k:', label='Correct velocity')
    axes[0, 0].set(ylabel='CV-fitted image velocity (px/frame)', title='Diagnostic: mean conditional on CV-certified tracks')
    axes[0, 1].set(ylabel='Valid velocity fits / evaluated sequences', title='Fit availability; counts saved in companion CSV')
    axes[1, 0].set(ylabel='CV-certified / visible target frames', title='CV-certification coverage (not model-neutral)')
    axes[1, 1].set(ylabel='Successful / visible target frames', title='Frame localization success (periodic ≤2 px)')
    axes[0, 0].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel('True horizontal velocity'); ax.grid(alpha=.2)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.set_ylim(-.03, 1.03)
    fig.suptitle('Diagnostic velocity fitted to generated images; horizon 40, bilinear.\nCV certification assumes near-constant motion and can reject correct frames; compare with frame success and evaluator ceilings.')
    save(fig, root, 'image_velocity_and_coverage', figures)


def rendering_plot(data, root, figures):
    subset = heldout(data[(data.suite == 'core') & (data.horizon == 40)])
    families = ['physical', 'physical_nearest_horizontal', 'physical_mixed_horizontal']
    matrices = [np.full((3, 2), np.nan), np.full((3, 2), np.nan)]
    annotations = [['' for _ in range(2)] for _ in range(3)]
    for i, family in enumerate(families):
        for j, mode in enumerate(['bilinear', 'nearest']):
            selected, cohort = prefer_postselection(subset[(subset.model_family == family) & (subset['mode'] == mode)])
            if selected.empty:
                continue
            stats = training_seed_summary(selected)
            for k, metric in enumerate(['mse', 'latent_velocity_error']):
                matrices[k][i, j] = stats['metrics'][metric]['mean']
            annotations[i][j] = f'n={stats["n_training_seeds"]}, {cohort}'
    if not np.isfinite(matrices[0]).any():
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in zip(axes, matrices, ['Image MSE', 'Velocity L2 error (px/frame)']):
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap='Blues', aspect='auto')
        for i in range(3):
            for j in range(2):
                label = f'{matrix[i,j]:.4g}\n{annotations[i][j]}' if np.isfinite(matrix[i,j]) else 'pending'
                ax.text(j, i, label, ha='center', va='center', fontsize=9)
        ax.set_xticks([0, 1], ['Bilinear test', 'Nearest test'])
        ax.set_yticks([0, 1, 2], ['Bilinear training', 'Nearest training', 'Mixed training'])
        ax.set_title(title); fig.colorbar(image, ax=ax, shrink=.7)
    fig.suptitle('Rendering transfer matrix; held-out speeds; horizon 40. Cell values are means across model seeds.')
    save(fig, root, 'rendering_matrix', figures)


def horizon_plot(data, root, figures):
    subset = heldout(data[(data.suite == 'time') & (data['mode'] == 'bilinear')])
    subset = subset[subset.scenario.str.contains('_regular_')]
    if subset.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for family in ['physical', 'relaxed', 'offset', 'mlp_velocity']:
        selected, cohort = prefer_postselection(subset[subset.model_family == family])
        if selected.empty:
            continue
        for ax, metric in zip(axes, ['mse', 'transport_ade', 'frame_joint_success']):
            values = [(h, training_seed_summary(g)['metrics'][metric]) for h, g in selected.groupby('horizon')]
            x = [h for h, _ in values]; y = [v['mean'] for _, v in values]
            sd = [v['sd'] if np.isfinite(v['sd']) else 0 for _, v in values]
            ax.errorbar(x, y, yerr=sd, marker='o', capsize=3, color=COLORS.get(family),
                        label=f'{LABELS[family]} (n={selected.train_seed.nunique()}, {cohort})')
    for ax, title in zip(axes, ['Image MSE', 'Internal transport ADE (px)', 'Frame localization success (periodic ≤2 px)']):
        ax.set(xlabel='Prediction horizon', title=title); ax.grid(alpha=.2)
    axes[2].set_ylim(0, 1.05); axes[0].legend(fontsize=7)
    fig.suptitle('Train horizon 40, test to 160; regular context; held-out speeds. Error bars = training-seed SD.')
    save(fig, root, 'time_generalization', figures)


def quantization_plot(data, root, figures):
    subset = data[(data.horizon == 40) & (data['mode'] == 'bilinear') & (data.train_seed == 2026)
                  & data.suite.isin(['core', 'intervention'])]
    common_speeds = [.37, .5, .63, 1, 1.23, 1.5]
    subset = subset[np.logical_or.reduce([np.isclose(subset.true_vx, v) for v in common_speeds])]
    groups = [['physical', 'est_q1_transport', 'est_q05_transport'],
              ['oracle_cont', 'oracle_q1_transport', 'oracle_q05_transport']]
    if not subset.model_family.isin(sum(groups, [])).any():
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, families in enumerate(groups):
        for family in families:
            selected = subset[subset.model_family == family]
            if selected.empty:
                continue
            # Seed2026 and the same six velocities hold the trained checkpoint
            # and dataset fixed; do not average other trained models into control.
            summary = []
            for speed, group in selected.groupby('true_vx'):
                by_seed = [summarize(seedgroup) for _, seedgroup in group.groupby('train_seed')]
                velocities = group.groupby('train_seed').latent_vx.mean()
                summary.append((speed, float(velocities.mean()), np.mean([x['mse'] for x in by_seed])))
            summary = np.asarray(summary)
            label = LABELS.get(family, family)+f' (n={selected.train_seed.nunique()})'
            line, = axes[row, 0].plot(summary[:, 0], summary[:, 1], marker='o',
                                      markerfacecolor='none', linewidth=1, label=label)
            # A mean of snapped per-sequence velocities can lie between grid
            # points. Show the actual distribution so that this average cannot
            # be mistaken for continuously valued individual predictions.
            for speed, group in selected.groupby('true_vx'):
                ordered = group.sort_values(['seed', 'case_id'])
                jitter = np.linspace(-.018, .018, len(ordered)) if len(ordered) > 1 else np.zeros(1)
                axes[row, 0].scatter(float(speed) + jitter, ordered.latent_vx,
                                     s=14, alpha=.45, color=line.get_color(),
                                     edgecolors='none', zorder=3)
            axes[row, 1].plot(summary[:, 0], summary[:, 2], marker='o', label=label)
        axes[row, 0].plot([0, 2], [0, 2], 'k:', label='Correct velocity')
        axes[row, 0].set(ylabel='Used latent horizontal velocity',
                         title=['Estimated velocity', 'Oracle velocity diagnostic'][row]
                         + '\nDots: individual scenes; open markers/lines: means')
        axes[row, 1].set(ylabel='Image MSE', title='Quantize future transport; keep memory fixed')
        for ax in axes[row]:
            ax.set_xlabel('True horizontal velocity'); ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
    fig.suptitle('Transport-only quantization: physical checkpoint seed 2026; the same six paired speeds.\nMemory is fixed within each pair. Velocity dots have small horizontal jitter; these are internal diagnostics.')
    save(fig, root, 'quantization_transport', figures)


def learning_plot(root, figures):
    selected_path = root/'selected_runs.json'
    if not selected_path.exists():
        return
    selected = json.loads(selected_path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7)); present = False
    for family in CORE:
        runs = selected.get(family, {}).get('runs', [])
        if not runs:
            continue
        path = root/'training'/runs[0]/'training.csv'
        if not path.exists():
            continue
        logs = pd.read_csv(path).dropna(subset=['val_mse'])
        if logs.empty:
            continue
        present = True
        axes[0].plot(logs.step, logs.val_mse, label=LABELS[family], color=COLORS[family])
        if family not in ('offset', 'relaxed'):
            axes[1].plot(logs.step, logs.val_velocity_l2, label=LABELS[family], color=COLORS[family])
    if not present:
        plt.close(fig); return
    axes[0].set(xlabel='Training updates', ylabel='Validation image MSE', yscale='log'); axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Training updates', ylabel='Fitted/regressed velocity L2 (px/frame)', yscale='log')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.axvline(5000, color='gray', linestyle=':'); ax.grid(alpha=.2)
    fig.suptitle('Selected learning rates, HPO training seed 2026. Curves diagnose optimization, not convergence.\nOffset/relaxed velocity lines omitted: their logged correspondence velocity is not their effective predicted motion.')
    save(fig, root, 'learning_curves', figures)


def rollout_plot(data, root, figures):
    subset = data[(data.horizon == 40) & (data['mode'] == 'bilinear') & np.isclose(data.true_vx, .37)]
    subset = subset[subset.suite.isin(['core', 'intervention'])]
    found = []
    for family in ['physical', 'raw_untrained', 'mlp_velocity', 'est_q1_transport', 'est_q05_transport']:
        group = subset[(subset.model_family == family) & (subset.seed == 0)]
        if group.empty:
            continue
        row = group.sort_values('train_seed').iloc[0]
        path = root/row.evaluation_path/'examples'/f'{row.case_id}.npz'
        if path.exists():
            found.append((family, int(row.train_seed), path))
    if not found:
        return
    with np.load(found[0][2]) as example:
        indices = example['target_indices']; truth = example['frames'][indices]
        visible = np.flatnonzero(example['visibility'][indices] >= .7)
    if len(visible) < 2:
        return
    times = visible[np.linspace(0, len(visible)-1, min(6, len(visible))).astype(int)]
    rows = [('Ground truth', truth)]
    for family, seed, path in found:
        with np.load(path) as example:
            rows.append((LABELS.get(family, family)+f'\ntrain seed {seed}', example['predictions']))
    fig, axes = plt.subplots(len(rows), len(times), figsize=(2*len(times), 1.9*len(rows)), squeeze=False)
    for i, (label, frames) in enumerate(rows):
        for j, t in enumerate(times):
            ax = axes[i, j]; ax.imshow(frames[t], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
            if i == 0:
                ax.set_title(f'+{t+1}')
    fig.suptitle('Fixed test seed 0, speed 0.37, bilinear. Times selected from GT visibility only; common intensity [0,1].')
    save(fig, root, 'selected_rollout', figures)


def table_html(groups, suite='core', split='heldout3', cohort='post_hpo', families=None):
    selected = [g for g in groups if g['suite'] == suite and g['split'] == split and g['cohort'] == cohort
                and g['horizon'] == 40 and g['mode'] == 'bilinear']
    if families is not None:
        selected = [g for g in selected if g['model_family'] in families]
    if not selected:
        return '<p>No completed runs in this partition yet.</p>'
    selected.sort(key=lambda x: x['model_family'])
    header = '<table><tr><th>Model</th><th>Training seeds</th><th>MSE</th><th>Internal transport ADE</th><th>Frame success</th><th>Continuity ADE</th><th>Continuity coverage</th></tr>'
    rows = []
    for row in selected:
        cells = [html.escape(LABELS.get(row['model_family'], row['model_family'])), str(row['n_training_seeds'])]
        for metric in ['mse', 'transport_ade', 'frame_joint_success', 'continuity_ade', 'continuity_coverage']:
            mean, sd = row[metric+'_mean'], row[metric+'_sd']
            percentage = metric in ('frame_joint_success', 'continuity_coverage')
            value = (f'{mean:.1%}' if percentage else f'{mean:.5g}') if np.isfinite(mean) else 'unavailable'
            if np.isfinite(sd):
                value += ' ± '+(f'{sd:.1%}' if percentage else f'{sd:.2g}')
            cells.append(value)
        rows.append('<tr>'+''.join(f'<td>{cell}</td>' for cell in cells)+'</tr>')
    return header+''.join(rows)+'</table>'


def direction_table_html(groups):
    selected = [g for g in groups if g['suite'] == '2d' and g['split'] == 'all'
                and g['cohort'] == 'post_hpo' and g['horizon'] == 80
                and g['model_family'] in ['physical_mixed_2d', 'mlp_velocity_mixed_2d']]
    if not selected:
        return '<p>No completed post-HPO 2D runs yet.</p>'
    rows = []
    for row in sorted(selected, key=lambda r: (r['mode'], r['direction_split'], r['model_family'])):
        cells = [LABELS.get(row['model_family'], row['model_family']), row['mode'],
                 row['direction_split'], str(row['n_training_seeds'])]
        for metric in ['mse', 'transport_ade', 'frame_joint_success', 'continuity_coverage']:
            value, sd = row[metric+'_mean'], row[metric+'_sd']
            cells.append((f'{value:.4g}' if np.isfinite(value) else 'unavailable')+
                         (f' ± {sd:.2g}' if np.isfinite(sd) else ''))
        rows.append('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in cells)+'</tr>')
    return '<table><tr><th>Model</th><th>Renderer</th><th>Directions</th><th>Seeds</th><th>MSE</th><th>Internal transport ADE</th><th>Frame success</th><th>Continuity coverage</th></tr>'+''.join(rows)+'</table>'


def ceiling_html(root):
    path = root/'measurement_ceiling.json'
    if not path.exists():
        return '<p>Perfect-prediction evaluator calibration is pending.</p>', {}
    result = json.loads(path.read_text())
    if not result.get('complete'):
        return '<p>Perfect-prediction evaluator calibration is running.</p>', result
    rows = []
    for row in result['summaries']:
        cells = [str(row[key]) for key in ['suite', 'mode', 'scenario', 'direction_split', 'horizon']]
        for metric in ['frame_joint_success', 'certified_joint_success', 'continuity_coverage']:
            cells.append(f'{row[metric]:.1%}' if row.get(metric) is not None else 'unavailable')
        rows.append('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in cells)+'</tr>')
    table = '<table><tr><th>Suite</th><th>Renderer</th><th>Context</th><th>Directions</th><th>Horizon</th><th>Frame success</th><th>CV-certified success</th><th>Continuity coverage</th></tr>'+''.join(rows)+'</table>'
    return '<p>These predictions are exactly the ground-truth frames; MSE is zero. Any failed certification or missing continuity coverage is an evaluator limitation.</p><details><summary>Perfect-frame measurement ceilings for every suite and horizon</summary>'+table+'</details>', result


def qualitative_artifacts(root):
    """Expose only current provenance-listed generated panels, without writing them."""
    paths = [root/'figures/qualitative_provenance.json', root/'qualitative_provenance.json']
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        return '<p>Qualitative panel provenance is not available yet.</p>', {}
    provenance = json.loads(path.read_text())
    relative = str(path.relative_to(root))
    if provenance.get('metric_schema') != METRICS_VERSION:
        return '<p>Qualitative panels use a different metric schema and are excluded.</p>', dict(provenance=relative, compatible=False, figures=[])
    shown, unavailable = [], []
    for figure in provenance.get('figures', []):
        if not figure.get('generated'):
            continue
        stem = figure.get('stem', '')
        if not stem.startswith('qualitative_') or Path(stem).name != stem:
            raise ValueError(f'Unexpected qualitative figure stem: {stem}')
        png, pdf = root/'figures'/f'{stem}.png', root/'figures'/f'{stem}.pdf'
        if not png.exists() or not pdf.exists():
            unavailable.append(stem); continue
        shown.append(dict(stem=stem, png=str(png.relative_to(root)), pdf=str(pdf.relative_to(root))))
    selection = provenance.get('selection', {})
    intro = (f'<p>Panels below are listed as generated in <a href="{html.escape(relative)}">qualitative_provenance.json</a> '
             f'and have both PNG and PDF artifacts. Selection uses fixed test seed {selection.get("test_seed", "unknown")} '
             'and predetermined horizons, without filtering by prediction error or visibility. '
             'Uniform contact sheets retain every specified time, including views where the digit is outside the camera. '
             'Missing panels inside a sheet denote pending examples, not model failures.</p>')
    panels = []
    for figure in shown:
        name = figure['stem'].removeprefix('qualitative_').replace('_', ' ')
        caption = html.escape(name.title())
        panels.append(f'<details><summary>{caption} — PNG and PDF</summary><a href="{html.escape(figure["pdf"])}"><img loading="lazy" src="{html.escape(figure["png"])}" alt="{caption}"></a></details>')
    if unavailable:
        intro += '<p>Provenance-listed panels with missing PNG/PDF files are omitted until both artifacts are available.</p>'
    return intro+''.join(panels), dict(provenance=relative, compatible=True,
                                      generated=provenance.get('generated'), selection=selection,
                                      figures=shown, unavailable_files=unavailable)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/continuous_velocity/paper_v1'))
    parser.add_argument('--bootstrap', type=int, default=2000)
    args = parser.parse_args(); root = args.root.resolve()
    (root/'figures').mkdir(parents=True, exist_ok=True)
    data, state = load_complete(root)
    groups = group_tables(data, root)
    comparisons = paired_comparisons(data, args.bootstrap)
    calibration_html, calibration = ceiling_html(root)
    qualitative_html, qualitative = qualitative_artifacts(root)
    review_path = root/'independent_review.md'
    review_link = '<a href="independent_review.md">Independent adversarial review and follow-up checks</a>' if review_path.exists() else 'Independent review artifact pending'
    figures = []
    if not data.empty:
        for function in [mechanism_plot, image_velocity_plot, rendering_plot, horizon_plot, quantization_plot, rollout_plot]:
            function(data, root, figures)
    learning_plot(root, figures)
    summary = dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), status=state, groups=groups,
                   paired_comparisons=comparisons, figures=figures,
                   metric_schema_version=METRICS_VERSION, measurement_ceiling=calibration,
                   qualitative_artifacts=qualitative,
                   primary_cohort='Training seeds >2026; seed2026 was used for learning-rate selection and is reported separately.',
                   weighting='Within each model: pooled frame/pixel ratios; across trained models: mean and sample SD.',
                   uncertainty='Conditional paired test-seed intervals do not quantify future training-seed variability.')
    write_json(root/'summary.json', summary)
    missing = ''.join(f'<li>{html.escape(x["job"])} / {html.escape(x["suite"])}</li>' for x in state['missing_jobs'])
    figure_html = ''.join(f'<h2>{name.replace("_", " ").title()}</h2><a href="figures/{name}.pdf"><img src="figures/{name}.png"></a>' for name in figures)
    title = 'Completed evidence campaign' if state['complete'] else 'Evidence campaign — intermediate results'
    report = f'''<!doctype html><meta charset="utf-8"><title>Continuous velocity evidence campaign</title>
<style>body{{font:16px/1.6 system-ui;max-width:1250px;margin:40px auto;padding:0 25px;color:#18252b}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.status{{padding:15px;background:#f1f5f7}}code{{background:#eee}}</style>
<h1>{title}</h1><p class="status">{state['completed_jobs']} completed evaluation jobs / {state['planned_jobs']} currently planned; {state['completed_model_sequences']:,} completed model-sequences. Plan final: {state['plan_final']}. Only complete manifests enter this report. Planning remains dynamic until all training stages finish.</p>
<p>This report separates mechanism controls, independently trained model seeds, rendering transfer, and temporal generalization. The architecture matches spatial features, fits one continuous velocity, then uses that velocity for memory alignment and future transport. The fitted velocity does not feed back into initial correspondence matching.</p>
<p>The MLP velocity control shares the same strictly constant-velocity transport law. Physical-versus-MLP comparisons replace the estimator and also change its initial function and optimization problem; they do not test whether constant-velocity transport is necessary. Physical versus relaxed is the cleaner same-initial-prediction test of strict motion versus learned corrections. Independent review verified finite, nonzero image-only training gradients through the MLP head and shared learned components; gradient connectivity does not establish equally easy optimization. See {review_link}.</p>
<h2>Primary post-selection results</h2><p>Held-out speeds 0.37, 0.63, 1.23; bilinear observations; horizon 40. Values are mean ± sample SD across training seeds greater than 2026. Learning-rate selection used seed 2026, which is excluded from this primary table. Different controls may currently have different completed seed counts.</p>
{table_html(groups, families=CORE)}
<h2>All available core training seeds</h2><p>This supplementary table includes HPO seed 2026. Each paired comparison explicitly intersects training seeds and case IDs; the machine-readable result records available and matched seed sets.</p>{table_html(groups, cohort='all', families=CORE)}
<h2>Nontrained references</h2><p>These reference implementations/checkpoints remain visible regardless of whether their initialization identifier is 2026. Untrained neural models may vary initialization; their SD describes initialization variability, not training. A single deterministic raw reference has no estimated seed SD.</p>{table_html(groups, cohort='all', families=['raw_untrained', 'physical_untrained'])}
<h2>Oracle velocity diagnostics</h2><p>True velocity is given only in these explicitly labeled interventions. These within-checkpoint diagnostics are separate from the post-HPO trained-model comparison.</p>{table_html(groups, suite='intervention', cohort='all', families=['oracle_cont', 'oracle_q1_both', 'oracle_q05_both', 'oracle_q1_transport', 'oracle_q05_transport'])}
<h2>Extended-budget optimization diagnostics</h2><p>These longer runs use the HPO seed and are reported separately from the fixed 5,000-update multi-seed comparison.</p>{table_html(groups, cohort='all', families=[f+'_extended' for f in CORE])}
<h2>Two-dimensional motion: seen and held-out directions</h2><p>Horizon 80, post-HPO training seeds, separated by test renderer and direction split. Physical-versus-MLP paired intervals at horizons 20, 40, and 80 are available in summary.json; each comparison intersects model seeds and case IDs. Speed magnitudes are not matched between seen and held-out directions: absolute held-out performance and within-split model comparisons are valid, but differences between direction splits cannot be attributed solely to angle.</p>{direction_table_html(groups)}
<h2>Evaluation definitions — schema 2</h2><p><strong>Image MSE and frame-localization success are the primary image outcomes.</strong> Frame success requires a sufficiently visible target, a present detection without detected duplication, and per-frame periodic localization error ≤2 px. It uses no velocity fit and does not measure unwrapped trajectory agreement: whole-world winding errors can have zero periodic localization error.</p>
<p><strong>Continuity ADE</strong> is a separate unwrapped trajectory diagnostic. Its last accepted context detection fixes a global winding once; future winding follows bounded observed increments, at most 2.5 px/frame per axis plus 1 px localization allowance. A gap permitting half-world traversal or an implausible/nonunique increment makes winding permanently unknown. No constant-velocity fit or GT reset repairs a gap. This conservative measure may have low coverage, which is reported alongside its conditional error.</p>
<p><strong>Internal transport ADE</strong> compares each model's displacement output with true accumulated motion. It is an internal motion-readout diagnostic, not object-position ADE, and does not include appearance or initial-position error. GT motion is used only by the evaluator. Legacy <code>ade</code>, <code>coverage</code>, and <code>joint_success</code> retain their old definitions and are explicitly CV-certified diagnostics; schema 2 adds named <code>certified_*</code> aliases. CV certification can reject correct frames and nonlinear predictions, so it is not a primary model-neutral outcome.</p>
<p>Rates pool visible frames; MSE pools all pixels. Each training seed is summarized separately before means and SDs are calculated. The known digit template is supplied only to the evaluator. Detector localization has a pixel-scale floor.</p>
<h2>Perfect-prediction measurement ceilings</h2>{calibration_html}<p><a href="measurement_ceiling.csv">Per-case perfect-prediction measurements</a> · <a href="measurement_ceiling.json">Ceiling summary and source hashes</a></p>
<p>Paired test-seed confidence intervals resample complete paired scenes, preserving their speed conditions, and use the same pooled estimators as point scores. The intervals are conditional on the trained model set. Training-seed mean/SD and paired training-seed differences are reported separately in <a href="summary.json">summary.json</a>; test examples are not substitutes for independent training runs.</p>
<h2>Interpretation limits</h2><p>Initial matching, continuous least-squares fitting, fixed appearance reuse, learned fusion, and decoder learning are distinct components. Inference-time quantization can change either both alignment and transport or transport alone; only the latter holds canonical memory fixed. Oracle velocities are diagnostic references, not guaranteed image-error bounds, and are never formal model inputs: a learned memory or renderer may compensate for velocity-estimation bias.</p>
<p>This is a single rigid object, constant-velocity setting with known camera translation. A frozen FloWM comparison does not establish architectural superiority over a FloWM trained on matched data. Extended training curves diagnose optimization but do not prove convergence. Irregular contexts and equal-count sparse contexts are separate controls. Context blackout is an observation-gap test, not a controlled geometric out-of-view intervention. Shared GPU timings are not speed benchmarks.</p>
<p>The fixed detector's calibration found limited recall for closely overlapping copies; write “no detected ghosts,” never “no ghosts.” A localized prediction can still be blurred or fragmented. See <a href="calibration/summary.json">calibration results</a>, <a href="calibration/controls.png">controls</a>, and <a href="calibration/manifest.json">hashes and settings</a>.</p>
{figure_html}
<h2>Fixed qualitative comparisons and uniform contact sheets</h2>{qualitative_html}
<h2>Artifacts and provenance</h2><p><a href="group_summary.csv">Group means and SDs</a> · <a href="per_training_seed.csv">Individual model-seed scores</a> · <a href="image_velocity_by_speed.csv">Image-derived velocities and valid-fit counts</a> · <a href="summary.json">Paired analyses and completion details</a> · <a href="protocol.json">Protocol</a> · <a href="evaluation_jobs.json">Evaluation jobs</a> · <a href="../../../experiments/continuous_velocity/paper/literature.md">Primary-source literature and claim boundaries</a></p>
<p>{review_link} · <a href="{html.escape(qualitative.get('provenance', 'figures/qualitative_provenance.json'))}">Qualitative selection, source examples, and hashes</a></p>
<h2>Pending evaluation jobs</h2><p>Missing jobs are excluded from every numerical conclusion. Additional jobs may appear while training is in progress.</p><ul>{missing or '<li>None in the current plan.</li>'}</ul>'''
    (root/'report.html').write_text(report)
    print(json.dumps(dict(complete=state['complete'], completed_jobs=state['completed_jobs'],
                         planned_jobs=state['planned_jobs'], groups=len(groups),
                         paired_comparisons=len(comparisons), figures=figures)))


if __name__ == '__main__':
    main()
