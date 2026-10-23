"""Factual report for the kinetic study from completed evaluations.

Run: python experiments/continuous_velocity/kinetic/report.py --root outputs/continuous_velocity/kinetic_v1

Every score pools frames/pixels within one fitted model (schema-2 ratios of
sums), then reports mean and sample SD across training seeds. Primary
hypotheses use a paired t-test across training seeds on the prespecified
metric with Holm correction; the paired scene-cluster bootstrap conditional on
the fitted models is reported separately. Missing or incomplete jobs are listed,
never silently dropped.
"""
import argparse
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
from scipy import stats

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from paper.metrics import summarize  # noqa: E402
from paper.statistics import paired_seed_comparison  # noqa: E402
from paper.report import clean, write_json  # noqa: E402

HELDOUT = [.37, .63, 1.23]
PRIMARY = [
    dict(name='P1', a='A_spectral', b='A_base', suite='core', metric='mse', horizon=40, mode='bilinear',
         claim='spectral alignment and rendering reduce MSE', kind='superiority'),
    dict(name='P2', a='A_newton', b='A_base', suite='core', metric='mse', horizon=40, mode='bilinear',
         claim='Newton refinement reduces MSE', kind='superiority'),
    dict(name='P3', a='A_order2', b='A_base', suite='core', metric='mse', horizon=40, mode='bilinear',
         claim='second order costs less than 1e-4 MSE on constant velocity', kind='noninferiority', margin=1e-4),
    dict(name='P4', a='B_order2', b='B_base', suite='accel', metric='mse', horizon=40, mode='bilinear',
         claim='second order reduces MSE under acceleration', kind='superiority'),
]
SECONDARY = [('A_alignS', 'A_base', 'core'), ('A_renderS', 'A_base', 'core'), ('A_spectral_newton', 'A_base', 'core'),
             ('A_order2_k4', 'A_base', 'core'), ('A_order2_k0', 'A_base', 'core'),
             ('A_order2', 'A_base', 'time'), ('A_spectral', 'A_base', 'time'), ('A_newton', 'A_base', 'time'),
             ('B_order2', 'B_relaxed', 'accel'), ('B_order2', 'B_offset', 'accel'), ('B_order2', 'B_order2_k4', 'accel'),
             ('B_order2', 'B_order2_k0', 'accel'), ('B_order2', 'B_order2_zeroacc_both', 'accel'),
             ('B_order2', 'B_order2_zeroacc_transport', 'accel'), ('B_order2', 'B_base', 'accel_long'),
             ('A_order2', 'A_base', 'accel'), ('A_order2', 'A_order2_zeroacc_both', 'accel'), ('B_base', 'A_base', 'accel'),
             ('B_relaxed', 'B_base', 'accel'), ('B_offset', 'B_base', 'accel')]
N_PRIMARY = 4
HASHED = {'kinetic__memory.py': 'experiments/continuous_velocity/kinetic/memory.py',
          'kinetic__transport.py': 'experiments/continuous_velocity/kinetic/transport.py',
          'kinetic__scenes.py': 'experiments/continuous_velocity/kinetic/scenes.py',
          'model.py': 'experiments/continuous_velocity/model.py'}
TABLE_METRICS = ['mse', 'frame_joint_success', 'missing_rate', 'ghost_rate', 'transport_ade',
                 'correspondence_velocity_error', 'latent_velocity_error', 'continuity_ade', 'continuity_coverage']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000', '#999999']


def hash_mismatches(root, manifest):
    """Training-time versus evaluation-time hashes of the model sources."""
    run_id = manifest['job'].get('run_id')
    if not run_id:
        return []
    training = root/'training'/run_id/'manifest.json'
    if not training.exists():
        return [f'{run_id}: training manifest missing']
    trained = json.loads(training.read_text())['source_sha256']
    evaluated = manifest['source_sha256']
    return [f'{run_id}: {key}' for key, path in HASHED.items()
            if key in trained and path in evaluated and trained[key] != evaluated[path]]


def load(root):
    frames, manifests, incomplete, mismatches = [], [], [], []
    for manifest_path in sorted((root/'evaluation').glob('*/*/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('complete'):
            incomplete.append(str(manifest_path.parent.relative_to(root)))
            continue
        bad = hash_mismatches(root, manifest)
        if bad:
            mismatches.extend(bad)
            incomplete.append(str(manifest_path.parent.relative_to(root))+' (source hash mismatch, excluded)')
            continue
        data = pd.read_csv(manifest_path.parent/'metrics.csv')
        data['suite'] = manifest['suite']
        data['confirmatory'] = manifest.get('seed_start', 0) > 0
        frames.append(data)
        manifests.append(manifest)
    if not frames:
        raise SystemExit('No complete evaluations found')
    data = pd.concat(frames, ignore_index=True)
    data['abs_acceleration'] = data['acceleration_x'].abs()
    return data, manifests, incomplete, mismatches


def heldout(data):
    mask = np.isclose(data.initial_vy, 0) & np.isclose(data.acceleration_x, 0)
    mask &= np.logical_or.reduce([np.isclose(data.initial_vx, v) for v in HELDOUT])
    return data[mask]


def per_seed_scores(rows, metric):
    """Pooled ratio-of-sums metrics via summarize; per-sequence means otherwise."""
    scores = {}
    for seed, part in rows.groupby('train_seed'):
        summary = summarize(part)
        scores[int(seed)] = (summary[metric] if metric in summary
                             else float(np.nanmean(part[metric].astype(float).values)))
    return scores


def family_table(data, suite, horizon, mode, selector=None):
    rows = data[(data.suite == suite) & (data.horizon == horizon) & (data['mode'] == mode) & (~data.confirmatory)]
    if selector is not None:
        rows = selector(rows)
    table = []
    for family, part in rows.groupby('model_family'):
        per_seed = {metric: per_seed_scores(part, metric) for metric in TABLE_METRICS}
        entry = dict(family=family, n_training_seeds=part.train_seed.nunique(),
                     n_sequences=int(part.case_id.nunique()), suite=suite, horizon=horizon, mode=mode,
                     spec={k[5:]: part[k].iloc[0] for k in part.columns if k.startswith('spec_')})
        for metric in TABLE_METRICS:
            values = np.array(list(per_seed[metric].values()), float)
            entry[metric] = dict(mean=float(np.nanmean(values)), sd=float(np.nanstd(values, ddof=1)) if len(values) > 1 else float('nan'),
                                 per_seed=per_seed[metric])
        table.append(entry)
    return table


def contrast(data, a, b, suite, metric, horizon, mode, selector, kind='superiority', margin=None):
    rows = data[(data.suite == suite) & (data.horizon == horizon) & (data['mode'] == mode) & (~data.confirmatory)]
    rows = selector(rows)
    ra, rb = rows[rows.model_family == a], rows[rows.model_family == b]
    result = dict(a=a, b=b, suite=suite, metric=metric, horizon=horizon, mode=mode, kind=kind, margin=margin)
    if ra.empty or rb.empty:
        result['status'] = 'missing'
        return result
    sa, sb = per_seed_scores(ra, metric), per_seed_scores(rb, metric)
    seeds = sorted(set(sa) & set(sb))
    if len(seeds) < 2:
        result['status'] = 'insufficient seeds'
        return result
    diff = np.array([sa[s]-sb[s] for s in seeds])
    se = diff.std(ddof=1)/np.sqrt(len(seeds))
    if se > 0:
        t = stats.ttest_rel([sa[s] for s in seeds], [sb[s] for s in seeds])
        t_statistic, p_two = float(t.statistic), float(t.pvalue)
    else:
        t_statistic, p_two = float('inf') if diff.mean() != 0 else 0., (0. if diff.mean() != 0 else 1.)
    result.update(status='ok', training_seeds=seeds, per_seed_difference=diff.tolist(),
                  mean_difference=float(diff.mean()), sd_difference=float(diff.std(ddof=1)),
                  paired_t=t_statistic, p_two_sided=p_two, n=len(seeds))
    upper95 = diff.mean()+stats.t.ppf(.95, len(seeds)-1)*se
    result['one_sided_upper95'] = float(upper95)
    if kind == 'noninferiority':
        result['noninferior'] = bool(upper95 < margin)
        # One-sided test of H0: difference >= margin.
        if se > 0:
            result['p_noninferiority'] = float(stats.t.cdf((diff.mean()-margin)/se, len(seeds)-1))
        else:
            result['p_noninferiority'] = 0. if diff.mean() < margin else 1.
    try:
        boot = paired_seed_comparison(ra[ra.train_seed.isin(seeds)].to_dict('records'),
                                      rb[rb.train_seed.isin(seeds)].to_dict('records'))
        result['conditional_bootstrap'] = boot['conditional_test_intervals'].get(metric, dict(error='not bootstrapped'))
    except (ValueError, KeyError) as error:
        result['conditional_bootstrap'] = dict(error=str(error))
    return result


def holm(results, family_size=N_PRIMARY):
    """Holm step-down over the fixed primary family; a missing contrast counts as p = 1.

    Superiority is declared only when the adjusted p is below .05 AND the mean
    difference favours A (negative for error metrics).
    """
    pvalues = []
    for r in results:
        if r.get('status') != 'ok':
            pvalues.append(1.)
        else:
            pvalues.append(r['p_noninferiority'] if r['kind'] == 'noninferiority' else r['p_two_sided'])
    m = max(family_size, len(pvalues))
    order = np.argsort(pvalues)
    adjusted = [None]*len(pvalues)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (m-rank)*pvalues[index])
        adjusted[index] = min(1., running)
    for r, p, q in zip(results, pvalues, adjusted):
        r.update(p_raw=p, p_holm=q, holm_family_size=m)
        if r.get('status') != 'ok':
            r['significant_at_05_holm'] = False
        elif r['kind'] == 'noninferiority':
            r['significant_at_05_holm'] = bool(q < .05)
        else:
            r['significant_at_05_holm'] = bool(q < .05 and r['mean_difference'] < 0)
    return results


def figure_acceleration(data, root, figures):
    rows = data[(data.suite == 'accel') & (data.horizon == 40) & (data['mode'] == 'bilinear') & (~data.confirmatory)]
    families = [f for f in ['A_base', 'B_base', 'B_order2', 'B_order2_k4', 'B_order2_k0', 'B_relaxed', 'B_offset',
                            'B_order2_zeroacc_both'] if f in set(rows.model_family)]
    if not families:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric, label in zip(axes, ['mse', 'frame_joint_success'], ['Image MSE, horizon 40', 'Frame localization success']):
        for color, family in zip(COLORS, families):
            part = rows[rows.model_family == family]
            levels = sorted(part.acceleration_x.unique())
            means, sds = [], []
            for level in levels:
                scores = list(per_seed_scores(part[np.isclose(part.acceleration_x, level)], metric).values())
                means.append(np.mean(scores)); sds.append(np.std(scores, ddof=1) if len(scores) > 1 else 0)
            ax.errorbar(levels, means, yerr=sds, color=color, lw=2, marker='o', ms=5, capsize=2, label=family)
        ax.set_xlabel('true acceleration (px/frame$^2$)'); ax.set_ylabel(label)
        ax.grid(alpha=.25); ax.spines[['top', 'right']].set_visible(False)
        if metric == 'mse':
            ax.set_yscale('log')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Bilinear rendering, held-out initial speeds, mean ± SD over training seeds', fontsize=10)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(root/'figures'/f'accel_mse_success.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig); figures.append('accel_mse_success')


def figure_horizon(data, root, figures):
    rows = data[(data.suite == 'time') & (~data.confirmatory)]
    families = [f for f in ['A_base', 'A_spectral', 'A_newton', 'A_order2', 'A_order2_k0', 'A_alignS', 'A_renderS'] if f in set(rows.model_family)]
    if not families:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, mode in zip(axes, ['bilinear', 'nearest']):
        for color, family in zip(COLORS, families):
            part = rows[(rows.model_family == family) & (rows['mode'] == mode)]
            horizons = sorted(part.horizon.unique())
            means = [np.mean(list(per_seed_scores(part[part.horizon == h], 'mse').values())) for h in horizons]
            ax.plot(horizons, means, color=color, lw=2, marker='o', ms=5, label=family)
        ax.set_title(f'{mode} rendering', fontsize=10); ax.set_xlabel('horizon (frames)'); ax.set_yscale('log')
        ax.grid(alpha=.25); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Image MSE (mean over training seeds)'); axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(root/'figures'/f'time_mse.{suffix}', dpi=160, bbox_inches='tight')
    plt.close(fig); figures.append('time_mse')


def markdown_table(table, metrics):
    lines = ['| family | seeds | ' + ' | '.join(metrics) + ' |', '|---|---:|' + '---:|'*len(metrics)]
    for entry in sorted(table, key=lambda e: e['family']):
        cells = [f"{entry[m]['mean']:.6g} ± {entry[m]['sd']:.2g}" if np.isfinite(entry[m]['sd']) else f"{entry[m]['mean']:.6g}"
                 for m in metrics]
        lines.append(f"| {entry['family']} | {entry['n_training_seeds']} | " + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('outputs/continuous_velocity/kinetic_v1'))
    args = parser.parse_args()
    root = args.root
    (root/'report').mkdir(exist_ok=True)
    (root/'figures').mkdir(exist_ok=True)
    data, manifests, incomplete, mismatches = load(root)
    figures = []
    tables = {}
    tables['core_heldout_bilinear_h40'] = family_table(data, 'core', 40, 'bilinear', heldout)
    tables['core_heldout_nearest_h40'] = family_table(data, 'core', 40, 'nearest', heldout)
    tables['core_all_bilinear_h40'] = family_table(data, 'core', 40, 'bilinear')
    tables['time_bilinear_h160'] = family_table(data, 'time', 160, 'bilinear')
    tables['time_nearest_h160'] = family_table(data, 'time', 160, 'nearest')
    tables['accel_bilinear_h40'] = family_table(data, 'accel', 40, 'bilinear')
    tables['accel_nearest_h40'] = family_table(data, 'accel', 40, 'nearest')
    tables['accel_bilinear_h40_nonzero'] = family_table(data, 'accel', 40, 'bilinear', lambda r: r[r.acceleration_x != 0])
    tables['accel_bilinear_h40_zero'] = family_table(data, 'accel', 40, 'bilinear', lambda r: r[r.acceleration_x == 0])
    tables['accel_long_bilinear_h80'] = family_table(data, 'accel_long', 80, 'bilinear')
    tables['accel_long_nearest_h80'] = family_table(data, 'accel_long', 80, 'nearest')
    by_level = {}
    accel = data[(data.suite == 'accel') & (data.horizon == 40) & (~data.confirmatory)]
    for (family, mode, level), part in accel.groupby(['model_family', 'mode', 'acceleration_x']):
        by_level.setdefault(family, {}).setdefault(mode, {})[f'{level:+g}'] = {
            m: float(np.mean(list(per_seed_scores(part, m).values()))) for m in ('mse', 'frame_joint_success', 'transport_ade')}
    primary = [contrast(data, p['a'], p['b'], p['suite'], p['metric'], p['horizon'], p['mode'],
                        heldout if p['suite'] == 'core' else (lambda r: r), p['kind'], p.get('margin')) for p in PRIMARY]
    for p, r in zip(PRIMARY, primary):
        r.update(name=p['name'], claim=p['claim'])
    primary = holm(primary)
    secondary = [contrast(data, a, b, suite, 'mse', 40 if suite != 'time' else 160, 'bilinear',
                          heldout if suite == 'core' else (lambda r: r)) for a, b, suite in SECONDARY]
    secondary += [contrast(data, a, b, suite, 'mse', 40 if suite != 'time' else 160, 'nearest',
                           heldout if suite == 'core' else (lambda r: r)) for a, b, suite in SECONDARY]
    secondary += [contrast(data, p['a'], p['b'], p['suite'], 'frame_joint_success', p['horizon'], p['mode'],
                           heldout if p['suite'] == 'core' else (lambda r: r)) for p in PRIMARY]
    secondary += [contrast(data, 'A_newton', 'A_base', 'core', 'correspondence_velocity_error', 40, mode, heldout) for mode in ('bilinear', 'nearest')]
    selected = json.loads((root/'selected_runs.json').read_text()) if (root/'selected_runs.json').exists() else {}
    best = selected.get('combined', {}).get('selected_transport')
    combined = []
    if best and best != 'base':
        for group, suite in (('A', 'core'), ('A', 'time'), ('B', 'accel'), ('B', 'accel_long')):
            horizon = dict(core=40, time=160, accel=40, accel_long=80)[suite]
            combined.append(contrast(data, f'{group}_order2_{best}', f'{group}_base', suite, 'mse', horizon, 'bilinear',
                                     heldout if suite == 'core' else (lambda r: r)))
            combined.append(contrast(data, f'{group}_order2_{best}', f'{group}_order2', suite, 'mse', horizon, 'bilinear',
                                     heldout if suite == 'core' else (lambda r: r)))
        combined.append(contrast(data, f'A_{best}', 'A_base', 'core', 'mse', 40, 'bilinear', heldout))
    confirm = {}
    conf = data[data.confirmatory]
    if len(conf):
        for suite in sorted(conf.suite.unique()):
            for mode in ('bilinear', 'nearest'):
                confirm[f'{suite}_{mode}_h40'] = family_table(conf.assign(confirmatory=False), suite, 40, mode,
                                                              heldout if suite == 'core' else None)
        confirm['contrasts'] = [contrast(conf.assign(confirmatory=False), p['a'], p['b'], p['suite'], p['metric'], p['horizon'],
                                         p['mode'], heldout if p['suite'] == 'core' else (lambda r: r), p['kind'], p.get('margin'))
                                for p in PRIMARY]
        for r, p in zip(confirm['contrasts'], PRIMARY):
            r.update(name=p['name'])
        if best and best != 'base':
            confirm['combined'] = [contrast(conf.assign(confirmatory=False), f'{g}_order2_{best}', f'{g}_base', suite, 'mse', 40,
                                            'bilinear', heldout if suite == 'core' else (lambda r: r))
                                   for g, suite in (('A', 'core'), ('B', 'accel'))]
    extended = {}
    for label in ('hpo5k', '20k'):
        for suite, horizon in (('core', 40), ('time', 160), ('accel', 40)):
            extended[f'{label}_{suite}_bilinear_h{horizon}'] = family_table(
                data, suite, horizon, 'bilinear',
                lambda r, label=label, suite=suite: (heldout(r) if suite == 'core' else r)[
                    (heldout(r) if suite == 'core' else r).model_family.str.endswith('_'+label)])
    figure_acceleration(data, root, figures)
    figure_horizon(data, root, figures)
    summary = dict(generated=time.strftime('%Y-%m-%dT%H:%M:%S%z'), complete_evaluations=len(manifests),
                   incomplete_evaluations=incomplete, source_hash_mismatches=mismatches, metric_rows=int(len(data)),
                   sequences=int(data.groupby(['suite', 'case_id']).ngroups), selection=selected,
                   tables=tables, accel_by_level=by_level, primary=primary, secondary=secondary,
                   combined_variant=dict(selected_transport=best, contrasts=combined),
                   confirmatory=confirm, extended=extended, figures=figures,
                   notes=['Scores pool frames/pixels within a fitted model; mean ± SD across training seeds.',
                          'Primary p-values: paired t-test across training seeds, Holm-corrected over P1-P4.',
                          'Conditional bootstrap intervals resample test scene clusters with fitted models fixed.',
                          'Group B models train on accelerated data; their speeds are not held out.',
                          'Nonsignificance is not equivalence except for the prespecified P3 non-inferiority test.'])
    write_json(root/'report'/'summary.json', summary)
    lines = ['# Kinetic study results (auto-generated)', '',
             f'Complete evaluations: {len(manifests)}; incomplete or excluded: {len(incomplete)}; '
             f'suite-sequences: {data.groupby(["suite", "case_id"]).ngroups}; source hash mismatches: {len(mismatches)}.', '']
    for key in ('core_heldout_bilinear_h40', 'core_heldout_nearest_h40', 'time_bilinear_h160', 'time_nearest_h160',
                'accel_bilinear_h40', 'accel_bilinear_h40_nonzero', 'accel_nearest_h40', 'accel_long_bilinear_h80'):
        lines += [f'## {key}', '', markdown_table(tables[key], ['mse', 'frame_joint_success', 'transport_ade',
                                                                 'correspondence_velocity_error']), '']
    lines += ['## Primary hypotheses (paired across training seeds, Holm-corrected)', '']
    for r in primary:
        if r.get('status') != 'ok':
            lines.append(f"- {r['name']} {r['a']} vs {r['b']}: {r.get('status')}")
            continue
        extra = f", non-inferior at margin {r['margin']:g}: {r['noninferior']}" if r['kind'] == 'noninferiority' else ''
        ci = r.get('conditional_bootstrap', {}).get('ci95')
        lines.append(f"- {r['name']} {r['a']} − {r['b']} ({r['suite']} {r['mode']} h{r['horizon']} {r['metric']}): "
                     f"mean {r['mean_difference']:+.3g} ± {r['sd_difference']:.2g} over {r['n']} seeds, "
                     f"p={r['p_raw']:.3g}, Holm p={r['p_holm']:.3g}, significant={r['significant_at_05_holm']}{extra}; "
                     f"conditional bootstrap CI {ci}")
    if combined:
        lines += ['', f'## Validation-selected combined variant (order2_{best}; descriptive)', '']
        for r in combined:
            if r.get('status') == 'ok':
                lines.append(f"- {r['a']} − {r['b']} ({r['suite']} {r['metric']} h{r['horizon']}): mean {r['mean_difference']:+.3g} "
                             f"± {r['sd_difference']:.2g}, p={r['p_two_sided']:.3g}")
            else:
                lines.append(f"- {r['a']} − {r['b']} ({r['suite']}): {r.get('status')}")
    lines += ['', '## Secondary contrasts (descriptive, bilinear MSE)', '']
    for r in secondary:
        if r.get('status') == 'ok' and r['mode'] == 'bilinear' and r['metric'] == 'mse':
            lines.append(f"- {r['a']} − {r['b']} ({r['suite']} h{r['horizon']}): mean {r['mean_difference']:+.3g} ± {r['sd_difference']:.2g}, p={r['p_two_sided']:.3g}")
    lines += ['', '## Accel suite by acceleration level (bilinear, h40, mean over seeds)', '']
    for family in sorted(by_level):
        if 'bilinear' not in by_level[family]:
            continue
        cells = ', '.join(f"{lvl}: {v['mse']:.4g}/{v['frame_joint_success']:.2%}" for lvl, v in sorted(by_level[family]['bilinear'].items(), key=lambda kv: float(kv[0])))
        lines.append(f'- {family}: {cells}')
    if confirm:
        lines += ['', '## Confirmatory scenes 32-63 (primary pairs, same statistics, unadjusted)', '']
        for r in confirm['contrasts']:
            if r.get('status') != 'ok':
                lines.append(f"- {r.get('name', '')} {r['a']} − {r['b']}: {r.get('status')}")
            elif r['kind'] == 'noninferiority':
                lines.append(f"- {r['name']} {r['a']} − {r['b']} ({r['suite']} {r['metric']}): mean {r['mean_difference']:+.3g}, "
                             f"one-sided upper 95% {r['one_sided_upper95']:+.3g}, non-inferior at {r['margin']:g}: {r['noninferior']}")
            else:
                lines.append(f"- {r['name']} {r['a']} − {r['b']} ({r['suite']} {r['metric']}): mean {r['mean_difference']:+.3g}, p={r['p_two_sided']:.3g}")
    if extended:
        lines += ['', '## Seed-2026 selection runs: 5k budget vs 20k extension (bilinear MSE)', '']
        for key, table in extended.items():
            for entry in sorted(table, key=lambda e: e['family']):
                lines.append(f"- {key} {entry['family']}: mse {entry['mse']['mean']:.6g}, success {entry['frame_joint_success']['mean']:.2%}")
    (root/'report'/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
