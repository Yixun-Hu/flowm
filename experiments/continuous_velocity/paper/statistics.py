"""Fast paired test bootstrap, separate from variability across trained models.

Every metric uses metrics.summarize's pooled numerator/denominator definition.
Training seeds are never counted as additional independent test sequences.
"""
import numpy as np
try:
    from .metrics import RATIOS, summarize
except ImportError:  # Direct-file unit tests.
    from metrics import RATIOS, summarize

MEANS = ['latent_velocity_error', 'image_velocity_error', 'endpoint_error', 'transport_endpoint_error']
METRICS = list(RATIOS)+MEANS


def records(rows):
    return rows.to_dict('records') if hasattr(rows, 'to_dict') else list(rows)


def mean_sd(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    return dict(mean=float(values.mean()) if len(values) else np.nan,
                sd=float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                n=int(len(values)), minimum=float(values.min()) if len(values) else np.nan,
                maximum=float(values.max()) if len(values) else np.nan)


def training_seed_summary(rows, seed_key='train_seed'):
    """Summarize each fitted model first, then report mean and sample SD.

    This reports optimization/data-stream variability, not a confidence interval
    inferred by pooling all model-sequence pairs as independent observations.
    """
    rows = records(rows)
    seeds = sorted({row[seed_key] for row in rows})
    per_seed = {str(seed): summarize(row for row in rows if row[seed_key] == seed) for seed in seeds}
    return dict(n_training_seeds=len(seeds), per_seed=per_seed,
                metrics={m: mean_sd([value[m] for value in per_seed.values()]) for m in METRICS},
                interpretation='Mean and sample SD across independently trained models; each model score pools frames/pixels.')


def _pair(rows_a, rows_b, keys=None):
    a, b = records(rows_a), records(rows_b)
    if not a or not b:
        raise ValueError('Paired comparison needs nonempty rows')
    if keys is None:
        keys = ['case_id', 'horizon'] if 'case_id' in a[0] else ['seed', 'mode', 'scenario', 'true_vx', 'true_vy', 'horizon']
    def index(rows):
        result = {tuple(row.get(key) for key in keys): row for row in rows}
        if len(result) != len(rows):
            raise ValueError('Duplicate pairing keys; filter to one training seed and evaluation group')
        return result
    ia, ib = index(a), index(b)
    if ia.keys() != ib.keys():
        raise ValueError('Models must have identical paired evaluation keys')
    keys_sorted = sorted(ia)
    return [ia[k] for k in keys_sorted], [ib[k] for k in keys_sorted]


def _blocks(rows, clusters, cluster):
    """Return [cluster, metric, numerator/denominator] sufficient statistics."""
    blocks = np.zeros((len(clusters), len(METRICS), 2), np.float64)
    lookup = {value: idx for idx, value in enumerate(clusters)}
    for row in rows:
        block = blocks[lookup[row[cluster]]]
        for j, metric in enumerate(METRICS):
            if metric in RATIOS:
                num, den = RATIOS[metric]
                block[j] += [float(row.get(num, 0)), float(row.get(den, 0))]
            else:
                value = float(row.get(metric, np.nan))
                if np.isfinite(value):
                    block[j] += [value, 1]
    return blocks


def _divide(values):
    num, den = values[..., 0], values[..., 1]
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def _weights(n, repeats, seed):
    # Exactly equivalent to drawing n clusters with replacement, with bounded
    # memory independent of the number of rows within each cluster.
    return np.random.default_rng(seed).multinomial(n, np.full(n, 1./n), size=repeats)


def _intervals(point, samples):
    result = {}
    for j, metric in enumerate(METRICS):
        sample = samples[:, j]
        finite = sample[np.isfinite(sample)]
        result[metric] = dict(difference=float(point[j]),
                              ci95=np.quantile(finite, [.025, .975]).tolist() if len(finite) else [np.nan, np.nan],
                              valid_bootstraps=int(len(finite)))
    return result


def paired_test_bootstrap(rows_a, rows_b, cluster='seed', keys=None, repeats=2000, seed=2026):
    """Test-sequence cluster CI for ONE fixed pair of trained models.

    Positive difference means A > B. All speeds/scenarios sharing a test seed
    stay together. The estimator is the same pooled ratio as the point score.
    """
    a, b = _pair(rows_a, rows_b, keys)
    for rows in (a, b):
        if len({row['train_seed'] for row in rows if 'train_seed' in row}) > 1:
            raise ValueError('Use paired_seed_comparison for multiple training seeds')
    clusters = sorted({row[cluster] for row in a})
    if any(ra[cluster] != rb[cluster] for ra, rb in zip(a, b)):
        raise ValueError('Paired rows disagree on test cluster')
    ba, bb = _blocks(a, clusters, cluster), _blocks(b, clusters, cluster)
    weights = _weights(len(clusters), repeats, seed)
    sa = np.einsum('bc,cmk->bmk', weights, ba, optimize=True)
    sb = np.einsum('bc,cmk->bmk', weights, bb, optimize=True)
    point = _divide(ba.sum(0))-_divide(bb.sum(0))
    samples = _divide(sa)-_divide(sb)
    return dict(n_pairs=len(a), n_test_clusters=len(clusters), weighting='pooled ratios of sums',
                uncertainty='paired test-sequence uncertainty conditional on this fixed model pair',
                metrics=_intervals(point, samples))


def paired_seed_comparison(rows_a, rows_b, train_seed_key='train_seed',
                           cluster='seed', keys=None, repeats=2000, seed=2026):
    """Separate paired training variability from conditional test uncertainty.

    Reports per-training-seed score differences and their mean/sample SD. A second
    interval resamples test clusters jointly across the fixed trained model set,
    taking the mean of each model-pair's pooled difference. That interval is NOT
    a confidence interval over future random training runs.
    """
    a, b = records(rows_a), records(rows_b)
    seeds_a = {row[train_seed_key] for row in a}
    seeds_b = {row[train_seed_key] for row in b}
    if seeds_a != seeds_b:
        raise ValueError('Paired training-seed comparison requires identical training seed sets')
    if not seeds_a:
        raise ValueError('Empty paired training seed sets')
    seeds = sorted(seeds_a)
    per_seed, points, samples = {}, [], []
    clusters_reference = None
    for training_seed in seeds:
        ra, rb = _pair([r for r in a if r[train_seed_key] == training_seed],
                       [r for r in b if r[train_seed_key] == training_seed], keys)
        clusters = sorted({r[cluster] for r in ra})
        if clusters_reference is not None and clusters != clusters_reference:
            raise ValueError('All training seeds must use the same paired test clusters')
        clusters_reference = clusters
        if any(x[cluster] != y[cluster] for x, y in zip(ra, rb)):
            raise ValueError('Paired rows disagree on test cluster')
        ba, bb = _blocks(ra, clusters, cluster), _blocks(rb, clusters, cluster)
        # Same random draws across training seeds preserve the shared test set.
        weights = _weights(len(clusters), repeats, seed)
        sa = np.einsum('bc,cmk->bmk', weights, ba, optimize=True)
        sb = np.einsum('bc,cmk->bmk', weights, bb, optimize=True)
        point = _divide(ba.sum(0))-_divide(bb.sum(0))
        sample = _divide(sa)-_divide(sb)
        points.append(point); samples.append(sample)
        per_seed[str(training_seed)] = dict(n_pairs=len(ra), metrics=_intervals(point, sample))
    points, samples = np.asarray(points), np.asarray(samples)
    # An unavailable metric stays unavailable rather than changing the training
    # seed mixture from one bootstrap draw to another.
    ensemble_point = points.mean(0)
    ensemble_samples = samples.mean(0)
    return dict(n_training_seeds=len(seeds), n_test_clusters=len(clusters_reference),
                training_seeds=seeds, per_seed=per_seed,
                paired_training_differences={metric: mean_sd(points[:, j]) for j, metric in enumerate(METRICS)},
                conditional_test_intervals=_intervals(ensemble_point, ensemble_samples),
                uncertainty='Training mean/SD and fixed-model-ensemble paired test intervals are distinct; no training-seed CI is claimed.',
                weighting='Mean across trained pairs of within-pair pooled ratios of sums')
