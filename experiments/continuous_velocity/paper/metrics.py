"""Version 2: separate frame localization, continuity, and CV-fit diagnostics.

The oracle digit template is evaluation metadata, never a model input. Position
Frame localization explicitly uses periodic distance and does not measure winding.
Continuity uses one GT-selected coordinate origin and bounded observed increments,
without a constant-velocity fit. Legacy fitted-trajectory scores remain available
as CV-certified diagnostics. All aggregation/bootstrap use ratios of sums.
"""
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'experiments/fractional_velocity'))
from measurement import Detector

METRICS_VERSION = 2


def continuity_unwrap(wrapped, times, anchor, anchor_time, speed_bound=2.5, localization_slack=1.):
    """Model-neutral observed continuity with permanent unknown-winding flags.

    The caller fixes a single initial winding. Each axis must have exactly one
    feasible integer winding under speed_bound*elapsed+localization_slack. Gaps
    permitting half-world traversal are conservatively marked ambiguous even
    when the observed endpoint could admit a single bounded branch. Missing
    observations do not reset the anchor. No global motion law or GT reanchoring.
    """
    wrapped, times = np.asarray(wrapped, float), np.asarray(times, float)
    result = np.full_like(wrapped, np.nan)
    unknown = np.zeros(len(wrapped), bool)
    previous = np.asarray(anchor, float).copy()
    previous_time = float(anchor_time)
    ambiguous = not np.isfinite(previous).all()
    for i, (position, timestamp) in enumerate(zip(wrapped, times)):
        if ambiguous:
            unknown[i] = True
            continue
        elapsed = timestamp-previous_time
        if elapsed <= 0:
            raise ValueError('Continuity timestamps must follow the anchor')
        bound = speed_bound*elapsed+localization_slack
        if bound >= 25:
            ambiguous = True; unknown[i] = True
            continue
        if not np.isfinite(position).all():
            continue
        delta = position-previous
        lower = np.ceil((-bound-delta)/50).astype(int)
        upper = np.floor((bound-delta)/50).astype(int)
        if np.any(lower != upper):
            ambiguous = True; unknown[i] = True
            continue
        current = position+50*lower
        result[i] = current
        previous, previous_time = current, timestamp
    return result, unknown


def periodic_fit_at(wrapped, times, speed_bound=2.5):
    """Infer winding and a constant velocity from positions and actual times.

    Rejected/ambiguous fits are missing, rather than circularly scored against
    truth. This evaluator assumes approximately constant predicted motion and
    explicitly reports its rejection rate; MSE has no such model assumption.
    """
    wrapped, times = np.asarray(wrapped, float), np.asarray(times, float)
    empty = (np.full_like(wrapped, np.nan), np.full(2, np.nan), np.full(2, np.nan))
    valid = np.isfinite(wrapped).all(1)
    if valid.sum() < 5 or np.ptp(times[valid]) < 10:
        return empty
    span = np.ptp(times[valid])
    mesh = min(.01, 2. / span)
    velocities = np.linspace(-speed_bound, speed_bound, int(np.ceil(2*speed_bound/mesh))+1)
    origin_time = times[valid][0]
    ts = times[valid] - origin_time
    inferred_v, inferred_origin, tracks = [], [], []
    for axis in range(2):
        observations = wrapped[valid, axis]
        phase = 2*np.pi/50*(observations[None] - velocities[:, None]*ts)
        phasor = np.exp(1j*phase).mean(1)
        score = np.abs(phasor)
        k = int(score.argmax()); speed = velocities[k]
        competing = score[np.abs(velocities-speed) >= .3].max()
        if score[k] < .90 or competing >= score[k]-.01:
            return empty
        origin = np.angle(phasor[k])*50/(2*np.pi)
        for _ in range(3):
            q = observations + 50*np.round((origin+speed*ts-observations)/50)
            keep = np.abs(q-(origin+speed*ts)) <= 4
            if keep.sum() < 5:
                return empty
            speed, origin = np.polyfit(ts[keep], q[keep], 1)
        intercept = origin-origin_time*speed
        line = intercept+times*speed
        q = wrapped[:, axis]+50*np.round((line-wrapped[:, axis])/50)
        inferred_v.append(speed); inferred_origin.append(intercept); tracks.append(q)
    track = np.stack(tracks, 1)
    velocity, intercept = np.array(inferred_v), np.array(inferred_origin)
    line = intercept+times[:, None]*velocity
    track[np.linalg.norm(track-line, axis=1) > 4] = np.nan
    return track, velocity, intercept


def evaluate_case(case, pred, diagnostics=None, horizons=(20, 40, 80, 160), tolerance=2.):
    """Return metric rows and full-horizon diagnostic arrays for one sequence.

    `case` provides full frames/positions/cameras/visibility/template and may
    provide context_indices, target_indices, times, and true_velocity/velocity.
    A shorter horizon is fitted only using that prefix of generated frames.
    Diagnostics accept velocity[2], latent_velocity[2], or latent_vx/latent_vy.
    """
    pred = np.asarray(pred)
    ci = np.asarray(case.get('context_indices', np.arange(50)), int)
    ti = np.asarray(case.get('target_indices', np.arange(50, 50+len(pred))), int)
    if len(ti) != len(pred):
        raise ValueError('Number of prediction and target frames differs')
    times = np.asarray(case.get('times', np.arange(len(case['frames']))), float)
    context_times, target_times = times[ci], times[ti]
    if np.any(np.diff(context_times) <= 0) or np.any(np.diff(target_times) <= 0):
        raise ValueError('Evaluation timestamps must be strictly increasing')
    detector = Detector(case['template'])
    ctx = [detector(frame) for frame in np.asarray(case['frames'])[ci]]
    detections = [detector(frame) for frame in pred]
    cameras = np.asarray(case['cameras'])
    ctx_wrapped = (np.array([d['pos'] for d in ctx])+cameras[ci]) % 50
    ctx_wrapped[np.array([d['ghost'] for d in ctx])] = np.nan
    ctx_track, tracker_v, tracker_intercept = periodic_fit_at(ctx_wrapped, context_times)
    # Fix global coordinate origin once at the earliest identified context point.
    found = np.flatnonzero(np.isfinite(ctx_track).all(1))
    origin_shift = np.zeros(2)
    if len(found):
        k = found[0]
        origin_shift = 50*np.round((np.asarray(case['positions'])[ci[k]]-ctx_track[k])/50)
    baseline = tracker_intercept+target_times[:, None]*tracker_v+origin_shift
    anchor_time = context_times[-1]
    anchor = tracker_intercept+anchor_time*tracker_v
    wrapped = (np.array([d['pos'] for d in detections])+cameras[ti]) % 50
    present = np.array([d['present'] for d in detections])
    ghost = np.array([d['ghost'] for d in detections])
    wrapped[ghost] = np.nan
    visible = np.asarray(case['visibility'])[ti] >= .35
    gt = np.asarray(case['positions'])[ti]
    truth = np.asarray(case['frames'])[ti]
    true_v = np.asarray(case.get('true_velocity', case.get('velocity', [case.get('speed', 0), 0])), float)
    diagnostics = diagnostics or {}
    latent = diagnostics.get('latent_velocity', diagnostics.get('velocity'))
    if latent is None:
        latent = [diagnostics.get('latent_vx', np.nan), diagnostics.get('latent_vy', np.nan)]
    latent = np.asarray(latent, float)
    # Per-frame periodic localization is deliberately independent of all fits.
    # It cannot expose whole-world winding errors and is named accordingly.
    frame_delta = (wrapped-gt+25) % 50-25
    frame_errors = np.linalg.norm(frame_delta, axis=1)
    frame_usable = visible & present & ~ghost & np.isfinite(frame_errors)
    frame_success = frame_usable & (frame_errors <= tolerance)
    # Start continuity at the last accepted context detection, avoiding any
    # requirement that its earlier context fit one constant-velocity line.
    last_context = np.flatnonzero(np.isfinite(ctx_wrapped).all(1))
    continuity_anchor = np.full(2, np.nan)
    continuity_anchor_time = context_times[-1]
    continuity_anchor_index = -1
    if len(last_context):
        k = last_context[-1]
        continuity_anchor = ctx_wrapped[k].copy()
        continuity_anchor += 50*np.round((np.asarray(case['positions'])[ci[k]]-continuity_anchor)/50)
        continuity_anchor_time = context_times[k]
        continuity_anchor_index = int(ci[k])
    continuity_track, continuity_unknown = continuity_unwrap(
        wrapped, target_times, continuity_anchor, continuity_anchor_time)
    continuity_errors = np.linalg.norm(continuity_track-gt, axis=1)
    continuity_usable = visible & present & ~ghost & np.isfinite(continuity_errors)
    rows, arrays = [], {}
    requested = sorted(set(int(h) for h in horizons if h <= len(pred)))
    if not requested:
        requested = [len(pred)]
    for h in requested:
        track, pred_v, intercept = periodic_fit_at(wrapped[:h], target_times[:h])
        if np.isfinite(anchor).all() and np.isfinite(intercept).all():
            track += 50*np.round((anchor-(intercept+anchor_time*pred_v))/50)+origin_shift
        else:
            track[:] = np.nan
        errors = np.linalg.norm(track-gt[:h], axis=1)
        base_errors = np.linalg.norm(baseline[:h]-gt[:h], axis=1)
        vis = visible[:h]
        usable = vis & present[:h] & ~ghost[:h] & np.isfinite(errors)
        success = usable & (errors <= tolerance)
        difference = (pred[:h]-truth[:h]).astype(np.float64)
        error_sum = float(errors[usable].sum())
        count = int(usable.sum())
        row = dict(metrics_version=METRICS_VERSION, seed=int(case['seed']), mode=case.get('mode', 'unknown'),
                   scenario=case.get('scenario', 'unknown'), speed=float(true_v[0]),
                   true_vx=float(true_v[0]), true_vy=float(true_v[1]), horizon=h,
                   elapsed_time=float(target_times[h-1]-anchor_time),
                   grid_distance=float(np.linalg.norm(true_v-np.floor(true_v+.5))),
                   visible_frames=int(vis.sum()), scored_frames=count,
                   success_frames=int(success.sum()), certified_success_frames=int(success.sum()),
                   certified_scored_frames=count, certified_error_sum=error_sum,
                   frame_success_frames=int(frame_success[:h].sum()),
                   frame_scored_frames=int(frame_usable[:h].sum()),
                   frame_error_sum=float(frame_errors[:h][frame_usable[:h]].sum()),
                   continuity_scored_frames=int(continuity_usable[:h].sum()),
                   continuity_error_sum=float(continuity_errors[:h][continuity_usable[:h]].sum()),
                   continuity_ambiguous_frames=int((vis & continuity_unknown[:h]).sum()),
                   continuity_success_frames=int((continuity_usable[:h] & (continuity_errors[:h] <= tolerance)).sum()),
                   continuity_anchor_index=continuity_anchor_index,
                   tolerance=float(tolerance),
                   missing_frames=int((vis & ~present[:h]).sum()),
                   ghost_frames=int((vis & ghost[:h]).sum()),
                   ambiguous_frames=int((vis & present[:h] & ~ghost[:h] & ~np.isfinite(errors)).sum()),
                   error_sum=error_sum, ade=error_sum/count if count else np.nan,
                   endpoint_error=float(errors[-1]) if usable[-1] else np.nan,
                   squared_error_sum=float((difference**2).sum()), pixel_count=int(difference.size),
                   mse=float((difference**2).mean()),
                   predicted_vx=float(pred_v[0]), predicted_vy=float(pred_v[1]),
                   image_velocity_error=float(np.linalg.norm(pred_v-true_v)),
                   latent_vx=float(latent[0]), latent_vy=float(latent[1]),
                   latent_velocity_error=float(np.linalg.norm(latent-true_v)),
                   tracker_vx=float(tracker_v[0]), tracker_vy=float(tracker_v[1]),
                   tracker_error_sum=float(base_errors[vis & np.isfinite(base_errors)].sum()),
                   tracker_scored_frames=int((vis & np.isfinite(base_errors)).sum()),
                   context_detected=int(sum(d['present'] for d in ctx)),
                   context_visible=int((np.asarray(case['visibility'])[ci] >= .35).sum()))
        for key in ['case_id', 'digit_index', 'label', 'appearance', 'context', 'absence']:
            if key in case:
                row[key] = case[key]
        rows.append(row)
        arrays = dict(track=track, baseline=baseline[:h], errors=errors, tracker_errors=base_errors,
                      present=present[:h], ghost=ghost[:h], visible=vis, usable=usable,
                      success=success, target_times=target_times[:h], context_times=context_times,
                      certified_track=track, certified_errors=errors, certified_success=success,
                      frame_errors=frame_errors[:h], frame_success=frame_success[:h],
                      frame_usable=frame_usable[:h], continuity_track=continuity_track[:h],
                      continuity_errors=continuity_errors[:h], continuity_usable=continuity_usable[:h],
                      continuity_unknown=continuity_unknown[:h],
                      amplitude=np.array([d['amplitude'] for d in detections[:h]]),
                      quality=np.array([d['quality'] for d in detections[:h]]))
    return rows, arrays


RATIOS = {'missing_rate': ('missing_frames', 'visible_frames'),
          'ghost_rate': ('ghost_frames', 'visible_frames'),
          'ambiguous_rate': ('ambiguous_frames', 'visible_frames'),
          'coverage': ('scored_frames', 'visible_frames'),
          'joint_success': ('success_frames', 'visible_frames'),
          'certified_joint_success': ('certified_success_frames', 'visible_frames'),
          'certified_coverage': ('certified_scored_frames', 'visible_frames'),
          'certified_ade': ('certified_error_sum', 'certified_scored_frames'),
          'frame_joint_success': ('frame_success_frames', 'visible_frames'),
          'frame_localization_error': ('frame_error_sum', 'frame_scored_frames'),
          'frame_coverage': ('frame_scored_frames', 'visible_frames'),
          'continuity_ade': ('continuity_error_sum', 'continuity_scored_frames'),
          'continuity_coverage': ('continuity_scored_frames', 'visible_frames'),
          'continuity_ambiguity': ('continuity_ambiguous_frames', 'visible_frames'),
          'continuity_joint_success': ('continuity_success_frames', 'visible_frames'),
          'transport_ade': ('transport_error_sum', 'transport_count'),
          'ade': ('error_sum', 'scored_frames'),
          'mse': ('squared_error_sum', 'pixel_count'),
          'tracker_ade': ('tracker_error_sum', 'tracker_scored_frames')}


def _rows(rows):
    return rows.to_dict('records') if hasattr(rows, 'to_dict') else list(rows)


def summarize(rows):
    """Pool frames/pixels; velocity errors average finite sequence estimates."""
    rows = _rows(rows)
    result = {'n_sequences': len(rows)}
    for metric, (num, den) in RATIOS.items():
        numerator = sum(float(row.get(num, 0)) for row in rows)
        denominator = sum(float(row.get(den, 0)) for row in rows)
        result[metric] = numerator/denominator if denominator else np.nan
    for metric in ['latent_velocity_error', 'image_velocity_error', 'endpoint_error', 'transport_endpoint_error']:
        values = np.asarray([row.get(metric, np.nan) for row in rows], float)
        good = np.isfinite(values)
        result[metric] = float(values[good].mean()) if good.any() else np.nan
        result[metric+'_count'] = int(good.sum())
    result['visible_frames'] = int(sum(row['visible_frames'] for row in rows))
    result['scored_frames'] = int(sum(row['scored_frames'] for row in rows))
    return result


def paired_bootstrap(rows_a, rows_b, cluster='seed', keys=None, repeats=2000, seed=2026):
    """Paired test-cluster bootstrap of the same pooled estimator as summarize.

    Call separately for each training seed to avoid treating optimization seeds
    as test examples. Cross-training-seed mean/SD should be reported separately.
    Every paired cluster, including all of its speeds, is sampled together.
    """
    a, b = _rows(rows_a), _rows(rows_b)
    if not a or not b:
        raise ValueError('Paired comparison needs nonempty rows')
    if keys is None:
        keys = ['case_id', 'horizon'] if 'case_id' in a[0] else ['seed', 'mode', 'scenario', 'true_vx', 'true_vy', 'horizon']
    def index(rows):
        mapping = {tuple(row.get(key) for key in keys): row for row in rows}
        if len(mapping) != len(rows):
            raise ValueError('Duplicate pairing keys: compare one training seed at a time')
        return mapping
    ia, ib = index(a), index(b)
    if ia.keys() != ib.keys():
        raise ValueError('Paired rows must have identical evaluation keys')
    clusters = sorted({row[cluster] for row in a})
    pair_keys = sorted(ia)
    block_a = [[ia[key] for key in pair_keys if ia[key][cluster] == c] for c in clusters]
    block_b = [[ib[key] for key in pair_keys if ib[key][cluster] == c] for c in clusters]
    point_a, point_b = summarize(a), summarize(b)
    metrics = list(RATIOS)+['latent_velocity_error', 'image_velocity_error', 'endpoint_error', 'transport_endpoint_error']
    values = {metric: [] for metric in metrics}
    rng = np.random.default_rng(seed)
    for selection in rng.integers(len(clusters), size=(repeats, len(clusters))):
        sa = summarize(row for idx in selection for row in block_a[idx])
        sb = summarize(row for idx in selection for row in block_b[idx])
        for metric in metrics:
            values[metric].append(sa[metric]-sb[metric])
    output = {}
    for metric in metrics:
        samples = np.asarray(values[metric]); samples = samples[np.isfinite(samples)]
        output[metric] = {'difference': point_a[metric]-point_b[metric],
                          'ci95': np.quantile(samples, [.025, .975]).tolist() if len(samples) else [np.nan, np.nan],
                          'valid_bootstraps': len(samples)}
    return dict(n_clusters=len(clusters), n_pairs=len(a), weighting='pooled ratios of sums', metrics=output)
