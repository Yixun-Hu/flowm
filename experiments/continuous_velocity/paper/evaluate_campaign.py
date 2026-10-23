"""Evaluate selected models as training artifacts become available.

Validation choices are made exclusively by campaign.py. This worker never
changes training or hyperparameters after looking at test measurements.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.data import ROOT
from paper.evaluate import LOADED_SOURCE_HASHES, evaluate_job


def atomic_json(path, value):
    """Unique temporary names also make accidental concurrent publication safe."""
    path = Path(path)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, prefix=path.name+'.',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    """Completion markers may be observed during a non-atomic upstream write."""
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def job_shard(job, suite, shard_count):
    """Stable ownership independent of Python hash salt and plan insertion order."""
    if shard_count < 1:
        raise ValueError('shard_count must be positive')
    key = (job['name']+'\0'+suite).encode()
    return int.from_bytes(hashlib.sha256(key).digest(), 'big') % shard_count


def shard_jobs(planned, shard_index=0, shard_count=1):
    if not 0 <= shard_index < shard_count:
        raise ValueError('shard_index must be in [0, shard_count)')
    return [(job, suite) for job, suite in planned
            if job_shard(job, suite, shard_count) == shard_index]


def load_selected(out):
    selected = read_json(out/'selected_runs.json')
    for family, specification in read_json(out/'additional_runs.json').items():
        if family not in selected:
            selected[family] = specification
        else:
            selected[family]['runs'] = list(dict.fromkeys(
                selected[family]['runs']+specification['runs']))
    return selected


def training_complete(out):
    """Both the main campaign and the declared additional-seed worker must finish."""
    return all(read_json(out/name).get('complete') is True for name in
               ('training_all_complete.json', 'additional_training_complete.json'))


def manifest_complete(out, job, suite):
    manifest = read_json(out/'evaluation'/job['name']/suite/'manifest.json')
    return (manifest.get('complete') is True and manifest.get('job') == job
            and manifest.get('suite') == suite
            and manifest.get('metric_schema_version') == 2
            and manifest.get('source_sha256') == LOADED_SOURCE_HASHES)


def completion_status(out, planned, training_done=None):
    """Count actual current-plan artifacts, never a process-local done cache."""
    if training_done is None:
        training_done = training_complete(out)
    pending = [(job['name'], suite) for job, suite in planned
               if not manifest_complete(out, job, suite)]
    return dict(complete=bool(training_done and not pending),
                completed_jobs=len(planned)-len(pending), total_jobs=len(planned),
                training_complete=bool(training_done), pending_jobs=pending,
                updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'))


def lock_exclusive(handle, description):
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'Another evaluation worker already owns {description}') from exc


def jobs(selected):
    result = []
    raw = dict(name='raw_untrained', kind='raw_untrained', family='raw_untrained', train_seed=2026)
    for suite in ('core', 'time', 'gap', 'appearance', '2d'):
        result.append((raw, suite))
    for seed in (2026, 2027, 2028):
        result.append((dict(name=f'physical_untrained_s{seed}', kind='physical_untrained',
                            family='physical_untrained', train_seed=seed), 'core'))
    for family, selection in selected.items():
        for run in selection['runs']:
            seed = int(run.split('_s')[-1].split('_')[0])
            kind = 'mlp_velocity' if family.startswith('mlp_velocity') else (
                'physical' if family.startswith('physical') else family)
            job = dict(name=f'{family}_s{seed}', kind=kind, family=family, train_seed=seed,
                       run_id=run, checkpoint='budget_5000.pt')
            if family.endswith('_2d'):
                result.append((job, '2d'))
                continue
            result.append((job, 'core'))
            if family in ('physical', 'relaxed', 'offset', 'mlp_velocity', 'physical_mixed_horizontal'):
                result.append((job, 'time'))
            if family in ('physical', 'physical_mixed_horizontal') and seed in (2026, 2027, 2028):
                for suite in ('gap', 'appearance'):
                    result.append((job, suite))
            if family == 'physical' and seed in (2026, 2027, 2028):
                for intervention in ('raw_matching', 'mean_memory'):
                    variant = f'{intervention}_intervention'
                    result.append(({**job, 'name': f'{variant}_s{seed}', 'family': variant,
                                    'intervention': intervention}, 'core'))
                for spacing, label in ((1., 'q1'), (.5, 'q05')):
                    variant = f'est_{label}_both'
                    result.append(({**job, 'name': f'{variant}_s{seed}', 'family': variant,
                                    'intervention': 'quantize', 'spacing': spacing, 'scope': 'both'}, 'intervention'))
                # Pure transport/oracle diagnostics use a fixed fitted checkpoint.
                # Replication over heldout sequences is distinct from training-seed stability.
                if seed == 2026:
                    for spacing, label in ((1., 'q1'), (.5, 'q05')):
                        variant = f'est_{label}_transport'
                        result.append(({**job, 'name': f'{variant}_s{seed}', 'family': variant,
                                        'intervention': 'quantize', 'spacing': spacing, 'scope': 'transport'}, 'intervention'))
                    for spacing, label in ((None, 'cont'), (1., 'q1'), (.5, 'q05')):
                        for scope in (('both',) if spacing is None else ('both', 'transport')):
                            variant = 'oracle_cont' if spacing is None else f'oracle_{label}_{scope}'
                            result.append(({**job, 'name': f'{variant}_s{seed}', 'family': variant,
                                            'oracle': True, 'spacing': spacing, 'scope': scope}, 'intervention'))
        if selection.get('extended_run'):
            kind = family
            variant = family+'_extended'
            job = dict(name=f'{variant}_s2026', kind=kind, family=variant, train_seed=2026,
                       run_id=selection['extended_run'], checkpoint='budget_20000.pt')
            result.extend((job, suite) for suite in ('core', 'time'))
    # Core estimates become visible first, followed by mechanism and robustness.
    order = {'core': 0, 'intervention': 1, '2d': 2, 'time': 3, 'gap': 4, 'appearance': 5}
    return sorted(result, key=lambda x: order[x[1]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='paper_v1')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error('--shard-count must be positive and 0 <= --shard-index < --shard-count')
    out = ROOT/f'outputs/continuous_velocity/{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    shard_label = f'shard_{args.shard_index}_of_{args.shard_count}'
    with (out/f'evaluation_worker_{shard_label}.lock').open('a+') as worker_lock:
        lock_exclusive(worker_lock, shard_label)
        # Every owned artifact passes through evaluate_job once per startup so
        # its full checkpoint/source/protocol signature is checked on resume.
        done = set()
        while True:
            # Read completion markers first. If true, selections were already
            # published by both upstream workers before their final markers.
            training_done = training_complete(out)
            planned = jobs(load_selected(out))
            owned = shard_jobs(planned, args.shard_index, args.shard_count)
            if args.shard_index == 0:
                atomic_json(out/'evaluation_jobs.json',
                            [{'job': job, 'suite': suite} for job, suite in planned])
            local = completion_status(out, owned, training_done)
            local.update(shard_index=args.shard_index, shard_count=args.shard_count)
            atomic_json(out/f'evaluation_status_{shard_label}.json', local)
            global_status = None
            if args.shard_index == 0:
                global_status = completion_status(out, planned, training_done)
                global_status['shard_count'] = args.shard_count
                atomic_json(out/'evaluation_status.json', global_status)
            # Membership depends only on job name+suite, never changing plan
            # indices. New selections cannot move a running job to another shard.
            def identity(job, suite):
                return (job['name'], suite, json.dumps(job, sort_keys=True))
            pending = [(job, suite) for job, suite in owned
                       if identity(job, suite) not in done
                       or not manifest_complete(out, job, suite)]
            if pending:
                job, suite = pending[0]
                job_dir = out/'evaluation'/job['name']/suite
                job_dir.mkdir(parents=True, exist_ok=True)
                with (job_dir/'driver.lock').open('a+') as job_lock:
                    # Also guards against accidental workers using different
                    # shard counts or an overlapping single-worker invocation.
                    lock_exclusive(job_lock, f"{job['name']}/{suite}")
                    result = evaluate_job(job, suite, tag=args.tag, device=args.device)
                    if not result.get('complete'):
                        raise RuntimeError(f"Evaluation did not complete: {job['name']}/{suite}")
                done.add(identity(job, suite))
                continue
            if args.shard_index == 0:
                if global_status['complete']:
                    break
                # Once training is final, coordinator waits for every shard's
                # current-plan manifests even without --watch.
                if not args.watch and not training_done:
                    break
            elif local['complete'] or not args.watch:
                break
            time.sleep(10)


if __name__ == '__main__':
    main()
