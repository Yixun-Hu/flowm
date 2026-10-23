"""Audit complete artifacts independently of the report's statistical summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='paper_v1')
    args = parser.parse_args()
    out = ROOT/f'outputs/continuous_velocity/{args.tag}'
    for filename in ['training_all_complete.json', 'additional_training_complete.json', 'evaluation_status.json']:
        assert json.loads((out/filename).read_text())['complete'], filename
    plan = json.loads((out/'evaluation_jobs.json').read_text())
    assert len({(x['job']['name'], x['suite']) for x in plan}) == len(plan), 'Duplicate planned jobs'
    count = rows_total = 0
    sources = {}
    checkpoints = {}
    for entry in plan:
        path = out/'evaluation'/entry['job']['name']/entry['suite']
        manifest = json.loads((path/'manifest.json').read_text())
        assert manifest['complete']
        assert manifest['metric_schema_version'] == 2
        assert manifest['job'] == entry['job']
        with (path/'metrics.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == manifest['metric_rows']
        assert len({(r['case_id'], r['horizon']) for r in rows}) == len(rows)
        sequences = len({r['case_id'] for r in rows})
        assert sequences == manifest['expected_sequences'] == manifest['sequences']
        assert sorted(map(int, manifest['completed_seeds'])) == list(range(manifest['seeds']))
        for row in rows:
            visible, scored, success = [int(row[k]) for k in ('visible_frames', 'scored_frames', 'success_frames')]
            assert 0 <= success <= scored <= visible
            assert 0 <= int(row['frame_success_frames']) <= int(row['frame_scored_frames']) <= visible
            assert 0 <= int(row['continuity_success_frames']) <= int(row['continuity_scored_frames']) <= visible
            assert int(row['transport_count']) == int(row['horizon'])
            assert int(row['metrics_version']) == 2
            assert int(row['missing_frames'])+int(row['ghost_frames']) <= visible
            assert int(row['seed']) < 64
        count += sequences
        rows_total += len(rows)
        for source, expected in manifest['source_sha256'].items():
            actual = sources.setdefault(source, sha(ROOT/source))
            assert actual == expected, source
        checkpoint = manifest['model']['checkpoint']
        if checkpoint:
            actual = checkpoints.setdefault(checkpoint, sha(checkpoint))
            assert actual == manifest['model']['checkpoint_sha256'], checkpoint
    training = []
    for manifest_path in sorted((out/'training').glob('*/manifest.json')):
        if manifest_path.parent.name.startswith('aborted_'):
            continue
        manifest = json.loads(manifest_path.read_text())
        assert manifest['complete'], manifest_path
        assert not manifest['velocity_labels_used_for_training']
        assert not manifest['position_labels_used_for_training']
        assert not manifest['template_given_to_model']
        for source, expected in manifest['source_sha256'].items():
            assert sha(manifest_path.parent/'source'/source) == expected
        checkpoint_dir = Path(manifest['checkpoint_directory'])
        for budget in (5000, 20000):
            path = checkpoint_dir/f'budget_{budget}.pt'
            if manifest['config']['steps'] >= budget:
                saved = torch.load(path, map_location='cpu', weights_only=False)
                assert saved['step'] <= budget
                assert saved['config']['seed'] == manifest['config']['seed']
                assert saved['config']['kind'] == manifest['config']['kind']
        training.append(dict(run=manifest_path.parent.name, kind=manifest['config']['kind'],
                             seed=manifest['config']['seed'], updates=manifest['config']['steps']))
    logs = out/'logs'
    logs.mkdir(exist_ok=True)
    for name in ['campaign', 'evaluation', 'evaluation_v2', 'evaluation_shard0',
                 'evaluation_shard1', 'additional', 'early_extension']:
        source = Path(f'/tmp/flowm_paper_{name}.log')
        if source.exists():
            shutil.copyfile(source, logs/source.name)
    completion = dict(complete=True, completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      completed_evaluation_jobs=len(plan), completed_model_sequences=count,
                      metric_rows=rows_total, completed_training_runs=len(training),
                      optimizer_updates=sum(x['updates'] for x in training), training=training,
                      source_sha256=sources, checkpoint_sha256=checkpoints,
                      scope='Controlled single-object evidence campaign; paper acceptance/novelty is not established.')
    (out/'completion.json').write_text(json.dumps(completion, indent=2))
    print(json.dumps({k: v for k, v in completion.items() if k not in ('training', 'source_sha256', 'checkpoint_sha256')}, indent=2))


if __name__ == '__main__':
    main()
