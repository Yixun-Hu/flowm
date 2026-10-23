"""CPU-only checks for stable evaluation ownership and global completion."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.evaluate_campaign import (
    LOADED_SOURCE_HASHES, atomic_json, completion_status, job_shard, jobs,
    load_selected, lock_exclusive, manifest_complete, shard_jobs,
    training_complete)


class CampaignTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='flowm-campaign-test-')
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.plan = [(dict(name=f'model_{i}', kind='physical'), suite)
                     for i in range(12) for suite in ('core', 'time')]

    def finish_training(self):
        for name in ('training_all_complete.json', 'additional_training_complete.json'):
            atomic_json(self.root/name, {'complete': True})

    def finish_job(self, specification, suite_name, **changes):
        path = self.root/'evaluation'/specification['name']/suite_name/'manifest.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(complete=True, job=specification, suite=suite_name, metric_schema_version=2,
                        source_sha256=LOADED_SOURCE_HASHES)
        manifest.update(changes)
        atomic_json(path, manifest)

    def test_shards_are_disjoint_complete_and_default_preserves_all(self):
        left, right = [shard_jobs(self.plan, i, 2) for i in (0, 1)]
        keys = lambda values: {(job['name'], suite) for job, suite in values}
        self.assertTrue(left and right)
        self.assertFalse(keys(left) & keys(right))
        self.assertEqual(keys(left) | keys(right), keys(self.plan))
        self.assertEqual(shard_jobs(self.plan), self.plan)
        self.assertTrue(all(job_shard(job, suite, 1) == 0 for job, suite in self.plan))

    def test_existing_ownership_stable_when_plan_grows_or_reorders(self):
        original = {(job['name'], suite): job_shard(job, suite, 2)
                    for job, suite in self.plan}
        expanded = [(dict(name='new_first'), 'core')] + list(reversed(self.plan))
        expanded += [(dict(name='new_last'), 'appearance')]
        for job, suite in expanded:
            if (job['name'], suite) in original:
                self.assertEqual(job_shard(job, suite, 2), original[job['name'], suite])
        with self.assertRaises(ValueError):
            shard_jobs(self.plan, 2, 2)
        with self.assertRaises(ValueError):
            job_shard(self.plan[0][0], 'core', 0)

    def test_global_completion_requires_remote_manifests_and_both_training_markers(self):
        owned = shard_jobs(self.plan, 0, 2)
        other = shard_jobs(self.plan, 1, 2)
        for job, suite in owned:
            self.finish_job(job, suite)
        self.finish_training()
        self.assertTrue(completion_status(self.root, owned)['complete'])
        status = completion_status(self.root, self.plan)
        self.assertFalse(status['complete'])
        self.assertEqual(status['completed_jobs'], len(owned))
        self.assertEqual(len(status['pending_jobs']), len(other))
        for job, suite in other:
            self.finish_job(job, suite)
        self.assertTrue(completion_status(self.root, self.plan)['complete'])
        (self.root/'additional_training_complete.json').unlink()
        self.assertFalse(completion_status(self.root, self.plan)['complete'])
        atomic_json(self.root/'additional_training_complete.json', {'complete': False})
        self.assertFalse(training_complete(self.root))
        # A partially written upstream marker also cannot signal completion.
        (self.root/'additional_training_complete.json').write_text('{')
        self.assertFalse(training_complete(self.root))

    def test_new_full_plan_job_revokes_completion(self):
        self.finish_training()
        for job, suite in self.plan:
            self.finish_job(job, suite)
        self.assertTrue(completion_status(self.root, self.plan)['complete'])
        expanded = self.plan+[(dict(name='later_selected_model'), 'core')]
        self.assertFalse(completion_status(self.root, expanded)['complete'])

    def test_stale_job_schema_or_source_never_counted_complete(self):
        job, suite = self.plan[0]
        self.finish_job(job, suite)
        self.assertTrue(manifest_complete(self.root, job, suite))
        for changes in ({'job': {**job, 'checkpoint': 'different.pt'}},
                        {'suite': 'gap'}, {'metric_schema_version': 1},
                        {'source_sha256': {}}, {'complete': False}):
            self.finish_job(job, suite, **changes)
            self.assertFalse(manifest_complete(self.root, job, suite))

    def test_selection_merge_and_atomic_plan(self):
        first = 'physical_bilinear_horizontal_s2027_lr0.001'
        second = 'physical_bilinear_horizontal_s2029_lr0.001'
        atomic_json(self.root/'selected_runs.json', {'physical': {'runs': [first]}})
        atomic_json(self.root/'additional_runs.json', {'physical': {'runs': [first, second]}})
        selection = load_selected(self.root)
        self.assertEqual(selection['physical']['runs'], [first, second])
        planned = jobs(selection)
        payload = [{'job': job, 'suite': suite} for job, suite in planned]
        atomic_json(self.root/'evaluation_jobs.json', payload)
        self.assertEqual(json.loads((self.root/'evaluation_jobs.json').read_text()), payload)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_duplicate_job_owner_is_rejected(self):
        path = self.root/'job.lock'
        with path.open('a+') as first, path.open('a+') as second:
            lock_exclusive(first, 'test job')
            with self.assertRaisesRegex(RuntimeError, 'already owns'):
                lock_exclusive(second, 'test job')


if __name__ == '__main__':
    unittest.main()
