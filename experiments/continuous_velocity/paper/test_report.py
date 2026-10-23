import json
from pathlib import Path
import sys
import tempfile
import unittest
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.report import load_complete, paired_comparisons, group_tables, table_html, qualitative_artifacts


def row(family='physical', train_seed=2026, seed=0):
    return dict(metrics_version=2, model_family=family, train_seed=train_seed, seed=seed, case_id=f'{seed}',
                horizon=40, suite='core', mode='bilinear', scenario='mnist_regular_released',
                true_vx=.37, true_vy=0., visible_frames=10, scored_frames=9,
                success_frames=8, missing_frames=1, ghost_frames=0, ambiguous_frames=0,
                certified_success_frames=8, certified_scored_frames=9, certified_error_sum=2,
                frame_success_frames=9, frame_scored_frames=9, frame_error_sum=1,
                continuity_scored_frames=7, continuity_error_sum=2, continuity_ambiguous_frames=2,
                continuity_success_frames=6, transport_error_sum=3, transport_count=40,
                error_sum=2, squared_error_sum=3, pixel_count=100,
                tracker_error_sum=1, tracker_scored_frames=10,
                latent_velocity_error=.01, image_velocity_error=.02, endpoint_error=.3)


class ReportTest(unittest.TestCase):
    def test_qualitative_panels_require_current_generated_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); figures = root/'figures'; figures.mkdir()
            for stem in ['qualitative_valid', 'qualitative_stale', 'qualitative_unlisted']:
                for suffix in ['png', 'pdf']:
                    (figures/f'{stem}.{suffix}').write_bytes(b'unchanged artifact')
            provenance = dict(metric_schema=2, generated='now', selection={'test_seed':0}, figures=[
                {'stem':'qualitative_valid', 'generated':True},
                {'stem':'qualitative_stale', 'generated':False},
                {'stem':'qualitative_missing', 'generated':True}])
            (figures/'qualitative_provenance.json').write_text(json.dumps(provenance))
            rendered, metadata = qualitative_artifacts(root)
            self.assertIn('qualitative_valid.png', rendered)
            self.assertNotIn('qualitative_stale.png', rendered)
            self.assertNotIn('qualitative_unlisted.png', rendered)
            self.assertEqual(metadata['unavailable_files'], ['qualitative_missing'])
            self.assertEqual((figures/'qualitative_valid.png').read_bytes(), b'unchanged artifact')
            provenance['metric_schema'] = 1
            (figures/'qualitative_provenance.json').write_text(json.dumps(provenance))
            rendered, metadata = qualitative_artifacts(root)
            self.assertFalse(metadata['compatible'])
            self.assertNotIn('qualitative_valid.png', rendered)

    def test_incomplete_manifest_never_enters_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for complete in [True, False]:
                name = str(complete)
                path = root/'evaluation'/name/'core'; path.mkdir(parents=True)
                manifest = dict(job={'name':name, 'family':'physical'}, model={'kind':'physical'},
                                suite='core', complete=complete, sequences=1, completed_seeds=[0], metric_rows=1,
                                source_sha256={'metrics.py':'samehash'})
                (path/'manifest.json').write_text(json.dumps(manifest))
                pd.DataFrame([row()]).to_csv(path/'metrics.csv', index=False)
            data, status = load_complete(root)
            self.assertEqual(len(data), 1)
            self.assertEqual(status['completed_jobs'], 1)
            self.assertEqual(len(status['discovered_incomplete']), 1)

    def test_complete_requires_actual_plan_markers_and_uniform_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'evaluation_status.json').write_text(json.dumps({'complete': True}))
            (root/'evaluation_jobs.json').write_text(json.dumps([{'job': {'name':'model'}, 'suite':'core'}]))
            path = root/'evaluation'/'model'/'core'; path.mkdir(parents=True)
            manifest = dict(job={'name':'model', 'family':'physical'}, model={'kind':'physical'},
                            suite='core', complete=True, sequences=1, completed_seeds=[0], metric_rows=1,
                            metric_schema_version=2, source_sha256={'metrics.py':'samehash'})
            (path/'manifest.json').write_text(json.dumps(manifest))
            pd.DataFrame([row()]).to_csv(path/'metrics.csv', index=False)
            self.assertFalse(load_complete(root)[1]['complete'])
            (root/'training_all_complete.json').write_text('{}')
            (root/'additional_training_complete.json').write_text('{}')
            self.assertTrue(load_complete(root)[1]['complete'])
            (root/'evaluation_jobs.json').write_text(json.dumps([{'job': {'name':'pending'}, 'suite':'core'}]))
            self.assertFalse(load_complete(root)[1]['complete'])
            other = root/'evaluation'/'different'/'core'; other.mkdir(parents=True)
            altered = dict(manifest, job={'name':'different', 'family':'physical'}, source_sha256={'metrics.py':'differenthash'})
            (other/'manifest.json').write_text(json.dumps(altered))
            pd.DataFrame([row()]).to_csv(other/'metrics.csv', index=False)
            with self.assertRaises(ValueError):
                load_complete(root)

    def test_pairing_intersects_model_seeds_and_preserves_partition(self):
        rows = [row('physical', t, s) for t in [2026, 2027, 2028] for s in range(3)]
        rows += [row('relaxed', t, s) for t in [2026, 2027] for s in range(3)]
        comparisons = paired_comparisons(pd.DataFrame(rows), repeats=20)
        post = next(x for x in comparisons if x['cohort'] == 'post_hpo')
        self.assertEqual(post['matched_training_seeds'], [2027])
        self.assertEqual(post['available_training_seeds_a'], [2026, 2027, 2028])
        self.assertEqual(post['n_training_seeds'], 1)

    def test_primary_table_excludes_hpo_only_model(self):
        data = pd.DataFrame([row('physical', 2026), row('relaxed', 2027)])
        with tempfile.TemporaryDirectory() as temp:
            groups = group_tables(data, Path(temp))
        primary = table_html(groups)
        self.assertNotIn('Physical fit', primary)
        self.assertIn('Physical + corrections', primary)
        self.assertIn('Physical fit', table_html(groups, cohort='all'))

    def test_intervention_and_generalization_routing(self):
        rows = []
        for mode in ['bilinear', 'nearest']:
            for family in ['physical', 'raw_matching_intervention', 'mean_memory_intervention',
                           'physical_nearest_horizontal', 'physical_mixed_horizontal']:
                rows.append(dict(row(family, 2027), mode=mode))
            for family in ['physical', 'relaxed']:
                rows.append(dict(row(family, 2027), mode=mode, suite='time', horizon=160))
        comparisons = paired_comparisons(pd.DataFrame(rows), repeats=20)
        post = [r for r in comparisons if r['cohort'] == 'post_hpo']
        self.assertTrue(any(r['b'] == 'raw_matching_intervention' and r['suite'] == 'core' for r in post))
        for mode in ['bilinear', 'nearest']:
            self.assertTrue(any(r['b'] == 'physical_mixed_horizontal' and r['mode'] == mode for r in post))
            self.assertTrue(any(r['b'] == 'relaxed' and r['horizon'] == 160 and r['mode'] == mode for r in post))


if __name__ == '__main__':
    unittest.main()
