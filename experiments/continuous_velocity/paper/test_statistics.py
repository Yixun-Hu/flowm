import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import summarize
from statistics import training_seed_summary, paired_test_bootstrap, paired_seed_comparison


def sample_rows(training_seed=2026, error=1):
    result = []
    for seed, visible, missing in [(0, 1, 1), (1, 9, 0), (2, 7, 2)]:
        result.append(dict(train_seed=training_seed, seed=seed, case_id=str(seed), horizon=40,
                           visible_frames=visible, scored_frames=visible-missing,
                           success_frames=visible-missing, missing_frames=missing,
                           ghost_frames=0, ambiguous_frames=0,
                           certified_success_frames=visible-missing,
                           certified_scored_frames=visible-missing,
                           certified_error_sum=error*(visible-missing),
                           frame_success_frames=visible-missing,
                           frame_scored_frames=visible-missing,
                           frame_error_sum=error*(visible-missing),
                           continuity_scored_frames=visible-missing,
                           continuity_error_sum=error*(visible-missing),
                           continuity_ambiguous_frames=missing,
                           continuity_success_frames=visible-missing,
                           transport_error_sum=error*10, transport_count=10,
                           transport_endpoint_error=error,
                           error_sum=error*(visible-missing),
                           squared_error_sum=error*10, pixel_count=10,
                           tracker_error_sum=0, tracker_scored_frames=visible,
                           latent_velocity_error=error*.1))
    return result


class StatisticsTest(unittest.TestCase):
    def test_fast_point_matches_pooled_summary(self):
        a = sample_rows(); b = [dict(r, missing_frames=0, squared_error_sum=0, error_sum=0) for r in a]
        comparison = paired_test_bootstrap(a, b, repeats=100)
        for metric in ['missing_rate', 'coverage', 'joint_success', 'mse', 'ade',
                       'frame_joint_success', 'continuity_ade', 'continuity_coverage',
                       'certified_joint_success', 'transport_ade', 'transport_endpoint_error']:
            self.assertAlmostEqual(comparison['metrics'][metric]['difference'], summarize(a)[metric]-summarize(b)[metric])
        self.assertAlmostEqual(comparison['metrics']['missing_rate']['difference'], 3/17)

    def test_identical_models_have_zero_interval(self):
        a = sample_rows()
        comparison = paired_test_bootstrap(a, a, repeats=100)
        self.assertEqual(comparison['metrics']['mse']['ci95'], [0, 0])

    def test_training_mean_sd_not_pooled_pseudoreplication(self):
        a = sample_rows(2026, 1)+sample_rows(2027, 3)
        summary = training_seed_summary(a)
        self.assertEqual(summary['n_training_seeds'], 2)
        self.assertAlmostEqual(summary['metrics']['mse']['mean'], 2)
        self.assertAlmostEqual(summary['metrics']['mse']['sd'], np.sqrt(2))
        with self.assertRaises(ValueError):
            paired_test_bootstrap(a, a, repeats=100)

    def test_paired_seed_and_test_uncertainty_separate(self):
        a = sample_rows(2026, 1)+sample_rows(2027, 3)
        b = sample_rows(2026, 0)+sample_rows(2027, 0)
        result = paired_seed_comparison(a, b, repeats=100)
        self.assertAlmostEqual(result['paired_training_differences']['mse']['mean'], 2)
        self.assertAlmostEqual(result['paired_training_differences']['mse']['sd'], np.sqrt(2))
        # Every test sequence has the same MSE in each fitted model, so varying
        # the test set adds no uncertainty despite nonzero training-seed SD.
        self.assertEqual(result['conditional_test_intervals']['mse']['ci95'], [2, 2])

    def test_mismatched_pairing_rejected(self):
        a = sample_rows()
        with self.assertRaises(ValueError):
            paired_test_bootstrap(a, a[:-1], repeats=20)
        with self.assertRaises(ValueError):
            paired_seed_comparison(a, sample_rows(2027), repeats=20)


if __name__ == '__main__':
    unittest.main()
