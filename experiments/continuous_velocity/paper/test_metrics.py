import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import periodic_fit_at, evaluate_case, summarize, paired_bootstrap, continuity_unwrap
from measurement import render, translate


class MetricsTest(unittest.TestCase):
    def test_multiple_world_crossings_and_irregular_times(self):
        times = np.r_[np.arange(0, 50, 3), np.arange(50, 210)]
        true_v = np.array([1.37, -1.23])
        truth = np.array([48., 3.])+times[:, None]*true_v
        wrapped = truth % 50
        wrapped[25:35] = np.nan
        track, velocity, origin = periodic_fit_at(wrapped, times)
        np.testing.assert_allclose(velocity, true_v, atol=1e-6)
        finite = np.isfinite(track).all(1)
        delta = track[finite]-truth[finite]
        np.testing.assert_allclose(delta, np.broadcast_to(delta[0], delta.shape), atol=1e-6)

    def test_missing_prediction_is_joint_failure(self):
        case = self.case()
        rows, _ = evaluate_case(case, np.zeros_like(case['frames'][50:]))
        summary = summarize(rows[-1:])
        self.assertEqual(summary['missing_rate'], 1.)
        self.assertEqual(summary['joint_success'], 0.)
        self.assertEqual(summary['frame_joint_success'], 0.)
        self.assertEqual(summary['coverage'], 0.)

    def case(self):
        template = np.zeros((50, 50), np.float32)
        template[4:12, 4:12] = np.array([.2, .4, .8, 1, .7, .5, .3, .1])[None]
        positions = np.array([4., 4.])+np.arange(210)[:, None]*np.array([1., 0.])
        cameras = np.zeros((210, 2))
        frames, visibility = render(template, positions, cameras, True)
        return dict(seed=0, mode='bilinear', template=template, frames=frames,
                    positions=positions, cameras=cameras, visibility=visibility,
                    velocity=np.array([1., 0.]), times=np.arange(210))

    def test_long_drift_is_not_circularly_forgiven(self):
        case = self.case()
        positions = np.repeat(case['positions'][49:50], 160, axis=0)
        pred, _ = render(case['template'], positions, case['cameras'][50:], True)
        rows, arrays = evaluate_case(case, pred)
        self.assertGreater(arrays['errors'][-1], 159.)
        self.assertGreater(arrays['continuity_errors'][-1], 159.)
        self.assertLess(arrays['frame_errors'][99], 1e-6)
        self.assertGreater(rows[-1]['ade'], 50.)
        self.assertLess(rows[-1]['predicted_vx'], .001)

    def test_perfect_frames_are_success_with_detector_floor(self):
        case = self.case()
        rows, _ = evaluate_case(case, case['frames'][50:])
        summary = summarize(rows[-1:])
        self.assertEqual(summary['mse'], 0.)
        self.assertGreater(summary['joint_success'], .98)
        self.assertGreater(summary['frame_joint_success'], .98)

    def test_short_horizon_does_not_use_future_detections(self):
        case = self.case()
        prediction = case['frames'][50:].copy()
        reference, _ = evaluate_case(case, prediction)
        prediction[20:] = 0
        altered, _ = evaluate_case(case, prediction)
        for key in ['ade', 'predicted_vx', 'scored_frames', 'success_frames', 'mse',
                    'frame_success_frames', 'continuity_scored_frames', 'continuity_error_sum']:
            self.assertEqual(reference[0][key], altered[0][key])

    def test_nonconstant_motion_keeps_frame_score_and_continuity(self):
        case = self.case()
        case['positions'][:] = [10, 10]
        case['velocity'][:] = 0
        case['frames'], case['visibility'] = render(case['template'], case['positions'], case['cameras'], True)
        position = np.tile([10., 10.], (160, 1))
        position[:, 0] += 6*np.sin(2*np.pi*np.arange(160)/40)
        prediction, _ = render(case['template'], position, case['cameras'][50:], True)
        rows, arrays = evaluate_case(case, prediction)
        measured = summarize(rows[-1:])
        self.assertAlmostEqual(measured['frame_joint_success'], .25)
        self.assertAlmostEqual(measured['continuity_coverage'], 1.)
        self.assertGreater(measured['continuity_ade'], 3.)
        self.assertEqual(measured['certified_joint_success'], 0.)
        self.assertTrue(np.isfinite(arrays['continuity_track']).all())

    def test_continuity_single_anchor_large_winding_and_permanent_ambiguity(self):
        times = np.arange(50, 210)
        truth = np.array([210., 310.])+(times-49)[:, None]*np.array([1., -1.])
        track, unknown = continuity_unwrap(truth % 50, times, np.array([210., 310.]), 49)
        np.testing.assert_allclose(track, truth)
        self.assertFalse(unknown.any())
        wrapped = truth % 50
        wrapped[4:20] = np.nan
        track, unknown = continuity_unwrap(wrapped, times, np.array([210., 310.]), 49)
        self.assertTrue(unknown[20:].all())
        self.assertFalse(np.isfinite(track[20:]).any())

    def test_ghost_is_frame_failure(self):
        case = self.case()
        case['positions'][:] = [10, 10]
        case['velocity'][:] = 0
        case['frames'], case['visibility'] = render(case['template'], case['positions'], case['cameras'], True)
        primary = translate(case['template'], [10, 10])[:32, :32]
        secondary = translate(case['template'], [18, 10])[:32, :32]
        pred = np.repeat(np.clip(primary+secondary, 0, 1)[None], 160, axis=0)
        rows, _ = evaluate_case(case, pred)
        measured = summarize(rows[-1:])
        self.assertEqual(measured['ghost_rate'], 1.)
        self.assertEqual(measured['frame_joint_success'], 0.)

    def test_paired_point_uses_pooled_ratio(self):
        a = []
        for seed, visible, missing in [(0, 1, 1), (1, 9, 0)]:
            a.append(dict(seed=seed, case_id=str(seed), horizon=40,
                          visible_frames=visible, scored_frames=visible-missing,
                          success_frames=visible-missing, missing_frames=missing,
                          ghost_frames=0, ambiguous_frames=0, error_sum=0,
                          squared_error_sum=0, pixel_count=10,
                          tracker_error_sum=0, tracker_scored_frames=visible))
        b = [dict(row, missing_frames=0) for row in a]
        result = paired_bootstrap(a, b, repeats=50)
        self.assertAlmostEqual(result['metrics']['missing_rate']['difference'], .1)
        self.assertAlmostEqual(summarize(a)['missing_rate'], .1)
        self.assertEqual(paired_bootstrap(a, a, repeats=50)['metrics']['mse']['ci95'], [0., 0.])


if __name__ == '__main__':
    unittest.main()
