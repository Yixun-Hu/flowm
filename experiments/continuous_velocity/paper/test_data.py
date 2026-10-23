"""Behavioral checks for paired rendering, temporal protocols and split integrity."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('paper_data_under_test', HERE / 'data.py')
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_signed_render_matches_reference(self):
        digit = data.test_digits()[0][23]
        positions = np.array([[-.37, .63], [2.45, -5.6], [50.2, -100.7]])
        cameras = np.array([[-16., -16.], [3., -5.], [12., 7.]])
        template = np.zeros((50, 50), np.float32)
        template[:28, :28] = digit
        for mode in ['nearest', 'bilinear']:
            actual = data.render(torch.tensor(digit)[None],
                                 torch.tensor(positions, dtype=torch.float32)[None],
                                 torch.tensor(cameras, dtype=torch.float32)[None], mode)[0].numpy()
            expected, _ = data._measurement.render(template, positions, cameras,
                                                    interpolate=mode == 'bilinear')
            np.testing.assert_allclose(actual, expected, atol=8e-6)

    def test_camera_extension_preserves_accumulation(self):
        case = data.make_case(0, [.37, -.63], horizon=160)
        self.assertEqual(case['frames'].shape, (210, 32, 32))
        np.testing.assert_array_equal(case['actions'][200:], case['actions'][:10])
        np.testing.assert_allclose(np.diff(case['cameras'], axis=0), case['actions'][1:])
        np.testing.assert_allclose(case['positions'][209] - case['positions'][0],
                                   np.array([.37, -.63]) * 209)

    def test_pairing_preserves_appearance_start_camera(self):
        a = data.make_case(3, .37, mode='nearest', horizon=40)
        b = data.make_case(3, [-1.23, .63], mode='bilinear', horizon=40)
        self.assertEqual(a['digit_index'], b['digit_index'])
        np.testing.assert_array_equal(a['template'], b['template'])
        np.testing.assert_array_equal(a['cameras'], b['cameras'])
        np.testing.assert_array_equal(a['positions'][0], b['positions'][0])

    def test_irregular_context_retains_true_timestamps(self):
        regular = data.make_case(4, .63, horizon=80)
        irregular = data.make_case(4, .63, horizon=80, context='irregular')
        indices = irregular['context_indices']
        self.assertEqual(len(indices), 25)
        self.assertEqual((indices[0], indices[-1]), (0, 49))
        self.assertGreater(len(np.unique(np.diff(indices))), 1)
        np.testing.assert_array_equal(regular['frames'], irregular['frames'])
        inputs = data.model_batch([irregular])
        np.testing.assert_array_equal(inputs['context_times'][0].numpy(), indices)
        self.assertEqual(set(inputs), {'context', 'cameras', 'context_times',
                                       'target_cameras', 'target_times'})
        self.assertEqual(inputs['context'].shape, (1, 25, 32, 32))
        self.assertEqual(inputs['target_times'][0, -1], 129)

    def test_blackout_changes_only_available_observations(self):
        regular = data.make_case(5, .37, horizon=40)
        gap = data.make_case(5, .37, horizon=40, context='gap', absence=20)
        np.testing.assert_array_equal(regular['frames'], gap['frames'])
        self.assertEqual(gap['context_times'][-1], 29)
        self.assertEqual(gap['target_times'][0], 50)
        self.assertEqual(len(gap['context_indices']), 30)

    def test_two_d_holdouts_and_bounds(self):
        velocity = data._sample_velocities(np.random.default_rng(33), 1000, '2d')
        self.assertFalse(data.heldout_direction(velocity).any())
        self.assertLessEqual(np.abs(velocity).max(), 2)
        self.assertTrue((velocity[:, 0] < 0).any())
        self.assertTrue((velocity[:, 1] < 0).any())
        self.assertTrue(data.heldout_direction([.37, .37]))
        self.assertTrue(data.heldout_direction([-.37, -.37]))
        self.assertFalse(data.heldout_direction([.37, -.37]))
        self.assertFalse(data.heldout_direction([0., 0.]))

    def test_training_splits_pairing_and_modes(self):
        dataset = data.TrainingData('cpu', seed=17, train_size=100)
        first = dataset.batch(4)
        dataset.rng = np.random.default_rng(17)
        dataset.render_mode = 'nearest'
        second = dataset.batch(4)
        dataset.rng = np.random.default_rng(17)
        dataset.render_mode = 'mixed'
        mixed = dataset.batch(4)
        for batch in [first, second, mixed]:
            self.assertLess(batch['digit_indices'].max(), 100)
            self.assertGreaterEqual(batch['camera_episodes'].min(), 64)
            self.assertLess(batch['camera_episodes'].max(), 224)
            self.assertEqual(batch['target_times'].max(), 89)
            for key in ['digit_indices', 'camera_episodes', 'true_velocity', 'cameras']:
                np.testing.assert_array_equal(first[key], batch[key])
        self.assertTrue((first['render_modes'] == 'bilinear').all())
        self.assertTrue((second['render_modes'] == 'nearest').all())
        for i, mode in enumerate(mixed['render_modes']):
            reference = first if mode == 'bilinear' else second
            np.testing.assert_array_equal(mixed['context'][i], reference['context'][i])
        validation = dataset.batch(8, validation=True)
        self.assertGreaterEqual(validation['digit_indices'].min(), 55000)
        self.assertLess(validation['digit_indices'].max(), 60000)
        self.assertGreaterEqual(validation['camera_episodes'].min(), 224)

    def test_test_camera_split_enforced(self):
        with self.assertRaises(ValueError):
            data.make_case(64, .37)


if __name__ == '__main__':
    unittest.main()
