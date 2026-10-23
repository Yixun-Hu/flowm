"""CPU checks for scientific controls, requiring no downloaded data."""
import unittest
from unittest.mock import patch
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data import render
from model import ContinuousMemory, CorrespondenceAttention
from paper.mechanism import (
    MechanismCorrespondence, VelocityIntervention, build_model,
    mean_memory_copy, raw_matching_copy)


torch.set_num_threads(1)


def synthetic_batch():
    yy, xx = torch.meshgrid(torch.arange(28), torch.arange(28), indexing='ij')
    sprite = (.8*torch.exp(-((xx-10.)**2+(yy-11.)**2)/15)
              + .5*torch.exp(-((xx-17.)**2+(yy-16.)**2)/6)).clamp(0, 1)
    times = torch.arange(18).float()[None]
    velocity = torch.tensor([[.37, -.63]])
    positions = torch.tensor([[[45., 3.]]])+times[:, :, None]*velocity[:, None]
    cameras = positions.floor()-2
    frames = render(sprite[None], positions, cameras)
    return dict(context=frames[:, :12], cameras=cameras[:, :12],
                context_times=times[:, :12], target_cameras=cameras[:, 12:],
                target_times=times[:, 12:]), velocity


class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.inputs, self.velocity = synthetic_batch()

    def test_factored_correspondence_reproduces_pilot(self):
        original = CorrespondenceAttention()
        control = MechanismCorrespondence()
        control.load_state_dict(original.state_dict())
        args = [self.inputs[k] for k in ('context', 'cameras', 'context_times')]
        expected = original(*args)
        actual = control(*args)
        for left, right in zip(expected[:4], actual[:4]):
            torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-6)
        for key in expected[4]:
            torch.testing.assert_close(expected[4][key], actual[4][key])

    def test_intervention_identity_and_oracle_reproduce_original(self):
        model = ContinuousMemory()
        for override in (None, self.velocity):
            expected, expected_aux = model(**self.inputs, velocity_override=override)
            actual, actual_aux = VelocityIntervention(model)(**self.inputs, velocity_override=override)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            for key in ('velocity', 'offsets', 'memory'):
                torch.testing.assert_close(expected_aux[key], actual_aux[key], rtol=0, atol=0)

    def test_quantization_controls_alignment_and_transport(self):
        model = ContinuousMemory()
        quantized = torch.round(self.velocity/.5)*.5
        expected, expected_aux = model(**self.inputs, velocity_override=quantized)
        actual, aux = VelocityIntervention(model, .5)(**self.inputs, velocity_override=self.velocity)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(aux['memory'], expected_aux['memory'], rtol=0, atol=0)
        torch.testing.assert_close(aux['velocity'], quantized)
        self.assertTrue(aux['oracle_velocity_used'])
        # Estimated-velocity intervention never consumes a true velocity.
        _, baseline_aux = model(**self.inputs)
        _, estimated_aux = VelocityIntervention(model, 1)(**self.inputs)
        torch.testing.assert_close(estimated_aux['velocity'], baseline_aux['velocity'].round())
        self.assertFalse(estimated_aux['oracle_velocity_used'])

    def test_raw_baseline_has_no_parameters_and_crosses_boundary(self):
        model = build_model('raw_untrained')
        self.assertEqual(sum(p.numel() for p in model.parameters()), 0)
        output, auxiliary = model(**self.inputs)
        self.assertEqual(output.shape, (1, 6, 32, 32))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(auxiliary['velocity'], self.velocity, atol=.06, rtol=0)

    def test_transport_only_quantization_preserves_canonical_memory(self):
        model = ContinuousMemory()
        _, original = model(**self.inputs, velocity_override=self.velocity)
        _, intervention = VelocityIntervention(model, 1, scope='transport')(
            **self.inputs, velocity_override=self.velocity)
        torch.testing.assert_close(original['memory'], intervention['memory'], atol=0, rtol=0)
        torch.testing.assert_close(intervention['alignment_velocity'], self.velocity)
        torch.testing.assert_close(intervention['velocity'], self.velocity.round())

    def test_raw_matching_is_independent_of_learned_parameters(self):
        model = raw_matching_copy(ContinuousMemory())
        _, first = model(**self.inputs)
        with torch.no_grad():
            for parameter in model.correspondence.parameters():
                parameter.normal_(0, 2)
        _, second = model(**self.inputs)
        torch.testing.assert_close(first['velocity'], second['velocity'], rtol=0, atol=0)

    def test_mlp_uses_no_physical_fit_and_gets_gradients(self):
        model = build_model('mlp_velocity')
        with patch('paper.mechanism._physical_fit',
                   side_effect=AssertionError('MLP must not call physical fit')):
            prediction, aux = model(**self.inputs)
        (prediction.square().mean()+aux['velocity'].square().sum()).backward()
        head_gradients = [p.grad for p in model.correspondence.velocity_head.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in head_gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in head_gradients), 0)
        self.assertGreater(model.correspondence.log_temperature.grad.abs().item(), 0)

    def test_mean_memory_copy_preserves_other_weights(self):
        model = ContinuousMemory()
        control = mean_memory_copy(model)
        for key, value in control.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key])
        # Initial learned logits are zero, therefore initial memory is the mean.
        expected, _ = model(**self.inputs)
        actual, _ = control(**self.inputs)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
