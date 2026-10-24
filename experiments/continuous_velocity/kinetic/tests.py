"""CPU tests for the kinetic extension. Run: python experiments/continuous_velocity/kinetic/tests.py"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from model import ContinuousMemory, world_embed  # noqa: E402
from paper.mechanism import _observations  # noqa: E402
from paper.data import make_case  # noqa: E402
from paper.test_mechanism import synthetic_batch  # noqa: E402
from kinetic.transport import (_hermitian_weights, _interpolant, raised_cosine_taper,  # noqa: E402
                               refine_peak, spectral_shift)
from kinetic.memory import KineticMemory, build_kinetic_model, fit_motion, kinetic_observations  # noqa: E402
from kinetic.scenes import (KineticTrainingData, SPEED_BOUND, make_kinetic_case, max_speed,  # noqa: E402
                            sample_acceleration)

torch.set_num_threads(1)
YY, XX = torch.meshgrid(torch.arange(50.), torch.arange(50.), indexing='ij')


def blob(cx, cy, s2=8.):
    dx = (XX-cx+25) % 50-25
    dy = (YY-cy+25) % 50-25
    return torch.exp(-(dx**2+dy**2)/s2)


class TransportTests(unittest.TestCase):
    def test_refinement_is_rejected_when_the_denominator_is_at_the_floor(self):
        # Audit reproduction: S = 2N/D is identically 0.5 (zero gradient), but
        # with D below the clamp floor the clamped derivatives are inconsistent;
        # such a case must not be accepted.
        y, x = torch.meshgrid(torch.arange(50, dtype=torch.float64),
                              torch.arange(50, dtype=torch.float64), indexing='ij')
        d = 1e-6*(1+.2*torch.cos(2*torch.pi*x/50)+.2*torch.cos(2*torch.pi*y/50))
        ns, ds = torch.fft.rfft2(d/4), torch.fft.rfft2(d)
        start = torch.tensor([.2, -.3], dtype=torch.float64)
        refined, accepted = refine_peak(ns, ds, start)
        self.assertFalse(bool(accepted))
        self.assertTrue(torch.equal(refined, start))
        # Same shape of problem, denominator well above the floor: refinement still works.
        refined, accepted = refine_peak(ns*1e6, ds*1e6, start)
        self.assertTrue(torch.isfinite(refined).all())

    def test_integer_shift_equals_roll(self):
        torch.manual_seed(0)
        h = torch.randn(3, 4, 5, 10, 13)
        s = torch.randint(-7, 8, (3, 4, 2)).float()
        out = spectral_shift(h, s)
        ref = torch.stack([torch.stack([torch.roll(h[b, v], shifts=(int(s[b, v, 1]), int(s[b, v, 0])), dims=(-2, -1))
                                        for v in range(4)]) for b in range(3)])
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=0)
        per_batch = torch.randint(-7, 8, (3, 2)).float()
        out = spectral_shift(h, per_batch)
        ref = torch.stack([torch.roll(h[b], shifts=(int(per_batch[b, 1]), int(per_batch[b, 0])), dims=(-2, -1))
                           for b in range(3)])
        torch.testing.assert_close(out, ref, atol=2e-5, rtol=0)

    def test_fractional_shift_of_band_limited_blob_and_single_composition(self):
        g = blob(10., 20., 12.)
        shifted = spectral_shift(g, torch.tensor([.37, -1.63]))
        torch.testing.assert_close(shifted, blob(10.37, 18.37, 12.), atol=1e-4, rtol=0)
        # Two shifts of a band-limited (Nyquist-free) map compose; the model uses one shift per use.
        spectrum = torch.fft.rfft2(g)
        spectrum[25, :] = 0; spectrum[:, 25] = 0
        band = torch.fft.irfft2(spectrum, s=(50, 50))
        twice = spectral_shift(spectral_shift(band, torch.tensor([.5, .25])), torch.tensor([-.5, -.25]))
        torch.testing.assert_close(twice, band, atol=1e-5, rtol=0)

    def test_shift_gradient_direction(self):
        s = torch.tensor([.3, -.2], requires_grad=True)
        y = spectral_shift(blob(10., 12.), s)
        ((y-blob(10.8, 12.5))**2).sum().backward()
        self.assertTrue(s.grad[0] < 0 and s.grad[1] < 0)

    def test_interpolant_matches_lattice_and_finite_differences(self):
        torch.manual_seed(1)
        f = torch.randn(2, 50, 50)
        weights = _hermitian_weights(50, 'cpu')
        values = _interpolant(torch.fft.rfft2(f), torch.tensor([[3., 7.], [49., 25.]]), weights)
        torch.testing.assert_close(values, torch.stack([f[0, 7, 3], f[1, 25, 49]]), atol=1e-4, rtol=0)
        spectrum = torch.fft.rfft2(blob(20.3, 10.7)[None])
        p = torch.tensor([[20.1, 10.9]]); eps = 3e-2
        ex, ey = torch.tensor([[eps, 0.]]), torch.tensor([[0., eps]])
        fx = _interpolant(spectrum, p, weights, 1, 0)
        fd = (_interpolant(spectrum, p+ex, weights)-_interpolant(spectrum, p-ex, weights))/(2*eps)
        self.assertLess(abs(fx.item()-fd.item()), 2e-3)
        fxy = _interpolant(spectrum, p, weights, 1, 1)
        fdxy = (_interpolant(spectrum, p+ex+ey, weights)-_interpolant(spectrum, p+ex-ey, weights)
                - _interpolant(spectrum, p-ex+ey, weights)+_interpolant(spectrum, p-ex-ey, weights))/(4*eps*eps)
        self.assertLess(abs(fxy.item()-fdxy.item()), 2e-3)

    def test_newton_refinement_recovers_subpixel_peak_and_rejects_nonpeaks(self):
        a, b = blob(10., 10.)[None], blob(10.37, 9.63)[None]
        numerator = torch.fft.rfft2(a).conj()*torch.fft.rfft2(b)
        constant = a.square().sum()+b.square().sum()   # lattice D is a sum over pixels
        denominator = torch.fft.rfft2(constant*torch.ones(1, 50, 50))
        refined, accepted = refine_peak(numerator, denominator, torch.tensor([[0., 0.]]))
        self.assertTrue(bool(accepted[0]))
        torch.testing.assert_close(refined, torch.tensor([[.37, -.37]]), atol=2e-3, rtol=0)
        # Starting far from any maximum (on a saddle/minimum) is rejected and returns the start.
        start = torch.tensor([[25., 25.]])
        refined, accepted = refine_peak(numerator, denominator, start)
        self.assertFalse(bool(accepted[0]))
        torch.testing.assert_close(refined, start)
        # Integer-lattice maxima stay in place.
        numerator = torch.fft.rfft2(a).conj()*torch.fft.rfft2(blob(13., 8.)[None])
        refined, accepted = refine_peak(numerator, denominator, torch.tensor([[3., -2.]]))
        self.assertTrue(bool(accepted[0]))
        torch.testing.assert_close(refined, torch.tensor([[3., -2.]]), atol=1e-3, rtol=0)

    def test_taper_interior_is_one_and_border_is_zero(self):
        taper = raised_cosine_taper(32, 3)
        self.assertEqual(taper[3:-3, 3:-3].min().item(), 1.)
        self.assertLess(taper[0, 16].item(), .1)
        self.assertLess(taper[16, 31].item(), .1)
        self.assertEqual(taper.shape, (32, 32))


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.dt = torch.tensor([[1., 2., 4., 1., 2., 4., 2., 1.]])
        midpoints = torch.tensor([[-40., -30., -20., -10., -5., -1., -25., -33.]])
        self.s = self.dt*midpoints
        self.v = torch.tensor([[.37, -.63]])
        self.a = torch.tensor([[.004, -.002]])
        self.d = self.dt[:, :, None]*self.v[:, None]+self.s[:, :, None]*self.a[:, None]
        self.w = torch.tensor([[.3, .9, .5, .7, .2, .8, .6, .4]])

    def test_order_two_recovers_motion_exactly_and_order_one_matches_parent(self):
        v, a, stats = fit_motion(self.d, self.dt, self.s, self.w, order=2, kappa=0.)
        torch.testing.assert_close(v, self.v, atol=1e-5, rtol=0)
        torch.testing.assert_close(a, self.a, atol=1e-6, rtol=0)
        v, a, stats = fit_motion(self.d, self.dt, self.s, self.w, order=2, kappa=4.)
        torch.testing.assert_close(a, self.a, atol=1e-6, rtol=0)   # noiseless: SE ~ 0, gate open
        torch.testing.assert_close(stats['acceleration_gate'], torch.ones(1, 2), atol=1e-4, rtol=0)
        v1, a1, _ = fit_motion(self.d, self.dt, self.s, self.w, order=1)
        torch.testing.assert_close(a1, torch.zeros(1, 2))
        self.assertGreater((v1-self.v).abs().max().item(), .01)  # order 1 is biased by curvature

    def test_gate_closes_for_pure_noise_and_degenerate_design(self):
        torch.manual_seed(3)
        trials = 200
        dt = self.dt.expand(trials, -1); s = self.s.expand(trials, -1); w = self.w.expand(trials, -1)
        d = dt[:, :, None]*self.v[:, None]+.05*torch.randn(trials, 8, 2)
        _, gated, stats = fit_motion(d, dt, s, w, order=2, kappa=9.)
        self.assertLess((stats['acceleration_gate'] > 0).float().mean().item(), .1)
        self.assertLess(gated.abs().mean().item(), stats['acceleration_raw'].abs().mean().item())
        # All pairs share one midpoint: s is collinear with dt and the fit falls back to order 1.
        s_degenerate = self.dt*(-20.)
        v, a, stats = fit_motion(self.d, self.dt, s_degenerate, self.w, order=2, kappa=4.)
        v1, _, _ = fit_motion(self.d, self.dt, s_degenerate, self.w, order=1)
        torch.testing.assert_close(a, torch.zeros(1, 2))
        self.assertFalse(bool(stats['acceleration_gate'].any()))

    def test_gradients_flow_through_order_two_fit(self):
        d = self.d.clone().requires_grad_(); w = self.w.clone().requires_grad_()
        v, a, _ = fit_motion(d, self.dt, self.s, w, order=2, kappa=4.)
        (v.square().sum()+1e4*a.square().sum()).backward()
        self.assertTrue(torch.isfinite(d.grad).all() and torch.isfinite(w.grad).all())
        self.assertGreater(d.grad.abs().sum().item(), 0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.inputs, self.velocity = synthetic_batch()

    def test_baseline_reproduces_parent_model_exactly(self):
        torch.manual_seed(1)
        parent = ContinuousMemory('physical')
        kinetic = KineticMemory()
        kinetic.load_state_dict(parent.state_dict(), strict=True)
        expected, expected_aux = parent(**self.inputs)
        actual, actual_aux = kinetic(**self.inputs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for key in ('velocity', 'offsets', 'memory'):
            torch.testing.assert_close(actual_aux[key], expected_aux[key], atol=0, rtol=0)
        self.assertEqual(set(parent.state_dict()), set(kinetic.state_dict()))

    def test_observations_match_parent_without_refinement(self):
        torch.manual_seed(2)
        model = ContinuousMemory('physical')
        images, cameras, times = (self.inputs[k] for k in ('context', 'cameras', 'context_times'))
        b, t, h, w = images.shape
        learned = model.correspondence.encoder(images.reshape(b*t, 1, h, w)).reshape(b, t, -1, h, w)
        fields, mask = world_embed(torch.cat([images[:, :, None], learned], 2), cameras)
        temperature = model.correspondence.log_temperature.exp().clamp(5, 150)
        expected = _observations(fields, mask, times, model.correspondence.shifts, temperature, model.correspondence.confidence)
        actual = kinetic_observations(fields, mask, times, model.correspondence.shifts, temperature, model.correspondence.confidence)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
        reference = times[:, -1:]
        i = torch.cat([torch.linspace(0, t-lag-1, 12).long() for lag in (1, 2, 4)])
        j = torch.cat([torch.linspace(0, t-lag-1, 12).long()+lag for lag in (1, 2, 4)])
        s = .5*((times[:, j]-reference)**2-(times[:, i]-reference)**2)
        torch.testing.assert_close(actual['s'], s)

    def test_all_variants_run_backward_and_are_camera_origin_invariant(self):
        for align in ('bilinear', 'spectral'):
            for render in ('bilinear', 'spectral'):
                for refine in ('none', 'newton'):
                    for order in (1, 2):
                        torch.manual_seed(5)
                        model = build_kinetic_model(dict(align=align, render=render, refine=refine, order=order))
                        prediction, aux = model(**self.inputs)
                        self.assertEqual(prediction.shape, (1, 6, 32, 32))
                        self.assertTrue(torch.isfinite(prediction).all())
                        torch.testing.assert_close(aux['velocity'], self.velocity, atol=.05, rtol=0)
                        (prediction-.5).square().mean().backward()
                        self.assertGreater(model.correspondence.log_temperature.grad.abs().item(), 0)
                        with torch.no_grad():
                            shifted = dict(self.inputs)
                            shifted['cameras'] = shifted['cameras']+torch.tensor([7., -11.])
                            shifted['target_cameras'] = shifted['target_cameras']+torch.tensor([7., -11.])
                            moved, _ = model(**shifted)
                        torch.testing.assert_close(moved, prediction, atol=5e-4, rtol=0)
                        if refine == 'newton':
                            self.assertGreater(aux['refined'].float().mean().item(), .5)

    def test_second_order_offsets_follow_positions_and_zero_acceleration_intervention(self):
        model = build_kinetic_model(dict(order=2))
        dt = torch.tensor([[-49., -10., 0., 1., 40.]])
        v, a = torch.tensor([[.37, .2]]), torch.tensor([[.004, -.001]])
        offsets = model.motion_offsets(dt, v, a, torch.zeros(1, 15))
        torch.testing.assert_close(offsets, dt[:, :, None]*v[:, None]+.5*dt[:, :, None]**2*a[:, None])
        _, aux = model(**self.inputs, velocity_override=v, acceleration_override=a)
        torch.testing.assert_close(aux['acceleration'], a)
        elapsed = (self.inputs['target_times']-self.inputs['context_times'][:, -1:])[:, :, None]
        # transport-only: alignment keeps (v, a); future transport uses v alone.
        _, aux = model(**self.inputs, velocity_override=v, acceleration_override=a, zero_acceleration='transport')
        torch.testing.assert_close(aux['acceleration'], torch.zeros(1, 2))
        torch.testing.assert_close(aux['offsets'], elapsed*v[:, None])
        torch.testing.assert_close(aux['velocity'], v)
        # both: the constant-velocity profile of the same robust normal equations replaces v.
        _, full = model(**self.inputs)
        _, both = model(**self.inputs, zero_acceleration='both')
        torch.testing.assert_close(both['velocity'], full['profiled_velocity'])
        torch.testing.assert_close(both['offsets'], elapsed*full['profiled_velocity'][:, None])
        with self.assertRaises(ValueError):
            model(**self.inputs, zero_acceleration=True)

    def test_initialization_matches_parent_and_controls_have_curvature_features(self):
        torch.manual_seed(2026); parent = ContinuousMemory('physical')
        torch.manual_seed(2026); kinetic = build_kinetic_model(dict(align='spectral', refine='newton', order=2))
        for key, value in parent.state_dict().items():
            torch.testing.assert_close(kinetic.state_dict()[key], value, atol=0, rtol=0)
        for kind in ('relaxed', 'offset'):
            model = build_kinetic_model(dict(kind=kind))
            prediction, aux = model(**self.inputs)
            self.assertEqual(aux['summary'].shape, (1, 15))
            self.assertTrue(torch.isfinite(prediction).all())
            prediction.square().mean().backward()
            self.assertEqual(model.offset_head[0].in_features, 16)
        # relaxed starts exactly at the physical prediction.
        torch.manual_seed(3); physical = build_kinetic_model(dict())
        torch.manual_seed(3); relaxed = build_kinetic_model(dict(kind='relaxed'))
        relaxed.load_state_dict({**relaxed.state_dict(), **{k: v for k, v in physical.state_dict().items()}})
        torch.testing.assert_close(relaxed(**self.inputs)[0], physical(**self.inputs)[0], atol=0, rtol=0)
        # Summaries resolve time: with constant velocity the early and late halves
        # agree; under acceleration the late half moves faster by about 25 a.
        from data import render
        yy, xx = torch.meshgrid(torch.arange(28), torch.arange(28), indexing='ij')
        sprite = (.8*torch.exp(-((xx-10.)**2+(yy-11.)**2)/15)+.5*torch.exp(-((xx-17.)**2+(yy-16.)**2)/6)).clamp(0, 1)
        times = torch.arange(50).float()[None]
        for a, expected_gap in ((0., 0.), (.01, .257)):
            positions = torch.tensor([[[5., 20.]]])+times[:, :, None]*torch.tensor([[.3, 0.]])+.5*times[:, :, None]**2*torch.tensor([[a, 0.]])
            cameras = positions.floor()-2   # tracking camera keeps the sprite in view
            frames = render(sprite[None], positions, cameras)
            _, aux = physical(context=frames, cameras=cameras, context_times=times,
                              target_cameras=cameras[:, :1], target_times=times[:, -1:]+1)
            summary = aux['summary'][0]
            gap = (summary[2]-summary[0]).item()   # lag-1 group: late-half vx minus early-half vx
            self.assertAlmostEqual(gap, expected_gap, delta=.08)

    def test_spectral_alignment_support_is_bounded(self):
        model = build_kinetic_model(dict(align='spectral'))
        images, cameras, times = (self.inputs[k] for k in ('context', 'cameras', 'context_times'))
        fields, mask, _ = model.correspondence.observe(images, cameras, times, model.taper)
        offsets = torch.tensor([[[.37*k, -.63*k] for k in range(-11, 1)]])
        aligned, visible = model._align(fields, mask, offsets)
        self.assertGreaterEqual(visible.min().item(), 0.)
        self.assertLessEqual(visible.max().item(), 1.+1e-5)
        values = aligned/visible.clamp_min(1e-4)
        # Raw intensities stay plausible on this synthetic batch (real data can show
        # isolated taper-ramp pixels near 1.9; the fraction above 1.5 is below 1e-5).
        self.assertLess(values[:, :, 0].abs().max().item(), 1.5)
        self.assertTrue(bool(((visible > 0) & (visible < .05)).sum() == 0))

    def test_spectral_alignment_with_oracle_motion_reconstructs_the_scene(self):
        for align, render in (('spectral', 'spectral'), ('spectral', 'bilinear'), ('bilinear', 'spectral')):
            model = build_kinetic_model(dict(align=align, render=render))
            prediction, _ = model(**self.inputs, velocity_override=self.velocity)
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertLess(prediction.min().item(), 0.05)
            self.assertGreater(prediction.max().item(), 0.5)


class SceneTests(unittest.TestCase):
    def test_zero_acceleration_reproduces_paper_case(self):
        base = make_case(3, .63, mode='bilinear', horizon=40)
        case = make_kinetic_case(3, .63, 0., mode='bilinear', horizon=40)
        np.testing.assert_array_equal(case['frames'], base['frames'])
        np.testing.assert_array_equal(case['positions'], base['positions'])
        np.testing.assert_allclose(case['velocity'], base['velocity'])
        elapsed = case['target_times']-case['context_times'][-1]
        np.testing.assert_allclose(case['true_offsets'], elapsed[:, None]*base['velocity'][None], atol=1e-4)

    def test_accelerated_case_obeys_the_law_and_bounds(self):
        case = make_kinetic_case(5, .37, .004, mode='nearest', horizon=40)
        times = case['times']
        expected = case['positions'][0][None]+times[:, None]*np.array([.37, 0.])+.5*times[:, None]**2*np.array([.004, 0.])
        np.testing.assert_allclose(case['positions'], expected, atol=1e-4)
        np.testing.assert_allclose(case['velocity'], [.37+.004*49, 0.])
        np.testing.assert_allclose(case['true_offsets'][-1], case['positions'][89]-case['positions'][49], atol=1e-4)
        self.assertEqual(case['frames'].shape, (90, 32, 32))
        with self.assertRaises(ValueError):
            make_kinetic_case(5, 1.23, .012, horizon=80)
        self.assertLessEqual(max_speed(1.23, .012, 90), SPEED_BOUND)

    def test_sampler_respects_speed_bound_and_zero_fraction(self):
        rng = np.random.default_rng(0)
        draws = np.array([sample_acceleration(rng, 1.9) for _ in range(2000)])
        self.assertTrue(np.all(np.abs(1.9+draws*89) <= SPEED_BOUND+1e-9))
        self.assertGreater((draws == 0).mean(), .2); self.assertLess((draws == 0).mean(), .4)
        self.assertGreater((draws < 0).sum(), (draws > 0).sum())   # positive side is bounded by the speed cap
        nonzero = np.abs(draws[draws != 0])
        self.assertGreaterEqual(nonzero.min(), 2e-4); self.assertLessEqual(nonzero.max(), 2e-2)

    def test_training_streams_share_scenes_with_the_paper_sampler(self):
        try:
            from paper.data import TrainingData
            reference = TrainingData('cpu', seed=12).batch(4)
        except FileNotFoundError:
            self.skipTest('MNIST assets unavailable')
        accelerated = KineticTrainingData('cpu', seed=12, motion='accelerated').batch(4)
        np.testing.assert_array_equal(accelerated['digit_indices'], reference['digit_indices'])
        np.testing.assert_array_equal(accelerated['camera_episodes'], reference['camera_episodes'])
        torch.testing.assert_close(accelerated['initial_velocity'], reference['true_velocity'])
        torch.testing.assert_close(accelerated['true_velocity'],
                                   accelerated['initial_velocity']+49*accelerated['true_acceleration'])
        plain = KineticTrainingData('cpu', seed=12, motion='horizontal').batch(4)
        torch.testing.assert_close(plain['context'], reference['context'])
        self.assertEqual(set(accelerated)-set(reference), {'true_acceleration', 'initial_velocity'})


if __name__ == '__main__':
    unittest.main()
