"""Training streams and paired test scenes with optional constant acceleration.

Wraps the paper study's samplers without modifying them. Trajectories obey
``p(t) = start + v0 t + a t^2 / 2`` in unwrapped coordinates. Acceleration is
drawn from a separate generator so that digits, camera episodes, starting
positions and initial velocities follow exactly the paper stream for a seed.
Only ``model_batch`` keys (and the keys of ``TrainingData.batch`` with the same
names) may reach a model; velocities, accelerations, positions and templates are
evaluation metadata.
"""
from pathlib import Path
import sys

import numpy as np
import torch

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from paper.data import (ROOT, HOLDOUT, TrainingData, _context_indices, _measurement,  # noqa: E402
                        _sample_velocities, heldout_direction, make_case, model_batch, render)

MOTIONS = ('horizontal', '2d', 'accelerated')
SPEED_BOUND = 2.4          # |v(t)| over the training sequence (t <= 89) and every test grid
ACCELERATION_RANGE = (2e-4, 2e-2)
ZERO_ACCELERATION_FRACTION = .3
TRAIN_LENGTH = 90


def sample_acceleration(rng, initial_velocity, horizon=TRAIN_LENGTH, bound=SPEED_BOUND):
    """Zero with probability .3, else a signed log-uniform magnitude kept within the speed bound.

    Rejection sampling (not clipping) avoids point masses at the bound.
    """
    if rng.random() < ZERO_ACCELERATION_FRACTION:
        return 0.
    low, high = np.log10(ACCELERATION_RANGE[0]), np.log10(ACCELERATION_RANGE[1])
    for _ in range(1000):
        magnitude = 10**rng.uniform(low, high)
        a = magnitude if rng.random() < .5 else -magnitude
        if abs(initial_velocity) <= bound and abs(initial_velocity+a*(horizon-1)) <= bound:
            return float(a)
    return 0.


def max_speed(initial_velocity, acceleration, length):
    """Largest |v(t)| for t in [0, length-1] on a one-dimensional law."""
    v0, a = float(initial_velocity), float(acceleration)
    return max(abs(v0), abs(v0+a*(length-1)))


class KineticTrainingData(TrainingData):
    """Paper sampler with an ``accelerated`` motion setting (horizontal v0, horizontal a).

    ``true_velocity`` in a batch is the velocity at the last context time,
    the quantity the model estimates; ``true_acceleration`` is the constant a.
    """
    def __init__(self, device, seed=1234, render_mode='bilinear', motion='horizontal',
                 train_size=55000, irregular_context=False):
        if motion not in MOTIONS:
            raise ValueError(f'Unknown motion setting: {motion}')
        super().__init__(device, seed=seed, render_mode=render_mode,
                         motion='horizontal' if motion == 'accelerated' else motion,
                         train_size=train_size, irregular_context=irregular_context)
        self.kinetic_motion = motion
        self.acceleration_rng = np.random.default_rng(777000+int(seed))

    def batch(self, batch_size=8, validation=False):
        if self.kinetic_motion != 'accelerated':
            result = super().batch(batch_size, validation)
            result['true_acceleration'] = torch.zeros_like(result['true_velocity'])
            result['initial_velocity'] = result['true_velocity'].clone()
            return result
        rng = self.rng
        indices = (rng.integers(55000, 60000, batch_size) if validation
                   else rng.integers(0, self.train_size, batch_size))
        episodes = (rng.integers(224, 256, batch_size) if validation
                    else rng.integers(64, 224, batch_size))
        velocities = _sample_velocities(rng, batch_size, 'horizontal')
        starts = rng.integers(0, 50, (batch_size, 2))
        nearest = rng.random(batch_size) < .5
        if self.render_mode != 'mixed':
            nearest[:] = self.render_mode == 'nearest'
        context_indices = _context_indices(rng, 'irregular' if self.irregular_context else 'regular')
        accelerations = np.array([[sample_acceleration(self.acceleration_rng, v[0]), 0.]
                                  for v in velocities], np.float32)
        targets = np.array([50, 54, 59, 69, 79, 89])
        time_indices = np.concatenate([context_indices, targets])
        times = torch.as_tensor(time_indices, device=self.device, dtype=torch.float32)[None].expand(batch_size, -1)
        velocity = torch.as_tensor(velocities, device=self.device)
        acceleration = torch.as_tensor(accelerations, device=self.device)
        positions = (torch.as_tensor(starts, device=self.device, dtype=torch.float32)[:, None]
                     + times[:, :, None]*velocity[:, None]
                     + .5*times[:, :, None].square()*acceleration[:, None])
        cameras = self.actions[episodes].cumsum(1)[:, time_indices]-16
        digits = self.images[indices]
        frames = render(digits, positions, cameras, mode='nearest' if nearest.all() else 'bilinear')
        if nearest.any() and not nearest.all():
            selected = torch.as_tensor(nearest, device=self.device)
            frames[selected] = render(digits[selected], positions[selected], cameras[selected], mode='nearest')
        n_context = len(context_indices)
        last = times[:, n_context-1:n_context]
        return dict(
            context=frames[:, :n_context], cameras=cameras[:, :n_context].float(),
            context_times=times[:, :n_context],
            target_cameras=cameras[:, n_context:].float(),
            target_times=times[:, n_context:], target=frames[:, n_context:],
            true_velocity=velocity+last*acceleration, true_acceleration=acceleration,
            initial_velocity=velocity,
            digit_indices=indices, camera_episodes=episodes,
            render_modes=np.where(nearest, 'nearest', 'bilinear'),
        )


def make_kinetic_case(seed, velocity, acceleration=0., mode='bilinear', horizon=40,
                      context='regular', appearance='mnist', camera='released', absence=0):
    """Paired scene from ``make_case`` with a constant-acceleration trajectory.

    Zero acceleration returns the paper case unchanged apart from added keys.
    ``velocity`` is the initial velocity (at t=0); ``case['velocity']`` is the
    velocity at the last context time, ``case['initial_velocity']`` the input,
    ``case['true_offsets']`` the unwrapped displacement of every target frame
    from the last context frame, valid for any motion law.
    """
    case = make_case(seed, velocity, mode=mode, horizon=horizon, context=context,
                     appearance=appearance, camera=camera, absence=absence)
    acceleration = np.asarray(acceleration, dtype=np.float64)
    if acceleration.ndim == 0:
        acceleration = np.array([acceleration, 0.], dtype=np.float64)
    if acceleration.shape != (2,):
        raise ValueError('Acceleration must be a scalar ax or [ax, ay]')
    initial = np.asarray(case['velocity'], dtype=np.float64)
    times = case['times']
    length = len(times)
    if max(max_speed(initial[k], acceleration[k], length) for k in range(2)) > SPEED_BOUND+1e-9:
        raise ValueError('Speed bound exceeded during the sequence; the search/metric bounds assume 2.5 px/frame')
    if np.any(acceleration != 0):
        start = case['positions'][0]  # times[0] == 0
        positions = (start[None]+times[:, None]*initial[None]
                     + .5*times[:, None].astype(np.float64)**2*acceleration[None]).astype(np.float32)
        frames, visibility = _measurement.render(case['template'], positions, case['cameras'],
                                                 interpolate=mode != 'nearest')
        case.update(positions=positions, frames=frames, visibility=visibility)
    last_context_time = float(case['context_times'][-1])
    case['initial_velocity'] = initial
    case['acceleration'] = acceleration
    case['velocity'] = initial+last_context_time*acceleration
    case['speed'] = float(case['velocity'][0])
    anchor = case['positions'][case['context_indices'][-1]]
    case['true_offsets'] = (case['positions'][case['target_indices']]-anchor[None]).astype(np.float32)
    elapsed = case['target_times']-last_context_time
    case['true_mean_velocity'] = case['true_offsets']/elapsed[:, None]
    case['direction_split'] = 'heldout_direction' if heldout_direction(case['velocity']) else 'seen_direction'
    return case


def kinetic_model_batch(cases, device='cpu'):
    return model_batch(cases, device)


__all__ = ['KineticTrainingData', 'make_kinetic_case', 'kinetic_model_batch', 'sample_acceleration',
           'max_speed', 'HOLDOUT', 'ROOT', 'MOTIONS', 'SPEED_BOUND']
