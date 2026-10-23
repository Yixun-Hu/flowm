"""Paired data for the continuous-motion mechanism and generalization study.

The released cameras have 200 integer actions. For forecasts through t=209 we
repeat the same episode's actions from its beginning *before* accumulating camera
position. This preserves integer actions, but the seam is not a newly sampled
episode from the original policy. Every method sees the identical camera path.

Only ``model_batch`` outputs should enter a predictive model. Templates, true
positions/velocities, rendering labels and visibility are evaluation metadata.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data/fractional_velocity'
_spec = importlib.util.spec_from_file_location(
    '_continuous_velocity_pilot_data', Path(__file__).resolve().parents[1] / 'data.py')
_pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pilot)
_measurement_spec = importlib.util.spec_from_file_location(
    '_fractional_velocity_measurement_data', ROOT / 'experiments/fractional_velocity/measurement.py')
_measurement = importlib.util.module_from_spec(_measurement_spec)
_measurement_spec.loader.exec_module(_measurement)
read_digits = _pilot.read_digits
render = _pilot.render
HOLDOUT = tuple(_pilot.HOLDOUT)
SPEEDS = tuple(_pilot.SPEEDS)
HELDOUT_DIRECTIONS_DEG = (45., 225.)
DIRECTION_HALF_WIDTH_DEG = 10.
TWO_D_VELOCITIES = (
    (.37, .37), (.63, .63), (1.23, 1.23),
    (-.37, -.37), (-.63, -.63), (-1.23, -1.23),
    (.63, -.63), (-.63, .63), (1.23, -.37), (-.37, 1.23),
    (1., 0.), (0., -1.),
)


@lru_cache(maxsize=1)
def camera_bank():
    """The complete action bank; split episodes 0:64 / 64:224 / 224:256."""
    return np.stack([
        torch.load(DATA / f'released/dynamic_validation_200/0/{i:04d}.pt',
                   map_location='cpu', weights_only=False)['actions'].numpy()
        for i in range(256)
    ]).astype(np.float32)


def extended_actions(episode, length):
    """Repeat actions, without resetting accumulated camera coordinates."""
    bank = camera_bank()
    if not 0 <= int(episode) < len(bank):
        raise ValueError(f'Camera episode outside released bank: {episode}')
    original = bank[int(episode)]
    return np.tile(original, (int(np.ceil(length / len(original))), 1))[:length].copy()


def heldout_direction(velocity):
    """True in either deliberately excluded angular wedge (zero is allowed)."""
    velocity = np.asarray(velocity)
    angle = np.degrees(np.arctan2(velocity[..., 1], velocity[..., 0])) % 360
    distances = np.stack([
        np.abs((angle - center + 180) % 360 - 180)
        for center in HELDOUT_DIRECTIONS_DEG
    ], axis=-1)
    return ((distances.min(axis=-1) < DIRECTION_HALF_WIDTH_DEG)
            & (np.linalg.norm(velocity, axis=-1) > 1e-8))


def _sample_velocities(rng, count, motion):
    if motion == 'horizontal':
        x = rng.uniform(0, 2, count)
        for i in range(count):
            while min(abs(x[i] - v) for v in HOLDOUT) < .03:
                x[i] = rng.uniform(0, 2)
        integer = rng.random(count) < .15
        x[integer] = rng.integers(0, 3, integer.sum())
        return np.column_stack([x, np.zeros(count)]).astype(np.float32)
    if motion != '2d':
        raise ValueError(f'Unknown motion setting: {motion}')
    result = []
    while len(result) < count:
        candidate = (rng.integers(-2, 3, 2) if rng.random() < .15
                     else rng.uniform(-2, 2, 2))
        if not heldout_direction(candidate):
            result.append(candidate)
    return np.asarray(result, dtype=np.float32)


def _context_indices(rng, context='regular', absence=0):
    if context == 'regular':
        return np.arange(50)
    if context == 'irregular':
        # Same elapsed window and final observation as regular evaluation.
        # Twenty-five samples, nonuniform gaps of one to three frames.
        indices = np.arange(0, 50, 2) + rng.integers(0, 2, 25)
        indices[0], indices[-1] = 0, 49
        return indices
    if context == 'gap':
        if not 1 <= absence <= 40:
            raise ValueError('Observation blackout must remove 1 to 40 final context frames')
        return np.arange(50 - absence)
    raise ValueError(f'Unknown context setting: {context}')


class TrainingData:
    """TrainingData-compatible sampler, with independent digit/camera splits.

    ``seed`` controls sample order. Use one seed per training replicate, and the
    same seed across model variants for paired optimization data. Rendering mode
    selection consumes the same RNG draw in all modes, so changing rendering does
    not change digits, positions, velocities or cameras for a fixed seed.
    Validation velocities obey the training holdouts; heldout angles and speeds
    are reserved for the test protocols.
    """
    def __init__(self, device, seed=1234, render_mode='bilinear',
                 motion='horizontal', train_size=55000, irregular_context=False):
        if render_mode not in ('bilinear', 'nearest', 'mixed'):
            raise ValueError(f'Unknown rendering mode: {render_mode}')
        if motion not in ('horizontal', '2d'):
            raise ValueError(f'Unknown motion setting: {motion}')
        if not 1 <= train_size <= 55000:
            raise ValueError('train_size must be in 1:55000 (validation remains disjoint)')
        images, _ = read_digits(True)
        self.images = torch.from_numpy(images).to(device).float() / 255
        self.actions = torch.from_numpy(camera_bank()).to(device)
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.render_mode = render_mode
        self.motion = motion
        self.train_size = int(train_size)
        self.irregular_context = irregular_context

    def batch(self, batch_size=8, validation=False):
        rng = self.rng
        indices = (rng.integers(55000, 60000, batch_size) if validation
                   else rng.integers(0, self.train_size, batch_size))
        episodes = (rng.integers(224, 256, batch_size) if validation
                    else rng.integers(64, 224, batch_size))
        velocities = _sample_velocities(rng, batch_size, self.motion)
        starts = rng.integers(0, 50, (batch_size, 2))
        nearest = rng.random(batch_size) < .5  # Draw regardless of rendering mode.
        if self.render_mode != 'mixed':
            nearest[:] = self.render_mode == 'nearest'
        context_indices = _context_indices(
            rng, 'irregular' if self.irregular_context else 'regular')
        targets = np.array([50, 54, 59, 69, 79, 89])
        time_indices = np.concatenate([context_indices, targets])
        times = torch.as_tensor(time_indices, device=self.device,
                                dtype=torch.float32)[None].expand(batch_size, -1)
        velocity = torch.as_tensor(velocities, device=self.device)
        positions = (torch.as_tensor(starts, device=self.device, dtype=torch.float32)[:, None]
                     + times[:, :, None] * velocity[:, None])
        cameras = self.actions[episodes].cumsum(1)[:, time_indices] - 16
        digits = self.images[indices]
        frames = render(digits, positions, cameras,
                        mode='nearest' if nearest.all() else 'bilinear')
        if nearest.any() and not nearest.all():
            selected = torch.as_tensor(nearest, device=self.device)
            frames[selected] = render(digits[selected], positions[selected],
                                      cameras[selected], mode='nearest')
        n_context = len(context_indices)
        return dict(
            context=frames[:, :n_context], cameras=cameras[:, :n_context].float(),
            context_times=times[:, :n_context],
            target_cameras=cameras[:, n_context:].float(),
            target_times=times[:, n_context:], target=frames[:, n_context:],
            true_velocity=velocity,
            # Metadata below must not be forwarded to a neural model.
            digit_indices=indices, camera_episodes=episodes,
            render_modes=np.where(nearest, 'nearest', 'bilinear'),
        )


@lru_cache(maxsize=1)
def test_digits():
    images, labels = read_digits(False)
    return images.astype(np.float32) / 255, labels


def _shape(seed):
    """Deterministic unfamiliar rigid appearance, independent of MNIST assets."""
    rng = np.random.default_rng(500000 + int(seed))
    yy, xx = np.indices((28, 28))
    result = np.zeros((28, 28), np.float32)
    for _ in range(3):
        x, y = rng.uniform(6, 22, 2)
        rx, ry = rng.uniform(3, 7, 2)
        ellipse = ((xx - x) / rx)**2 + ((yy - y) / ry)**2 <= 1
        result = np.maximum(result, ellipse * rng.uniform(.4, 1))
    return result.astype(np.float32)


def make_case(seed, velocity, mode='bilinear', horizon=160, context='regular',
              appearance='mnist', camera='released', absence=0):
    """Create a paired case; trajectories always use absolute, unwrapped positions.

    Frames/cameras/times contain the complete regular sequence. ``context_indices``
    indicates what is actually observed, allowing irregular timestamps and a
    final-context observation blackout without changing any test scene.
    The blackout is not a geometric out-of-view control and must be labeled so.
    """
    if not 0 <= seed < 64:
        raise ValueError('Test seeds 0:64 reserve camera episodes disjoint from training')
    if mode not in ('bilinear', 'nearest', 'half_offset'):
        raise ValueError(f'Unknown rendering mode: {mode}')
    if horizon < 1:
        raise ValueError('Prediction horizon must be positive')
    velocity = np.asarray(velocity, dtype=np.float64)
    if velocity.ndim == 0:
        velocity = np.array([velocity, 0], dtype=np.float64)
    if velocity.shape != (2,):
        raise ValueError('Velocity must be a scalar vx or [vx,vy]')
    rng = np.random.default_rng(seed)
    images, labels = test_digits()
    index = int(rng.integers(len(images)))
    start = rng.integers(0, 50, 2).astype(np.float32)
    sprite = images[index] if appearance == 'mnist' else _shape(seed)
    if appearance not in ('mnist', 'shapes'):
        raise ValueError(f'Unknown appearance: {appearance}')
    template = np.zeros((50, 50), np.float32)
    template[:28, :28] = sprite
    length = 50 + int(horizon)
    actions = extended_actions(seed, length)
    if camera == 'stationary':
        actions[:] = 0
    elif camera != 'released':
        raise ValueError(f'Unknown camera setting: {camera}')
    cameras = actions.cumsum(0) - 16
    times = np.arange(length, dtype=np.float32)
    offset = np.array([.5, .5], np.float32) if mode == 'half_offset' else 0.
    positions = start + times[:, None] * velocity[None] + offset
    # Reuse the original numpy renderer so visibility has exactly the established
    # definition: observed squared energy / total current rendered squared energy.
    frames, visibility = _measurement.render(
        template, positions, cameras, interpolate=mode != 'nearest')
    # A separate stream makes irregular subsampling independent of velocity/mode.
    context_indices = _context_indices(np.random.default_rng(800000 + seed), context, absence)
    target_indices = np.arange(50, length)
    direction_split = ('heldout_direction' if heldout_direction(velocity)
                       else 'seen_direction')
    scenario = f'{appearance}_{context}_{camera}'
    if context == 'gap':
        scenario += f'_{absence}'
    return dict(
        seed=int(seed), speed=float(velocity[0]), velocity=velocity,
        mode=mode, digit_index=index if appearance == 'mnist' else -1,
        label=int(labels[index]) if appearance == 'mnist' else -1,
        appearance=appearance, scenario=scenario, direction_split=direction_split,
        template=template, positions=positions, actions=actions, cameras=cameras,
        frames=frames, visibility=visibility, times=times,
        context_indices=context_indices, target_indices=target_indices,
        context_times=times[context_indices], target_times=times[target_indices],
        horizon=int(horizon), context=context, absence=int(absence),
        camera_extension='same-episode actions replayed after t=199 before cumsum',
    )


def model_batch(cases, device='cpu'):
    """Build only the five allowed model inputs; shapes must agree across cases."""
    if not cases:
        raise ValueError('Cannot build an empty model batch')
    lengths = {(len(c['context_indices']), len(c['target_indices'])) for c in cases}
    if len(lengths) != 1:
        raise ValueError('Batch cases must have equal context and target lengths')
    arrays = dict(
        context=[c['frames'][c['context_indices']] for c in cases],
        cameras=[c['cameras'][c['context_indices']] for c in cases],
        context_times=[c['context_times'] for c in cases],
        target_cameras=[c['cameras'][c['target_indices']] for c in cases],
        target_times=[c['target_times'] for c in cases],
    )
    return {key: torch.as_tensor(np.stack(value), device=device, dtype=torch.float32)
            for key, value in arrays.items()}
