"""CPU reconstruction of FloWM's released dynamic_po MNIST World.

Conventions and evidence
------------------------
* Arrays use [y, x]; metadata vectors use (x, y). The world is a 50x50
  torus, sprites are 28x28 MNIST TRAIN images anchored at their top left,
  and overlapping intensities are summed before clipping to [0, 1].
  ``verify_generator.py`` tests these conventions against decoded releases,
  including alternative anchoring, overlap, and window conventions.
* Integer trajectories are (p0 + v*t) % 50. Camera actions apply BEFORE
  frame t, with action[0] = (0, 0), centre = cumsum(actions) % 50, and
  window top left = centre - 16. The verifier checks both metadata identities.
* The recovered release camera law explores uniformly chosen unseen torus
  pixels, taking the shortest wrapped step clipped to [-10,10] per axis.
  Once the 32x32 window has covered the world, actions are iid uniform integers
  in [-10,10]^2. Actions 0 and 50 are zero. ``action_law='release'`` is the
  default; ``'uniform'`` retains iid actions after action 0. Labels are iid
  uniform classes, followed by a uniform TRAIN image within each class.
  Original seeds and RNG call order are not claimed to have been recovered.
* Local IDX headers establish 60,000 training images. Matching label-restricted
  TRAIN candidates and re-encoding provides the evidence for sprite identity.
* A released stream's FFmpeg header and x264 SEI specify RGB input encoded as
  H.264 High/yuv420p, CRF 10, 8 fps, medium preset, one encoder thread, using
  imageio-ffmpeg's FFmpeg 7.0.2 / x264 core 164 r3191. Multiplication by 255
  followed by uint8 truncation is an assumption compatible with integer-release
  bit-exact recovery, not uniquely identifiable from it. Subpixel modes instead
  use numpy.rint (round half to even). RGB channels before encoding are identical.
* ``datasets/video/base_video.py:load_video`` uses decord CPU get_batch,
  float32, TCHW, /255. We preserve all decoded RGB channels for verification;
  the unused grayscale helper in mnist_world.py is NOT the pipeline decoder.
* Subpixel modes use p(t) = (p0 + v0*t + .5*a*t**2) mod 50, with metadata
  strictly in [0,50), including floating remainder edge cases. p0 is integer
  by default; p0_continuous=True samples U[0,50)^2. Nearest rendering uses
  numpy.rint (half to even), but metadata retains the real positions. Bilinear
  translation distributes each sprite BEFORE overlap clipping.
* Fractional v0 is uniform in area in the speed_max disc. Accelerated 'plan'
  shares this same v0 under the same seed: choose a=0 with probability .3;
  otherwise propose a uniform direction and log-uniform magnitude [2e-4,2e-2]
  px/frame^2, rejecting only a until the speed bound holds. The branch is kept
  fixed during rejection, preserving the .3 atom. If speed_max=0, a=0 always.
  Both endpoint speeds bound the entire interval by convexity. bound_horizon
  (default 200 frames, t=0..199) is independent of the rendered n_frames;
  no bound is promised outside that horizon.
* Accelerated 'disc_endpoints' retains joint rejection of uniform velocity and
  acceleration discs, with default a_max=2*speed_max/(bound_horizon-1). For
  horizon>1 this gives acceptance 1/4 and iid uniform-disc endpoint velocities;
  v0 is consequently not shared with fractional mode after a rejection.
  A custom acceleration_max changes that law. Sampler settings and pooled
  acceptance counts/rates live in generation.json, never in episode tensors.

Local verification on validation episodes 0-199 (70 frames) and long-validation
episodes 0-49 (200 frames) recovered all 24,000 decoded frames bit-exactly after
re-encoding. Every episode's raw-render MSE was <=5e-5; alternative crop,
anchoring, maximum, and overwrite conventions all scored worse. The report at
outputs/flowm_bench/generator_verification.json records the individual evidence.

No downloads, GPU access, or optional framework imports are needed. Run with
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 and the flowm environment's Python.
"""

from __future__ import annotations

import gzip
import json
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import numpy.typing as npt
import torch

WORLD_SIZE = 50
WINDOW_SIZE = 32
SPRITE_SIZE = 28
ROOT = Path(__file__).resolve().parents[2]
MNIST_DIR = ROOT / "data/fractional_velocity"
FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
Motion = Literal["integer", "fractional", "accelerated"]
Renderer = Literal["nearest", "bilinear"]
ActionLaw = Literal["release", "uniform"]
AccelerationLaw = Literal["plan", "disc_endpoints"]
_SY, _SX = np.indices((SPRITE_SIZE, SPRITE_SIZE))
_PIXELS = np.arange(SPRITE_SIZE**2)


def _read_idx(path: Path, expected_shape: tuple[int, ...]) -> npt.NDArray[np.uint8]:
    with gzip.open(path, "rb") as stream:
        header = stream.read(4)
        if header != bytes((0, 0, 8, len(expected_shape))):
            raise ValueError(f"Invalid uint8 IDX header in {path}: {header!r}")
        shape = struct.unpack(">" + "I" * len(expected_shape), stream.read(4 * len(expected_shape)))
        payload = stream.read()
    if shape != expected_shape or len(payload) != int(np.prod(shape)):
        raise ValueError(f"Invalid IDX dimensions/payload in {path}: {shape}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(shape)


@lru_cache(maxsize=4)
def load_mnist_train(directory: str | Path = MNIST_DIR) -> npt.NDArray[np.uint8]:
    """Read and cache the local, read-only uint8 [60000,28,28] TRAIN images."""
    return _read_idx(Path(directory) / "train-images-idx3-ubyte.gz", (60000, 28, 28))


@lru_cache(maxsize=4)
def load_mnist_train_labels(directory: str | Path = MNIST_DIR) -> npt.NDArray[np.uint8]:
    """Read the corresponding local TRAIN labels, without downloading."""
    return _read_idx(Path(directory) / "train-labels-idx1-ubyte.gz", (60000,))


@lru_cache(maxsize=1)
def _class_indices() -> tuple[IntArray, ...]:
    return tuple(np.flatnonzero(load_mnist_train_labels() == label) for label in range(10))


def sample_digits(rng: np.random.Generator, n_digits: int) -> tuple[IntArray, IntArray]:
    """Uniform classes, then uniform TRAIN indices conditional on each class."""
    labels = rng.integers(0, 10, n_digits, dtype=np.int64)
    indices = np.array([rng.choice(_class_indices()[label]) for label in labels], dtype=np.int64)
    return labels, indices


def wrap_positions(positions: np.ndarray) -> np.ndarray:
    """Modulo 50, mapping a rounded floating remainder of exactly 50 back to 0."""
    wrapped = np.remainder(positions, WORLD_SIZE)
    return np.where(wrapped == WORLD_SIZE, 0, wrapped)


def sample_actions(
    rng: np.random.Generator, n_frames: int, action_range: int = 10,
    context_frames: int = 50, action_law: ActionLaw = "release",
) -> IntArray:
    """Port of the review's recovered release_actions.py (including tie +25)."""
    if action_law == "uniform":
        actions = rng.integers(-action_range, action_range + 1, (n_frames, 2), dtype=np.int64)
        actions[0] = 0
        return actions
    if action_law != "release":
        raise ValueError(f"Unknown action law: {action_law}")
    seen = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=bool)
    centre = np.zeros(2, dtype=np.int64)
    actions = np.zeros((n_frames, 2), dtype=np.int64)

    def look() -> None:
        seen[np.ix_((centre[1] - 16 + np.arange(32)) % 50,
                    (centre[0] - 16 + np.arange(32)) % 50)] = True

    look()
    for t in range(1, n_frames):
        if t == context_frames:
            continue
        if not seen.all():
            ys, xs = np.nonzero(~seen)
            j = rng.integers(len(xs))
            delta = (np.array([xs[j], ys[j]]) - centre) % WORLD_SIZE
            delta[delta > WORLD_SIZE // 2] -= WORLD_SIZE
            actions[t] = np.clip(delta, -action_range, action_range)
        else:
            actions[t] = rng.integers(-action_range, action_range + 1, size=2)
        centre = (centre + actions[t]) % WORLD_SIZE
        look()
    return actions


def as_numpy(value: Any) -> np.ndarray:
    """Convert CPU tensors or array-like metadata without changing dtype."""
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("This generator only accepts CPU tensors")
        return value.detach().numpy()
    return np.asarray(value)


def load_metadata(path: str | Path) -> dict[str, Any]:
    """Load released metadata with explicit CPU placement and restricted loading."""
    return torch.load(path, map_location="cpu", weights_only=True)


def _validate_sprites(sprites: np.ndarray, positions: np.ndarray) -> FloatArray:
    sprites = np.asarray(sprites, dtype=np.float32)
    if sprites.ndim != 3 or sprites.shape[1:] != (28, 28):
        raise ValueError("sprites must have shape [N,28,28]")
    if positions.shape != (len(sprites), 2) or not np.isfinite(positions).all():
        raise ValueError("positions must be finite [N,2] (x,y)")
    if not np.isfinite(sprites).all() or np.any((sprites < 0) | (sprites > 1)):
        raise ValueError("sprites must be finite intensities in [0,1]")
    return sprites


def _add_sprites(world: FloatArray, sprites: FloatArray, positions: IntArray) -> None:
    x = (positions[:, 0, None, None] + _SX) % WORLD_SIZE
    y = (positions[:, 1, None, None] + _SY) % WORLD_SIZE
    np.add.at(world.ravel(), (y * WORLD_SIZE + x).ravel(), sprites.ravel())


def render_world(sprites: np.ndarray, positions: np.ndarray) -> FloatArray:
    """Vectorised top-left torus placement; additive compositing then clipping."""
    positions = np.asarray(positions)
    sprites = _validate_sprites(sprites, positions)
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("render_world requires integer positions; use render_world_subpixel")
    world = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.float32)
    _add_sprites(world, sprites, positions)
    return np.clip(world, 0, 1, out=world)


def render_world_subpixel(
    sprites: np.ndarray, positions: np.ndarray, mode: Renderer = "bilinear"
) -> FloatArray:
    """Place real-valued positions using periodic nearest or bilinear translation.

    Interpolate each sprite's unclipped contribution, sum all contributions, and
    only then clip. This prevents overlap saturation from changing interpolation.
    """
    positions = np.asarray(positions, dtype=np.float64)
    sprites = _validate_sprites(sprites, positions)
    if mode == "nearest":
        return render_world(sprites, np.rint(positions).astype(np.int64))
    if mode != "bilinear":
        raise ValueError(f"Unknown renderer: {mode}")
    base = np.floor(positions).astype(np.int64)
    frac = positions - base
    world = np.zeros((WORLD_SIZE, WORLD_SIZE), dtype=np.float32)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        weight = (frac[:, 0] if dx else 1 - frac[:, 0]) * (
            frac[:, 1] if dy else 1 - frac[:, 1])
        _add_sprites(world, sprites * weight.astype(np.float32)[:, None, None], base + (dx, dy))
    return np.clip(world, 0, 1, out=world)


def crop_window(world: np.ndarray, camera_centre: np.ndarray) -> np.ndarray:
    """Return the wrapped 32x32 view with top left at (centre_x-16, centre_y-16)."""
    centre = np.asarray(camera_centre)
    if world.shape != (50, 50) or centre.shape != (2,):
        raise ValueError("Expected world [50,50] and camera_centre [2]")
    if not np.issubdtype(centre.dtype, np.integer):
        raise ValueError("Camera centres must be integer pixel coordinates")
    x = (centre[0] - 16 + np.arange(32)) % 50
    y = (centre[1] - 16 + np.arange(32)) % 50
    return world[np.ix_(y, x)]


def camera_centres(metadata: Mapping[str, Any]) -> np.ndarray:
    """Join stored context and target camera centres (do not recompute them)."""
    return np.concatenate((as_numpy(metadata["camera_centers_input"]),
                           as_numpy(metadata["camera_centers_target"])))


def render_metadata(metadata: Mapping[str, Any], indices: np.ndarray) -> FloatArray:
    """Render every stored position/centre; no extrapolation or frame subsampling."""
    sprites = load_mnist_train()[indices].astype(np.float32) / 255.0
    positions = as_numpy(metadata["digit_positions_over_time"])
    centres = camera_centres(metadata)
    if positions.shape != (len(indices), len(centres), 2):
        raise ValueError("Positions and camera sequence lengths disagree")
    renderer = metadata.get("renderer", "nearest")
    frames = np.empty((len(centres), 32, 32), dtype=np.float32)
    for t, centre in enumerate(centres):
        world = (render_world(sprites, positions[:, t]) if np.issubdtype(positions.dtype, np.integer)
                 else render_world_subpixel(sprites, positions[:, t], renderer))
        frames[t] = crop_window(world, centre)
    return frames


def decode_video_uint8(path: str | Path) -> npt.NDArray[np.uint8]:
    """Decode on CPU with decord get_batch exactly as BaseVideoDataset does (THWC)."""
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return reader.get_batch(list(range(len(reader)))).asnumpy()


def decode_video(path: str | Path) -> FloatArray:
    """Pipeline-normalised float32 [T,3,H,W]; keep all RGB channels."""
    return decode_video_uint8(path).transpose(0, 3, 1, 2).astype(np.float32) / 255.0


def _gray_uint8(frames: np.ndarray) -> npt.NDArray[np.uint8]:
    frames = as_numpy(frames)
    if frames.ndim == 4:
        if frames.shape[1:] == (3, 32, 32):
            frames = frames[:, 0]
        elif frames.shape[1:] == (32, 32, 3):
            frames = frames[..., 0]
        else:
            raise ValueError("Expected TCHW/THWC RGB or THW grayscale frames")
    if frames.ndim != 3 or frames.shape[1:] != (32, 32):
        raise ValueError("Expected frames with spatial size 32x32")
    if frames.dtype == np.uint8:
        return frames
    if not np.isfinite(frames).all() or np.any((frames < 0) | (frames > 1)):
        raise ValueError("Float frames must be in [0,1]")
    return np.rint(frames * 255).astype(np.uint8)


@dataclass
class DigitMatches:
    """Label-restricted nearest indices and MAEs in [0,1], with evidence counts.

    Full 28x28 visibility can be absent in a 70-frame clip. In that case the
    explicit fallback pools individually visible, non-overlapped sprite pixels
    over time; it never treats clipped pixels as observations of the sprite.
    """

    indices: IntArray
    best_mae: npt.NDArray[np.float64]
    second_best_mae: npt.NDArray[np.float64]
    full_visible_frames: IntArray
    sample_counts: IntArray
    covered_pixels: IntArray
    methods: list[str]

    @property
    def margins(self) -> npt.NDArray[np.float64]:
        return self.second_best_mae - self.best_mae

    @property
    def mae_ratios(self) -> npt.NDArray[np.float64]:
        """Runner-up/best MAE; infinity for a unique exact match, 1 for a tie."""
        return np.divide(self.second_best_mae, self.best_mae,
                         out=np.where(self.second_best_mae > 0, np.inf, 1.0),
                         where=self.best_mae > 0)


def _mae_scores(candidates: np.ndarray, observations: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Exact pooled MAE via per-sprite-pixel uint8 histograms, without huge tensors."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return np.full(len(candidates), np.inf)
    histogram = np.bincount(cols * 256 + observations[rows, cols], minlength=784 * 256)
    histogram = histogram.reshape(784, 256)
    values = np.arange(256)
    counts = np.cumsum(histogram, axis=1)
    sums = np.cumsum(histogram * values, axis=1)
    costs = (values * counts - sums + sums[:, -1, None] - sums
             - values * (counts[:, -1, None] - counts))
    return costs[_PIXELS[None, :], candidates.reshape(-1, 784)].sum(axis=1) / (len(rows) * 255.0)


def match_digit_indices(episode_pt: Mapping[str, Any] | str | Path, frames: np.ndarray) -> DigitMatches:
    """Recover TRAIN indices using label-restricted MAE on clean visible pixels.

    Start conservatively by excluding every other digit's entire 28x28 box.
    Prefer fully visible sprite frames when they cover >=128 sprite pixels;
    otherwise pool partial views. Refine using recovered nonzero sprite support
    to expose clean pixels inside overlapping boxes. All candidates of the given
    label are scored, and the best/runner-up margin is returned. Indices with
    zero margin are ambiguous, not a claim of unique source identity.
    """
    metadata = load_metadata(episode_pt) if isinstance(episode_pt, (str, Path)) else episode_pt
    gray = _gray_uint8(frames)
    positions = as_numpy(metadata["digit_positions_over_time"])
    centres = camera_centres(metadata)
    labels = as_numpy(metadata["digit_labels"])
    if not np.issubdtype(positions.dtype, np.integer):
        raise ValueError("Sprite recovery is for released integer-motion metadata")
    if positions.shape != (len(labels), len(gray), 2) or len(centres) != len(gray):
        raise ValueError("Metadata/frame dimensions disagree")
    images = load_mnist_train()
    train_labels = load_mnist_train_labels()
    count = len(labels)
    candidate_indices = [np.flatnonzero(train_labels == label) for label in labels]
    indices = np.zeros(count, dtype=np.int64)
    best = np.full(count, np.inf)
    second = np.full(count, np.inf)
    full_counts = np.zeros(count, dtype=np.int64)
    sample_counts = np.zeros(count, dtype=np.int64)
    coverage = np.zeros(count, dtype=np.int64)
    methods = [""] * count
    coordinates = []
    for i in range(count):
        wx = (positions[i, :, 0, None] + _SX.ravel()) % 50
        wy = (positions[i, :, 1, None] + _SY.ravel()) % 50
        cx = (wx - centres[:, 0, None] + 16) % 50
        cy = (wy - centres[:, 1, None] + 16) % 50
        visible = (cx < 32) & (cy < 32)
        full = visible.all(axis=1)
        observations = gray[np.arange(len(gray))[:, None], cy % 32, cx % 32]
        other_coordinates = [((wx - positions[j, :, 0, None]) % 50,
                              (wy - positions[j, :, 1, None]) % 50)
                             for j in range(count)]
        coordinates.append((visible, full, observations, other_coordinates))
        full_counts[i] = full.sum()
    # Three passes allow mutually overlapping digits to reveal one another.
    for iteration in range(3):
        previous = indices.copy()
        for i, (visible, full, observations, others) in enumerate(coordinates):
            clean = visible.copy()
            for j, (x, y) in enumerate(others):
                if i == j:
                    continue
                overlap = (x < 28) & (y < 28)
                if iteration:
                    overlap &= images[indices[j], y % 28, x % 28] > 0
                clean &= ~overlap
            full_clean = clean & full[:, None]
            use_full = np.count_nonzero(full_clean.any(axis=0)) >= 128
            mask = full_clean if use_full else clean
            scores = _mae_scores(images[candidate_indices[i]], observations, mask)
            order = np.argsort(scores, kind="stable")[:2]
            indices[i] = candidate_indices[i][order[0]]
            best[i], second[i] = scores[order]
            sample_counts[i] = mask.sum()
            coverage[i] = np.count_nonzero(mask.any(axis=0))
            methods[i] = ("full_visible" if use_full else "pooled_partial") + (
                "_support_mask" if iteration else "_box_mask")
        if iteration and np.array_equal(indices, previous):
            break
    if not np.isfinite(best).all():
        raise ValueError("No clean observations for one or more digits; identity is unidentifiable")
    return DigitMatches(indices, best, second, full_counts, sample_counts, coverage, methods)


@dataclass
class Episode:
    """Lossless floating frames and CPU NumPy metadata for one generated episode."""

    frames: FloatArray
    actions: IntArray
    digit_labels: IntArray
    digit_init_positions: np.ndarray
    digit_velocities: np.ndarray
    digit_accelerations: np.ndarray
    digit_positions_over_time: np.ndarray
    camera_centers_input: IntArray
    camera_centers_target: IntArray
    digit_train_indices: IntArray
    renderer: Renderer
    motion: Motion
    settings: dict[str, Any]
    acceleration_proposals: int = 0
    acceleration_acceptances: int = 0

    @property
    def acceleration_acceptance_rate(self) -> float | None:
        return (self.acceleration_acceptances / self.acceleration_proposals
                if self.acceleration_proposals else None)

    @property
    def camera_centers(self) -> IntArray:
        return np.concatenate((self.camera_centers_input, self.camera_centers_target))

    def metadata(self) -> dict[str, Any]:
        """Collatable tensors plus the renderer needed to reconstruct real positions.

        Sampler provenance belongs in generation.json. In particular, never
        attach an inapplicable None acceptance rate to a DataLoader sample.
        """
        result = {name: torch.from_numpy(getattr(self, name).copy()) for name in (
            "actions", "digit_labels", "digit_init_positions", "digit_velocities",
            "digit_accelerations", "digit_positions_over_time", "camera_centers_input",
            "camera_centers_target", "digit_train_indices")}
        result["renderer"] = self.renderer
        return result

    def provenance(self) -> dict[str, Any]:
        result = dict(self.settings, motion=self.motion, renderer=self.renderer,
                      label_law="uniform_class_then_uniform_train_index",
                      quantization="truncate" if self.motion == "integer" else "rint_half_to_even")
        if self.acceleration_proposals:
            result.update(acceleration_proposals=self.acceleration_proposals,
                          acceleration_acceptances=self.acceleration_acceptances,
                          acceleration_acceptance_rate=self.acceleration_acceptance_rate)
        return result


def _disc(rng: np.random.Generator, radius: float, count: int) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi, count)
    length = radius * np.sqrt(rng.uniform(0, 1, count))
    return length[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))


def sample_accelerations(
    rng: np.random.Generator, velocity: np.ndarray, speed_max: float,
    bound_horizon: int, law: AccelerationLaw, acceleration_max: float,
    max_proposals: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fixed-v0 conditional 'plan', or joint disc rejection; see module law notes."""
    velocity = velocity.copy()
    acceleration = np.zeros_like(velocity)
    horizon = bound_horizon - 1
    proposals = 0
    for i in range(len(velocity)):
        zero = law == "plan" and (speed_max == 0 or rng.random() < .3)
        while True:
            if proposals >= max_proposals:
                raise RuntimeError(f"Acceleration rejection sampler exhausted {max_proposals} proposals")
            proposals += 1
            if law == "plan":
                if zero:
                    a = np.zeros(2)
                else:
                    magnitude = np.exp(rng.uniform(np.log(2e-4), np.log(2e-2)))
                    angle = rng.uniform(0, 2 * np.pi)
                    a = magnitude * np.array([np.cos(angle), np.sin(angle)])
            else:
                a = _disc(rng, acceleration_max, 1)[0]
            if np.linalg.norm(velocity[i] + a * horizon) <= speed_max:
                acceleration[i] = a
                break
            if law == "disc_endpoints":
                velocity[i] = _disc(rng, speed_max, 1)[0]
    return velocity, acceleration, proposals


def generate_episode(
    rng: np.random.Generator,
    n_frames: int,
    motion: Motion = "integer",
    renderer: Renderer = "nearest",
    n_digits: int = 5,
    action_range: int = 10,
    speed_max: float = 2.0,
    *,
    acceleration_max: float | None = None,
    bound_horizon: int = 200,
    acceleration_law: AccelerationLaw = "plan",
    action_law: ActionLaw = "release",
    p0_continuous: bool = False,
    context_frames: int = 50,
    max_proposals: int = 1_000_000,
) -> Episode:
    """Sample an episode with the conventions in this module's docstring.

    Integer mode always samples each velocity component in {-1,0,1}; speed_max
    applies to the continuous modes. Motion is sampled before camera actions so
    a fixed seed gives the same p0/v0/a for different clip lengths. Fractional
    and accelerated plan modes also share p0/v0. Only disc_endpoints accepts
    acceleration_max; plan uses its specified mixture and magnitude interval.
    """
    for name, value, minimum in (("n_frames", n_frames, 1), ("n_digits", n_digits, 1),
                                 ("action_range", action_range, 0),
                                 ("context_frames", context_frames, 0),
                                 ("bound_horizon", bound_horizon, 1),
                                 ("max_proposals", max_proposals, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if motion not in ("integer", "fractional", "accelerated") or renderer not in ("nearest", "bilinear"):
        raise ValueError("Unknown motion or renderer")
    if acceleration_law not in ("plan", "disc_endpoints") or action_law not in ("release", "uniform"):
        raise ValueError("Unknown acceleration or action law")
    if p0_continuous and motion == "integer":
        raise ValueError("p0_continuous is only supported for subpixel modes")
    if acceleration_max is not None and acceleration_law != "disc_endpoints":
        raise ValueError("acceleration_max is only supported by disc_endpoints")
    if not np.isfinite(speed_max) or speed_max < 0:
        raise ValueError("speed_max must be finite and nonnegative")
    horizon = bound_horizon - 1
    if acceleration_max is None:
        acceleration_max = 2 * speed_max / max(horizon, 1)
    if not np.isfinite(acceleration_max) or acceleration_max < 0:
        raise ValueError("acceleration_max must be finite and nonnegative")
    if (motion == "accelerated" and acceleration_law == "disc_endpoints"
            and speed_max == 0 and acceleration_max > 0 and horizon):
        raise ValueError("Positive acceleration has zero acceptance when speed_max=0 and horizon>1")
    labels, indices = sample_digits(rng, n_digits)
    initial = (rng.uniform(0, 50, size=(n_digits, 2)) if p0_continuous
               else rng.integers(0, 50, size=(n_digits, 2), dtype=np.int64))
    initial = wrap_positions(initial)
    proposals = 0
    if motion == "integer":
        velocity = rng.integers(-1, 2, size=(n_digits, 2), dtype=np.int64)
        acceleration = np.zeros_like(velocity)
        positions = wrap_positions(initial[:, None] + velocity[:, None] * np.arange(n_frames)[None, :, None])
    else:
        velocity = _disc(rng, speed_max, n_digits)
        acceleration = np.zeros_like(velocity)
        if motion == "accelerated":
            velocity, acceleration, proposals = sample_accelerations(
                rng, velocity, speed_max, bound_horizon, acceleration_law,
                acceleration_max, max_proposals)
        time = np.arange(n_frames, dtype=np.float64)[None, :, None]
        positions = wrap_positions(initial[:, None] + velocity[:, None] * time + .5 * acceleration[:, None] * time**2)
    actions = sample_actions(rng, n_frames, action_range, context_frames, action_law)
    centres = np.cumsum(actions, axis=0) % 50
    split = min(context_frames, n_frames)
    settings = dict(n_frames=n_frames, n_digits=n_digits, action_range=action_range,
                    action_law=action_law, speed_max=speed_max, bound_horizon=bound_horizon,
                    acceleration_law=acceleration_law, p0_continuous=p0_continuous,
                    context_frames=context_frames)
    if acceleration_law == "disc_endpoints":
        settings["acceleration_max"] = acceleration_max
    episode = Episode(np.empty((n_frames, 32, 32), dtype=np.float32), actions, labels, initial,
                      velocity, acceleration, positions, centres[:split], centres[split:], indices,
                      renderer, motion, settings, proposals, n_digits if proposals else 0)
    episode.frames = render_metadata(episode.metadata(), indices)
    return episode


def frames_to_uint8(
    frames: np.ndarray, *, quantization: Literal["truncate", "rint"] = "truncate",
) -> npt.NDArray[np.uint8]:
    """Truncation is a release assumption; subpixel exports use half-to-even rint."""
    pixels = np.clip(frames, 0, 1) * 255
    if quantization == "rint":
        pixels = np.rint(pixels)
    elif quantization != "truncate":
        raise ValueError(f"Unknown quantization: {quantization}")
    return pixels.astype(np.uint8)


def encode_video(
    path: str | Path, frames: np.ndarray, *, quantization: Literal["truncate", "rint"] = "truncate",
) -> None:
    """Write RGB replicas using the released libx264/CRF10/8fps/yuv420p settings."""
    import imageio_ffmpeg

    pixels = frames_to_uint8(frames, quantization=quantization)
    if pixels.ndim != 3 or pixels.shape[1:] != (32, 32) or not len(pixels):
        raise ValueError("Expected nonempty [T,32,32] frames")
    writer = imageio_ffmpeg.write_frames(
        str(path), (32, 32), fps=8, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=None, macro_block_size=16,
        output_params=["-crf", "10", "-preset", "medium", "-threads", "1"],
        ffmpeg_log_level="error")
    writer.send(None)
    try:
        for frame in pixels:
            writer.send(np.repeat(frame[..., None], 3, axis=2))
    finally:
        writer.close()


def _write_episode_files(
    directory: str | Path, idx: int | str, episode: Episode, *, frames_uint8: bool = False
) -> tuple[Path, Path]:
    """Write <idx>.mp4/.pt; optionally save pre-codec uint8 frames as <idx>.npy.

    The NPY preserves the quantised encoder input losslessly, not subpixel float
    precision. Existing episode files are refused to prevent accidental loss.
    """
    directory = Path(directory)
    name = str(idx)
    if not name or name in (".", "..") or Path(name).name != name:
        raise ValueError("idx must be a plain file stem")
    paths = [directory / f"{name}{suffix}" for suffix in (".mp4", ".pt", ".npy")]
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Episode {name} already exists in {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    quantization = "truncate" if episode.motion == "integer" else "rint"
    encode_video(paths[0], episode.frames, quantization=quantization)
    torch.save(episode.metadata(), paths[1])
    if frames_uint8:
        np.save(paths[2], frames_to_uint8(episode.frames, quantization=quantization), allow_pickle=False)
    return paths[0], paths[1]


def _write_generation(path: Path, episodes: dict[str, Any], **settings: Any) -> None:
    proposals = sum(item.get("acceleration_proposals", 0) for item in episodes.values())
    acceptances = sum(item.get("acceleration_acceptances", 0) for item in episodes.values())
    result = dict(schema_version=1, **settings, episodes=episodes)
    if proposals:
        result.update(acceleration_proposals=proposals, acceleration_acceptances=acceptances,
                      acceleration_acceptance_rate=acceptances / proposals)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    path.chmod(0o644)


def write_episode(
    directory: str | Path, idx: int | str, episode: Episode, *, frames_uint8: bool = False,
) -> tuple[Path, Path]:
    """Write a standalone episode plus directory/generation.json provenance.

    For release sharding and a single sidecar per split, use write_split.
    Existing episode files are never overwritten.
    """
    manifest = Path(directory) / "generation.json"
    records = json.loads(manifest.read_text())["episodes"] if manifest.exists() else {}
    paths = _write_episode_files(directory, idx, episode, frames_uint8=frames_uint8)
    records[str(idx)] = episode.provenance()
    _write_generation(manifest, records)
    return paths


def write_split(
    out_dir: str | Path, split_name: str, n_episodes: int, *, seed: int = 0,
    n_frames: int = 70, frames_uint8: bool = False, **episode_options: Any,
) -> Path:
    """Write out_dir/split_name/<shard>/<id:04d>.{mp4,pt}, 256 per shard.

    IDs restart at zero within each shard, as in the release. The split must be
    new (no overwrite or stale loader cache). FloWM only loads split names in
    MNISTWorldVideoDataset._ALL_SPLITS: put new processes under a separate
    save_dir, reusing e.g. dynamic_validation, rather than inventing split names.
    Sampler names, settings and pooled/per-episode acceptance rates are stored
    once in split_name/generation.json. The episode .pt files remain collatable.
    """
    if not split_name or Path(split_name).name != split_name or split_name in (".", ".."):
        raise ValueError("split_name must be a plain directory name")
    if not isinstance(n_episodes, (int, np.integer)) or n_episodes < 1:
        raise ValueError("n_episodes must be a positive integer")
    directory = Path(out_dir) / split_name
    if directory.exists() or (Path(out_dir) / "metadata" / f"{split_name}.pt").exists():
        raise FileExistsError(f"Split or loader cache already exists: {directory}")
    rng = np.random.default_rng(seed)
    records = {}
    for i in range(n_episodes):
        episode = generate_episode(rng, n_frames, **episode_options)
        shard, local_id = divmod(i, 256)
        stem = f"{local_id:04d}"
        _write_episode_files(directory / str(shard), stem, episode, frames_uint8=frames_uint8)
        records[f"{shard}/{stem}"] = episode.provenance()
    _write_generation(directory / "generation.json", records, seed=seed, split_name=split_name,
                      n_episodes=n_episodes, episodes_per_shard=256)
    return directory
