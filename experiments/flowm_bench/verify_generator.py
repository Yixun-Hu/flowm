"""Verify the MNIST World reconstruction using local data and CPU only.

Evidence: released .pt trajectories/cameras test the 50-pixel torus and timing;
label-restricted MNIST TRAIN matching tests sprite provenance; decoded RGB
frames test top-left sprites, additive clipping, and the centre-minus-16 crop.
Alternative crop/anchor/compositing conventions are measured as controls.
Re-encoding tests the CRF10/8fps/yuv420p settings against every decoded pixel.
Decoding is decord CPU get_batch, float32 RGB /255, as in base_video.py (not
mnist_world.py's separate, unused grayscale conversion). Raw-render errors
therefore include lossy compression and RGB decoding, even for exact recovery.

Episode ranges are inclusive global ordinals after NUMERIC directory/file
sorting, e.g. 0-199 selects 0/0000.pt through 0/0199.pt. --all selects every
episode of the chosen split. By default, verification uses 100 random episodes
and the training audit samples 8,000 files uniformly without replacement.
Selections and seeds are recorded. Camera histograms at t=1..12 compare the
recovered release law with the training audit; labels test uniform classes.
Chi-square diagnostics test marginals, not RNG identity or independence.
Every verified video has its stream settings inspected. Reports for different
splits coexist in one JSON. API checks include the real FloWM dataset and
default_collate, fixed-horizon acceleration laws, and subpixel conventions.

Run directly (avoids the repository's experiments package import side effects):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    /home/yixunhu/miniconda3/envs/flowm/bin/python -B \
    experiments/flowm_bench/verify_generator.py --random-episodes 100
  # Use --split dynamic_validation_200 --episodes 0-49 for the longer clips.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

# Set before importing numerical libraries; no CUDA device is ever selected.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True

import imageio_ffmpeg
import numpy as np
import torch

# Direct sibling import also keeps framework initialization out of this CLI.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mnist_world_gen import (
    ROOT, as_numpy, camera_centres, crop_window, decode_video, decode_video_uint8,
    encode_video, frames_to_uint8, generate_episode, load_metadata,
    load_mnist_train, load_mnist_train_labels, match_digit_indices,
    render_metadata, render_world, render_world_subpixel, sample_actions, sample_digits,
    sample_accelerations, wrap_positions, write_episode, write_split, _disc,
)


def episode_paths(directory: Path) -> list[Path]:
    """Stable numeric ordering of sharded <shard>/<id>.pt episode files."""
    paths = list(directory.glob("*/*.pt"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata in {directory}")
    return sorted(paths, key=lambda path: (int(path.parent.name), int(path.stem)))


def select_episodes(specification: str, total: int) -> list[int]:
    """Parse inclusive ranges and comma-separated ordinals; reject silent truncation."""
    selected: set[int] = set()
    for part in specification.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", part):
            raise ValueError(f"Invalid episode range: {part!r}")
        ends = [int(value) for value in part.split("-")]
        start, end = ends[0], ends[-1]
        if start > end or end >= total:
            raise ValueError(f"Episode range {part!r} is outside 0-{total - 1}")
        selected.update(range(start, end + 1))
    return sorted(selected)


def metadata_checks(metadata: dict[str, Any]) -> dict[str, bool]:
    """Check integer kinematics, action timing, bounds, and context length exactly."""
    actions = as_numpy(metadata["actions"])
    initial = as_numpy(metadata["digit_init_positions"])
    velocity = as_numpy(metadata["digit_velocities"])
    positions = as_numpy(metadata["digit_positions_over_time"])
    centres = camera_centres(metadata)
    expected = (initial[:, None] + velocity[:, None] * np.arange(len(actions))[None, :, None]) % 50
    return {
        "positions_equal_p0_plus_vt_mod50": np.array_equal(positions, expected),
        "centres_equal_cumsum_actions_mod50": np.array_equal(centres, np.cumsum(actions, axis=0) % 50),
        "first_action_zero": bool(np.all(actions[0] == 0)),
        "action_50_zero": bool(len(actions) <= 50 or np.all(actions[50] == 0)),
        "first_camera_zero": bool(np.all(centres[0] == 0)),
        "actions_int64": actions.dtype == np.int64,
        "positions_integer": bool(np.issubdtype(positions.dtype, np.integer)),
        "actions_in_minus10_plus10": bool(np.all(np.abs(actions) <= 10)),
        "velocities_in_minus1_plus1": bool(np.all(np.isin(velocity, (-1, 0, 1)))),
        "initial_positions_in_0_49": bool(np.all((initial >= 0) & (initial < 50))),
        "positions_in_0_50": bool(np.all((positions >= 0) & (positions < 50))),
        "context_length_50": len(metadata["camera_centers_input"]) == min(50, len(actions)),
    }


def inspect_stream(path: Path) -> dict[str, Any]:
    """Inspect the local FFmpeg stream header, decord FPS, and embedded x264 SEI.

    imageio-ffmpeg ships FFmpeg but no ffprobe in this environment. Its -i
    stream diagnostic exposes pixel format/profile; decord independently probes
    frame count/FPS. Neither command decodes on or initialises a GPU.
    """
    from decord import VideoReader, cpu

    executable = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([executable, "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, check=False)
    lines = [line.strip() for line in probe.stderr.splitlines()
             if "Video:" in line or "encoder" in line or "Duration:" in line]
    data = path.read_bytes()
    start = data.find(b"x264 - core")
    sei = ""
    if start >= 0:
        sei = data[start:].split(b"\x00", 1)[0].decode("ascii", errors="replace").strip()
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(), "header": lines,
            "fps": float(reader.get_avg_fps()), "frames": len(reader), "x264_sei": sei,
            "expected_settings_present": (reader.get_avg_fps() == 8 and "yuv420p" in probe.stderr
                                          and "h264 (High)" in probe.stderr and "32x32" in probe.stderr
                                          and "crf=10.0" in sei and "threads=1" in sei)}


def convention_controls(metadata: dict[str, Any], indices: np.ndarray, decoded: np.ndarray) -> dict[str, Any]:
    """Measure competing conventions on every frame of an episode."""
    sprites = load_mnist_train()[indices].astype(np.float32) / 255
    positions = as_numpy(metadata["digit_positions_over_time"])
    centres = camera_centres(metadata)
    variants = {name: [] for name in (
        "centre_minus16_add_clip", "centre_as_top_left", "centre_minus15", "centre_minus17",
        "sprite_centre_anchor", "maximum_overlap", "overwrite_overlap")}
    for t, centre in enumerate(centres):
        world = render_world(sprites, positions[:, t])
        for name, offset in (("centre_minus16_add_clip", 0), ("centre_as_top_left", 16),
                             ("centre_minus15", 1), ("centre_minus17", -1), ("sprite_centre_anchor", 14)):
            variants[name].append(crop_window(world, centre + offset))
        layers = np.stack([render_world(sprite[None], position[None])
                           for sprite, position in zip(sprites, positions[:, t])])
        variants["maximum_overlap"].append(crop_window(layers.max(axis=0), centre))
        overwrite = np.zeros((50, 50), dtype=np.float32)
        for sprite, (x, y) in zip(sprites, positions[:, t]):
            overwrite[np.ix_((y + np.arange(28)) % 50, (x + np.arange(28)) % 50)] = sprite
        variants["overwrite_overlap"].append(crop_window(overwrite, centre))
    mse = {name: float(np.mean((np.stack(frames)[:, None] - decoded)**2, dtype=np.float64))
           for name, frames in variants.items()}
    return {"mse": mse, "expected_convention_best": min(mse, key=mse.get) == "centre_minus16_add_clip"}


def distribution_test(counts: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    """Pearson multinomial marginal diagnostic, with a declared 1e-3 threshold."""
    from scipy.stats import chisquare

    if probabilities is None:
        probabilities = np.full(len(counts), 1 / len(counts))
    expected = counts.sum() * probabilities / probabilities.sum()
    statistic, pvalue = chisquare(counts, expected)
    return {"counts": counts.tolist(), "probabilities": (counts / counts.sum()).tolist(),
            "expected_probabilities": probabilities.tolist(), "chi_square": float(statistic),
            "p_value": float(pvalue), "reject_at_0_001": bool(pvalue < 0.001)}


def training_statistics(root: Path, limit: int, seed: int = 20261024) -> dict[str, Any]:
    paths = episode_paths(root / "dynamic_training")
    if limit > len(paths):
        raise ValueError(f"Requested {limit} training episodes, only {len(paths)} available")
    ordinals = (np.arange(len(paths)) if limit == 0 else
                np.sort(np.random.default_rng(seed).choice(len(paths), limit, replace=False)))
    selected = [paths[i] for i in ordinals]
    hist = {"actions": np.zeros((2, 21), dtype=np.int64),
            "velocities": np.zeros((2, 3), dtype=np.int64),
            "positions": np.zeros((2, 50), dtype=np.int64)}
    labels = np.zeros(10, dtype=np.int64)
    failures: Counter[str] = Counter()
    action_time = np.zeros((70, 21), dtype=np.int64)
    for number, path in enumerate(selected, 1):
        data = load_metadata(path)
        for name, key, offset in (("actions", "actions", 10), ("velocities", "digit_velocities", 1),
                                  ("positions", "digit_init_positions", 0)):
            values = as_numpy(data[key])
            if name == "actions":
                values = values[1:]
            if np.any((values + offset < 0) | (values + offset >= hist[name].shape[1])):
                raise ValueError(f"Out-of-range {name} in {path}")
            for axis in range(2):
                hist[name][axis] += np.bincount(values[:, axis] + offset, minlength=hist[name].shape[1])
        actions = as_numpy(data["actions"])
        if len(actions) == 70:
            np.add.at(action_time, (np.repeat(np.arange(70), 2), (actions + 10).ravel()), 1)
        labels += np.bincount(as_numpy(data["digit_labels"]), minlength=10)
        failures.update(name for name, passed in metadata_checks(data).items() if not passed)
        if number % 1000 == 0:
            print(f"Training metadata: {number}/{len(selected)}", flush=True)
    result: dict[str, Any] = {"available_episodes": len(paths), "sampled_episodes": len(selected),
                              "sampling": "uniform without replacement" if limit else "all",
                              "seed": seed, "ordinals": ordinals.tolist(),
                              "first_file": str(selected[0]), "last_file": str(selected[-1]),
                              "metadata_check_failures": dict(failures)}
    for name, counts in hist.items():
        result[name] = {"x": distribution_test(counts[0]), "y": distribution_test(counts[1]),
                        "pooled": distribution_test(counts.sum(axis=0))}
        print(f"{name} histogram x: {counts[0].tolist()}", flush=True)
        print(f"{name} histogram y: {counts[1].tolist()}", flush=True)
        print(f"  pooled uniform p={result[name]['pooled']['p_value']:.5g}", flush=True)
    train_labels = np.bincount(load_mnist_train_labels(), minlength=10).astype(np.float64)
    result["labels_uniform"] = distribution_test(labels)
    result["labels_uniform_train_index"] = distribution_test(labels, train_labels / train_labels.sum())
    result["actions_by_timestep_counts"] = action_time.tolist()
    result["action_50_always_zero"] = bool(action_time[50, 10] == 2 * len(selected))
    result["action_phase_distributions"] = {
        "early_context_1_7": distribution_test(action_time[1:8].sum(axis=0)),
        "later_context_8_49": distribution_test(action_time[8:50].sum(axis=0)),
        "target_51_69": distribution_test(action_time[51:].sum(axis=0)),
    }
    print(f"action[50] always zero: {result['action_50_always_zero']}", flush=True)
    for name, distribution in result["action_phase_distributions"].items():
        print(f"{name} histogram: {distribution['counts']}; uniform p={distribution['p_value']:.5g}", flush=True)
    print(f"labels histogram: {labels.tolist()}; p(uniform classes)={result['labels_uniform']['p_value']:.5g}; "
          f"p(uniform TRAIN indices)={result['labels_uniform_train_index']['p_value']:.5g}", flush=True)
    result["interpretation"] = (
        "Marginal goodness-of-fit only, not proof of independent sampling. Early camera actions explore "
        "unseen pixels, becoming uniform after coverage; action[50] is zero. Labels are uniform classes. "
        "Original release seeds and RNG call order are unknown.")
    return result


def verify_sampling(statistics: dict[str, Any], count: int = 2000) -> dict[str, Any]:
    """Compare the production camera sampler with independently audited releases."""
    from scipy.stats import chi2_contingency

    rng = np.random.default_rng(731)
    simulated = np.zeros((70, 21), dtype=np.int64)
    uniform = np.zeros(21, dtype=np.int64)
    episode_labels = np.zeros(10, dtype=np.int64)
    for i in range(count):
        episode = generate_episode(rng, 70)
        actions = episode.actions
        episode_labels += np.bincount(episode.digit_labels, minlength=10)
        np.add.at(simulated, (np.repeat(np.arange(70), 2), (actions + 10).ravel()), 1)
        uniform += np.bincount((sample_actions(rng, 70, action_law="uniform")[1:] + 10).ravel(), minlength=21)
        if (i + 1) % 200 == 0:
            print(f"Generated sampling audit: {i + 1}/{count}", flush=True)
    observed = np.asarray(statistics["actions_by_timestep_counts"])
    comparisons = []
    for t in range(1, 13):
        table = np.stack((observed[t], simulated[t]))
        # Drop structural zero bins (no pseudocounts); Bonferroni across 12 tests.
        table = table[:, table.sum(axis=0) > 0]
        chi2, pvalue, dof, _ = chi2_contingency(table, correction=False)
        comparisons.append(dict(t=t, chi_square=float(chi2), p_value=float(pvalue), dof=int(dof),
                                reject_familywise_0_001=bool(pvalue < .001 / 12)))
    labels, indices = sample_digits(np.random.default_rng(732), 100_000)
    np.testing.assert_array_equal(load_mnist_train_labels()[indices], labels)
    label_test = distribution_test(np.bincount(labels, minlength=10))
    episode_label_test = distribution_test(episode_labels)
    uniform_test = distribution_test(uniform)
    action_zero = bool(simulated[0, 10] == 2 * count and simulated[50, 10] == 2 * count)
    passed = (action_zero and statistics["action_50_always_zero"]
              and not any(row["reject_familywise_0_001"] for row in comparisons)
              and not label_test["reject_at_0_001"] and not episode_label_test["reject_at_0_001"]
              and not uniform_test["reject_at_0_001"])
    return dict(passed=passed, generated_episodes=count, generated_actions_by_timestep=simulated,
                release_comparisons=comparisons, action_0_and_50_zero=action_zero,
                uniform_option=uniform_test, generated_labels=label_test,
                generated_episode_labels=episode_label_test,
                note="Camera histograms use fully rendered generated episodes. "
                     "Camera threshold is familywise .001/12; other marginals use .001.")


def verify_acceleration_statistics(count: int = 10_000) -> dict[str, Any]:
    """Realised joint distributions, with accepted/proposed pooled across digits."""
    result = {}
    horizon = 199
    for law in ("plan", "disc_endpoints"):
        rng = np.random.default_rng(811)
        initial = _disc(rng, 2, count)
        velocity, acceleration, proposals = sample_accelerations(
            rng, initial, 2, horizon + 1, law, 4 / horizon, 1_000_000)
        speed = np.linalg.norm(velocity, axis=1)
        magnitude = np.linalg.norm(acceleration, axis=1)
        moving = (speed > 0) & (magnitude > 0)
        dot = np.sum(velocity * acceleration, axis=1)
        end_speed = np.linalg.norm(velocity + horizon * acceleration, axis=1)
        if np.any(end_speed > 2 + 1e-12) or np.any(speed > 2 + 1e-12):
            raise AssertionError("Acceleration sampler exceeded its bound horizon")
        zero_test = distribution_test(np.array([(magnitude == 0).sum(), (magnitude > 0).sum()]),
                                      np.array([.3, .7])) if law == "plan" else None
        if law == "plan":
            np.testing.assert_array_equal(velocity, initial)
            if zero_test["reject_at_0_001"] or np.any((magnitude[moving] < 2e-4) | (magnitude[moving] > 2e-2)):
                raise AssertionError("Plan mixture or nonzero acceleration range failed")
        if law == "disc_endpoints" and abs(count / proposals - .25) > .01:
            raise AssertionError("Default disc-endpoint acceptance differs from 1/4")
        midpoint = {}
        for length in (70, 200):
            values = np.linalg.norm(velocity + ((length - 1) / 2) * acceleration, axis=1)
            midpoint[str(length)] = dict(t=(length - 1) / 2, mean=float(values.mean()),
                                         median=float(np.median(values)), max=float(values.max()))
        result[law] = dict(
            digits=count, bound_horizon=200, H=horizon, proposals=proposals, acceptances=count,
            pooled_acceptance_rate=count / proposals, zero_fraction=float(np.mean(magnitude == 0)),
            zero_mixture_test=zero_test,
            mean_cos_v0_a_nonzero=float(np.mean(dot[moving] / (speed[moving] * magnitude[moving]))),
            mean_a_dot_v0_over_speed_squared_times_H=float(np.mean(dot / speed**2) * horizon),
            mean_a_dot_v0_over_mean_speed_squared_times_H=float(dot.mean() / np.mean(speed**2) * horizon),
            median_acceleration=float(np.median(magnitude)),
            median_nonzero_acceleration=float(np.median(magnitude[moving])),
            mean_initial_speed=float(speed.mean()), mean_endpoint_speed=float(end_speed.mean()),
            max_endpoint_speed=float(end_speed.max()), mid_episode_speed=midpoint)
    return result


def verify_loader(save_dir: Path, n_episodes: int, n_frames: int = 70) -> dict[str, Any]:
    """Instantiate the real dataset, call every __getitem__, and default_collate pairs.

    Build the loader's ordinary timestamp cache serially to avoid its four-worker
    metadata scan (and /dev/shm requirements). No dataset methods are replaced;
    video decoding, conditions, extra metadata and collation use the real path.
    All caches live under the generated save_dir, never under released data.
    """
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate

    sys.path.insert(0, str(ROOT))
    from datasets.video.mnist_world import MNISTWorldVideoDataset

    split = "dynamic_validation"
    assert split in MNISTWorldVideoDataset._ALL_SPLITS
    paths = episode_paths(save_dir / split)
    assert len(paths) == n_episodes
    streams = [inspect_stream(path.with_suffix(".mp4")) for path in paths]
    assert all(stream["expected_settings_present"] and stream["frames"] == n_frames for stream in streams)
    cache = save_dir / "metadata"
    cache.mkdir(exist_ok=True)
    for name in MNISTWorldVideoDataset._ALL_SPLITS:
        cache_path = cache / f"{name}.pt"
        if not cache_path.exists():
            videos = [path.with_suffix(".mp4") for path in paths] if name == split else []
            torch.save(dict(video_paths=videos, video_pts=[torch.arange(n_frames) for _ in videos],
                            video_fps=[8.0 for _ in videos]), cache_path)
    cfg = OmegaConf.create(dict(
        save_dir=str(save_dir), resolution=[32, 32], downsampling_factor=[1, 1, 1],
        latent=dict(enable=False, type="online", suffix=None, is_preprocessing_expt=False),
        cond_loading_style="relative_pos_offset", subdataset_size=None, external_cond_dim=2,
        external_cond_stack=False, external_cond_processing="mask_first", max_frames=n_frames,
        n_frames=n_frames, frame_skip=1, filter_min_len=None, path_list=None,
        training_splits=["dynamic_training"], num_default_clips=n_episodes,
        cond_alignment="t-1->t", data_cond_alignment="t-1->t"))
    dataset = MNISTWorldVideoDataset(cfg, split=split, purpose="validation")
    assert len(dataset) == n_episodes
    shapes = []
    loaded = 0
    for start in range(0, len(dataset), 2):
        items = [dataset[i] for i in range(start, min(start + 2, len(dataset)))]
        for item in items:
            metadata_path = Path(item["metadata"]["path"]).with_suffix(".pt")
            raw = load_metadata(metadata_path)
            assert all(value is not None for value in raw.values())
            assert not any(key in raw for key in ("acceleration_acceptance_rate", "action_sampler", "motion"))
            np.testing.assert_array_equal(item["videos"].numpy(), decode_video(metadata_path.with_suffix(".mp4")))
            np.testing.assert_array_equal(item["conds"].numpy(), as_numpy(raw["actions"]))
        batch = default_collate(items)
        assert batch["videos"].device.type == "cpu"
        assert tuple(batch["videos"].shape) == (len(items), n_frames, 3, 32, 32)
        assert tuple(batch["conds"].shape) == (len(items), n_frames, 2)
        shapes.append(list(batch["videos"].shape))
        loaded += len(items)
    manifest = json.loads((save_dir / split / "generation.json").read_text())
    assert len(manifest["episodes"]) == n_episodes
    assert all(entry["action_law"] == "release" for entry in manifest["episodes"].values())
    return dict(passed=True, episodes=loaded, collated_batches=len(shapes), video_shapes=shapes,
                stream_checks=len(streams), split=split, save_dir=str(save_dir),
                provenance=manifest)


def verify_smoke(out_dir: Path, count: int = 8) -> dict[str, Any]:
    """One existing loader split name per motion, under separate save directories."""
    result = {}
    for motion in ("integer", "fractional", "accelerated"):
        directory = out_dir / motion
        write_split(directory, "dynamic_validation", count, seed=914, motion=motion,
                    renderer="nearest" if motion == "integer" else "bilinear",
                    p0_continuous=motion != "integer")
        result[motion] = verify_loader(directory, count)
    return result


def verify_api() -> dict[str, Any]:
    """Small independent checks for wrap, interpolation, motion bounds, and I/O."""
    # Frozen output of the supplied review/release_actions.py, not this generator.
    reference_actions = sample_actions(np.random.default_rng(0), 200)
    assert hashlib.sha256(reference_actions.astype("<i8").tobytes()).hexdigest() == (
        "b6d8eafdf9abebc9973464b21a63fbb39a941f1dc3a2fbd59149833abec262f8")
    rng = np.random.default_rng(71)
    sprites = rng.random((5, 28, 28), dtype=np.float32)
    positions = np.array([[49, 49], [0, 0], [-1, 50], [25, 25], [40, 30]])
    reference = np.zeros((50, 50), dtype=np.float32)
    for sprite, (x, y) in zip(sprites, positions):
        for sy in range(28):
            for sx in range(28):
                reference[(y + sy) % 50, (x + sx) % 50] += sprite[sy, sx]
    np.testing.assert_array_equal(render_world(sprites, positions), np.clip(reference, 0, 1))
    np.testing.assert_array_equal(render_world_subpixel(sprites, positions, "bilinear"), render_world(sprites, positions))
    np.testing.assert_array_equal(render_world_subpixel(sprites, positions + .2, "nearest"), render_world(sprites, positions))
    impulse = np.zeros((1, 28, 28), dtype=np.float32)
    impulse[0, 0, 0] = 1
    expected = np.zeros((50, 50), dtype=np.float32)
    expected[49, 49], expected[49, 0], expected[0, 49], expected[0, 0] = .375, .125, .375, .125
    np.testing.assert_array_equal(render_world_subpixel(impulse, np.array([[49.25, 49.5]])), expected)
    np.testing.assert_array_equal(render_world_subpixel(impulse, np.array([[-.75, -.5]])), expected)
    np.testing.assert_array_equal(render_world_subpixel(impulse, np.array([[49.5, 49.5]]), "nearest"),
                                  render_world(impulse, np.array([[50, 50]])))
    np.testing.assert_array_equal(render_world_subpixel(impulse, np.array([[.5, 1.5]]), "nearest"),
                                  render_world(impulse, np.array([[0, 2]])))
    np.testing.assert_array_equal(frames_to_uint8(np.array([.5, 1.5, 2.5]) / 255, quantization="rint"),
                                  [0, 2, 2])
    np.testing.assert_array_equal(frames_to_uint8(np.array([.5, 1.5, 2.5]) / 255), [0, 1, 2])
    edge = np.array([-np.nextafter(0.0, 1.0), -1e-16, 50.0, 100.0, -50.0])
    assert np.any(edge % 50 == 50)  # Exercise the floating remainder regression.
    wrapped = wrap_positions(edge)
    assert np.all((wrapped >= 0) & (wrapped < 50))
    world = np.arange(2500).reshape(50, 50)
    np.testing.assert_array_equal(crop_window(world, np.array([0, 0])),
                                  np.array([[world[(y - 16) % 50, (x - 16) % 50] for x in range(32)] for y in range(32)]))
    rates: dict[str, float | None] = {}
    for motion in ("integer", "fractional", "accelerated"):
        for renderer in ("nearest", "bilinear"):
            episode = generate_episode(np.random.default_rng(123), 70, motion=motion, renderer=renderer)
            repeat = generate_episode(np.random.default_rng(123), 70, motion=motion, renderer=renderer)
            np.testing.assert_array_equal(episode.frames, repeat.frames)
            np.testing.assert_array_equal(episode.camera_centers, np.cumsum(episode.actions, axis=0) % 50)
            np.testing.assert_array_equal(episode.actions[[0, 50]], np.zeros((2, 2)))
            velocity = episode.digit_velocities[:, None] + episode.digit_accelerations[:, None] * np.arange(200)[None, :, None]
            if np.max(np.linalg.norm(velocity, axis=-1)) > 2 + 1e-12:
                raise AssertionError("Speed bound exceeded")
            if not (np.all(episode.frames >= 0) and np.all(episode.frames <= 1)):
                raise AssertionError("Frame intensities out of bounds")
            rates[f"{motion}_{renderer}"] = episode.acceleration_acceptance_rate
    for continuous in (False, True):
        fractional = generate_episode(np.random.default_rng(824), 70, motion="fractional", p0_continuous=continuous)
        accelerated = generate_episode(np.random.default_rng(824), 70, motion="accelerated", p0_continuous=continuous)
        np.testing.assert_array_equal(fractional.digit_velocities, accelerated.digit_velocities)
        np.testing.assert_array_equal(fractional.digit_init_positions, accelerated.digit_init_positions)
        if continuous:
            assert np.any(fractional.digit_init_positions != np.rint(fractional.digit_init_positions))
        for law in ("plan", "disc_endpoints"):
            short = generate_episode(np.random.default_rng(824), 70, motion="accelerated",
                                     acceleration_law=law, p0_continuous=continuous)
            long = generate_episode(np.random.default_rng(824), 200, motion="accelerated",
                                    acceleration_law=law, p0_continuous=continuous)
            np.testing.assert_array_equal(short.digit_velocities, long.digit_velocities)
            np.testing.assert_array_equal(short.digit_accelerations, long.digit_accelerations)
            np.testing.assert_array_equal(short.digit_positions_over_time, long.digit_positions_over_time[:, :70])
            np.testing.assert_array_equal(short.frames, long.frames[:70])
            np.testing.assert_array_equal(short.actions, long.actions[:70])
            assert np.all((long.digit_positions_over_time >= 0) & (long.digit_positions_over_time < 50))
            # Nearest-rendered metadata must preserve the unrounded trajectory.
            assert np.any(short.digit_positions_over_time != np.rint(short.digit_positions_over_time))
    for horizon in (1, 70, 400):
        for law in ("plan", "disc_endpoints"):
            bounded = generate_episode(rng, 2, motion="accelerated", acceleration_law=law,
                                       bound_horizon=horizon)
            endpoint = bounded.digit_velocities + (horizon - 1) * bounded.digit_accelerations
            assert np.max(np.linalg.norm(endpoint, axis=1)) <= 2 + 1e-12
    for length in (1, 2):
        zero = generate_episode(rng, length, motion="accelerated", speed_max=0)
        np.testing.assert_array_equal(zero.digit_positions_over_time,
                                      np.repeat(zero.digit_init_positions[:, None], length, axis=1))
    try:
        generate_episode(rng, 2, motion="accelerated", n_digits=2, max_proposals=1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Acceleration rejection ignored max_proposals")
    with tempfile.TemporaryDirectory(prefix="flowm-api-") as temporary:
        video, metadata_path = write_episode(temporary, "sample", episode, frames_uint8=True)
        restored = load_metadata(metadata_path)
        np.testing.assert_array_equal(render_metadata(restored, episode.digit_train_indices), episode.frames)
        np.testing.assert_array_equal(np.load(Path(temporary) / "sample.npy"),
                                      frames_to_uint8(episode.frames, quantization="rint"))
        manifest = json.loads((Path(temporary) / "generation.json").read_text())
        assert manifest["episodes"]["sample"]["action_law"] == "release"
        assert all(value is not None for value in restored.values())
        decoded = decode_video(video)
        if decoded.shape != (70, 3, 32, 32) or decoded.dtype != np.float32:
            raise AssertionError("Incorrect pipeline decoder shape/dtype")
        try:
            write_episode(temporary, "sample", episode)
        except FileExistsError:
            pass
        else:
            raise AssertionError("write_episode overwrote an existing file")
        loaders = verify_smoke(Path(temporary) / "loader", count=2)
        # Cross the real release shard boundary using minimal one-frame clips.
        sharded = write_split(Path(temporary) / "sharding", "dynamic_validation", 257,
                              n_frames=1, n_digits=1)
        assert len(list((sharded / "0").glob("*.pt"))) == 256
        assert len(list((sharded / "1").glob("*.pt"))) == 1
        assert (sharded / "0/0255.mp4").exists() and (sharded / "1/0000.mp4").exists()
        assert not (sharded / "0/generation.json").exists()
        try:
            write_split(Path(temporary) / "sharding", "dynamic_validation", 1)
        except FileExistsError:
            pass
        else:
            raise AssertionError("write_split reused an existing split")
    joint = verify_acceleration_statistics()
    return {"passed": True, "checks": "review action reference digest, independent integer oracle, bilinear periodic impulse, integer equivalence, "
            "half-to-even rendering/quantization, floating modulo, crop wrap, reproducibility, fixed-horizon speed bounds, "
            "shared fractional/plan v0, 70/200 trajectory prefixes, continuous p0, real nearest metadata, "
            "zero speed/T=1, rejection limit, metadata/NPY roundtrip, real loader/default_collate (two per motion), "
            "CPU decoder shape, 257-episode shard boundary, overwrite protection",
            "acceleration_acceptance_rates": rates, "acceleration_statistics": joint,
            "loader_checks": {motion: {key: value for key, value in item.items() if key != "provenance"}
                              for motion, item in loaders.items()}}


def verify_episode(path: Path, ordinal: int, temporary: Path, controls: bool) -> dict[str, Any]:
    stream = inspect_stream(path.with_suffix(".mp4"))
    metadata = load_metadata(path)
    pixels = decode_video_uint8(path.with_suffix(".mp4"))
    decoded = pixels.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    matches = match_digit_indices(metadata, decoded)
    rendered = render_metadata(metadata, matches.indices)
    difference = rendered[:, None] - decoded
    frame_mse = np.mean(difference**2, axis=(1, 2, 3), dtype=np.float64)
    recoded_path = temporary / "reencoded.mp4"
    encode_video(recoded_path, rendered)
    recoded = decode_video_uint8(recoded_path)
    identical = np.all(pixels == recoded, axis=(1, 2, 3))
    record = {
        "ordinal": ordinal, "path": str(path), "n_frames": len(pixels),
        "mse": float(frame_mse.mean()), "max_abs_error": float(np.max(np.abs(difference))),
        "frame_mse": frame_mse.tolist(), "mse_le_5e_minus5": bool(frame_mse.mean() <= 5e-5),
        "matching": dict(asdict(matches),
                         second_best_to_best_mae_ratio=[float(x) if np.isfinite(x) else "inf"
                                                       for x in matches.mae_ratios],
                         incomplete_coverage=matches.covered_pixels < 784,
                         incomplete_coverage_digit_indices=np.flatnonzero(matches.covered_pixels < 784)),
        "match_margins": matches.margins.tolist(), "stream": stream,
        "metadata_checks": metadata_checks(metadata),
        "decoded_rgb_channels_identical": bool(np.all(pixels == pixels[..., :1])),
        "reencoded_identical_frames": int(identical.sum()),
        "reencoded_all_frames_identical": bool(identical.all()),
        "reencoded_mismatched_frame_indices": np.flatnonzero(~identical).tolist(),
        "reencoded_mse": float(np.mean(((pixels.astype(np.float32) - recoded) / 255)**2, dtype=np.float64)),
    }
    if controls:
        record["convention_controls"] = convention_controls(metadata, matches.indices, decoded)
    return record


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    margins = np.concatenate([record["match_margins"] for record in records])
    ratios = np.array([value for record in records for value in
                       record["matching"]["second_best_to_best_mae_ratio"]], dtype=np.float64)
    coverage = np.concatenate([record["matching"]["covered_pixels"] for record in records])
    failed_checks = Counter(name for record in records for name, passed in record["metadata_checks"].items() if not passed)
    return {
        "episodes": len(records), "frames": sum(record["n_frames"] for record in records),
        "mean_episode_mse": float(np.mean([record["mse"] for record in records])),
        "min_episode_mse": min(record["mse"] for record in records),
        "max_episode_mse": max(record["mse"] for record in records),
        "max_abs_error": max(record["max_abs_error"] for record in records),
        "fraction_mse_le_5e_minus5": sum(record["mse_le_5e_minus5"] for record in records) / len(records),
        "reencoded_identical_frames": sum(record["reencoded_identical_frames"] for record in records),
        "reencoded_identical_episodes": sum(record["reencoded_all_frames_identical"] for record in records),
        "min_match_margin": float(margins.min()), "median_match_margin": float(np.median(margins)),
        "max_match_margin": float(margins.max()), "zero_margin_digits": int(np.sum(margins == 0)),
        "min_second_best_to_best_mae_ratio": float(ratios.min()) if np.isfinite(ratios.min()) else "inf",
        "median_second_best_to_best_mae_ratio": float(np.median(ratios)) if np.isfinite(np.median(ratios)) else "inf",
        "max_second_best_to_best_mae_ratio": float(ratios.max()) if np.isfinite(ratios.max()) else "inf",
        "min_covered_pixels": int(coverage.min()), "median_covered_pixels": float(np.median(coverage)),
        "digits_with_incomplete_coverage": int(np.sum(coverage < 784)),
        "digits_without_full_sprite_view": sum(int(np.sum(record["matching"]["full_visible_frames"] == 0)) for record in records),
        "matching_methods": dict(Counter(method for record in records for method in record["matching"]["methods"])),
        "metadata_check_failures": dict(failed_checks),
        "stream_checks": len(records),
        "stream_check_failures": sum(not record["stream"]["expected_settings_present"] or
                                     record["stream"]["frames"] != record["n_frames"] for record in records),
        "convention_control_failures": sum(not record["convention_controls"]["expected_convention_best"]
                                           for record in records if "convention_controls" in record),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="dynamic_validation",
                        choices=("dynamic_validation", "dynamic_validation_200", "dynamic_training"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--episodes", help="Inclusive numeric episode ordinals, e.g. 0-199,300")
    selection.add_argument("--random-episodes", type=int, help="Uniform random sample without replacement (default 100)")
    selection.add_argument("--all", action="store_true", help="Verify every episode in the selected split")
    parser.add_argument("--seed", type=int, default=20261024, help="Validation selection seed")
    parser.add_argument("--training-seed", type=int, default=20261025, help="Independent training audit selection seed")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mnist_world")
    parser.add_argument("--training-episodes", type=int, default=8000, help="Training files to inspect; 0 means all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/flowm_bench/generator_verification.json")
    parser.add_argument("--smoke-dir", type=Path, help="Write and load eight episodes per motion here (must be new)")
    args = parser.parse_args()
    if args.training_episodes < 0:
        parser.error("--training-episodes must be nonnegative")
    torch.set_num_threads(1)
    started = time.monotonic()
    paths = episode_paths(args.data_root / args.split)
    if args.all:
        ordinals = list(range(len(paths)))
        selection_description = "all"
    elif args.episodes:
        ordinals = select_episodes(args.episodes, len(paths))
        selection_description = args.episodes
    else:
        count = 100 if args.random_episodes is None else args.random_episodes
        if not 1 <= count <= len(paths):
            parser.error(f"--random-episodes must be between 1 and {len(paths)}")
        ordinals = sorted(np.random.default_rng(args.seed).choice(len(paths), count, replace=False).tolist())
        selection_description = f"uniform without replacement, n={count}, seed={args.seed}"
    print("Running generator API, acceleration and loader checks", flush=True)
    api = verify_api()
    smoke = verify_smoke(args.smoke_dir) if args.smoke_dir else {}
    print(f"{args.split}: {len(ordinals)}/{len(paths)} episodes; {selection_description}", flush=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="flowm-verify-") as temporary:
        for ordinal in ordinals:
            record = verify_episode(paths[ordinal], ordinal, Path(temporary), controls=True)
            records.append(record)
            print(f"{ordinal:5d}  MSE={record['mse']:.8g}  max={record['max_abs_error']:.6g}  "
                  f"margin_min={min(record['match_margins']):.6g}  "
                  f"ratio_min={min(record['matching']['second_best_to_best_mae_ratio'], key=float)}  "
                  f"covered_min={min(record['matching']['covered_pixels'])} "
                  f"flagged={len(record['matching']['incomplete_coverage_digit_indices'])}  "
                  f"reencode={record['reencoded_identical_frames']}/{record['n_frames']}", flush=True)
    statistics = training_statistics(args.data_root, args.training_episodes, args.training_seed)
    sampling = verify_sampling(statistics)
    summary = summarise(records)
    run = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "split": args.split,
           "episode_selection": selection_description, "ordinals": ordinals, "seed": args.seed,
           "available_episodes": len(paths),
           "cpu_only": True, "decoder": "decord CPU get_batch -> float32 RGB TCHW /255",
           "api_checks": api, "sampling_checks": sampling, "smoke_checks": smoke, "summary": summary,
           "episodes": records, "elapsed_seconds": time.monotonic() - started}
    output: dict[str, Any] = {"schema_version": 1, "runs": {}}
    if args.output.exists():
        output = json.loads(args.output.read_text())
        if output.get("schema_version") != 1:
            raise ValueError("Refusing to overwrite a report with an unknown schema")
    output["training_statistics"] = statistics
    output["runs"][args.split] = run
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Replace only this generated report, after a complete serialisation succeeds.
    payload = json.dumps(output, indent=2, default=_json_default, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, suffix=".tmp", delete=False) as stream_out:
        stream_out.write(payload)
        temporary_output = Path(stream_out.name)
    temporary_output.replace(args.output)
    args.output.chmod(0o644)
    print(f"SUMMARY {args.split}: {json.dumps(summary, sort_keys=True)}", flush=True)
    print(f"Report: {args.output}", flush=True)
    print(f"SAMPLING: {json.dumps(sampling, default=_json_default)}", flush=True)
    print(f"ACCELERATION: {json.dumps(api['acceleration_statistics'])}", flush=True)
    # Historical uniform-action audit mismatches remain diagnostics; the recovered
    # law, new sampler tests, and all deterministic checks must pass.
    return int(bool(summary["metadata_check_failures"] or summary["convention_control_failures"]
                    or summary["stream_check_failures"] or not sampling["passed"]
                    or statistics["metadata_check_failures"]
                    or summary["reencoded_identical_episodes"] != summary["episodes"]))


if __name__ == "__main__":
    raise SystemExit(main())
