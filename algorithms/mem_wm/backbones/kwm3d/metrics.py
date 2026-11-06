"""Analytic MiniWorld block silhouettes and pixel-weighted regional errors."""
import itertools
import math

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from torchmetrics import Metric

SPEED_BINS = ((0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
METRIC_NAMES = ("block_mse", "block_mse_speed_0.25_0.5", "block_mse_speed_0.5_0.75",
                "block_mse_speed_0.75_1.0", "block_mse_turn", "block_mse_nonturn")


def project_block_corners(position, image_size=(128, 128), fov_y=60.0,
                          camera_height=1.5, near=0.04):
    """Project 0.7 x 1.5 x 0.7 box corners, clipping edges to the near plane.

    position=(forward,right), base on y=0; MiniWorld gluPerspective + gluLookAt
    at pitch=0. Pixel centres have coordinates 0..W-1 / 0..H-1. Quarter-turn
    headings preserve the square ground footprint. Returns visible [N,2] xy.
    """
    forward, right = np.asarray(position, dtype=np.float64)
    corners = np.array([(forward + dx, right + dz, y - camera_height)
                        for dx, dz, y in itertools.product((-.35, .35), (-.35, .35), (0., 1.5))])
    points = list(corners[corners[:, 0] >= near])
    # Include intersections of the twelve box edges, so near-camera boxes have
    # valid silhouettes instead of exploding / disappearing behind the camera.
    for i in range(8):
        for axis in (1, 2, 4):
            j = i ^ axis
            if j <= i:
                continue
            a, b = corners[i], corners[j]
            if (a[0] < near) != (b[0] < near):
                points.append(a + (b - a) * ((near - a[0]) / (b[0] - a[0])))
    if not points:
        return np.empty((0, 2))
    points = np.asarray(points)
    height, width = image_size
    focal = height / (2 * math.tan(math.radians(fov_y) / 2))
    return np.stack(((width - 1) / 2 + focal * points[:, 1] / points[:, 0],
                     (height - 1) / 2 - focal * points[:, 2] / points[:, 0]), -1)


def project_block_masks(block_pos_agent, block_valid, image_size=(128, 128), *, per_block=False):
    """Boolean [...,N,H,W] silhouettes, or [...,H,W] union (no occlusion).

    Rasterization runs on CPU; output is returned on the metadata's device.
    """
    pos = torch.as_tensor(block_pos_agent)
    valid = torch.as_tensor(block_valid, dtype=torch.bool)
    if pos.shape[:-1] != valid.shape or pos.shape[-1] != 2:
        raise ValueError("positions [...,N,2] and valid [...,N] must match")
    height, width = image_size
    shape = valid.shape
    positions = pos.detach().cpu().numpy().reshape(-1, 2)
    active = valid.detach().cpu().numpy().reshape(-1)
    masks = np.zeros((len(active), height, width), dtype=np.bool_)
    for i in np.flatnonzero(active):
        points = project_block_corners(positions[i], image_size)
        if len(points) < 3 or not np.isfinite(points).all():
            continue
        hull = points[ConvexHull(points).vertices]
        # PIL polygon coordinates refer to pixel centres. The analytic hull is
        # rasterized without any colour-based refinement of the actual metric.
        canvas = Image.new("1", (width, height))
        ImageDraw.Draw(canvas).polygon([tuple(p) for p in hull], fill=1)
        masks[i] = np.asarray(canvas, dtype=bool)
    masks = torch.from_numpy(masks.reshape(*shape, height, width)).to(pos.device)
    return masks if per_block else masks.any(dim=-3)


def block_region_stats(pred, target, mask, *, per_clip=False):
    """Sum of squared RGB errors and the number of channel-pixels selected."""
    if pred.shape != target.shape or mask.shape != (*pred.shape[:-3], *pred.shape[-2:]):
        raise ValueError("expected matching [...,C,H,W] videos and [...,H,W] mask")
    mask = mask.to(device=pred.device, dtype=torch.bool).unsqueeze(-3)
    dims = tuple(range(1, pred.ndim)) if per_clip else tuple(range(pred.ndim))
    error = (pred.float() - target.float()).square().masked_fill(~mask, 0).sum(
        dims, dtype=torch.float64 if per_clip else None)
    count = mask.sum(dims) * pred.shape[-3]
    return error, count


def block_region_mse(pred, target, mask):
    """Pixel-weighted MSE; NaN for an empty region (never a false zero)."""
    error, count = block_region_stats(pred, target, mask)
    return torch.where(count > 0, error / count.clamp_min(1), error.new_tensor(float("nan")))


def block_region_splits(pred, target, metadata, actions, *, return_stats=False, per_clip=False):
    """All-block, speed-bin and incoming turn-step errors, over supplied frames.

    Intervals are [lo,hi), with 1.0 included in the last bin. Overlapping blocks
    count once within each split. Pass only predicted frames, excluding context.
    Prefer generator block_speed (object speed); raw block_vel_agent includes ego.
    """
    masks = project_block_masks(metadata["block_pos_agent"], metadata["block_valid"],
                                pred.shape[-2:], per_block=True).to(pred.device)
    valid = metadata["block_valid"].to(pred.device)
    speeds = metadata.get("block_speed")
    if speeds is None:
        speeds = metadata["block_vel_agent"].norm(dim=-1)
    speeds = speeds.to(pred.device)
    union = masks.any(-3)
    regions = [union]
    for lo, hi in SPEED_BINS:
        selected = valid & (speeds >= lo) & ((speeds <= hi) if hi == 1.0 else (speeds < hi))
        regions.append((masks & selected[..., None, None]).any(-3))
    if actions.ndim == 3 and actions.shape[-1] == 1:
        actions = actions.squeeze(-1)
    turn = ((actions == 0) | (actions == 1)).to(pred.device)
    regions.extend((union & turn[..., None, None], union & ~turn[..., None, None]))
    stats = {name: block_region_stats(pred, target, region, per_clip=per_clip) for name, region in zip(METRIC_NAMES, regions)}
    if return_stats:
        return stats
    return {name: torch.where(n > 0, e / n.clamp_min(1), e.new_tensor(float("nan")))
            for name, (e, n) in stats.items()}


class BlockRegionMetric(Metric):
    """Distributed sums, rather than averaging batches with unequal block area."""
    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state("squared_error", default=torch.zeros(len(METRIC_NAMES), dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("pixels", default=torch.zeros(len(METRIC_NAMES), dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, pred=None, target=None, metadata=None, actions=None, *, stats=None):
        if stats is None:
            stats = block_region_splits(pred, target, metadata, actions, return_stats=True)
        self.add_stats(stats)

    def add_stats(self, stats):
        for i, name in enumerate(METRIC_NAMES):
            error, count = stats[name]
            self.squared_error[i] += error.sum().to(self.squared_error)
            self.pixels[i] += count.sum().to(self.pixels)

    def compute(self):
        return {name: torch.where(self.pixels[i] > 0, self.squared_error[i] / self.pixels[i].clamp_min(1),
                                  self.squared_error.new_tensor(float("nan")))
                for i, name in enumerate(METRIC_NAMES)}


def saturation_mask(frame, saturation=0.35, brightness=0.15):
    """Independent RGB control; grey blocks are intentionally invisible here."""
    frame = torch.as_tensor(frame).float()
    high, low = frame.amax(-3), frame.amin(-3)
    return ((high - low) / high.clamp_min(1e-6) > saturation) & (high > brightness)
