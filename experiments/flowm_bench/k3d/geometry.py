"""Cone coordinates and empirical check of the agent-right / map-row sign."""
import argparse
from functools import lru_cache
import torch
from algorithms.mem_wm.backbones.flowm.flowm_utils import fov_mask_indices

WORLD, CENTER, BOX, BOX_SHAPE = 33, 16, (7, 16), (19, 17)
# Verified by geometry CLI on released-weight writes and by the turn convention.
ROW_SIGN = 1


@lru_cache(maxsize=1)
def cone_mask():
    mask, indices = fov_mask_indices((33, 33), (16, 16), heading_deg=0,
                                    fov_deg=60, samples_per_edge=5, min_coverage=0)
    assert len(indices) == 177
    assert indices.amin(0).tolist() == [7, 16]
    assert indices.amax(0).tolist() == [25, 32]
    return mask


def rounded_cells(position, sign=ROW_SIGN):
    """Half-open residual [-.5,.5); output row,column coordinates."""
    rounded = torch.floor(position + 0.5).long()
    return torch.stack((CENTER + sign * rounded[..., 1], CENTER + rounded[..., 0]), -1)


def visible(position, sign=ROW_SIGN):
    rc = rounded_cells(position, sign)
    inside = ((rc >= 0) & (rc < WORLD)).all(-1) & (position[..., 0] >= 1)
    return inside & cone_mask()[rc[..., 0].clamp(0, 32), rc[..., 1].clamp(0, 32)]


def velocity_to_map(velocity):
    return velocity * velocity.new_tensor([1, ROW_SIGN])


def camera_action(action):
    return torch.stack(((action == 2).float(), torch.zeros_like(action).float()), -1)


def turn_k(action):
    """KWM positive rotation is clockwise in row-down display coordinates."""
    return 1 if int(action) == 0 else -1 if int(action) == 1 else 0


def main():
    from .common import clips, clip_id, truth, write_csv, log
    from .lift import read_cache, tokens_to_box
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=20)
    args = p.parse_args()
    rows = []
    for path in clips("k3d_int_validation", args.clips):
        cache = read_cache(path)
        pos = truth(path)["block_pos_agent"][:140]
        for signal in ("tokens", "g"):
            box, _ = tokens_to_box(cache[signal].float())
            activity = box.norm(dim=1) if signal == "tokens" else box.mean(dim=1)
            centered = activity - activity.mean(0, keepdim=True)
            keep = visible(pos) & (pos[..., 1].abs() >= 1)
            tt, nn = keep.nonzero(as_tuple=True)
            plus, minus = rounded_cells(pos[tt, nn], 1), rounded_cells(pos[tt, nn], -1)
            for name, field in (("raw", activity), ("cell_centered", centered)):
                a = field[tt, plus[:, 0]-7, plus[:, 1]-16]
                b = field[tt, minus[:, 0]-7, minus[:, 1]-16]
                rows.append(dict(clip=clip_id(path), signal=signal, normalization=name,
                                 blocks=len(tt), plus_sum=a.sum().item(), minus_sum=b.sum().item(),
                                 paired_difference=(a-b).mean().item()))
    write_csv("geometry.csv", rows)
    for signal in ("tokens", "g"):
        for norm in ("raw", "cell_centered"):
            subset = [r for r in rows if r["signal"] == signal and r["normalization"] == norm]
            n = sum(r["blocks"] for r in subset)
            log(f"{signal}/{norm}: n={n}, plus={sum(r['plus_sum'] for r in subset)/n:.6f}, "
                f"minus={sum(r['minus_sum'] for r in subset)/n:.6f}")


if __name__ == "__main__":
    main()
