"""Blockworld coordinate conventions and scalar-channel C4 state actions."""
import torch
from kwm.rot import rot_vec, rotation_perms

# MiniWorld: a left turn sends an object ahead to the camera's right.
# FloWM sends +column to +row with rot90(-1), therefore right is +row.
# Independently checked by the pinhole projection / saturated-block overlay.
RIGHT_ROW_SIGN = 1
MAX_BLOCKS = 10


def agent_to_map(xz: torch.Tensor) -> torch.Tensor:
    """(forward, right) -> (column displacement, row displacement)."""
    return xz * xz.new_tensor((1, RIGHT_ROW_SIGN))


def turn_scalar_state(features, node_state, lattice, k, perms=None):
    """rot_state semantics for scalar [B,V,H,W,C], about the map centre.

    node_state is [B,V,4]: rho_x, rho_y, evidence, lock. Snapshots are
    invalidated by the caller, since a turned cone has no usable overlap.
    Unlike kwm.rot.rot_state this never permutes feature/orientation channels.
    """
    perms = rotation_perms(lattice) if perms is None else perms
    source = perms[(-k) % 4]
    rotated = torch.rot90(features[:, source], -k, (2, 3)).contiguous()
    state = node_state[:, source]
    return rotated, torch.cat((rot_vec(state[..., :2], k), state[..., 2:]), -1)
