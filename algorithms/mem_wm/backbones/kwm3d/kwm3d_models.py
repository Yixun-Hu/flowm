"""Kinetic FloWM-3D, importing both FloWM modules and KWM primitives."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from algorithms.mem_wm.backbones.flowm.flowm_models_3d import ViTBlockworldModel
from kwm.cells import AdaptiveNodeCell, _fixed_projection
from kwm.collision import named_nodes
from kwm.gconv import C4SymmetricDepthwise
from kwm.lock_nodes import LockState, lock_measurement
from kwm.model import oracle_nodes_from
from kwm.rot import rotation_perms, rot_vec
from kwm.transport import stream

from .geometry import MAX_BLOCKS, agent_to_map, turn_scalar_state

ARMS = ("c", "c_pw", "a_oracle", "a", "b")


class _LockArithmetic(AdaptiveNodeCell):
    """Reuse the installed cell's exact update arithmetic, without its CNN lift.

    kwm (k3d-base, 94803f5) routes the measurement through the overridable
    `_measure_lock` hook; this subclass passes the cone box, support mask and
    residual options there. A one-cell dummy feature bank makes the cell's
    otherwise unused feature transport negligible.
    """
    def __init__(self, world, sigma, step_clip, s_init):
        nn.Module.__init__(self)
        self.world, self.sigma = world, sigma
        self.step_clip, self.bound = step_clip, 0.75
        self.lam_rel, self.tr_rel, self.gate, self.coh_rel = 1e-2, 0.1, 0.6, 0.5
        self.pin_static, self.c4_numerics = True, True
        self.n_s = nn.Parameter(torch.tensor(float(s_init)))
        self.conv_u = nn.Identity()

    def imprint(self, u):
        return u

    def feature_update(self, u, feat, return_prelift=False):
        return feat

    def _measure_lock(self, *args, **kwargs):
        return lock_measurement(*args, box=self._box, support_mask=self._support,
                                sigma=self.sigma, radius=None, static_residual=True, **kwargs)

    def forward(self, previous, current, state, lattice, action, box, support):
        self._box, self._support = box, support
        dummy = current.new_zeros(state.shape[0], state.shape[1], 1, 1, 1)
        result = self._forward_lock(current, LockState(dummy, state, previous), action,
                                    lattice, 1.0, None, True)
        return result.nodes


class KineticBlockworldModel(ViTBlockworldModel):
    def __init__(self, arm="a", lattice=None, map_conv="c4", write=None,
                 checkpoint_steps=True, imprint_k=8, sigma=1.0, step_clip=0.3,
                 s_init=1.0, oracle_velocity="raw", **flowm_kwargs):
        arm = arm.lower().replace("-", "_")
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        expected_write = "per_node" if arm == "c" else "pooled"
        if write not in (None, "auto", expected_write):
            raise ValueError(f"{arm} requires write={expected_write}")
        for key, value in dict(v_range=1, v_channel_maxpool_decode=True,
                               v_channel_identity_emb=False, no_self_motion_equiv=False,
                               v_channel_mixing_mode="2d_v_channel_flattened").items():
            if key in flowm_kwargs and flowm_kwargs[key] != value:
                raise ValueError(f"kwm3d requires {key}={value!r}")
            flowm_kwargs[key] = value
        super().__init__(**flowm_kwargs)
        self.arm, self.write = arm, expected_write
        if oracle_velocity not in ("raw", "incoming_object"):
            raise ValueError("oracle_velocity must be raw or incoming_object")
        self.oracle_velocity = oracle_velocity
        self.checkpoint_steps = checkpoint_steps
        self.kinetic = arm in ("a", "a_oracle", "b")
        chosen = lattice or ("v17" if arm == "a" else "v9" if arm == "b" else "v5")
        if arm == "a" and chosen not in ("v9", "v17"):
            raise ValueError("A supports v9 or v17")
        if arm == "b" and chosen != "v9":
            raise ValueError("B uses v9")
        if arm in ("c", "c_pw") and chosen != "v5":
            raise ValueError("C and C-pw use v5")
        nodes = named_nodes(chosen)
        if arm == "a_oracle":
            nodes = torch.zeros(MAX_BLOCKS + 1, 2)
        self.register_buffer("lattice", nodes, persistent=False)
        self.register_buffer("perms", rotation_perms(nodes) if arm != "a_oracle"
                             else torch.empty(0, dtype=torch.long), persistent=False)
        self.map_processor.num_v = len(nodes)
        if self.kinetic:
            self.v_channel_mixing = None
            if map_conv == "c4":
                self.map_conv = C4SymmetricDepthwise(self.embed_dim)
            elif map_conv == "plain":
                self.map_conv = nn.Conv2d(self.embed_dim, self.embed_dim, 3, padding=1,
                                          groups=self.embed_dim, padding_mode="circular", bias=False)
                nn.init.dirac_(self.map_conv.weight, groups=self.embed_dim)
            else:
                raise ValueError("map_conv must be c4 or plain")
        if arm == "a":
            self.register_buffer("P", _fixed_projection(self.embed_dim, imprint_k))
            self.refiner = _LockArithmetic(self.map_processor.world_size, sigma, step_clip, s_init)
        self.last_nodes = self.last_node_state = None

    def _pooled_write(self, h, frame, mask, valid):
        p = self.map_processor
        tokens, idx = p.extract_view_tokens(h, mask)
        pos = p.map_pos_enc(h.shape[2], h.shape[3], device=h.device)[idx]
        pooled = tokens.masked_fill(~valid[:, :, None, None], -torch.inf).max(1).values
        image = p.patch(frame) + p.img_pos_enc(*p.patch.grid_size, device=frame.device)[None]
        joint = torch.cat((pooled + pos[None], image), 1)
        for block in p.blocks:
            joint = block(joint)
        proposed = p.norm(joint)[:, :len(idx), None].transpose(1, 2).expand_as(tokens)
        gate = p.update_gate(torch.cat((tokens, proposed), -1)) * valid[:, :, None, None]
        written = (1 - gate) * tokens + gate * proposed
        return p.overwrite_view_tokens(h, written, idx)

    def _write(self, h, frame, mask, valid):
        if self.write == "per_node":
            return self.map_processor(h, frame, mask)
        return self._pooled_write(h, frame, mask, valid)

    def observation_imprint(self, frame, mask=None):
        """Observation-only lift: shared processor on a zero, single-node map.

        With pooled write all zero-node inputs are identical, so one node is
        exactly equivalent to V zero nodes, with much smaller activation cost.
        """
        if mask is None:
            mask = self._get_fov_mask(frame.device)
        size = self.map_processor.world_size
        zero = frame.new_zeros(frame.shape[0], 1, size, size, self.embed_dim)
        valid = torch.ones(frame.shape[0], 1, dtype=torch.bool, device=frame.device)
        written = self._pooled_write(zero, frame, mask, valid)[:, 0]
        rc = mask.nonzero()
        r0, c0 = rc.amin(0).tolist()
        r1, c1 = (rc.amax(0) + 1).tolist()
        with torch.autocast(device_type=frame.device.type, enabled=False):
            imprint = torch.einsum("kc,bhwc->bkhw", self.P.float(), written.float())
        return imprint[:, :, r0:r1, c0:c1], (r0, c0), mask[r0:r1, c0:c1]

    def turn_state(self, h, state, k):
        return turn_scalar_state(h, state, self.lattice, k, self.perms)

    def _move(self, h, state, action, nodes):
        if not self.kinetic:
            return self.map_processor.roll_map(self.map_processor.shift_v_channels(h), action), state
        ego = torch.stack(((action == 2).float(), torch.zeros_like(action).float()), -1)
        # Transport in the pre-turn frame, as FloWM does. Oracle nodes supplied
        # in the destination frame are rotated back just for this transport.
        if self.arm == "a_oracle":
            old_nodes = nodes.clone()
            for code, k in ((0, 1), (1, -1)):
                selected = action == code
                old_nodes[selected] = rot_vec(nodes[selected], -k)
        else:
            old_nodes = self.lattice[None] + state[..., :2]
        h = stream(h.permute(0, 1, 4, 2, 3), old_nodes, ego, c4=True)
        h = h.permute(0, 1, 3, 4, 2).contiguous()
        out, new_state = h.clone(), state.clone()
        for code, k in ((0, 1), (1, -1)):
            selected = action == code
            if selected.any():
                if self.arm == "a_oracle":
                    # Object identities persist; there is no lattice permutation.
                    out[selected] = torch.rot90(h[selected], -k, (2, 3))
                else:
                    out[selected], new_state[selected] = self.turn_state(h[selected], state[selected], k)
        return out, new_state

    def _mix(self, h):
        b, v, height, width, channels = h.shape
        if self.kinetic:
            x = h.permute(0, 1, 4, 2, 3).reshape(b * v, channels, height, width)
            return self.map_conv(x).reshape(b, v, channels, height, width).permute(0, 1, 3, 4, 2).contiguous()
        # This layout is the original FloWM per-feature cross-V grouped conv.
        x = h.permute(0, 4, 1, 2, 3).contiguous().view(b, channels * v, height, width)
        return self.v_channel_mixing(x).view(b, channels, v, height, width).permute(0, 2, 3, 4, 1).contiguous()

    def _decode(self, h, mask, valid):
        tokens, idx = self.map_processor.extract_view_tokens(h, mask)
        pos = self.map_processor.map_pos_enc(h.shape[2], h.shape[3], device=h.device)[idx]
        tokens = (tokens + pos[None, None]).masked_fill(~valid[:, :, None, None], -torch.inf)
        return self.decoder(tokens.max(1).values)

    def _step(self, h, state, snapshot, snapshot_valid, frame, action, nodes, valid,
              mask, prediction, teacher_force, record_latent):
        h, state = self._move(h, state, action, nodes)
        h = self._mix(h) * valid[:, :, None, None, None]
        latent = h.norm(dim=-1).norm(dim=1) if record_latent else h.new_empty(0)
        out = self._decode(h, mask, valid) if prediction else frame
        obs = frame if teacher_force or not prediction else out
        h = self._write(h, obs, mask, valid)
        if self.arm == "a" and not prediction:
            current, box, support = self.observation_imprint(obs, mask)
            turning = (action == 0) | (action == 1)
            usable = snapshot_valid & ~turning
            if usable.any():
                ego = torch.stack(((action == 2).float(), torch.zeros_like(action).float()), -1)
                measured = self.refiner(snapshot[usable], current[usable], state[usable],
                                        self.lattice, ego[usable], box, support)
                state = state.clone()
                state[usable] = measured
            # Turn frames invalidate the pair, including the following step.
            snapshot = torch.where(turning[:, None, None, None], torch.zeros_like(current), current)
            snapshot_valid = ~turning
        # Prediction keeps evidence/rho/snapshot frozen, except the coordinate
        # change of rho and node labels in _move. No decoded-frame measurement.
        return h, state, snapshot, snapshot_valid, out, latent

    def _oracle(self, metadata, t, device):
        key = "block_vel_agent" if self.oracle_velocity == "raw" else "block_vel_object_incoming"
        if metadata is None or key not in metadata or "block_valid" not in metadata:
            raise ValueError(f"A-oracle needs batched {key} and block_valid metadata")
        vel = metadata[key][:, t].to(device=device, dtype=torch.float32)
        valid = metadata["block_valid"][:, t].to(device=device, dtype=torch.bool)
        if vel.shape[1:] != (MAX_BLOCKS, 2) or valid.shape != vel.shape[:2]:
            raise ValueError("oracle metadata must be padded to 10 blocks")
        vel = agent_to_map(vel).masked_fill(~valid[..., None], 0)
        return oracle_nodes_from(vel), torch.cat((torch.ones_like(valid[:, :1]), valid), 1)

    def forward(self, input_seq, input_actions, target_actions, target_seq=None,
                teacher_forcing_ratio=0.0, return_hidden=False, return_latent_video=False,
                metadata=None):
        b, context = input_seq.shape[:2]
        if input_actions.ndim == 3 and input_actions.shape[-1] == 1:
            input_actions = input_actions.squeeze(-1)
            target_actions = target_actions.squeeze(-1)
        if input_actions.ndim != 2 or target_actions.ndim != 2:
            raise ValueError("kwm3d requires integer action codes [B,T]")
        horizon = target_actions.shape[1]
        if context < 1 or horizon < 1:
            raise ValueError("at least one context and prediction frame is required")
        mask = self._get_fov_mask(input_seq.device)
        h = self.map_processor.init_map(b, input_seq.device)
        state = h.new_zeros(b, len(self.lattice), 4, dtype=torch.float32)
        bounds = mask.nonzero()
        box_shape = (bounds.amax(0) - bounds.amin(0) + 1).tolist()
        snapshot = h.new_zeros(b, self.P.shape[0] if self.arm == "a" else 0, *box_shape)
        snapshot_valid = torch.zeros(b, dtype=torch.bool, device=h.device)
        valid = torch.ones(b, len(self.lattice), dtype=torch.bool, device=h.device)
        nodes = self.lattice[None].expand(b, -1, -1)
        outputs, latents = [], []
        for t in range(context + horizon):
            prediction = t >= context
            j = t - context
            action = target_actions[:, j] if prediction else input_actions[:, t]
            if t == 0:
                action = torch.full_like(action, 4)  # FloWM discards the first ego action.
            if self.arm == "a_oracle":
                nodes, valid = self._oracle(metadata, t, h.device)
            force = (self.training and prediction and target_seq is not None
                     and torch.rand(1).item() < teacher_forcing_ratio)
            frame = (target_seq[:, j] if force else input_seq[:, -1]) if prediction else input_seq[:, t]
            args = (h, state, snapshot, snapshot_valid, frame, action, nodes, valid,
                    mask, prediction, force, return_latent_video and prediction)
            if self.checkpoint_steps and torch.is_grad_enabled():
                h, state, snapshot, snapshot_valid, out, latent = checkpoint(self._step, *args, use_reentrant=False)
            else:
                h, state, snapshot, snapshot_valid, out, latent = self._step(*args)
            if prediction:
                outputs.append(out)
                if return_latent_video:
                    latents.append(latent)
        self.last_node_state = state.detach()
        self.last_nodes = (nodes if self.arm == "a_oracle" else self.lattice[None] + state[..., :2]).detach()
        preds = torch.stack(outputs, 1)
        if return_latent_video:
            return preds, torch.stack(latents, 1).unsqueeze(2)
        return (preds, h) if return_hidden else preds

    def load_released(self, state_dict, rebuild_mixing=True):
        """Strict transfer of the released backbone; only new module keys may miss."""
        source = {k.removeprefix("main_model."): v for k, v in state_dict.items()}
        mixing = "v_channel_mixing.weight"
        # The released node-indexed convolution has no analogue in kinetic arms.
        if self.kinetic:
            if mixing not in source:
                raise AssertionError("released checkpoint lacks mixing weights")
            source.pop(mixing)
        elif rebuild_mixing:
            if mixing not in source:
                raise AssertionError("released checkpoint lacks mixing weights")
            nn.init.dirac_(self.v_channel_mixing.weight, groups=self.embed_dim)
            source[mixing] = self.v_channel_mixing.weight.detach().clone()
        expected = {"map_conv.weight"} if self.kinetic else set()
        if self.arm == "a":
            expected |= {"P", "refiner.n_s"}
        result = self.load_state_dict(source, strict=False)
        assert not result.unexpected_keys, result.unexpected_keys
        assert set(result.missing_keys) == expected, (result.missing_keys, expected)
        return result.missing_keys, result.unexpected_keys


def parameter_counts(**kwargs):
    """Print trainable parameter counts without retaining all five models."""
    counts = {}
    for arm in ARMS:
        model = KineticBlockworldModel(arm=arm, **kwargs)
        counts[arm] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{arm}: {counts[arm]:,}")
    return counts
