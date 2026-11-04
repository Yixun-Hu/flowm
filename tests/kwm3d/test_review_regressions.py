"""Regression gates promoted from review probes p2, p4 and p7."""
import math

import pytest
import torch
from kwm.collision import named_nodes
from kwm.rot import rot_vec

from algorithms.mem_wm.backbones.flowm.flowm_models_3d import ViTBlockworldModel
from algorithms.mem_wm.backbones.kwm3d import KineticBlockworldModel
from algorithms.mem_wm.backbones.kwm3d.geometry import agent_to_map, turn_scalar_state
from test_dataset import lightweight_dataset
from test_models import ROOT, SMALL


def test_b_integer_transport_and_turns_match_flowm():
    torch.manual_seed(0)
    baseline = ViTBlockworldModel(**{**SMALL, 'world_size': 16}, v_range=1)
    model = KineticBlockworldModel(arm='b', **{**SMALL, 'world_size': 16})
    v5 = named_nodes('v5')
    indices = [int((model.lattice == node).all(-1).nonzero().item()) for node in v5]
    h9 = torch.randn(1, 9, 33, 33, 32, dtype=torch.float64).float()
    h5 = h9[:, indices].clone()
    for code in (0, 1, 2, 4):
        action = torch.tensor([code])
        actual, _ = model._move(h9, torch.zeros(1, 9, 4), action, None)
        expected = baseline.map_processor.roll_map(baseline.map_processor.shift_v_channels(h5), action)
        torch.testing.assert_close(actual[:, indices], expected, rtol=0, atol=2e-6)
    for k, code in ((1, 0), (-1, 1)):
        actual, _ = model.turn_state(h9, torch.zeros(1, 9, 4), k)
        expected = baseline.map_processor.roll_map(h5, torch.tensor([code]))
        torch.testing.assert_close(actual[:, indices], expected, rtol=0, atol=0)
        turned, _ = turn_scalar_state(h5, torch.zeros(1, 5, 4), v5, k)
        torch.testing.assert_close(turned, expected, rtol=0, atol=0)


def test_c_training_parity_24_plus_8_with_turns_and_checkpointing():
    torch.manual_seed(0)
    baseline = ViTBlockworldModel(**SMALL, v_range=1, v_channel_maxpool_decode=True,
                                 v_channel_mixing_mode='2d_v_channel_flattened').train()
    with torch.no_grad():
        baseline.v_channel_mixing.weight.normal_(0, .2)
    model = KineticBlockworldModel(arm='c', checkpoint_steps=True, **SMALL).train()
    model.load_state_dict(baseline.state_dict(), strict=True)
    video = torch.rand(2, 32, 3, 32, 32)
    actions = torch.tensor([[4, 2, 0, 2, 1, 2, 4, 2], [4, 1, 2, 2, 0, 2, 4, 2]]).repeat(1, 4)
    # Seed 5 exercises both teacher-forced and autoregressive prediction steps.
    torch.manual_seed(5)
    coins = torch.rand(8) < .5
    assert coins.any() and not coins.all()
    results = []
    for net in (baseline, model):
        torch.manual_seed(5)
        output, hidden = net(video[:, :24], actions[:, :24], actions[:, 24:],
                             target_seq=video[:, 24:], teacher_forcing_ratio=.5, return_hidden=True)
        output.square().mean().backward()
        results.append((output.detach(), hidden.detach()))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    grads = dict(model.named_parameters())
    for name, parameter in baseline.named_parameters():
        actual = grads[name].grad
        assert (actual is None) == (parameter.grad is None), name
        if actual is not None:
            torch.testing.assert_close(actual, parameter.grad, rtol=0, atol=0, msg=name)


def test_oracle_real_clips_through_model_transport(monkeypatch):
    paths = sorted((ROOT / 'data/blockworld/k3d_v_validation').glob('*/*_rgb.mp4'))[:6]
    if not paths:
        pytest.skip('k3d_v_validation absent')
    import algorithms.mem_wm.backbones.kwm3d.kwm3d_models as module
    original_stream = module.stream
    captured = {}
    def capture_stream(h, nodes, ego, **kwargs):
        captured.update(nodes=nodes, ego=ego)
        return original_stream(h, nodes, ego, **kwargs)
    monkeypatch.setattr(module, 'stream', capture_stream)
    model = KineticBlockworldModel(arm='a_oracle', **{**SMALL, 'world_size': 16})
    yy, xx = torch.meshgrid(torch.arange(33.), torch.arange(33.), indexing='ij')
    phase = torch.exp(2j * math.pi * torch.stack((xx, yy)) / 33)
    worst = {mode: dict(turn=0., nonturn=0.) for mode in ('incoming_object', 'raw')}
    counts = dict(turn=0, nonturn=0)
    with torch.no_grad():
        for path in paths:
            if not path.with_name(path.stem.removesuffix('_rgb') + '_block_info.pt').exists():
                pytest.skip('block metadata absent')
            sample = lightweight_dataset(path, 30, n_frames=60)[0]
            meta = {key: value[None] for key, value in sample['metadata'].items() if torch.is_tensor(value)}
            actions = sample['conds'].long()
            positions = agent_to_map(meta['block_pos_agent'][0])
            for t in range(1, 60):
                code = int(actions[t])
                k = 1 if code == 0 else -1 if code == 1 else 0
                kind = 'turn' if k else 'nonturn'
                valid = meta['block_valid'][0, t] & meta['block_valid'][0, t-1]
                visible = valid & (positions[t-1].abs().amax(-1) <= 11) & (positions[t].abs().amax(-1) <= 11)
                h = torch.zeros(1, 11, 33, 33, 1)
                centers = positions[t-1] + 16
                h[0, 1:, :, :, 0] = torch.exp(-((xx - centers[:, 0, None, None])**2 +
                                                (yy - centers[:, 1, None, None])**2) / (2 * 1.5**2))
                for mode in worst:
                    model.oracle_velocity = mode  # raw is an explicit diagnostic.
                    nodes, _ = model._oracle(meta, t, torch.device('cpu'))
                    moved, _ = model._move(h, torch.zeros(1, 11, 4), actions[t:t+1], nodes)
                    # Use the actual pre-turn nodes and ego sent to stream,
                    # then the model's turn convention. No centroid approximation.
                    predicted = rot_vec(positions[t-1] + captured['nodes'][0, 1:] - captured['ego'][0], k)
                    if valid.any():
                        error = (predicted[valid] - positions[t, valid]).norm(dim=-1).max().item()
                        worst[mode][kind] = max(worst[mode][kind], error)
                    if mode == 'incoming_object' and visible.any():
                        mass = (moved[0, 1:, :, :, 0][:, None] * phase).sum((-2, -1))
                        centroid = (torch.angle(mass) / (2 * math.pi) * 33) % 33 - 16
                        assert (centroid[visible] - positions[t, visible]).norm(dim=-1).max() < 5e-4
                        counts[kind] += int(visible.sum())
    assert all(counts.values()), counts
    # The stored fp32 kinematics are zero to 4 decimal places; float32
    # addition/cancellation does not promise literal bitwise equality.
    assert max(worst['incoming_object'].values()) < 1e-5, worst
    assert min(worst['raw'].values()) > .1, worst
