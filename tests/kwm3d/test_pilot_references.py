import torch

from algorithms.mem_wm.backbones.kwm3d.metrics import block_regions
from algorithms.mem_wm.backbones.kwm3d.per_clip import clip_stats
from scripts.kwm3d_baseline_repeat import predictors


def test_predictors_use_only_context_and_preserve_rgb_mean():
    context = torch.arange(2*70*3*4*4, dtype=torch.float32).reshape(2, 70, 3, 4, 4) / 10000
    frames = predictors(context)
    assert set(frames) == {"repeat", "context_mean", "last_flat", "gray"}
    torch.testing.assert_close(frames["repeat"], context[:, -1:], rtol=0, atol=0)
    torch.testing.assert_close(frames["context_mean"], context.mean(1, keepdim=True), rtol=0, atol=0)
    for channel in range(3):
        for b in range(2):
            expected = context[b, -1, channel].mean()
            assert torch.all(frames["last_flat"][b, :, channel] == expected)
    assert torch.all(frames["gray"] == .5)


def test_shared_regions_match_original_repeat_arithmetic_at_both_horizons():
    torch.manual_seed(5)
    videos = torch.rand(1, 280, 3, 8, 8)
    metadata = dict(block_pos_agent=torch.tensor([3., 0.]).expand(1, 210, 1, 2),
                    block_valid=torch.ones(1, 210, 1, dtype=torch.bool),
                    block_speed=torch.full((1, 210, 1), .5))
    actions = torch.arange(210)[None] % 5
    regions = block_regions(metadata, actions, (8, 8))
    for horizon in (140, 280):
        target = videos[:, 70:horizon]
        pred = videos[:, 69:70].expand_as(target)
        meta = {k: v[:, :horizon-70] for k, v in metadata.items()}
        expected = clip_stats(pred, target, meta, actions[:, :horizon-70])
        actual = clip_stats(pred, target, meta, actions[:, :horizon-70],
                            regions=[r[:, :horizon-70] for r in regions])
        for name in expected:
            for x, y in zip(actual[name], expected[name]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
