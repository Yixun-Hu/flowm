"""Trained lift calls the same backbone write used by the estimator."""
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/flowm_bench'))
from k3d.trained_lift import pooled_tokens
from algorithms.mem_wm.backbones.kwm3d import KineticBlockworldModel


def test_trained_pooled_write_matches_estimator_imprint():
    torch.set_num_threads(8)
    model = KineticBlockworldModel(arm='a', embed_dim=32, map_proc_encoder_depth=1, dec_depth=1,
                                  num_heads=4, input_shape=(3,32,32), img_patch_size=8).eval()
    frames = torch.rand(2,3,32,32)
    with torch.inference_mode():
        tokens = pooled_tokens(model, frames)
        imprint, box, support = model.observation_imprint(frames)
        expected = tokens @ model.P.T
    torch.testing.assert_close(expected, imprint[:,:,support].transpose(1,2), rtol=1e-6, atol=1e-6)
