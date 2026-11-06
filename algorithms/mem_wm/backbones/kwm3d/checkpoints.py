"""Strict, CPU-only backbone restoration for post-pilot lift diagnostics."""
import hashlib
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from . import create_kwm3d_model


def checkpoint_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(path, arm):
    path = Path(path).resolve()
    config_path = path.parent.parent / ".hydra/config.yaml"
    if config_path.exists():
        cfg = OmegaConf.load(config_path)
        if cfg.algorithm.backbone.arm != arm:
            raise ValueError("--arm disagrees with checkpoint's saved Hydra configuration")
    else:
        root = Path(__file__).resolve().parents[4]
        with initialize_config_dir(version_base=None, config_dir=str(root / "configurations")):
            cfg = compose(config_name="config", overrides=["shortcode=exp/blockworld/kwm3d/train/pilot_c",
                                                          f"algorithm.backbone.arm={arm}"])
    model = create_kwm3d_model(cfg.algorithm.backbone)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "main_model."
    state = {k[len(prefix):]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)
