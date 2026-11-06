"""Shared CLI environment and Hydra composition for kinetic pilot tools."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "algorithms/mem_wm/backbones/kwm3d/.scratch"


def setup(device="cpu"):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # CUDA mode deliberately preserves the caller's visibility mask.
    for key, value in dict(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", WANDB_MODE="offline",
                           PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(SCRATCH / "tmp"),
                           MPLCONFIGDIR=str(SCRATCH / "matplotlib"),
                           XDG_CACHE_HOME=str(SCRATCH / "cache")).items():
        os.environ.setdefault(key, value)
    for name in ("tmp", "matplotlib", "cache"):
        (SCRATCH / name).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))


def compose_pilot(shortcode, overrides=()):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configurations")):
        return compose(config_name="config", overrides=[
            f"shortcode=exp/blockworld/kwm3d/{shortcode}", "ckpt_map=default", *overrides])
