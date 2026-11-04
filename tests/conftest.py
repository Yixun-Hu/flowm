"""CPU-only tests; all generated caches and temporary files live in kwm3d scratch."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "algorithms/mem_wm/backbones/kwm3d/.scratch"
sys.path.insert(0, str(ROOT))
for key, value in dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", WANDB_MODE="offline",
                       MPLCONFIGDIR=str(SCRATCH / "matplotlib"), XDG_CACHE_HOME=str(SCRATCH / "cache"),
                       TMPDIR=str(SCRATCH / "tmp")).items():
    os.environ[key] = value
for name in ("matplotlib", "cache", "tmp"):
    (SCRATCH / name).mkdir(parents=True, exist_ok=True)


def pytest_configure(config):
    config.option.basetemp = str(SCRATCH / "pytest-tmp")
    import torch
    torch.set_num_threads(8)
