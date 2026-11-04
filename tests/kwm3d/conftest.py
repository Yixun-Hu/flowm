"""CPU-only tests; all generated caches and temporary files live in kwm3d scratch."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "algorithms/mem_wm/backbones/kwm3d/.scratch"
sys.path.insert(0, str(ROOT))
import pytest


@pytest.fixture(scope="module")
def kwm3d_environment():
    """Restore process settings when these tests finish in a combined run."""
    import torch
    import tempfile
    settings = dict(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", WANDB_MODE="offline",
                    MPLCONFIGDIR=str(SCRATCH / "matplotlib"), XDG_CACHE_HOME=str(SCRATCH / "cache"),
                    TMPDIR=str(SCRATCH / "tmp"))
    old_threads, old_tempdir = torch.get_num_threads(), tempfile.tempdir
    with pytest.MonkeyPatch.context() as patch:
        for key, value in settings.items():
            patch.setenv(key, value)
        for name in ("matplotlib", "cache", "tmp"):
            (SCRATCH / name).mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(SCRATCH / "tmp")
        torch.set_num_threads(8)
        yield
        torch.set_num_threads(old_threads)
        tempfile.tempdir = old_tempdir


@pytest.fixture(autouse=True)
def kwm3d_runtime(kwm3d_environment):
    pass


@pytest.fixture
def tmp_path(kwm3d_environment):
    import tempfile
    with tempfile.TemporaryDirectory(prefix="pytest-", dir=SCRATCH / "tmp") as directory:
        yield Path(directory)
