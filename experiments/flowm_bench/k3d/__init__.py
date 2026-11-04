"""CPU-only released FloWM-3D stage-one experiments."""
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["WANDB_MODE"] = "offline"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_scratch = Path(__file__).resolve().parent / ".scratch"
_scratch.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(_scratch / "cache")
os.environ["MPLCONFIGDIR"] = str(_scratch / "mpl")
os.environ["TMPDIR"] = str(_scratch)

