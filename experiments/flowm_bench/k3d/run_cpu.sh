#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/experiments/flowm_bench:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$PWD/experiments/flowm_bench/k3d/.scratch"
export XDG_CACHE_HOME="$TMPDIR/cache" MPLCONFIGDIR="$TMPDIR/mpl"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
module="$1"
shift
exec /home/yixunhu/miniconda3/envs/flowm/bin/python -m "k3d.$module" "$@"
