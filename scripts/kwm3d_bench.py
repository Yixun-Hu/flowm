#!/usr/bin/env python3
"""CPU-only forward/backward timing; no optimizer, trainer, or GPU calls.

Each checkpoint mode runs in a fresh child process so ru_maxrss is independent.
One step means a complete 10-context + 5-prediction rollout and backward.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "algorithms/mem_wm/backbones/kwm3d/.scratch"
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", WANDB_MODE="offline",
                  PYTHONDONTWRITEBYTECODE="1", MPLCONFIGDIR=str(SCRATCH / "matplotlib"),
                  XDG_CACHE_HOME=str(SCRATCH / "cache"), TMPDIR=str(SCRATCH / "tmp"))
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu"], default="cpu")
    parser.add_argument("--arm", choices=["c", "c_pw", "a_oracle", "a", "b"], required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--checkpoint", choices=["both", "on", "off"], default="both")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.steps < 1 or args.batch < 1:
        parser.error("steps and batch must be positive")
    for name in ("matplotlib", "cache", "tmp"):
        (SCRATCH / name).mkdir(parents=True, exist_ok=True)
    if not args.worker:
        modes = ("off", "on") if args.checkpoint == "both" else (args.checkpoint,)
        for mode in modes:
            subprocess.run([sys.executable, __file__, "--worker", "--device", "cpu", "--arm", args.arm,
                            "--steps", str(args.steps), "--batch", str(args.batch), "--checkpoint", mode], check=True)
        return
    import resource
    import time
    import torch
    from algorithms.mem_wm.backbones.kwm3d import KineticBlockworldModel
    torch.manual_seed(1729)
    torch.set_num_threads(8)
    model = KineticBlockworldModel(arm=args.arm, checkpoint_steps=args.checkpoint == "on")
    x = torch.rand(args.batch, 10, 3, 128, 128)
    target = torch.rand(args.batch, 5, 3, 128, 128)
    actions = torch.tensor([4, 2, 4, 4, 0, 4, 2, 1, 4, 4, 2, 4, 0, 4, 4]).expand(args.batch, -1)
    metadata = {"block_vel_agent": torch.rand(args.batch, 15, 10, 2) * 1.5 - .75,
                "block_valid": torch.ones(args.batch, 15, 10, dtype=torch.bool)}
    times = []
    for _ in range(args.steps):
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        output = model(x, actions[:, :10], actions[:, 10:], metadata=metadata)
        (output - target).square().mean().backward()
        times.append(time.perf_counter() - started)
    print(json.dumps(dict(arm=args.arm, checkpoint=args.checkpoint, batch=args.batch, context=10, prediction=5,
                          steps=args.steps, seconds_per_step=sum(times) / len(times), timings=times,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                          parameters=sum(p.numel() for p in model.parameters()), threads=8)), flush=True)


if __name__ == "__main__":
    main()
