#!/usr/bin/env python3
"""Hydra-model rollout/backward/AdamW timing, in a fresh process per mode."""
import argparse
import json
import subprocess
import sys
from kwm3d_runtime import setup, compose_pilot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--arm', choices=['c', 'c_pw', 'a_oracle', 'a', 'b'], required=True)
    p.add_argument('--steps', type=int, default=3, help='timed optimizer steps after warmup')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--context', type=int, default=10)
    p.add_argument('--prediction', type=int, default=5)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--accumulate', type=int, default=1)
    p.add_argument('--checkpoint', choices=['both', 'on', 'off'], default='both')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.steps, args.batch, args.context, args.prediction, args.accumulate) < 1 or args.warmup < 0:
        p.error('steps, batch, context, prediction, accumulate must be positive; warmup nonnegative')
    setup(args.device)
    if not args.worker:
        for mode in ('off', 'on') if args.checkpoint == 'both' else (args.checkpoint,):
            cmd = [sys.executable, __file__, '--worker']
            for key, value in vars(args).items():
                if key != 'worker':
                    cmd.extend(['--' + key, str(mode if key == 'checkpoint' else value)])
            subprocess.run(cmd, check=True)
        return
    import resource
    import statistics
    import time
    import torch
    from omegaconf import OmegaConf
    from algorithms.mem_wm.backbones.kwm3d import create_kwm3d_model
    torch.manual_seed(0)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")  # same setting as experiments/base_exp.py
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable under CUDA_VISIBLE_DEVICES')
    cfg = compose_pilot('train/pilot_c', [f'algorithm.backbone.arm={args.arm}',
        f'algorithm.backbone.checkpoint_steps={str(args.checkpoint == "on").lower()}'])
    model = create_kwm3d_model(cfg.algorithm.backbone).to(args.device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.experiment.training.lr,
                             betas=tuple(cfg.experiment.training.optim.optimizer_beta),
                             weight_decay=cfg.experiment.training.weight_decay)
    total = args.context + args.prediction
    x = torch.rand(args.batch, args.context, 3, 128, 128, device=args.device)
    target = torch.rand(args.batch, args.prediction, 3, 128, 128, device=args.device)
    actions = torch.tensor(([4, 2, 4, 4, 0, 4, 2, 1] * ((total+7)//8))[:total], device=args.device).expand(args.batch, -1)
    metadata = {'block_vel_object_incoming': torch.rand(args.batch, total, 10, 2, device=args.device)*1.5-.75,
                'block_valid': torch.ones(args.batch, total, 10, dtype=torch.bool, device=args.device)}
    sync = (lambda: torch.cuda.synchronize()) if args.device == 'cuda' else (lambda: None)
    times = []
    for step in range(args.warmup + args.steps):
        optim.zero_grad(set_to_none=True)
        sync()
        if step == args.warmup and args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        for _ in range(args.accumulate):
            output = model(x, actions[:, :args.context], actions[:, args.context:],
                           target_seq=target, teacher_forcing_ratio=cfg.algorithm.teacher_forcing, metadata=metadata)
            ((output-target).square().mean()/args.accumulate).backward()
            del output
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.experiment.training.optim.gradient_clip_val)
        optim.step()
        sync()
        if step >= args.warmup:
            times.append(time.perf_counter()-started)
    result = dict(arm=args.arm, checkpoint=args.checkpoint, device=args.device, batch=args.batch,
                  accumulate=args.accumulate, context=args.context, prediction=args.prediction,
                  steps=args.steps, warmup=args.warmup, seconds_per_step_mean=statistics.mean(times),
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  seconds_per_step_median=statistics.median(times), timings=times,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  backbone=OmegaConf.to_container(cfg.algorithm.backbone, resolve=True))
    if args.device == 'cuda':
        result.update(max_memory_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      max_memory_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
