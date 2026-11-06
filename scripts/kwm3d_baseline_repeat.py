#!/usr/bin/env python3
"""CPU repeat-last baseline and reported-only references in one decode pass."""
import argparse
import json
import time
if __package__:
    from .kwm3d_runtime import setup, compose_pilot, ROOT
else:
    from kwm3d_runtime import setup, compose_pilot, ROOT


def predictors(context):
    """One fixed RGB frame per predictor; context is [B,70,C,H,W] in [0,1]."""
    import torch
    last = context[:, -1:]
    return dict(repeat=last, context_mean=context.mean(1, keepdim=True),
                last_flat=last.mean((-2, -1), keepdim=True).expand_as(last),
                gray=torch.full_like(last, .5))


def main():
    setup()
    import torch
    from datasets.video.blockworld import BlockWorldVideoDataset
    from algorithms.mem_wm.backbones.kwm3d.per_clip import PerClipDump, clip_stats
    from algorithms.mem_wm.backbones.kwm3d.metrics import block_regions
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", type=int, default=1000)
    p.add_argument("--output", default=str(ROOT / "experiments/flowm_bench/k3d/results"))
    args = p.parse_args()
    torch.set_num_threads(8)
    started = time.perf_counter()
    cfg = compose_pilot("infer/metrics_280/k3d_v_70ctx", [f"dataset.num_default_clips={args.clips}"])
    dataset = BlockWorldVideoDataset(cfg.dataset, split="k3d_v_validation", purpose="validation")
    assert len(dataset) == args.clips
    dumps = {(name, h): PerClipDump(70, h)
             for name in ("repeat", "context_mean", "last_flat", "gray") for h in (140, 280)}
    with torch.inference_mode():
        for i in range(len(dataset)):
            item = torch.utils.data.default_collate([dataset[i]])
            frames = predictors(item["videos"][:, :70])
            metadata = {k: v[:, 70:280] for k, v in item["metadata"].items() if k.startswith("block_")}
            regions = block_regions(metadata, item["conds"][:, 70:280], item["videos"].shape[-2:])
            for (name, horizon), dump in dumps.items():
                target = item["videos"][:, 70:horizon]
                pred = frames[name].expand_as(target)
                masks = [region[:, :horizon-70] for region in regions]
                dump.append(clip_stats(pred, target, metadata, item["conds"][:, 70:horizon], regions=masks),
                            item["metadata"])
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(dataset)} clips, {time.perf_counter()-started:.1f}s", flush=True)
    from pathlib import Path
    for (name, horizon), dump in dumps.items():
        prefix = "baseline" if name == "repeat" else "reference"
        directory = Path(args.output) / f"pilot_{prefix}_{name}_{horizon}"
        data = dump.write(directory)
        print(json.dumps(dict(predictor=name, horizon=horizon, clips=args.clips, output=str(directory),
                              pooled=dict(zip(data["metric_names"], data["pooled"].tolist())))), flush=True)
    print(f"Wall seconds: {time.perf_counter()-started:.3f}")


if __name__ == "__main__":
    main()
