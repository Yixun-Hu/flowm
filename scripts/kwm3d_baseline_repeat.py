#!/usr/bin/env python3
"""CPU repeat-last-context baseline using the pilot dataset and metric code."""
import argparse
import json
import time
from kwm3d_runtime import setup, compose_pilot, ROOT


def main():
    setup()
    import torch
    from datasets.video.blockworld import BlockWorldVideoDataset
    from algorithms.mem_wm.backbones.kwm3d.per_clip import PerClipDump, clip_stats
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clips", type=int, default=1000)
    p.add_argument("--output", default=str(ROOT / "experiments/flowm_bench/k3d/results"))
    args = p.parse_args()
    torch.set_num_threads(8)
    started = time.perf_counter()
    cfg = compose_pilot("infer/metrics_280/k3d_v_70ctx", [f"dataset.num_default_clips={args.clips}"])
    dataset = BlockWorldVideoDataset(cfg.dataset, split="k3d_v_validation", purpose="validation")
    assert len(dataset) == args.clips
    dumps = {h: PerClipDump(70, h) for h in (140, 280)}
    with torch.inference_mode():
        for i in range(len(dataset)):
            item = torch.utils.data.default_collate([dataset[i]])
            for horizon, dump in dumps.items():
                target = item["videos"][:, 70:horizon]
                pred = item["videos"][:, 69:70].expand_as(target)
                metadata = {k: v[:, 70:horizon] for k, v in item["metadata"].items() if k.startswith("block_")}
                dump.append(clip_stats(pred, target, metadata, item["conds"][:, 70:horizon]), item["metadata"])
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(dataset)} clips, {time.perf_counter()-started:.1f}s", flush=True)
    from pathlib import Path
    for horizon, dump in dumps.items():
        directory = Path(args.output) / f"pilot_baseline_repeat_{horizon}"
        data = dump.write(directory)
        print(json.dumps(dict(horizon=horizon, clips=args.clips, output=str(directory),
                              pooled=dict(zip(data["metric_names"], data["pooled"].tolist())))), flush=True)
    print(f"Wall seconds: {time.perf_counter()-started:.3f}")


if __name__ == "__main__":
    main()
