"""Strict checkpoint and dataset-alignment sanity check (70 context + 70 rollout)."""
import argparse
import time
import torch
from .common import clips, clip_id, ReadOnlyClips, load_model, write_csv, log


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    start = time.perf_counter()
    paths = clips("k3d_int_validation", args.clips)
    dataset = ReadOnlyClips(paths)
    model = load_model()
    rows = []
    for lo in range(0, len(paths), args.batch_size):
        batch = [dataset[i] for i in range(lo, min(lo + args.batch_size, len(paths)))]
        x = torch.stack([b["videos"] for b in batch])
        a = torch.stack([b["conds"] for b in batch]).long()
        log(f"sanity batch {lo}: x={tuple(x.shape)}, range={x.min():.3f}..{x.max():.3f}")
        y = model(x[:, :70], a[:, :70], a[:, 70:])
        mse = (y.clamp(0, 1) - x[:, 70:]).square().mean((1, 2, 3, 4))
        raw = (y - x[:, 70:]).square().mean((1, 2, 3, 4))
        frame_error = (y.clamp(0,1) - x[:,70:]).square().mean((2,3,4))
        for j in range(len(batch)):
            turn = (a[j,70:] == 0) | (a[j,70:] == 1)
            rows.append(dict(clip=clip_id(paths[lo+j]), mse=mse[j].item(), raw_mse=raw[j].item(),
                             turn_mse=frame_error[j,turn].mean().item(),
                             nonturn_mse=frame_error[j,~turn].mean().item(),
                             turn_frames=int(turn.sum()), nonturn_frames=int((~turn).sum())))
    elapsed = time.perf_counter() - start
    mean = sum(r["mse"] for r in rows) / len(rows)
    write_csv("released_sanity.csv", rows)
    write_csv("released_sanity_summary.csv", [dict(clips=len(rows), mse=mean, seconds=elapsed,
                                                  passed=mean < 0.003)])
    log(f"sanity MSE={mean:.9f}, seconds={elapsed:.1f}")
    assert mean < 0.003, "Loading/alignment sanity failed; do not proceed."


if __name__ == "__main__":
    main()
