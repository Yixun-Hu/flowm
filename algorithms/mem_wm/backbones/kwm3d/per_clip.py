"""Portable per-clip RGB SSE/counts. Inputs contain predicted frames only."""
import csv
import json
from pathlib import Path

import torch
from .metrics import METRIC_NAMES, block_region_splits

NAMES = ("mse", *METRIC_NAMES)


def clip_stats(pred, target, metadata, actions):
    pred, target = [torch.nan_to_num(x).clamp(0, 1).float() for x in (pred, target)]
    stats = block_region_splits(pred, target, metadata, actions, return_stats=True, per_clip=True)
    stats["mse"] = ((pred - target).square().flatten(1).sum(1, dtype=torch.float64),
                    torch.full((len(pred),), pred[0].numel(), dtype=torch.int64, device=pred.device))
    return stats


def pooled(dump):
    e, n = dump["squared_error"].sum(0), dump["element_count"].sum(0)
    return torch.where(n > 0, e / n.clamp_min(1), torch.nan)


class PerClipDump:
    def __init__(self, context, horizon):
        self.context, self.horizon = context, horizon
        self.paths, self.indices, self.errors, self.counts = [], [], [], []

    def append(self, stats, metadata):
        self.paths.extend(metadata["path"])
        self.indices.extend(torch.as_tensor(metadata["dataset_index"]).cpu().tolist())
        self.errors.append(torch.stack([stats[k][0] for k in NAMES], 1).detach().cpu().double())
        self.counts.append(torch.stack([stats[k][1] for k in NAMES], 1).detach().cpu().long())

    def data(self):
        errors, counts = torch.cat(self.errors), torch.cat(self.counts)
        if len(set(zip(self.paths, self.indices))) != len(self.paths):
            raise ValueError("Duplicate clip identity in validation dump")
        return dict(version=1, metric_names=list(NAMES), paths=self.paths,
                    dataset_indices=torch.tensor(self.indices), context_frames=self.context,
                    n_frames=self.horizon, squared_error=errors, element_count=counts,
                    block_pixel_count=counts[:, 1:] // 3)

    def write(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = self.data()
        data["pooled"] = pooled(data)
        torch.save(data, directory / "per_clip.pt")
        columns = ["dataset_index", "path"]
        columns += [f"{k}_{suffix}" for k in NAMES for suffix in ("squared_error", "element_count", "value")]
        columns += [f"{k}_pixel_count" for k in METRIC_NAMES]
        with (directory / "per_clip.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for i, path in enumerate(data["paths"]):
                row = [self.indices[i], path]
                for e, n in zip(data["squared_error"][i], data["element_count"][i]):
                    row.extend((float(e), int(n), float(e / n) if n else float("nan")))
                row.extend(data["block_pixel_count"][i].tolist())
                writer.writerow(row)
        (directory / "pooled.json").write_text(json.dumps(dict(zip(NAMES, data["pooled"].tolist())), indent=2) + "\n")
        return data
