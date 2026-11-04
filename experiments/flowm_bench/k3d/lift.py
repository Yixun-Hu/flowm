"""Observation-only released-weight writes, pre-gate tokens and reusable fp16 caches."""
import argparse
import time
import torch
from kwm.cells import _fixed_projection
from .geometry import cone_mask, BOX_SHAPE
from .common import SCRATCH, clips, clip_id, ReadOnlyClips, load_model, write_csv, log

CACHE_VERSION = 1


@torch.inference_mode()
def observation_lift(model, frames):
    assert frames.device.type == "cpu"
    processor = model.map_processor
    assert processor.v_channel_id_embed is None
    saved = {}

    def gate_hook(module, inputs, output):
        saved["g"] = output
        saved["h_new"] = inputs[0][..., 256:]

    handle = processor.update_gate.register_forward_hook(gate_hook)
    mask = model._get_fov_mask(frames.device)
    try:
        written = processor(processor.init_map(len(frames), frames.device), frames, mask)
        tokens = processor.extract_view_tokens(written, mask)[0]
        saved["tokens"] = tokens
        for name, value in saved.items():
            assert torch.allclose(value, value[:, :1].expand_as(value), atol=2e-5, rtol=2e-5), name
        assert torch.allclose(tokens, saved["g"] * saved["h_new"], atol=1e-6)
        return {k: v[:, 0].contiguous() for k, v in saved.items()}
    finally:
        handle.remove()


def tokens_to_box(tokens):
    """[N,177,C] -> [N,C,19,17], returning the exact boolean cone support."""
    support = cone_mask()[7:26, 16:33]
    box = tokens.new_zeros(len(tokens), tokens.shape[-1], *BOX_SHAPE)
    box[:, :, support] = tokens.transpose(1, 2)
    return box, support


def box_to_tokens(box):
    return box[:, :, cone_mask()[7:26, 16:33]].transpose(1, 2)


def projection(k=8):
    assert k in (8, 32)
    return _fixed_projection(256, k)


def projected_box(tokens, k=8):
    return tokens_to_box(tokens.float() @ projection(k).T)[0]


def cache_path(path):
    split = path.parent.parent.name
    return SCRATCH / "lifts" / split / (clip_id(path).replace("/", "_") + ".pt")


def read_cache(path):
    data = torch.load(cache_path(path), map_location="cpu", weights_only=True)
    assert data["version"] == CACHE_VERSION and data["tokens"].shape == (140, 177, 256)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="k3d_v_validation")
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()
    paths = clips(args.split, args.clips)
    dataset = ReadOnlyClips(paths)
    model = load_model()
    # SDPA avoids materializing the large multihead score bank on CPU.
    torch.backends.mha.set_fastpath_enabled(False)
    rows = []
    for i, path in enumerate(paths):
        start = time.perf_counter()
        dest = cache_path(path)
        if dest.exists():
            data = read_cache(path)
            rows.append(dict(clip=clip_id(path), seconds=data["seconds"], cached=True))
            continue
        item = dataset[i]
        x = item["videos"]
        batches = [observation_lift(model, x[t:t+args.batch_size]) for t in range(0, 140, args.batch_size)]
        data = {k: torch.cat([b[k] for b in batches]).half() for k in ("tokens", "g", "h_new")}
        elapsed = time.perf_counter() - start
        data.update(actions=item["conds"].long(), version=CACHE_VERSION, seconds=elapsed)
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, dest.with_suffix(".tmp"))
        dest.with_suffix(".tmp").replace(dest)
        rows.append(dict(clip=clip_id(path), seconds=elapsed, cached=False))
        log(f"lift {args.split} {i+1}/{len(paths)} {clip_id(path)} {elapsed:.2f}s")
        write_csv(f"lift_{args.split}.csv", rows)
    write_csv(f"lift_{args.split}.csv", rows)


if __name__ == "__main__":
    main()
