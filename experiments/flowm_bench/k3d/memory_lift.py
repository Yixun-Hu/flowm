"""Diagnostic released-model context writes with its own evolving map."""
import time
import torch
from .common import clips, clip_id, ReadOnlyClips, load_model, write_csv, log
from .followup import memory_cache_path, append_result, table, MEM, mask_campaign


@torch.inference_mode()
def context_writes(model, frames, actions):
    """Released forward's context loop, returning post-gate max-V cone tokens.

    The public forward always stacks predictions and cannot take T_out=0.
    This context-only adapter mirrors its shift, ego roll, mixing and write.
    It performs no decoder call, prediction, or modification of model code.
    """
    assert frames.device.type == "cpu"
    assert model.v_channel_mixing_mode == "2d_v_channel_flattened"
    assert not model.no_self_motion_equiv
    assert model.map_processor.v_channel_id_embed is None
    processor = model.map_processor
    memory = processor.init_map(len(frames), frames.device)
    mask = model._get_fov_mask(frames.device)
    tokens = []
    for t in range(frames.shape[1]):
        memory = processor.shift_v_channels(memory)
        if t > 0:
            memory = processor.roll_map(memory, actions[:, t])
        if model.v_channel_mixing is not None:
            b,v,h,w,c = memory.shape
            x = memory.permute(0,4,1,2,3).contiguous().view(b,c*v,h,w)
            x = model.v_channel_mixing(x)
            memory = x.view(b,c,v,h,w).permute(0,2,3,4,1).contiguous()
        memory = processor(memory, frames[:, t], mask)
        written = processor.extract_view_tokens(memory, mask)[0]
        assert written.shape[1:] == (5,177,256)
        tokens.append(written.max(dim=1).values.half())
    return torch.stack(tokens,1)


@torch.inference_mode()
def main():
    # Run sequentially after patch scoring; this extra attribution check follows
    # up the inconclusive whole-region SSD, without overlapping eight-thread jobs.
    from .stationary_attribution import main as attribute_stationarity
    attribute_stationarity()
    start = time.perf_counter()
    model = load_model()
    torch.backends.mha.set_fastpath_enabled(False)
    rows = []
    for split,n in (("k3d_null_noblocks",50),("k3d_null_static",50),("k3d_v_validation",100)):
        paths = clips(split,n)
        dataset = ReadOnlyClips(paths)
        for lo in range(0,n,8):
            ids = [i for i in range(lo,min(lo+8,n)) if not memory_cache_path(paths[i]).exists()]
            if not ids:
                continue
            tick = time.perf_counter()
            items = [dataset[i] for i in ids]
            frames = torch.stack([i["videos"] for i in items])
            actions = torch.stack([i["conds"] for i in items]).long()
            tokens = context_writes(model,frames,actions)
            seconds = time.perf_counter()-tick
            for j,i in enumerate(ids):
                dest = memory_cache_path(paths[i])
                dest.parent.mkdir(parents=True,exist_ok=True)
                torch.save(dict(tokens=tokens[j].contiguous(),actions=actions[j],seconds=seconds/len(ids),
                                version=1,source=MEM),dest.with_suffix(".tmp"))
                dest.with_suffix(".tmp").replace(dest)
            log(f"memory {split} {lo+len(ids)}/{n} {seconds:.2f}s ({seconds/len(ids)/140:.4f}s/frame)")
        for path in paths:
            d = torch.load(memory_cache_path(path),weights_only=True)
            rows.append(dict(source=MEM,split=split,clip=clip_id(path),frames=140,seconds=d["seconds"]))
        write_csv("followup_memory_lift_clips.csv",rows)
        subset = [r for r in rows if r["split"] == split]
        summary = dict(source=MEM,split=split,clips=len(subset),frames=140*len(subset),
                       seconds=sum(r["seconds"] for r in subset))
        append_result("memory-lift-"+split,"### 4. Memory-consistent imprint cache\n\n" +
            table([summary],["source","split","clips","frames","seconds"]) +
            "\n\nReleased weights and original internal flow, ego transform, velocity mixing and gated write; all 140 frames are observed context. Imprint is the cone **after** `(1-g)*h_old + g*h_new`, max over all five velocity channels, before the decoder's shared positional encoding. No predicted frames or decoder calls. This uses model memory and is explicitly a diagnostic variant, not the design's observation-only lift. Cached fp16 under `.scratch/lifts_mem/`.")
        # Persist each split's measurements as soon as its cache exists.
        from .followup import run, null_score, gate_score
        run(paths,source="mem")
        if "null" in split:
            null_score(paths,"mem",1.)
        else:
            gate_score(paths,"mem",1.)
    seconds = time.perf_counter()-start
    write_csv("followup_memory_time.csv",[dict(seconds=seconds)])
    log(f"DONE memory {seconds:.2f}s")


if __name__ == "__main__":
    main()
