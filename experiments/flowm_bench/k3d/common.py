"""Shared CPU setup and read-only access through the original dataset methods."""
import csv
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf
from datasets.video.blockworld import BlockWorldVideoDataset
from algorithms.mem_wm.backbones.flowm.flowm_models_3d import ViTBlockworldModel

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCRATCH = HERE / ".scratch"
RESULTS = HERE / "results"
CHECKPOINT = ROOT / "downloaded_checkpoints/hf_models/blockworld-models/blockworld/dynamic/flowm/v2_dynamic.ckpt"
torch.set_num_threads(8)


def clips(split, n=None):
    paths = sorted((ROOT / "data/blockworld" / split).glob("*/*_rgb.mp4"),
                   key=lambda p: (int(p.parent.name), int(p.stem.split("_")[0])))
    if n is not None and len(paths) < n:
        raise ValueError(f"{split}: requested {n}, found {len(paths)}")
    return paths[:n]


def clip_id(path):
    return f"{path.parent.name}/{path.stem.removesuffix('_rgb')}"


def companion(path, suffix):
    return path.with_name(path.name.replace("rgb.mp4", suffix))


def truth(path):
    return torch.load(companion(path, "block_info.pt"), map_location="cpu", weights_only=False)


class ReadOnlyClips(BlockWorldVideoDataset):
    """Use original __getitem__, decode, resize and action alignment, no NAS indexing.

    Only metadata construction and clip selection differ from the parent: choose
    frame zero deterministically and never call its writable metadata builder.
    """
    def __init__(self, paths, frames=140):
        self.cfg = OmegaConf.create(dict(cond_alignment="t-1->t", data_cond_alignment="t->t+1"))
        self.resolution = [128, 128]
        self.transform = self.build_transform()
        self.cond_loading_style = "action_int"
        self.external_cond_dim = 5
        self.n_frames = frames
        self.frame_skip = 1
        self.use_depth = self.use_preprocessed_latents = self.is_latent_preprocessing_expt = False
        self.purpose, self.split = "validation", "validation"
        self.metadata = []
        for p in paths:
            a = torch.load(companion(p, "actions.pt"), map_location="cpu", weights_only=False)
            self.metadata.append(dict(video_paths=p, video_pts=torch.arange(len(a["actions"]))))

    def get_clip_location(self, idx):
        return idx, 0

    def __len__(self):
        return len(self.metadata)


def load_model():
    model = ViTBlockworldModel(input_shape=(3, 128, 128), world_size=16, embed_dim=256,
                              img_patch_size=16, map_proc_encoder_depth=6, dec_depth=6,
                              num_heads=8, fov_deg=60, v_range=1,
                              v_channel_maxpool_decode=True,
                              v_channel_mixing_mode="2d_v_channel_flattened",
                              v_channel_identity_emb=False)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    assert len(state) == 240 and all(k.startswith("main_model.") for k in state)
    model.load_state_dict({k.removeprefix("main_model."): v for k, v in state.items()}, strict=True)
    return model.eval().requires_grad_(False)


def write_csv(name, rows):
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with (RESULTS / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)

