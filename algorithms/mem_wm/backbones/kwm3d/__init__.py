"""Opt-in kinetic 3D backbone factory."""
import ast
from .kwm3d_models import KineticBlockworldModel, parameter_counts

__all__ = ["create_kwm3d_model", "KineticBlockworldModel", "parameter_counts"]


def create_kwm3d_model(backbone_cfg):
    cfg = backbone_cfg
    if cfg.get("frame_skip", 1) != 1:
        raise ValueError("kwm3d dynamics require frame_skip=1; skipped ego actions are not composed")
    shape = cfg.get("input_shape", (cfg.get("input_channels", 3), 128, 128))
    if isinstance(shape, str):
        shape = ast.literal_eval(shape)
    options = {key: cfg[key] for key in
               ("arm", "lattice", "map_conv", "write", "checkpoint_steps", "imprint_k",
                "sigma", "step_clip", "s_init", "oracle_velocity", "residual_mode") if key in cfg}
    return KineticBlockworldModel(
        input_shape=tuple(shape), world_size=cfg.get("world_size", 16),
        embed_dim=cfg.get("hidden_channels", 256), img_patch_size=cfg.get("patch_size", 16),
        map_proc_encoder_depth=cfg.get("map_proc_encoder_depth", 6),
        dec_depth=cfg.get("decoder_depth", 6), num_heads=cfg.get("num_heads", 8),
        fov_deg=cfg.get("fov_deg", 60.0), **options)
