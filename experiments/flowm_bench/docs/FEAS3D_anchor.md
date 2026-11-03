# 3D feasibility, step 1: released FloWM-3D reproduced on Blockworld (2026-11-02)

Data: `flowm123/blockworld`, config `dynamic` (split `sunday_v2_validation`, 1000 clips, 128×128 RGB, 16 shards,
32 GB snapshot on the NAS under `data/_hf_snapshots/blockworld`, extracted to `data/blockworld/sunday_v2_validation`).
Weights: `flowm123/blockworld-models` (released `dynamic/flowm/v2_dynamic.ckpt`, 12.8 M parameters).
Command (GPU 1, batch 16, ~18 min for 1000 clips):

```
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -m main shortcode=exp/blockworld/flowm/infer/metrics_140/dynamic_70ctx \
  +name=flowm3d_released_dynamic_70ctx_1000 algorithm=flowm_video dataset=blockworld ckpt_map=default \
  wandb.mode=offline wandb.entity=local wandb.project=flowm_bench dataset.num_default_clips=1000 experiment.devices=1 \
  'algorithm.logging.metrics=[mse,psnr,ssim]' experiment.validation.batch_size=16
```

| 70 context + 70 predicted, dynamic | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Paper, Table 2 | 0.000603 | 32.19 | 0.9673 |
| Our run of the released weights | 0.000603 | 32.19 | 0.9673 |

Exact reproduction (FVD and LPIPS skipped: no offline I3D/LPIPS weights, as in Phase 0).

## What Blockworld is (for the kinetic 3D design)

15×15 grid room, agent camera at 128×128, 6–10 blocks. Block velocities are integer: one cell per step along x or y,
reversing at walls; no block–block or block–agent collisions. Actions are discrete: turn left/right 90°, move forward
one cell, no-op. FloWM-3D keeps a latent map (V = 5 integer velocity channels × 33 × 33 × 256), rolls each channel by
its velocity every step, and handles turns by rotating the map 90° and cyclically permuting the velocity channels
(its "self-motion equivariance"), i.e. exactly the C4 structure PR6 adds to the kinetic model. The generator is
external (`hlillemark/Miniworld`); it is not in the FloWM repository.

## Next (needs the user's decisions)

A kinetic 3D benchmark = Blockworld with continuous block velocities (and optionally constant accelerations), still
with the discrete agent actions so that the released FloWM-3D remains a valid baseline. Design choices to settle:
velocity law (uniform in a continuous range vs. speed classes), whether blocks accelerate, bounce rule for
non-integer positions, camera model (keep discrete turns/forward, or add continuous ego motion), clip length and
number, and whether to render with the Miniworld engine (depends on obtaining and modifying that repo).
