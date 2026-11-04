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

## Step 2 (2026-11-03): K3D-int control split through our generator pipeline

Generator: `Miniworld-k3d` branch k3d (7e740ec), released dynamic command, 1000 clips of 500 frames, software
rendering (xvfb + Mesa llvmpipe, CPU), no depth. Released checkpoint, same evaluation command with the split swapped.

| 70 context + 70 predicted | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Released validation (anchor) | 0.000603 | 32.19 | 0.9673 |
| K3D-int v1 (base fork, unseeded, 1000 clips) | 0.000756 | 31.21 | 0.9664 |
| K3D-int v2 (branch k3d, seeds 20261103+idx, 999 clips) | 0.000680 | 31.67 | 0.9673 |

The dynamics match the released clips (action mix forward 58.6 % / no-op 31.8 % / left 7 % / right 2.6 %, block
counts 6–10, temporal change 0.125 vs 0.128). The remaining gap is appearance: over 40 clips our software-rendered
frames are 2 % brighter and 16 % less sharp (mean gradient 0.0040 vs 0.0047), consistent with the llvmpipe
multisample fallback (4 instead of 8/16 samples) and texture filtering. One seeded episode rendered through NVIDIA
EGL on GPU 1 reproduced the trajectory exactly and was 16 % sharper than its software render; the two renders differ
by MSE 9e-5. Whether the splits are rendered on the GPU is the user's decision (see the session report).
The v1 and v2 numbers also show the clip-sampling variance of a 1000-clip set (different episodes, same law and
renderer): about 10 %, so the released-vs-ours gap of 13 % on v2 is within about one such step.

### Renderer check (2026-11-03, user-approved NVIDIA rendering on GPU 1)

The same 200 seeded K3D-int episodes (seeds 20261103–20261302; trajectories, actions and block states bit-identical
across renderers) rendered with Mesa llvmpipe (CPU) and with NVIDIA EGL on GPU 1, released checkpoint:

| 200 paired episodes, 70 + 70 | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Software render (llvmpipe, 4 multisamples) | 0.000720 | 31.43 | 0.9672 |
| NVIDIA render (GPU 1) | 0.000608 | 32.16 | 0.9682 |
| Released validation, 1000 clips (anchor) | 0.000603 | 32.19 | 0.9673 |

The renderer explains the whole gap: NVIDIA-rendered K3D-int scores the anchor. All kinetic 3D splits are therefore
rendered with NVIDIA EGL on GPU 1 (EGL device 1 = CUDA device 1, verified; ≈ 200 MB per render process, eight
processes, gated on free memory). Rendering throughput ≈ 1.3 s per 500-frame clip with eight processes.
