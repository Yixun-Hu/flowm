# flowm_bench: our continuous-velocity memory against FloWM on FloWM's benchmark

Status (2026-10-24): reconnaissance and Phase 0 complete; plan v1 reviewed and
being revised; generator re-implementation in progress. Nothing here modifies
the FloWM framework (`algorithms/`, `datasets/`, `configurations/`) or the
parent study (`experiments/continuous_velocity/`).

## Phase 0: FloWM reproduced on its own benchmark

Released checkpoint `dynamic_po_flowm.pt`, official pipeline, all 8000
`dynamic_validation` clips, 50 context + 20 predicted frames:

| | MSE | PSNR | SSIM |
|---|---:|---:|---:|
| Paper (Table 1) | 0.0005 | 32.99 | 0.9900 |
| Reproduction | 0.000502 | 32.99 | 0.9900 |

Command (offline wandb; `configurations/secrets/my_secrets.yaml` holds a dummy
key and is git-ignored):

```bash
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python -m main \
  shortcode=exp/mnist_world/flowm/infer/dynamic_po_70wf +name=flowm_released_70wf_8000 \
  algorithm=flowm_video dataset=mnist_world ckpt_map=default \
  wandb.mode=offline wandb.entity=local wandb.project=flowm_bench \
  dataset.num_default_clips=8000 experiment.devices=1 \
  'algorithm.logging.metrics=[mse,psnr,ssim]' experiment.validation.batch_size=64
```

Pipeline notes found on the way: `pytz` must be installed; the clip count is
`dataset.num_default_clips` (the `num_validation_clips` key is not read for
these split names); the released `dynamic_po_200wf` shortcode inherits
`v_range: 0` and `no_self_motion_equivariance: true` from `base_inference.yaml`
and must be overridden with `algorithm.backbone.v_range=2
algorithm.backbone.no_self_motion_equivariance=false`; the metadata cache
`data/mnist_world/metadata/<split>.pt` is never rebuilt once present.

## Documents

- `docs/PLAN_v1.md`: first plan (two tracks: released splits; fractional and
  accelerated generalisation of their generator).
- `docs/REVIEW_plan_v1_opus.md`: adversarial review. Main conclusions: on the
  released integer, constant-velocity splits the three contributions (Newton
  refinement, spectral transport, second-order motion) have nothing to act on;
  compute was underestimated about tenfold; KWM (kinetic-world-models) already
  defines the continuous-motion extension of this benchmark and must be a
  baseline. The plan is being revised to build on KWM (cloned beside this
  repository) with our components added as switchable ablations.

## Benchmark facts (verified against the downloaded data)

`dynamic_po`: 50x50 torus, 32x32 window (top-left = camera centre - 16), 5
MNIST-train digits composited additively with clip, integer velocities in
{-1,0,1}^2, camera actions target uniform unseen pixels with wrapped steps clipped to [-10,10]^2 until full window coverage, then iid uniform integers (actions 0 and 50 are zero), 180k training /
8000 validation episodes of 70 frames (and 200-frame validation), mp4 (x264
crf 10, 8 fps) + .pt metadata with labels, positions and velocities. Metrics
are torchmetrics MSE/PSNR/SSIM pooled over non-context frames.
