# Kinetic FloWM-3D (`kwm3d`)

Five opt-in backbones share FloWM's image patchifier, positional encodings,
attention blocks, sigmoid write gate and image decoder **by inheritance/import**.
The existing FloWM backbone remains available and requires no KWM dependency.
The default map is `[B,V,33,33,256]`, with a 177-cell cone at rows 7–25,
columns 16–32. Images are 128×128 with 64 patch tokens.

| Arm | Nodes | Transport | Write attention | Map convolution | Parameters |
| --- | --- | --- | --- | --- | ---: |
| C (`c`) | v5 | original integer rolls | 5×177 + 64 | original cross-V grouped conv | 12,830,976 |
| C-pw (`c_pw`) | v5 | original integer rolls | 177 + 64 | original cross-V grouped conv | 12,830,976 |
| A-oracle (`a_oracle`) | static + ten padded block identities | spectral, per sample | 177 + 64 | shared C4 depthwise | 12,774,144 |
| A (`a`) | v17 + adaptive offsets; v9 optional | spectral, per sample | 177 + 64 | shared C4 depthwise | 12,774,145 |
| B (`b`) | fixed v9 | spectral | 177 + 64 | shared C4 depthwise | 12,774,144 |

`parameter_counts()` prints these counts. Parameter counts exclude fixed buffers.
`named_nodes("v17")` is static + `{±0.25,±0.75}²`; v9 is `{−1,0,1}²`.
`map_conv: plain` selects a shared circular depthwise 3×3 ablation for the
kinetic arms. Both convolution choices start as an identity. C and C-pw retain
the original FloWM initialization unless explicitly loading screening weights.
`write: auto` chooses the arm's prescribed write; incompatible combinations fail.

## Dependency and source reuse

The tested editable dependency is `/home/yixunhu/codespace/kwm-wt-k3d`, branch
`k3d-base`, commit **2e73903d7ef5e6268cf6bca3a9703a3b9971ccf9**. It supplies
`stream`, `named_nodes`, `rotation_perms`, `rot_vec`, `C4SymmetricDepthwise`,
`oracle_nodes_from`, `_fixed_projection`, `LockState`, and `lock_measurement`.
No KWM or FloWM source has been vendored. Keep this dependency pinned when
reproducing the numbers; the private lock adapter depends on this version.

`_LockArithmetic` overrides the installed `AdaptiveNodeCell._measure_lock`
hook to pass the cone, box and residual options, and calls the inherited
`_forward_lock` for its exact update arithmetic. A one-cell dummy feature bank
bypasses its unrelated CNN recurrence; FloWM owns the actual feature recurrence.
The hook first appeared in `94803f5`; the pinned HEAD adds mask residuals and
all-node diagnostics. No module globals, function bindings or external source
files are modified.

## Writes, turns and estimation

C calls the original `MapLatentProcessor.forward`, `shift_v_channels` and
`roll_map`. Its cross-V convolution retains the original feature-major grouping.
For all other arms, max pooling over valid nodes yields 177 map tokens. Map
position encodings and the 64 encoded image patches enter the inherited joint
attention stack. Its shared proposal is written to every node with
`sigmoid(W[old_i, proposal])`. Invalid oracle nodes have zero features, velocity
and gate, and are excluded from both write pooling and decode pooling with
`-inf` masks. Static node zero is always valid.

Kinetic transport calls `stream` on `[B,V,C,H,W]`, with `c4=True`. Forward ego
motion subtracts `(1,0)`. Transport precedes turns, matching FloWM's ordering.
A left turn is `rot90(k=-1)` about `(16,16)`, followed by the inverse gather of
`rotation_perms`; offsets rotate with `rot_vec`. Evidence and lock bits follow
the node permutation. These are `rot_state` semantics specialized to **scalar**
features. The fork's literal `rot_state` permutes orientation channels and
assumes square snapshots, so calling it directly would be incorrect here.
Oracle channels track persistent block identities, with no lattice permutation;
destination-frame oracle velocities rotate back for pre-turn transport.

`geometry.RIGHT_ROW_SIGN = +1` is the single conversion constant. MiniWorld's
left turn sends an object ahead to the camera's right; FloWM's clockwise map
turn sends +column to +row. The analytic camera projection also places positive
agent-right positions on the right side of the RGB frame, confirmed in the mask
montage. A test checks this constant against the independently maintained
stage-1 `geometry.py` (skipping only when that file is absent).

A's estimator uses **only an observation imprint**: the inherited processor's
pooled write on a zero map, projected with fixed `_fixed_projection(C,K)`
(default K=8), cropped to the cone's 19×17 bounding box. One zero node is
algebraically equivalent to V identical zero nodes for pooled attention, and
avoids retaining redundant lift activations. Memory features do not enter this
estimator. `lock_measurement` uses `box=(7,16)`, the exact cone support,
`static_residual=residual_mode`, `sigma=1.0` (configurable) and `radius=None`.
`residual_mode: mask` is the default: ego-only change weights the plain matching
pair. `residual_mode: subtract` retains the old dipole residual as an ablation.
The fork's diagnostics expose estimates, evidence before mode selection and
lock bits for all nodes. Both modes are exercised through the backbone hook.

The installed cell supplies the exact fp32 lock update: cumulative evidence,
evidence-weighted gain, learned scalar `s` (initially 1), clipped step (default
0.3), offset bound 0.75, and static-node pinning. Turn steps supply no velocity
measurement across the turn, but retain their post-turn imprint as the next
snapshot: pair (t−1,t) is skipped and pair (t,t+1) remains valid. During prediction,
rho/evidence/lock/snapshot receive no measurements or writes. Turns still change
rho's coordinates and node labels. Decoded frames feed the feature recurrence.

Every arm uses the same `_step` call through non-reentrant per-step checkpointing
when gradients are enabled and `checkpoint_steps: true`. The teacher-forcing
coin is drawn outside this call. Evaluation under `no_grad()` avoids checkpoint
bookkeeping. The decoder takes max-over-valid-V cone tokens plus map positions.
A's observation lift is checkpointed separately; the measurement and lock update
run outside checkpointing. KWM evaluates search, clustering, gates and lock bits
once under `no_grad`, so backward cannot choose different discrete decisions.
The selected LK displacement derivative and learned scale `s` still retain their
graphs: putting the entire measurement under `no_grad` would remove an existing
gradient. This keeps a small estimator graph per measured pair in addition to
the saved feature maps. Tests assert no measurement calls during backward.

Cone bounds are cached at construction. The flattened search-mask key is also
prepared there, and passed when the dependency exposes `support_mask_key`.
Pinned KWM `2e73903` does **not** expose that argument, so its `LocalSearch`
still copies the mask to CPU per measurement. The minimal upstream proposal is
`.scratch/kwm_mask_key.diff`; it was not applied to the read-only dependency.

## Oracle metadata convention and maintainer decision

**The current generator contradicts the brief's interpretation of
`block_vel_agent`.** `generate_videos.py` lines 1730–1755 compute outgoing
`t→t+1` displacement and subtract camera displacement before writing that field.
`stream` also subtracts camera displacement. Consequently the requested literal
mode double-subtracts ego translation and is one transition out of alignment for
an incoming-step recurrence. This matters especially at turns and bounces.

The Python and configuration default **`oracle_velocity: incoming_object`** uses
`block_vel_world[t-1]`, expressed using destination heading `agent_dir[t]`, and
lets `stream` subtract ego motion once. The opt-in dataset loader supplies this
as `block_vel_object_incoming`; the requested raw metadata fields stay unchanged.
The first global frame has zero incoming displacement. Missing corrected
metadata raises an error, without silently falling back to raw velocities.
Both conventions mask padded slots identically and read metadata during
prediction as well as context. Maintainer decision (2026-11-04): `incoming_object` is the configured default; `raw` stays available as a
diagnostic only. The stage 1 gate reports both conventions as `literal` and `intrinsic_incoming`.

## Data, metrics and initialization

`dataset.load_block_info: false` keeps the previous outputs. When enabled,
`<idx>_block_info.pt` supplies `block_pos_agent [T,10,2]`,
`block_vel_agent [T,10,2]` and boolean `block_valid [T,10]` inside `metadata`.
The loader follows the actual video indices, clip start, purpose-dependent
`max_frames`/`n_frames`, padding and frame skip; temporal padding is invalid.
It also supplies `block_speed`, corrected incoming velocities and
`block_incoming_valid` with fixed keys. Missing block files produce all-zero
tensors with `block_valid=False`, so legacy and K3D samples collate together.
Older block files lacking speed use the prior velocity-norm fallback. Missing
incoming velocities remain zero with `block_incoming_valid=False`; the oracle
rejects such slots if `block_valid=True`, preserving the missing-data error.
Default PyTorch collation produces `[B,T,...]` tensors.
The new training split is registered without caching an absent split's empty
index. Opt-in integer-action padding/skipping avoids two pre-existing shape
errors; the disabled path retains its behavior. Dataset alignment supports
frame skipping, but the model factory rejects `frame_skip != 1`: composing
multiple ego actions and dynamics across skipped frames is not implemented.

`FloWMVideo` inspects the backbone's `forward` signature and only passes
`metadata=` to backbones declaring it. `algorithm.logging.metrics: [...,
block_mse]` enables extra distributed sum/count metrics without changing inputs
to the existing metrics. Only prediction frames count; `n_metrics_frames` is
respected. RGB predictions are clamped to [0,1], matching the existing logger.
Metric values are pixel-weighted, not averages of unequal-area batch means.
Empty regions report NaN. Logged keys are
`validation/prediction/block_mse`, its three `block_mse_speed_LO_HI` splits,
and `block_mse_turn` / `block_mse_nonturn`. Turn labels use the incoming action
at the predicted step. Speed bins are [0.25,0.5), [0.5,0.75), [0.75,1.0].
Generator `block_speed` is used when available; the generic helper falls back
to the supplied velocity norm for synthetic/custom metadata.

Masks project the eight corners of a 0.7×1.5×0.7 floor-standing box through the
MiniWorld pinhole camera (height 1.5, vertical FOV 60°, pitch 0, forward offset 0).
The near plane is 0.04; intersected box edges are clipped before hull filling.
Masks are boolean unions, ignoring occlusion as specified. Metric masks include
grey blocks. The independent colour control excludes grey blocks and uses
saturation >0.35 and brightness >0.15. The first 20 frames with nonempty colour
controls from a deterministic clip/time scan gave **median IoU 0.875727**;
12 empty-colour frames were skipped, without selecting by IoU. The overlay is
[.scratch/block_mask_check.png](.scratch/block_mask_check.png), with per-frame
results in `.scratch/block_mask_check.json`. Cyan marks analytic masks; magenta
marks the colour control. Occluded coloured blocks can produce low individual
IoU even with correct projection.

`load_released(state_dict, rebuild_mixing=True)` accepts the released
`state_dict` with or without `main_model.` prefixes. It loads the inherited
patchifier/attention/gate/decoder tensors and rebuilds C/C-pw's convolution with
`dirac_(..., groups=embed_dim)`. The kinetic map conv keeps its identity init;
A's fixed projection and scalar keep their initialization. Missing keys must be
exactly `{map_conv.weight}` for B/oracle and additionally `{P, refiner.n_s}` for
A, or empty for C/C-pw. Unexpected keys must be empty; unrelated omissions fail.
For C fidelity checks, **`rebuild_mixing=False`** loads the original released
mixing weights verbatim. The sinusoidal encodings have no trainable state.

## Configurations and CPU reproduction

Select `shortcode=exp/blockworld/kwm3d/train/{c,c_pw,a_oracle,a,b}`. These select
`algorithm=flowm_video`, `model_type=kwm3d`, batch 8, metadata loading,
checkpointing and `k3d_v_training` / `k3d_v_validation`. A defaults to v17, B to
v9. Append `exp/blockworld/kwm3d/train/pilot_50_40` for 90-frame clips.
Append `exp/blockworld/kwm3d/screening_finetune` for the 50+40 pilot, released
initialization with rebuilt mixing, a fresh optimizer and 5,000 steps; e.g. the
Hydra list is `shortcode=[exp/blockworld/kwm3d/train/a,exp/blockworld/kwm3d/screening_finetune]`.
That initial backbone implementation did not launch training; the Pilot section
below records the later two-step CPU smoke runs.

Inference uses `shortcode=exp/blockworld/kwm3d/infer/metrics_140/k3d_v_70ctx`,
with 70+70 frames and `[mse, psnr, ssim, block_mse]`. Set `load` and
`algorithm.backbone.arm` to the trained kinetic checkpoint. It deliberately does
not pretend a released FloWM checkpoint is a trained A checkpoint.

From the repository root:

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1
source algorithms/mem_wm/backbones/kwm3d/.scratch/env.sh
/home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/kwm3d -q -p no:cacheprovider
/home/yixunhu/miniconda3/envs/flowm/bin/python scripts/kwm3d_bench.py \
  --device cpu --arm a --steps 3 --batch 1
/home/yixunhu/miniconda3/envs/flowm/bin/python scripts/kwm3d_check_masks.py
```

The scratch `env.sh` is a local convenience, not a tracked dependency. Benchmark
and test entrypoints set their own cache directories. Run the mask-check script
with the same environment to keep third-party caches inside scratch.

## CPU measurements

Measured 2026-11-04 on an Intel Xeon Gold 5220R @ 2.20 GHz, using the supplied
Python environment, PyTorch 2.9.0+cu128
with CUDA hidden, fp32, eight OpenMP/MKL/PyTorch threads. No CUDA operations,
optimizer steps, training jobs or GPU queries were executed. A benchmark step
is **one full 10-context + 5-prediction rollout and backward**, batch 1, full
128×128 images, 33×33 maps, width 256, encoder/decoder depths 6. Means include
the first iteration, three iterations per mode. Each mode runs in a fresh
process. Peak memory is Linux process high-water RSS, including imports,
parameters, gradients and CPU allocator retention; it is not GPU allocated
memory. The host is shared, so timings are indicative rather than exclusive-CPU
comparisons. Inputs are random; oracle slots are all valid. A's real estimator
runs, including its observation-only lift.

| Arm | Checkpoint off: s/step | Peak RSS MiB | Checkpoint on: s/step | Peak RSS MiB |
| --- | ---: | ---: | ---: | ---: |
| c | 3.350 | 3161 | 4.077 | 2089 |
| c_pw | 1.589 | 2274 | 2.179 | 1684 |
| a_oracle | 2.069 | 2866 | 2.672 | 2283 |
| a | 3.800 | 4550 | 4.457 | 2941 |
| b | 1.861 | 2688 | 2.592 | 2055 |

Raw fix-pass timings are in `.scratch/fixpass-bench.log`. GPU memory, GPU timing and campaign
budgets remain unmeasured. In particular these batch-1, 15-frame CPU figures
cannot establish feasibility of batch-8, 90/140-frame GPU backpropagation.

**Expected GPU memory, to be measured:** non-reentrant per-step checkpointing
saves the input map, approximately **1.1 MB × V per sample-step**
(`33 × 33 × 256 × 4` bytes per node). For A at V=17 this is 19.0 MB per
sample-step: batch 8 retains approximately **21 GB at 140 steps** or
**13.6 GB at 90 steps**, before the recompute peak of one step, parameters,
gradients and other allocations. These are decimal MB/GB projections, not
measured GPU allocations. The benchmark reports exact map bytes and both
batch-8 projections for each arm.

## Verification, deviations and remaining limitations

**53 tests passed, no skips, in 40.90 seconds** on CPU. Only the existing
`pkg_resources` and timm import deprecation warnings were emitted.

Tests cover every arm's reduced forward/backward, C vs released FloWM on two
real clips (24 context + 8 predictions each, exact outputs and hidden states),
C training-mode parity over 24+8 frames with turns, teacher forcing 0.5 and
checkpointing (exact outputs and gradients), pooled token counts,
33×33 turn inverses and spectral commutation (1e-5), checkpoint gradient agreement
(1e-5 relative, width 32, depth 1, 4+3 frames), invalid oracle slots, strict
released loading, observation-only projection, nonzero lock-update derivatives,
static pinning, exactly one skipped measurement per turn, both residual modes,
integer transport versus FloWM (2e-6) and exact turns, real-clip oracle kinematics
(below 1e-5 cells, zero to four decimal places; spectral centroids below 5e-4),
configuration/factory integration,
metadata signature gating, analytic geometry, real mask IoU, dataset alignment,
padding, skipping and mixed missing-file collation. Real-data/checkpoint tests skip only when the
required artifacts are absent. Validation wiring is tested without launching a
Lightning trainer or downloading metric networks.

Differences and extensions relative to the brief/design:

1. The maintainer-selected incoming object velocity is the oracle default.
   The generator's raw camera-relative outgoing velocity remains an explicit
   diagnostic option, not a physically validated oracle ceiling.
2. Scalar turns specialize `rot_state` semantics, because the installed function
   rotates orientation channels. The pre-turn snapshot is replaced by the
   turn-step post-turn imprint without measuring across the turn. Frozen prediction
   snapshots are never read again for measurement.
3. The installed `_measure_lock` hook reuses exact lock arithmetic without
   copying code or modifying the read-only fork.
4. Implementation proceeded before stage-1 kill-test completion as explicitly
   requested in this task; it does not supersede D5's decision gate for training.
5. Corrected incoming velocities and object speed are additional metadata fields.
   Raw required fields and the disabled dataset path are preserved. Frame-skip
   physics is rejected at model creation, although metadata alignment is tested.
6. CPU peak memory is whole-process RSS; a full GPU benchmark and long-clip
   distributed/mixed-precision execution remain unverified. No empirical
   estimator quality, subcell lift quality or training superiority is claimed.

Other approximations and open work: fixed nominal camera/box dimensions and
quarter-turn headings; hull rasterization has pixel-boundary quantization;
boxes behind walls/other objects are not occlusion-tested; colour checks miss
grey blocks and are sensitive to compression/lighting; map transport uses a
33×33 torus; the lift/decoder with absolute positions are not turn-equivariant
(the C4 guarantee applies to map transport + symmetric convolution); rho is
componentwise bounded, with no explicit collision or bounce model; prediction
freezes node learning even through bounces; Python control flow/the KWM hook
has not been verified under `torch.compile`, TorchScript or export.
The optional plain convolution intentionally relinquishes C4 equivariance.

Maintainer decision (2026-11-04): use `incoming_object` for A-oracle and retain
`raw` for diagnostics. Stage 1 subsequently failed on the frozen released lift;
Addendum 2 authorizes the pilot below after the C4 chain finishes.

## Pilot (Addendum 2, CPU preparation 2026-11-05/06)

The protocol in `experiments/flowm_bench/docs/KINETIC3D_model_design.md`,
**Addendum 2**, governs this pilot. C and A-oracle start from scratch with seed 0,
50 context + 40 prediction frames, 6,000 **optimizer** steps, batch 8,
AdamW (1e-4, betas 0.9/0.99, weight decay 1e-3), constant learning rate,
gradient norm clipping 1, teacher forcing 0.125, fp32 and step checkpointing.
A-oracle uses `incoming_object`, including true velocities during prediction.
The pooled write and C4 map convolution follow the registered arm definition.
Neither arm uses EMA or released-weight initialization.

`pilot_c` composes the C/base and 50+40 shortcodes; `pilot_a_oracle` inherits
that complete pilot configuration and selects the oracle arm. Checkpoints are
retained every 1,000 optimizer steps, including `step-006000.ckpt`; there is no
checkpoint selection. Validation runs every 1,000 steps on four batches of eight
90-frame clips, logging MSE, PSNR and all block splits. Sanity validation is off.
`strategy: auto` selects the single-device strategy. The trainer now accepts
`experiment.accelerator=cpu` explicitly; the default remains `auto`.

Lightning **2.6.6** implements `global_step` as
`automatic_optimization.optim_progress.optimizer_steps` and tests `max_steps`
against that counter (`lightning/pytorch/loops/training_epoch_loop.py:113`).
An executable CPU test confirms accumulation 2 processes four batches for two
optimizer steps and validates at step 2 when the batch interval is four.
`pilot_bs4_accum2` therefore keeps `max_steps=6000`, sets batch 4/accumulation 2,
and changes the validation **batch** interval to 2,000. Checkpoint cadence
continues to count optimizer steps. Both arms must use the same fallback.

`seed: null` leaves existing process-seeding behavior unchanged. `seed: 0`
uses Lightning's `seed_everything(..., workers=True)` for Python, NumPy and
PyTorch. The dataloader generator uses the run seed; the existing worker hook
seeds Python/NumPy from each worker's distinct PyTorch initial seed. This covers
initialization, teacher-forcing draws, shuffled training batches and workers;
it does not promise bitwise deterministic CUDA kernels.

Pilot data access sets `metadata_read_only: true`: constructing a dataset
never creates NAS metadata, even for other registered splits. The training
index `data/blockworld/metadata/k3d_v_training.pt` now exists: **10,000 videos,
each with 500 frames**, verified in the Opus review. The original CPU training
smoke runs used `k3d_v_validation` because the index was absent at that time.

The new `validation_first_clips: true` selects the **first 1,000 videos in the
metadata index, starting at frame 0**, with no shuffle; the current validation
index is sorted by path (verified). Both horizons use the same videos in the
same order. `num_default_clips` truncates this list. This is an opt-in correction:
the older validation path samples seeded random starts and shuffles videos.
The dataset still uses the original CPU decoder, resize, RGB `/255` convention,
and incoming action alignment. The test split is never used.

### Maintainer GPU commands (not executed during preparation)

Run from `/home/yixunhu/codespace/flowm`. These commands are for the maintainer's
existing launcher, **only after `C4 CHAIN DONE` and the free-memory gate**.
No scheduling or GPU process was started by this work. The training metadata
and maintainer's real-split CPU dry run have been verified in the Opus review.

```bash
P=/home/yixunhu/miniconda3/envs/flowm/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/algorithms/mem_wm/backbones/kwm3d/.scratch/tmp"
export XDG_CACHE_HOME="$PWD/algorithms/mem_wm/backbones/kwm3d/.scratch/cache"
export MPLCONFIGDIR="$PWD/algorithms/mem_wm/backbones/kwm3d/.scratch/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" outputs/k3d_pilot
COMMON=(algorithm=flowm_video dataset=blockworld ckpt_map=default
        wandb.mode=offline wandb.entity=local wandb.project=flowm_bench
        experiment.devices=1 experiment.accelerator=gpu)

# Ten optimizer steps per arm: two warmups plus eight measured steps (<=50).
"$P" scripts/kwm3d_bench.py --device cuda --arm c --context 50 --prediction 40 \
  --batch 8 --accumulate 1 --warmup 2 --steps 8 --checkpoint on \
  > outputs/k3d_pilot/bench_c.log 2>&1
"$P" scripts/kwm3d_bench.py --device cuda --arm a_oracle --context 50 --prediction 40 \
  --batch 8 --accumulate 1 --warmup 2 --steps 8 --checkpoint on \
  > outputs/k3d_pilot/bench_a_oracle.log 2>&1
```

Each JSON record contains the resolved Hydra backbone, mean/median seconds per
optimizer step (including accumulation, clipping and AdamW), and CUDA allocated /
reserved peak MiB reset after warmup. CUDA visibility is preserved in CUDA mode;
CPU mode hides CUDA. CPU RSS is reported separately. Inputs are synthetic, with
ten valid oracle slots; timings exclude real-data loading and validation.
Do not infer a GPU budget from CPU timings.

The measured NAS data-loading floor is **0.353 s/batch** at batch 8, 8 workers,
and 90 frames; the first batch takes **3.8 s** (Opus review, shared host).
The budget projection uses **`max(gpu_step, 0.35)` seconds per arm**:
`6000 * (max(mean_C, 0.35) + max(mean_A_oracle, 0.35)) / 86400` GPU-days.
**If this exceeds four days, stop and report.** Report actual elapsed time as well,
including validation and I/O. If either arm's batch 8 cannot fit in the *free*
memory, benchmark **both** with `--batch 4 --accumulate 2`, recompute the cap,
and use the fallback shortcode for both trainings. `--checkpoint both` can
measure the optional off/on comparison; checkpoint-off at pilot batch/length
may OOM and is not the mode used for training.

Exactly one command launches each standard arm, sequentially:

```bash
"$P" -m main shortcode=exp/blockworld/kwm3d/train/pilot_c \
  +name=k3d_pilot_c "${COMMON[@]}" hydra.run.dir=outputs/k3d_pilot/c
"$P" -m main shortcode=exp/blockworld/kwm3d/train/pilot_a_oracle \
  +name=k3d_pilot_a_oracle "${COMMON[@]}" hydra.run.dir=outputs/k3d_pilot/a_oracle
```

For fallback, replace only the corresponding `shortcode=` argument with
`'shortcode=[exp/blockworld/kwm3d/train/pilot_c,exp/blockworld/kwm3d/train/pilot_bs4_accum2]'`
or
`'shortcode=[exp/blockworld/kwm3d/train/pilot_a_oracle,exp/blockworld/kwm3d/train/pilot_bs4_accum2]'`.
The CLI now validates Hydra shortcode lists as well as single shortcodes.

Four evaluations of the final checkpoints, then the statistics:

```bash
for arm in c a_oracle; do
  for horizon in 140 280; do
    "$P" -m main \
      "shortcode=exp/blockworld/kwm3d/infer/metrics_${horizon}/k3d_v_70ctx" \
      "+name=k3d_pilot_${arm}_${horizon}" "${COMMON[@]}" \
      "algorithm.backbone.arm=$arm" \
      "load=$PWD/outputs/k3d_pilot/$arm/checkpoints/step-006000.ckpt" \
      "hydra.run.dir=outputs/k3d_pilot/$arm/eval_$horizon" \
      "algorithm.logging.per_clip_dump=outputs/k3d_pilot/$arm/eval_$horizon/dump"
  done
done
"$P" scripts/kwm3d_pilot_stats.py \
  --c outputs/k3d_pilot/c/eval_140/dump \
  --a_oracle outputs/k3d_pilot/a_oracle/eval_140/dump \
  --baseline experiments/flowm_bench/k3d/results/pilot_baseline_repeat_140 \
  --reference context_mean=experiments/flowm_bench/k3d/results/pilot_reference_context_mean_140 \
  --reference last_flat=experiments/flowm_bench/k3d/results/pilot_reference_last_flat_140 \
  --reference gray=experiments/flowm_bench/k3d/results/pilot_reference_gray_140 \
  --other-c outputs/k3d_pilot/c/eval_280/dump \
  --other-a-oracle outputs/k3d_pilot/a_oracle/eval_280/dump \
  --output outputs/k3d_pilot/pilot_stats_140.md
"$P" scripts/kwm3d_pilot_stats.py --horizon 280 \
  --c outputs/k3d_pilot/c/eval_280/dump \
  --a_oracle outputs/k3d_pilot/a_oracle/eval_280/dump \
  --baseline experiments/flowm_bench/k3d/results/pilot_baseline_repeat_280 \
  --reference context_mean=experiments/flowm_bench/k3d/results/pilot_reference_context_mean_280 \
  --reference last_flat=experiments/flowm_bench/k3d/results/pilot_reference_last_flat_280 \
  --reference gray=experiments/flowm_bench/k3d/results/pilot_reference_gray_280 \
  --output outputs/k3d_pilot/pilot_stats_280.md
```

Evaluation metrics are exactly `[mse, psnr, ssim, block_mse]`; no FVD/LPIPS
weights are downloaded. The validation log retains `prediction/mse`,
`prediction/psnr`, `prediction/ssim` and `validation/prediction/block_mse*`.
The kwm3d epoch hook also computes/resets these metrics during training; the
older common hook omitted FIT metrics and retained their state across epochs.
`load=` passes a local checkpoint to Lightning. Relative paths resolve from the
repository working directory (Hydra does not change it here); absolute paths
are recommended above. Both arms restored through this path in CPU smoke runs.
A strict backbone check rejects missing, extra or mismatched tensors before the
common checkpoint mixin can fill or discard keys. The trained-lift loader uses
the saved `.hydra/config.yaml` beside the checkpoint directory when available;
otherwise it composes the full pilot model config and loads strictly.

### Dumps and decision rule

`algorithm.logging.per_clip_dump=null` is the default and allocates no dump.
When enabled it requires one device and one validation split, stores arrival
order, and writes `per_clip.pt`, `per_clip.csv`, and `pooled.json` at validation
end. Training writes into `DIR/step-NNNNNN/`; standalone evaluation writes into
`DIR/`. Only predicted frames count, respecting `n_metrics_frames` if set.
Dump-on execution checks pooled MSE and every block split against the actual
logged metric state with relative tolerance **1e-5** (including multiple batches).
A larger discrepancy logs a warning with both values and writes
`pooled_check_failed.json` beside the dump; evaluation and metric logging continue.

The portable torch payload has version 1, `paths`, `dataset_indices`,
`context_frames`, `n_frames`, `metric_names`, and `[clips,7]` tensors
`squared_error` (float64) and `element_count` (int64). Columns are full MSE,
all blocks, three speed bins, turn, non-turn. `block_pixel_count [clips,6]`
counts spatial pixels; `element_count` for these columns is three times that
count because SSE sums RGB channels. Thus MSE is SSE / element_count.
`pooled` is the ratio of column sums, never an average of batch or clip means.
CSV includes every numerator, denominator, per-clip value and block pixel count.
Empty regions have zero numerator/count and NaN per-clip MSE. Accumulation uses
float64 to avoid drift over 1,000 clips; masks and image values are unchanged.

The statistics script requires **exactly 1,000 clips in every dump** and checks
both arms, repeat-last and references for matching paths/order, dataset indices,
horizon, full-frame element counts and block pixel counts. It reports pooled
values, relative reductions, mean paired per-clip
A-oracle minus C, and the 2.5/97.5 percentiles of **10,000** clip bootstrap
resamples from NumPy Generator seed **20261105**. Empty-region clips are counted
and excluded from that split's paired mean, but remain in pooled sums. An
entirely empty primary region is rejected as undefined. Empty secondary splits
report NaN. Explicit `--smoke` permits smaller arm samples and subsets baseline
and reference dumps to the arms' paths, failing if any path is missing or its
index/counts differ. Output is labelled **SMOKE** and contains no pilot verdict.

Repeatable `--reference name=<dump>` adds **reference, not part of the criterion**
rows for pooled block MSE and full-frame MSE next to both arms, with flags for
whether each arm is below that reference. `context_mean` averages the 70 context
frames per pixel, `last_flat` repeats the last context frame's mean RGB colour
at every pixel, and `gray` is constant 0.5. These diagnostics address the review's
finding that even gray can beat repeat-last on block MSE; beating repeat-last
alone need not establish useful dynamics. The verdict never reads reference
values; repeat-last remains the registered INCONCLUSIVE baseline.

At **140 frames**, INCONCLUSIVE is checked first if both arms' pooled block MSE
are >= repeat-last. Otherwise PASS requires oracle <= 0.90*C **and** CI upper
bound < 0; otherwise FAIL. The 280-frame table is explicitly secondary. The
optional other-horizon arguments report a sign-disagreement flag. One seed's
clip CI does not measure training-seed variance. Report at this boundary and
stop before stage 3; an estimator training run still requires the trained-lift
gate to pass and the user's approval.

### Trained-lift gate, after the pilot (CPU)

```bash
bash experiments/flowm_bench/k3d/run_cpu.sh trained_lift \
  --checkpoint outputs/k3d_pilot/a_oracle/checkpoints/step-006000.ckpt \
  --arm a_oracle --clips 200
```

This runs the pooled observation-only lift, direct 3x3 patch-motion diagnostic,
and estimator gate with **mask residuals, K=8, sigma=1,
intrinsic_incoming truth**, on 140 observed frames per clip. The backbone's
`observation_write` is shared with arm A's `observation_imprint`; no lift is
reimplemented in the runner. Fixed projection and fp16 cache conventions match
stage 1. Model construction uses `create_kwm3d_model` and strict restoration.
Caches are under `experiments/flowm_bench/k3d/.scratch/trained_lifts/SHA256/ARM/`;
new CSVs and provenance are under `outputs/k3d_trained_lifts/SHA256/ARM/`.
Existing released-weight reports/CSVs are untouched. The existing `lift`,
`patch_motion`, and `estimator_gate` entrypoints also accept `--checkpoint`,
`--arm`, and `--clips` to run just the corresponding stage. Without these
arguments their released-weight paths retain their behavior.
`--arm c` is rejected because C trains per-node writes and has no trained pooled
observation write. Pooled arms `a_oracle`, `a`, `b`, and `c_pw` are supported.
Dispatch accepts `--split k3d_v_validation`; other splits and unsupported stage-1
options (such as `--ks` or `--sigmas`) fail with an explanation of the fixed
trained-lift settings, before checkpoint loading.

### CPU verification and limits

All executed commands hid CUDA and set `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
WANDB_MODE=offline`; Python was `/home/yixunhu/miniconda3/envs/flowm/bin/python`.
Caches/temp files and command logs live in the kwm3d `.scratch/`, except the
stage-1 runner's caches under its own `.scratch/`. No GPU computation, GPU
memory query, campaign training, NAS writes, KWM dependency edits, or commits
were performed. The editable dependency was `kwm-wt-k3d`, `k3d-base`, revision
`c7067d3c6b76cdd4eb8121c9bcef4d6957590c0c` (the stage-1 follow-up already
established parity with the earlier revision documented above).

The dry-run overrides are deliberately small: `experiment.accelerator=cpu`,
`experiment.devices=1`, batch 1, workers 0, max_steps 2, 8+4 frames, validation
at step 2 over two separate one-clip batches, and checkpoint every two steps.
Those original runs used `k3d_v_validation` before the training index existed. Each
checkpoint was then evaluated at full 70+70 on two clips and 70+210 on one clip.
All six dumps passed the runtime 1e-6 pooling checks. Both checkpoints contain
`global_step=2`, AdamW step counters of 2 and finite model weights. Paths are
`outputs/k3d_pilot_cpu_{c,a_oracle}/checkpoints/step-000002.ckpt`,
`train_dump/step-000002/`, and `eval_{140,280}/dump/` under each arm directory.

The original sandbox's first C command with `wandb.mode=offline` failed before model execution:
`ServicePollForTokenError: wandb-core exited with code 1`. Those CPU smoke commands
therefore used **`wandb.mode=disabled`**, and skipped video logging with
`algorithm.logging.max_num_videos=0`; the pilot shortcodes still use offline
W&B. Disabled logging exposed an existing uninitialized `resume` local in
`main.py`, which is fixed. **Offline W&B works on the maintainer's host**:
the Opus review verified it directly and through `main.py`, including with
the documented `.scratch/tmp` TMPDIR. The earlier failure was specific to the
sandbox. The initial C evaluation log used enum-form metric
labels; the hook now uses the existing `prediction/mse` spelling, covered by
tests and the subsequent oracle evaluations. Values were unaffected.

Original preparation tests: **93 passed, 5 warnings, 52.60 s pytest time / 55.12 s command wall**
for `tests/kwm3d tests/k3d`, below five minutes. Warnings concern existing
pkg_resources/timm/TF32 APIs and the intentionally single-process dataloaders.
Tests cover config composition, first-clip selection and read-only metadata,
multiple validation batches and metric parity, empty regions, strict C/oracle
state restoration, all three verdicts, margin/CI boundaries, bootstrap seed,
clip mismatches, horizon sign diagnostics, seeding, accumulation semantics,
pooled-lift parity and a real CPU benchmark subprocess (10+5, batch 1,
accumulation 2). The earlier full pass had 92 tests in 48.41 s; the additional
sign-diagnostic test and final default-off precision check prompted the final
pass. The final statistics subset also passed (17 tests, 9.10 s).

The CPU micro-benchmark command was:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python scripts/kwm3d_bench.py \
  --device cpu --arm c --context 10 --prediction 5 --batch 1 \
  --steps 1 --warmup 2 --checkpoint both
```

Output: `.scratch/pilot_bench_cpu.log`; approximately 43.87 s wall, measured
from log creation to final output. After two warmups, C took 2.45460 s/optimizer
step without checkpointing and 3.55622 s with checkpointing; corresponding
process peak RSS was 3152.64 / 1984.29 MiB. Each mode has one measured sample,
so mean equals median. The A-oracle CPU path with accumulation 2 is also
exercised by the subprocess test. Concurrent CPU baseline decoding and shared
host load make these smoke timings unsuitable for GPU budget extrapolation.

The trained-lift proof used the oracle step-2 checkpoint on two clips, hash
`3ca3aa6eaa79bc7a60b4a5b6390ef709aa123272937076bb191f44150005b9c3`.
It ran all three stages with K=8, sigma=1, mask residuals and
intrinsic-incoming truth. Its final command took 19.31 s (including imports;
existing lift caches reused); output is
`outputs/k3d_trained_lifts/3ca3aa6eaa79bc7a60b4a5b6390ef709aa123272937076bb191f44150005b9c3/a_oracle/`.
The first uncached proof is separately recorded below. Its numerical verdict
has no scientific meaning after two updates.

No pilot-protocol settings or kill thresholds were changed. The first-clip
option corrects the pre-existing random-start evaluation behavior to match
Addendum 2. CPU-only departures are the explicitly requested small sample,
short-clip and validation-training-split overrides, plus disabled W&B/video
logging after the offline service failure. The baseline shares one 280-frame
decode between horizons without changing either metric window. Post-pilot
results use a new checkpoint-specific output directory; no released result CSV
was overwritten. Only zero-map write reuse was factored out of the backbone;
its forward arithmetic and state-dict keys are unchanged.

Still unverified: actual GPU memory/timing, feasibility of batch 8 or fallback
batch 4, the four-day projection, and 6,000-step convergence/stability. Ensure the
correct arm and final checkpoint are supplied; wrong-arm or incomplete state
is now rejected strictly. A rerun using the same output directory can overwrite
same-step artifacts, so retain completed pilot directories before rerunning.

### Repeat-last baseline, references and command record

The baseline command was:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python scripts/kwm3d_baseline_repeat.py
```

It used 1,000 validation clips at both horizons, in the same first-video/frame-0
order as the evaluation shortcodes. Measured processing time was **788.298 s**;
command wall was approximately **797.84 s** from log creation to final output,
including imports. Log: `.scratch/baseline.log`. Both dump directories contain
`per_clip.pt`, `per_clip.csv` and `pooled.json`:
`experiments/flowm_bench/k3d/results/pilot_baseline_repeat_{140,280}/`.
These small sufficient-statistic artifacts are explicitly unignored for review.
The script now also writes `pilot_reference_{context_mean,last_flat,gray}_{140,280}/`
in that results directory, sharing one 280-frame decode and the analytic masks
across predictors and horizons. The CPU fix pass regenerates all 1,000 clips;
reference measurements and the repeat-last byte-identity hashes are recorded in
[.scratch/FIXPASS_PILOT.md](.scratch/FIXPASS_PILOT.md).

| Horizon | Predicted frames | Clips | Pooled block MSE | Full-frame MSE |
|---|---:|---:|---:|---:|
| 140 | 70 | 1,000 | 0.20110865545138992 | 0.04059602298643912 |
| 280 | 210 | 1,000 | 0.2025386558195779 | 0.04125877101735599 |

The original statistics CLI smoke runs succeeded. Their outputs were
`outputs/k3d_pilot_cpu_stats_140.md` and `outputs/k3d_pilot_cpu_stats_280.md`.
The old two-clip primary printed INCONCLUSIVE, the one-clip secondary printed FAIL,
and the pooled improvement signs agreed. **These are plumbing checks only**;
the arm sample counts are tiny, the models have only two updates, and the
baseline contained 1,000 clips. They do not decide the registered pilot.
Current smoke commands must pass `--smoke`; they subset the baseline and print
no verdict. The historical command records below predate this requirement.

The table records subprocess wall time (including imports) for every main-entry
smoke command, statistics command, and trained-lift proof. Logs use
`.scratch/LABEL.log`. The failed offline attempt's C log was reused on retry;
its W&B startup log remains under
`outputs/k3d_pilot_cpu_c/wandb/offline-run-20261105_235536-07bc1m06/logs/debug.log`.
The initial trained-lift proof's new result files were relocated under `outputs/`
and regenerated from its hash-isolated lift cache by the final proof; no released
result was removed. Exact expanded commands follow the table. The shared CPU
environment above applies to each. Auxiliary audit scripts verified checkpoints,
AdamW counters, all six dumps and the baseline identities/windows/counts; their
record is `.scratch/pilot_artifact_audit.json`.

| Command label | Wall seconds | Exit | Output location |
|---|---:|---:|---|
| dry_train_c | 12.108 | 1 | `outputs/k3d_pilot_cpu_c` |
| dry_train_c | 21.248 | 0 | `outputs/k3d_pilot_cpu_c` |
| dry_eval_c_140 | 31.253 | 0 | `outputs/k3d_pilot_cpu_c/eval_140/dump` |
| dry_eval_c_280 | 30.758 | 0 | `outputs/k3d_pilot_cpu_c/eval_280/dump` |
| dry_train_a_oracle | 19.047 | 0 | `outputs/k3d_pilot_cpu_a_oracle` |
| dry_eval_a_oracle_140 | 23.054 | 0 | `outputs/k3d_pilot_cpu_a_oracle/eval_140/dump` |
| dry_eval_a_oracle_280 | 24.197 | 0 | `outputs/k3d_pilot_cpu_a_oracle/eval_280/dump` |
| trained_lift_cpu | 22.097 | 0 | `outputs/k3d_trained_lifts/3ca3aa6eaa79bc7a60b4a5b6390ef709aa123272937076bb191f44150005b9c3/a_oracle/` |
| trained_lift_cpu_final | 19.305 | 0 | `outputs/k3d_trained_lifts/3ca3aa6eaa79bc7a60b4a5b6390ef709aa123272937076bb191f44150005b9c3/a_oracle/` |
| pilot_stats_tests | 11.495 | 0 | `.scratch/pilot_stats_tests.log` |
| pilot_tests_final | 55.117 | 0 | `.scratch/pilot_tests_final.log` |
| dry_stats_140 | 1.605 | 0 | `outputs/k3d_pilot_cpu_stats_140.md` |
| dry_stats_280 | 1.637 | 0 | `outputs/k3d_pilot_cpu_stats_280.md` |

<details>
<summary>Exact expanded CPU commands (in execution order)</summary>

dry_train_c — 12.108 s, exit 1:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m main shortcode=exp/blockworld/kwm3d/train/pilot_c +name=k3d_pilot_cpu_c algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=offline wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=cpu experiment.training.dataloader.num_workers=0 experiment.validation.dataloader.num_workers=0 algorithm.logging.max_num_videos=0 hydra.run.dir=outputs/k3d_pilot_cpu_c 'experiment.training.splits_included=[k3d_v_validation]' experiment.training.batch_size=1 experiment.validation.batch_size=1 dataset.max_frames=12 dataset.n_frames=12 algorithm.tasks.prediction.context_frames=8 experiment.training.max_steps=2 experiment.validation.val_every_n_step=2 experiment.validation.limit_batch=2 dataset.num_default_clips=2 experiment.training.checkpointing.every_n_train_steps=2 algorithm.logging.per_clip_dump=outputs/k3d_pilot_cpu_c/train_dump
```

dry_train_c — 21.248 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m main shortcode=exp/blockworld/kwm3d/train/pilot_c +name=k3d_pilot_cpu_c algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=disabled wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=cpu experiment.training.dataloader.num_workers=0 experiment.validation.dataloader.num_workers=0 algorithm.logging.max_num_videos=0 hydra.run.dir=outputs/k3d_pilot_cpu_c 'experiment.training.splits_included=[k3d_v_validation]' experiment.training.batch_size=1 experiment.validation.batch_size=1 dataset.max_frames=12 dataset.n_frames=12 algorithm.tasks.prediction.context_frames=8 experiment.training.max_steps=2 experiment.validation.val_every_n_step=2 experiment.validation.limit_batch=2 dataset.num_default_clips=2 experiment.training.checkpointing.every_n_train_steps=2 algorithm.logging.per_clip_dump=outputs/k3d_pilot_cpu_c/train_dump
```

dry_eval_c_140 — 31.253 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m main shortcode=exp/blockworld/kwm3d/infer/metrics_140/k3d_v_70ctx +name=k3d_pilot_cpu_c_140 algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=disabled wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=cpu experiment.training.dataloader.num_workers=0 experiment.validation.dataloader.num_workers=0 algorithm.logging.max_num_videos=0 hydra.run.dir=outputs/k3d_pilot_cpu_c/eval_140 algorithm.backbone.arm=c load=/home/yixunhu/codespace/flowm/outputs/k3d_pilot_cpu_c/checkpoints/step-000002.ckpt dataset.num_default_clips=2 experiment.validation.batch_size=1 algorithm.logging.per_clip_dump=outputs/k3d_pilot_cpu_c/eval_140/dump
```

dry_eval_c_280 — 30.758 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m main shortcode=exp/blockworld/kwm3d/infer/metrics_280/k3d_v_70ctx +name=k3d_pilot_cpu_c_280 algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=disabled wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=cpu experiment.training.dataloader.num_workers=0 experiment.validation.dataloader.num_workers=0 algorithm.logging.max_num_videos=0 hydra.run.dir=outputs/k3d_pilot_cpu_c/eval_280 algorithm.backbone.arm=c load=/home/yixunhu/codespace/flowm/outputs/k3d_pilot_cpu_c/checkpoints/step-000002.ckpt dataset.num_default_clips=1 experiment.validation.batch_size=1 algorithm.logging.per_clip_dump=outputs/k3d_pilot_cpu_c/eval_280/dump
```

dry_train_a_oracle — 19.047 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m main shortcode=exp/blockworld/kwm3d/train/pilot_a_oracle +name=k3d_pilot_cpu_a_oracle algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=disabled wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=cpu experiment.training.dataloader.num_workers=0 experiment.validation.dataloader.num_workers=0 algorithm.logging.max_num_videos=0 hydra.run.dir=outputs/k3d_pilot_cpu_a_oracle 'experiment.training.splits_included=[k3d_v_validation]' experiment.training.batch_size=1 experiment.validation.batch_size=1 dataset.max_frames=12 dataset.n_frames=12 algorithm.tasks.prediction.context_frames=8 experiment.training.max_steps=2 experiment.validation.val_every_n_step=2 experiment.validation.limit_batch=2 dataset.num_default_clips=2 experiment.training.checkpointing.every_n_train_steps=2 algorithm.logging.per_clip_dump=outputs/k3d_pilot_cpu_a_oracle/train_dump
```

dry_eval_a_oracle_140 — 23.054 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m main shortcode=exp/blockworld/kwm3d/infer/metrics_140/k3d_v_70ctx +name=k3d_pilot_cpu_a_oracle_140 algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=disabled wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=cpu experiment.training.dataloader.num_workers=0 experiment.validation.dataloader.num_workers=0 algorithm.logging.max_num_videos=0 hydra.run.dir=outputs/k3d_pilot_cpu_a_oracle/eval_140 algorithm.backbone.arm=a_oracle load=/home/yixunhu/codespace/flowm/outputs/k3d_pilot_cpu_a_oracle/checkpoints/step-000002.ckpt dataset.num_default_clips=2 experiment.validation.batch_size=1 algorithm.logging.per_clip_dump=outputs/k3d_pilot_cpu_a_oracle/eval_140/dump
```

dry_eval_a_oracle_280 — 24.197 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m main shortcode=exp/blockworld/kwm3d/infer/metrics_280/k3d_v_70ctx +name=k3d_pilot_cpu_a_oracle_280 algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=disabled wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=cpu experiment.training.dataloader.num_workers=0 experiment.validation.dataloader.num_workers=0 algorithm.logging.max_num_videos=0 hydra.run.dir=outputs/k3d_pilot_cpu_a_oracle/eval_280 algorithm.backbone.arm=a_oracle load=/home/yixunhu/codespace/flowm/outputs/k3d_pilot_cpu_a_oracle/checkpoints/step-000002.ckpt dataset.num_default_clips=1 experiment.validation.batch_size=1 algorithm.logging.per_clip_dump=outputs/k3d_pilot_cpu_a_oracle/eval_280/dump
```

trained_lift_cpu — 22.097 s, exit 0:

```bash
bash experiments/flowm_bench/k3d/run_cpu.sh trained_lift --checkpoint outputs/k3d_pilot_cpu_a_oracle/checkpoints/step-000002.ckpt --arm a_oracle --clips 2
```

trained_lift_cpu_final — 19.305 s, exit 0:

```bash
bash experiments/flowm_bench/k3d/run_cpu.sh trained_lift --checkpoint outputs/k3d_pilot_cpu_a_oracle/checkpoints/step-000002.ckpt --arm a_oracle --clips 2
```

pilot_stats_tests — 11.495 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/kwm3d/test_pilot_stats.py -q -p no:cacheprovider
```

pilot_tests_final — 55.117 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/kwm3d tests/k3d -q -p no:cacheprovider
```

dry_stats_140 — 1.605 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python scripts/kwm3d_pilot_stats.py --c outputs/k3d_pilot_cpu_c/eval_140/dump --a_oracle outputs/k3d_pilot_cpu_a_oracle/eval_140/dump --baseline experiments/flowm_bench/k3d/results/pilot_baseline_repeat_140 --other-c outputs/k3d_pilot_cpu_c/eval_280/dump --other-a-oracle outputs/k3d_pilot_cpu_a_oracle/eval_280/dump --output outputs/k3d_pilot_cpu_stats_140.md
```

dry_stats_280 — 1.637 s, exit 0:

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python scripts/kwm3d_pilot_stats.py --horizon 280 --c outputs/k3d_pilot_cpu_c/eval_280/dump --a_oracle outputs/k3d_pilot_cpu_a_oracle/eval_280/dump --baseline experiments/flowm_bench/k3d/results/pilot_baseline_repeat_280 --output outputs/k3d_pilot_cpu_stats_280.md
```

</details>

The working tree is intentionally uncommitted. The original preparation message
is `.scratch/COMMIT_MSG3.txt`; the review fix pass uses `.scratch/COMMIT_MSG4.txt`.
Its current test counts and artifact audit are in `.scratch/FIXPASS_PILOT.md`.

Final benchmark review aligned `torch.set_float32_matmul_precision("high")`
with `experiments/base_exp.py` and records that setting in each JSON line.
This preserves the trainer's fp32/TF32 policy on CUDA rather than benchmarking
with PyTorch's unrelated default. No CUDA execution occurred. The focused CPU
subprocess regression passed afterward: **1 test, 22.25 s pytest time /
24.412 s command wall**, output `.scratch/pilot_bench_final_test.log`.

```bash
/home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/kwm3d/test_pilot.py::test_benchmark_cpu_cli -q -p no:cacheprovider
```
