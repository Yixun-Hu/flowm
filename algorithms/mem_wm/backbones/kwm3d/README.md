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
`k3d-base`, commit **0190d250f6d2c6a8bc0ebd77e92bb90ad3301083**. It supplies
`stream`, `named_nodes`, `rotation_perms`, `rot_vec`, `C4SymmetricDepthwise`,
`oracle_nodes_from`, `_fixed_projection`, `LockState`, and `lock_measurement`.
No KWM or FloWM source has been vendored. Keep this dependency pinned when
reproducing the numbers; the private lock adapter depends on this version.

The fork exposes the masked measurement primitive but has no injectable
measurement hook in `AdaptiveNodeCell._forward_lock`. `_LockArithmetic` reuses
that installed function's code object with a private globals dictionary, binding
only `lock_measurement` to the cone/box options. It does **not** modify module
globals or copy the function's source. A one-cell dummy feature bank bypasses
its unrelated CNN recurrence; FloWM owns the actual feature recurrence.
`.scratch/kwm_patch.diff` proposes a small upstream `_measure_lock` hook. Once
that patch is adopted, replace the temporary binding adapter with the hook.
The external checkout was not modified.

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
`static_residual=True`, `sigma=1.0` (configurable) and `radius=None`.

The installed cell supplies the exact fp32 lock update: cumulative evidence,
evidence-weighted gain, learned scalar `s` (initially 1), clipped step (default
0.3), offset bound 0.75, and static-node pinning. Turn steps supply no velocity
measurement and invalidate the snapshot; the following observation seeds a new
pair, so the first valid measurement is one observation later. During prediction,
rho/evidence/lock/snapshot receive no measurements or writes. Turns still change
rho's coordinates and node labels. Decoded frames feed the feature recurrence.

Every arm uses the same `_step` call through non-reentrant per-step checkpointing
when gradients are enabled and `checkpoint_steps: true`. The teacher-forcing
coin is drawn outside this call. Evaluation under `no_grad()` avoids checkpoint
bookkeeping. The decoder takes max-over-valid-V cone tokens plus map positions.

## Oracle metadata convention: maintainer decision needed

**The current generator contradicts the brief's interpretation of
`block_vel_agent`.** `generate_videos.py` lines 1730–1755 compute outgoing
`t→t+1` displacement and subtract camera displacement before writing that field.
`stream` also subtracts camera displacement. Consequently the requested literal
mode double-subtracts ego translation and is one transition out of alignment for
an incoming-step recurrence. This matters especially at turns and bounces.

The default **`oracle_velocity: raw`** preserves the explicitly requested
`block_vel_agent[t]` behavior. It should not be interpreted as a physically
validated oracle ceiling on the current generated data. An additional,
explicit **`oracle_velocity: incoming_object`** mode uses
`block_vel_world[t-1]`, expressed using destination heading `agent_dir[t]`, and
lets `stream` subtract ego motion once. The opt-in dataset loader supplies this
as `block_vel_object_incoming`; the requested raw metadata fields stay unchanged.
The first global frame has zero incoming displacement. Missing corrected
metadata raises an error, without silently falling back to raw velocities.
Both conventions mask padded slots identically and read metadata during
prediction as well as context. The configuration default was not silently
changed while the clarification remained unanswered.

## Data, metrics and initialization

`dataset.load_block_info: false` keeps the previous outputs. When enabled,
`<idx>_block_info.pt` supplies `block_pos_agent [T,10,2]`,
`block_vel_agent [T,10,2]` and boolean `block_valid [T,10]` inside `metadata`.
The loader follows the actual video indices, clip start, purpose-dependent
`max_frames`/`n_frames`, padding and frame skip; temporal padding is invalid.
It also passes generator `block_speed` and the corrected incoming velocities
when available. Default PyTorch collation produces `[B,T,...]` tensors.
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
No training configuration was launched.

Inference uses `shortcode=exp/blockworld/kwm3d/infer/metrics_140/k3d_v_70ctx`,
with 70+70 frames and the released metric list plus `block_mse`. Set `load` and
`algorithm.backbone.arm` to the trained kinetic checkpoint. It deliberately does
not pretend a released FloWM checkpoint is a trained A checkpoint.

From the repository root:

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline
export PYTHONDONTWRITEBYTECODE=1
source algorithms/mem_wm/backbones/kwm3d/.scratch/env.sh
/home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/kwm3d -q \
  -o cache_dir=algorithms/mem_wm/backbones/kwm3d/.scratch/pytest-cache
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
| c | 4.047 | 3156 | 4.458 | 2171 |
| c_pw | 1.599 | 2281 | 2.481 | 1741 |
| a_oracle | 1.986 | 2900 | 2.925 | 2196 |
| a | 2.989 | 4282 | 4.773 | 2861 |
| b | 1.809 | 2736 | 2.591 | 1991 |

Raw timings are in `.scratch/bench.log`. GPU memory, GPU timing and campaign
budgets remain unmeasured. In particular these batch-1, 15-frame CPU figures
cannot establish feasibility of batch-8, 90/140-frame GPU backpropagation.

## Verification, deviations and remaining limitations

**40 tests passed, no skips, in 35.16 seconds** on CPU. Only the existing
`pkg_resources` and timm import deprecation warnings were emitted.

Tests cover every arm's reduced forward/backward, C vs released FloWM on two
real clips (3 context + 2 predictions each, tolerance 1e-6), pooled token counts,
33×33 turn inverses and spectral commutation (1e-5), checkpoint gradient agreement
(1e-5 relative, width 32, depth 1, 4+3 frames), invalid oracle slots, strict
released loading, observation-only projection, nonzero lock-update derivatives,
static pinning, turn snapshot invalidation, configuration/factory integration,
metadata signature gating, analytic geometry, real mask IoU, dataset alignment,
padding, skipping and collation. Real-data/checkpoint tests skip only when the
required artifacts are absent. Validation wiring is tested without launching a
Lightning trainer or downloading metric networks.

Differences and extensions relative to the brief/design:

1. The discovered generator velocity convention is retained literally by
   default and a corrected oracle option is added; see the warning above.
   This is an unresolved experimental choice, not evidence for an oracle ceiling.
2. Scalar turns specialize `rot_state` semantics, because the installed function
   rotates orientation channels. The rectangular observation snapshot is
   invalidated, not rotated into a fictitious full cone. Frozen prediction
   snapshots are never read again for measurement.
3. A temporary private-function binding plus a proposed external patch reuses
   exact lock arithmetic without copying code or modifying the read-only fork.
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
freezes node learning even through bounces; Python control flow/private KWM
binding has not been verified under `torch.compile`, TorchScript or export.
The optional plain convolution intentionally relinquishes C4 equivariance.

Maintainer decisions: choose raw versus corrected incoming oracle convention
before interpreting A-oracle results; review/adopt the upstream measurement
hook and pin its replacement revision; await the independent stage-1 kill tests;
measure GPU memory/time before assigning budgets; verify training split
availability and select a trained arm checkpoint for inference.
