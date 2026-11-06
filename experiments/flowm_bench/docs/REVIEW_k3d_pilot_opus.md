# Opus review of 6a99e3c: kinetic 3D C vs A-oracle pilot preparation

Everything ran on CPU (`CUDA_VISIBLE_DEVICES=""`). No GPU was used, no tracked file was changed, nothing was committed. Probe scripts and run outputs are in
`/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/review/pilot/`.
The temporary `base` worktree (at 6a99e3c~1) has been removed. I also deleted one stray `__pycache__/main.cpython-312.pyc` that my own probe had created in the repo. `git status` is clean.

**Bottom line: no blockers.** The plumbing is right, the two arms get the same treatment, and the evaluation reproduces independent computations. There is one major problem, and it is in the pre-registered criterion, not the code: the INCONCLUSIVE guard can almost never fire (M1). The user should hear about it before the GPU run. There is also one missing safety check in the statistics script (M2).

## Blockers

None found.

## Major

**M1. The INCONCLUSIVE guard has almost no power.** Repeat-last-frame is a very weak baseline on block-region MSE. I measured pooled `block_mse` over 100 validation clips (every 10th clip, 70+70) for several trivial predictors (`trivial.py`):

| Predictor | Pooled block_mse |
|---|---:|
| repeat-last (the registered baseline) | 0.2052 (full 1,000 clips: 0.2011) |
| mean of the context frames | 0.1667 |
| flat mean colour of the last frame | 0.1736 |
| constant mid-gray 0.5 | 0.1184 |

Any blurry, regress-to-the-mean model therefore "beats" repeat-last. A model after only 3 optimizer steps already sits at baseline level: C on clips 0–2 scores 0.172 / 0.218 / 0.210, against repeat-last 0.158 / 0.220 / 0.259.

So the "1/50 budget too small to tell" case will almost always come out as PASS or FAIL rather than INCONCLUSIVE. A FAIL could then mean "both arms only learned blur", not "the oracle ceiling does not beat integer transport".

The criterion is pre-registered, so I would not change it silently. Recommended:
- Tell the user before launch.
- Add reported-only reference rows to the statistics output: constant gray and context-mean, both cheap on CPU like `kwm3d_baseline_repeat.py`.
- Report full-frame MSE of each arm against repeat-last (0.0406 at 140, 0.0413 at 280). This is already a "reported" item.

**M2. The statistics script never checks that the baseline covers the same clips as the arms.** `scripts/kwm3d_pilot_stats.py:38-46` asserts that C and A-oracle have the same paths, indices and counts. The baseline only goes through `validate()`. I fed it a baseline built from clips 999, 998, 997 in reverse order against arm dumps of clips 0–2. It accepted it and returned a verdict (INCONCLUSIVE), using baseline pooled 0.1942 instead of 0.2011.

It also never requires 1,000 clips. An accidental `dataset.num_default_clips` override in an eval command would still produce a verdict.

For the real pilot the data does match:
- The baseline paths are exactly the validation metadata index order (first 1,000 = the whole split).
- The eval dataset uses the same paths, in the same order and the same relative format.
- The eval element counts equal the baseline rows (checked on clips 0–2).

Fix: assert `baseline["paths"] == c["paths"]`, the dataset indices, and `element_count` equality, plus `len == 1000`. Keep an explicit `--smoke` flag for the current smaller-sample use.

## Minor

1. **The benchmark leaves data loading out of the 4 GPU-day projection** (`scripts/kwm3d_bench.py`, README "Maintainer GPU commands"). I measured the real training loader (8 workers, batch 8, 90 frames, NAS, shared host with load average about 4.9): first batch 3.8 s, steady state 0.353 s/batch. Each batch is 135 MiB. That puts a floor of about 0.025 GPU-days per arm. It only matters if the GPU step is under about 0.35 s. Recommend projecting `6000 * (max(gpu_C, 0.35) + max(gpu_A, 0.35)) / 86400` and stating that.
2. **Possible GPU-only crash at the end of evaluation.** `algorithms/flowm_video.py` (`on_validation_epoch_end`) asserts that pooled MSE equals the torchmetrics MSE at `rtol=1e-6`. Torchmetrics sums each batch (about 27.5M elements) in float32 before the float64 state. On GPU, tree-reduction error may approach 1e-6. The dump is written before the assertion, so no data is lost, but the run would exit non-zero and the per-split logs would be skipped. It passes on CPU; I could not test GPU. Suggest a warning instead, or `rtol=1e-5`.
3. **The trained-lift gate has two rough edges** (`experiments/flowm_bench/k3d/trained_lift.py`):
   - `--arm c` is accepted, but `observation_write` is the pooled write, which C never trains (C uses `write=per_node`). Restrict the arm to a_oracle / a / b, or document this.
   - `dispatch_checkpoint` re-parses `sys.argv` with the trained-lift parser, so stage-1 flags such as `--split` give an argparse error. This fails loudly, not silently.
4. **The README is stale on two points:**
   - It says the training index was absent and must be prepared. It now exists: 10,000 videos, all 500 frames.
   - It says offline W&B failed. In my runs offline W&B works, both directly and through `main.py` with the training pipeline (`runs/fb/wandb/offline-run-*`), including with the README's `.scratch/tmp` TMPDIR. The coder's failure was most likely its sandbox.
5. **Teacher forcing uses one coin per frame per batch**, not per sample (`kwm3d_models.py:269-270`). In the batch-4×2 fallback, coins are drawn per micro-batch, so the realisation changes. Both arms are affected identically, so this is fair; it is just a note.
6. **Eval runs log 8 videos and in-training validation logs 32 videos** to offline W&B per validation. This costs local disk only; 157 GB is free on `/`.
7. **`enable_version_counter: false` lets a rerun into the same `hydra.run.dir` overwrite `step-*.ckpt`.** The README already warns about this.

## Verified correct (with evidence)

**Protocol fidelity, item 1.** I composed every config with Hydra exactly as `main.py` does (`validate_shortcode_params` + `unwrap_shortcuts` + compose, with the README COMMON overrides; `compose.py`, outputs in `cfg/`).
- The resolved `pilot_c` and `pilot_a_oracle` configs differ in exactly one line, `algorithm.backbone.arm`. The `.hydra/config.yaml` files saved by the real `main.py` dry runs differ only in arm, name and dump path.
- Resolved values:
  - frames: `max_frames = n_frames = 90`, `context_frames: 50`
  - schedule: `max_steps 6000`, batch 8, accumulate 1, `max_epochs -1`, constant learning rate. Checkpoint `lr` is still 1e-4 after 3 steps, so there is no warmup.
  - optimiser: AdamW (`optimizer_mixin.py:35-36`), lr 1e-4, betas [0.9, 0.99], weight decay 1e-3, `gradient_clip_val 1.0`
  - other training settings: teacher forcing 0.125, `precision 32-true`, `checkpoint_steps: true`, seed 0
  - data and initialisation: `splits_included [k3d_v_training]` / `[k3d_v_validation]`, `ema.enable false`, `load null`, `load_model_state null`
- The fallback list shortcode resolves to batch 4, `accumulate_grad_batches 2`, `max_steps 6000` and `val_every_n_step 2000` (batches), for both arms. Nothing else changes.
- Compared with the released `flowm/train/dynamic`: optimiser, learning rate, betas, weight decay, clip, precision, teacher forcing, context 50 and TF32 (`base_exp.py:48`, shared) are identical. The differences are all ones the Addendum intends:
  - batch 8 on 1 device instead of 8×2 DDP
  - 90-frame training clips instead of 140
  - `max_steps` 6000, checkpoints every 1000 steps
  - validation every 1000 steps on 4 batches, no sanity validation
  - the k3d splits
- The instantiated models implement the arm definitions. C: per-node write, non-kinetic, 5-node lattice. A-oracle: pooled write, kinetic, 11 slots, `C4SymmetricDepthwise` map conv, `incoming_object`.

**Data, item 2.** Checked on the real `k3d_v_training` with `metadata_read_only` (`data_probe.py`).
- 10,000 unique videos, all 500 frames, 411 start offsets each, `len = 4,110,000`. Training uses an identity index remap plus a shuffled `RandomSampler`, so every video and every start frame can be drawn.
- On 6 random (video, offset) pairs, the 90-frame clip with frame indices, conditioning actions and all `block_*` metadata equals the neighbouring clip shifted by one frame and the overlapping frames of the offset-0 clip.
- `conds[t] == raw actions[s+t-1]` (the `t-1→t` alignment), `block_pos_agent == block_info[video_pts[s:s+90]]`, `video_pts` is the identity, and `block_incoming_valid` is 1.0.

**Same samples for both arms.** Two-step-plus dry runs of each arm through the real `main.py` on `k3d_v_training` (`dryrun.py`):
- Identical first batches. Step 0: `k3d_v_training/14/0125` clip [308, 320]; step 1: `28/0076` [88, 100]; step 2: `23/0171` [404, 416].
- Identical global torch RNG state before and after every training step, including across the in-training validation at step 2. Hashes: d31ed…, 8cc94…, 7be8d…, a4aba….
- Identical first parameters after initialisation.
- The fallback run (accumulate 2, 2 workers, offline W&B) consumed the same samples in the same order: 4 batches for 2 optimizer steps.
- Data order comes only from the data-module generator seeded with the run seed. Teacher-forcing coins do not touch it.

**Evaluation clip selection.** `validation_first_clips` gives start offsets of 0 (`clip [0, 140]`) for videos in index order. The validation split has exactly 1,000 videos, so the evaluation uses all of them. Both horizons use the same paths. With the flag off, the code path is unchanged (`base_video.py:522`, purpose and flag gated).

**A-oracle at evaluation, item 2 cont.** `_oracle(metadata, t)` uses the clip's frame index `t` over the full 140/280-frame metadata, and `on_after_batch_transfer` passes the metadata untruncated. In-training validation uses the same path.

**Evaluation end to end, items 4 and 5.**
- I ran the real `main.py` evaluation (70+70, 3 clips, dump on) for both arms from my step-3 checkpoints.
- I then recomputed it independently: `load_checkpoint_model`, a direct forward pass on the first-clips dataset, and `clip_stats`. Per-clip SSE matches the dump to a relative 6.8e-9 (C) and 1.1e-8 (A-oracle). Counts are equal.
- Dump `n_frames` is 140 and context is 70. Only predicted frames count (`start = n_context`; actions are `target_actions[:, :T_pred]`, the same window as the baseline).
- The runtime 1e-6 pooled-vs-logged assertion passed.

**Strict checkpoint loading.**
- Loading the C checkpoint with `arm=a_oracle` fails with `Missing key(s) in state_dict: "map_conv.weight"`. Each arm loads its own checkpoint.
- Checkpoint contents: `global_step 3`, every AdamW `step` = 3, there is no EMA callback, and the only callback is ModelCheckpoint with `every_n_epochs 0`.
- The file name pattern produces `step-000003.ckpt` (so `step-006000.ckpt` at 6000), and `save_top_k -1` keeps every checkpoint.
- Evaluation uses `trainer.validate(ckpt_path=load)` on the model weights.

**Baseline, item 4.** Independent recomputation (direct decord decode, raw block_info, float64) for clips 0, 137, 500, 812 and 999 at both horizons. Full-frame and block SSE match within a relative 8e-8, and counts match exactly (`baseline_recompute.py`). The baseline order equals the metadata index order.

**Statistics.** The resampling unit is the clip. NumPy `default_rng(20261105)`, 10,000 resamples, `np.percentile` 2.5/97.5. Clips with an empty region are excluded from the paired values but kept in pooled sums. Pooled values are ratios of sums. The verdict order is INCONCLUSIVE (both arms ≥ baseline), then PASS (a ≤ 0.9·c and CI high < 0), then FAIL. This matches the Addendum.

**Shared files, item 3.** I diffed the released `flowm/infer/metrics_140/dynamic_70ctx` config resolved at 6a99e3c~1 against 6a99e3c. The only differences are the new default keys (`seed: null`, `accelerator: auto`, `metadata_read_only/validation_first_clips: false`) plus a local secrets file absent in the worktree. The released model is `flow_vit`, with no `block_mse` and `load_block_info: false`, so none of the new `flowm_video.py` branches run. Seeding stays at `deterministic 42` for both the generator and the workers.
- `persistent_workers` and `prefetch_factor` change only when `num_workers == 0`, where torch used to raise an error.
- `strategy=None → "auto"` is a pure fix: Lightning 2.6 rejects `strategy=None` with a ValueError.
- Hoisting `resume` in `main.py` keeps the same value.

**Tests.** `pytest tests/kwm3d tests/k3d -q -p no:cacheprovider`: 93 passed, 5 warnings, 53.25 s. Nothing regressed.

**GPU hazards, item 6.**
- `kwm3d_bench.py --device cuda` forwards every argument to the child process unchanged (verified in the child's argv), keeps `CUDA_VISIBLE_DEVICES`, and fails cleanly with "CUDA requested but unavailable" when CUDA is hidden.
- It resets peak memory after warmup.
- It builds the model from the same composed `pilot_c` config with the arm override, and uses the same AdamW, clip and teacher-forcing settings.
- The cap formula is in seconds converted to days, so it is correct.
- Nothing writes to the NAS: metadata is read-only, and outputs, W&B and caches are local.
- Nothing downloads: FVD and LPIPS are excluded, and `skip_download` is set.
- No state grows across steps: node states are detached, Lightning 2 does not keep step outputs, and the new kwm3d epoch-end hook resets video metrics each validation.
- The "5 modules in eval mode" warning refers only to the torchmetrics wrappers.
- In-training validation is 32 clips × 90 frames, which is negligible.
- Disk: 157 GB free; each checkpoint is about 154 MB, so 6 per arm.

**Trained-lift gate, item 7.**
- `pooled_tokens` calls the model's own `observation_write`, the same one `observation_imprint` uses (covered by tests).
- The cache is keyed by checkpoint SHA-256 and arm (`.scratch/trained_lifts/<sha>/<arm>`), and outputs go to `outputs/k3d_trained_lifts/<sha>/<arm>`, so released results are untouched.
- `load_checkpoint_model` reads the training `.hydra/config.yaml` next to the checkpoint and loads strictly. It worked on my dry-run checkpoint.

## Ranked fix list

1. **(M1)** Before the GPU run, raise with the user that repeat-last is beaten by constant gray (0.118 vs 0.205). Add reported-only gray and context-mean reference rows to the statistics output. Do not change the registered criterion without approval.
2. **(M2)** `kwm3d_pilot_stats.py`: assert that baseline paths, dataset indices and `element_count` equal the arms', and require 1,000 clips unless an explicit `--smoke` flag is given.
3. Include the measured loader floor (about 0.35 s/batch at 8 workers) in the 4 GPU-day projection.
4. Relax the runtime pooled-MSE assertion (`rtol 1e-5` or a warning) so a float32 GPU reduction cannot fail an otherwise finished evaluation.
5. Trained-lift: restrict or document `--arm c`, and pass through or strip stage-1 flags in `dispatch_checkpoint`.
6. Update the stale README statements (training index present; offline W&B works on this host).
