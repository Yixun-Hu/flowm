# Plan: our continuous-velocity memory against FloWM on FloWM's own MNIST World benchmark

Campaign tag: `flowm_bench_v1`. Repository `/home/yixunhu/codespace/flowm`, GPU 1 only (shared with another user's jobs, ~23 GB free), Python `/home/yixunhu/miniconda3/envs/flowm/bin/python`. Coding by gpt-6-astra via `codex exec` (workspace-write sandbox); adversarial review of this plan and of every code drop by Opus. Nothing in `experiments/continuous_velocity/` (parent study) or in the FloWM framework (`algorithms/`, `datasets/`, `configurations/`) is edited; new code lives in `experiments/flowm_bench/`.

## 0. The user's goal and what the benchmark allows

Goal as stated: show that Newton sub-pixel refinement, spectral transport and second-order (acceleration) motion let our model beat FloWM on FloWM's benchmark.

Facts about FloWM's benchmark (verified from the downloaded `dynamic_po` data, the configs and the paper appendix):

| Property | dynamic_po (main table) | consequence |
|---|---|---|
| 5 digits per episode, each with its own constant velocity | multi-object | our model estimates one rigid motion per scene; it cannot run on this data without a multi-object extension (Section 2) |
| digit velocities are integers in {-1,0,1}² | on-grid | spectral shift == `torch.roll` and the Newton step starts at the exact answer: **spectral transport and Newton refinement reduce to the baseline on this data, provably**. FloWM's 25 integer channels cover the velocity set exactly. |
| velocities are constant | no curvature | the acceleration term can only be neutral (gate closed) or harmful |
| camera moves by an integer action in [-10,10]² every frame, 50×50 torus, 32×32 window, H.264 mp4 frames | partial observability | identical world/window geometry to ours; our model takes integer camera origins = cumsum(actions) |
| metrics: torchmetrics MSE / PSNR / SSIM pooled over all non-context frames of all clips, 50 context + 20 or 150 predicted | | we reuse `algorithms/common/metrics/video/video_metric.py` unchanged |
| FloWM paper numbers (dynamic_po): MSE 0.0005 @20, 0.0018 @150 | | released checkpoint reproduces 0.00053 @20 on 32 episodes (parent study) |

So the three named contributions cannot, by construction, beat FloWM on the released splits: two of them are the identity on integer motion and the third has nothing to model. What *can* beat FloWM there is the architecture they live in: analytic transport of a fixed canonical memory (no autoregressive feedback, no recurrent decay over 150 frames) with estimated velocities. The three contributions become decisive only when the benchmark's motion is off-grid or accelerating, which FloWM's generator never produces.

Therefore two tracks, both run with FloWM's data format, metric code and context/horizon protocol:

- **Track A (their data, as released).** Multi-object extension of our model, trained on `dynamic_training` (70 frames), evaluated on `dynamic_validation` (20 predicted) and `dynamic_validation_200` (150 predicted), against (i) the released FloWM checkpoint and (ii) FloWM retrained by us with the official pipeline under the same budget. The three switches are run as ablations and are expected to be neutral; the claim under test is "continuous-memory transport beats recurrent velocity channels, especially at 150 frames". Also the `static_po` split (5 static digits, camera motion only) as a second released table.
- **Track B (their benchmark, generalized motion).** A verified re-implementation of the MNIST World generator (no generator exists in the repo; we have per-episode digit labels, initial positions, velocities and actions, so the re-implementation is checked against the released mp4 frames). Two new splits with everything else identical to `dynamic_po`: `dynamic_po_frac` (per-digit velocities continuous in [-2,2]², bilinear rendering, with held-out speed bands) and `dynamic_po_accel` (per-digit constant acceleration, speed bound kept). FloWM is trained on each (25 channels; and an 81-channel half-grid variant, since its design can be densified), with the same pipeline, budget and metrics as ours. This is where Newton, spectral and acceleration are tested.

Primary claims (prespecified, Holm-corrected within each track, paired across training seeds; details in Section 5):
- A1: ours (full) < FloWM (retrained, matched) in MSE@150 on `dynamic_validation_200`.
- A2: ours (full) ≤ FloWM (retrained) in MSE@20 on `dynamic_validation`, non-inferiority margin 1e-4.
- B1: ours (newton+spectral) < FloWM-81 on `dynamic_po_frac`, MSE@20 and @150.
- B2: ours (order 2) < ours (order 1) and < FloWM-81 on `dynamic_po_accel`, MSE@150.
- Secondary: per-contribution ablations (base / +newton / +spectral / +order2 / full) on each split; PSNR and SSIM reported for every primary contrast; the released checkpoint reported alongside the retrained FloWM.

## 1. Phase 0: reproduce FloWM on its benchmark (no new modelling)

1. Official pipeline, released checkpoint: `python -m main shortcode=exp/mnist_world/flowm/infer/dynamic_po_70wf +name=... algorithm=flowm_video dataset=mnist_world ckpt_map=default wandb.mode=offline dataset.num_validation_clips=8000 experiment.devices=[1]`, and the 200-frame split with the `200wf` shortcode **plus** `algorithm.backbone.v_range=2 algorithm.backbone.no_self_motion_equivariance=false` (the released `200wf` config inherits `v_range 0` and no self-motion equivariance from `base_inference.yaml`; this is a config oversight found during reconnaissance and must be stated in the write-up). Metrics MSE/PSNR/SSIM only (FVD/LPIPS dropped: no I3D weights offline and not reported in the paper's MNIST table).
2. Standalone evaluator `experiments/flowm_bench/evaluate.py` that loads any model behind a common interface `predict(frames[B,50,1,32,32], actions[B,T,2]) -> frames[B,T-50,1,32,32]`, reads episodes directly (decord), and calls `VideoMetric` with the same `context_mask`. Acceptance: on the same 8000 clips it reproduces the official pipeline's numbers for the released checkpoint to 1e-6 (MSE) and 1e-3 (PSNR, SSIM). All later numbers come from this evaluator so that every model is scored by literally the same code.
3. Record: paper 0.0005/0.0018; official pipeline on 8000 clips; evaluator on 8000 clips. Any discrepancy with the paper is reported, not hidden (paper may have used 16 clips: `num_validation_clips: 16` in `configurations/dataset/mnist_world.yaml`).
4. Retrain FloWM with the official training shortcode on GPU 1 (`experiment.devices=[1]`, batch 32 to match the paper's 32 = 4×8; the released checkpoint's optimizer step count is 45,000, epoch 8 → budget 45k steps; constant lr 1e-4; checkpoints every 5k; validation on 16 clips every 1k steps as configured, final selection by validation loss). Three seeds. This is the matched FloWM for Track A and the reference for budget matching (same number of training sequences seen).

## 2. Phase 1: multi-object continuous-velocity memory (`experiments/flowm_bench/model.py`)

Reuses `experiments/continuous_velocity/kinetic/` (`KineticCorrespondence`, `spectral_shift`, `refine_peak`, `fit_motion`) by import, never by edit.

Inputs: 50 context frames (mp4-decoded, mean of 3 channels, [0,1]), integer camera origins `c_t = cumsum(a)` (window top-left = camera centre − 16; `t−1→t` alignment as in the dataset config), future camera origins for the target frames. No positions, labels or velocities enter the forward path.

Design: **K estimated velocity hypotheses instead of 25 fixed integer channels.**
1. **Correspondence density.** As the parent: encoder features, masked FFT correspondence over frame pairs at lags {1,2,4}, giving per-pair normalized similarity on the displacement lattice. With several digits the density is multimodal; the lag-pooled similarity (displacement/lag → velocity, resampled onto a common velocity grid of step 0.25 px/frame over [-2.5,2.5]²) has one peak per distinct digit velocity.
2. **Mode extraction.** Non-maximum suppression on the pooled velocity map → up to K=8 candidate velocities with their peak strengths (K ≥ 5 digits; distinct velocities ≤ 5; extra slots are low-weight). Differentiable through the peak values (positions are integers; optionally refined). Order-invariant: hypotheses are a set, processed identically.
3. **Per-hypothesis motion fit.** For each hypothesis k, the parent's robust WLS fit over the 36 pairs is run on the displacement sub-lattice around `v_k Δt` (window ±1.5 px): `refine='newton'` refines each pair's displacement on the trigonometric interpolant; `order=2` adds the gated acceleration. Output `(v_k, a_k)` continuous.
4. **Per-hypothesis canonical memory.** Align all 50 context frames with offsets `v_k τ + ½ a_k τ²` (bilinear or spectral switch) and fuse by the parent's normalized convolution → `M_k`. Content moving with `v_k` is coherent in `M_k`; other digits smear. A per-pixel **coherence weight** `w_k(u)` = exp(−β · temporal variance of the aligned stack at u) / normalizer (β learned) gives a soft assignment of appearance to hypotheses; the memory kept is `M_k ⊙ w_k`.
5. **Rendering.** For target time τ and camera c: each `M_k ⊙ w_k` is transported by `v_k τ + ½ a_k τ²` (bilinear/spectral), cropped to the window, composited by `softmax_k(γ · w_k)`-weighted sum (γ learned) with a max-over-k alternative as an ablation (FloWM's readout), then the parent's decoder. Fixed memory, direct rendering at absolute time: no autoregression.
6. **Controls built in.** `K=1` reduces to the parent model (sanity test on the kinetic test suite). `hypotheses='integer_grid'` replaces estimated modes by the fixed 25 integer velocities (FloWM's velocity set) with the same memory/rendering — isolates "estimated continuous hypotheses" from "analytic transport memory".
7. **Training.** Image-only MSE on the 20 target frames exactly as FloWM's loss (`flowm_video.py`), Adam, lr selected in {3e-4, 1e-3} on seed 2026 by validation loss, batch 32, budget matched to FloWM's by number of sequences seen (45k × 32 ≈ 1.44M sequences, i.e. 8 epochs of 180k), gradient clip 1.0, fp32. Three seeds (2027–2029) per variant; five if time permits.

Compute: each hypothesis costs one parent-model alignment (50 frames × 50×50 field); K=8 ≈ 8× the parent's per-step cost (parent ≈ 20 ms/step at batch 8). Estimate ≈ 0.5 s/step at batch 32 → 45k steps ≈ 6 h per run on a shared A6000. 5 variants × 3 seeds ≈ 90 h; plus FloWM retraining ≈ 3 × 4 h. Feasible within a week on one GPU; shards by variant.

Known risks, stated up front: digits overlap and occlude (coherence weights degrade at crossings); digits never seen in context cannot be predicted by any model (floor shared with FloWM); hypothesis count K < number of distinct velocities loses a digit (K=8 vs ≤5 distinct velocities, safe for `dynamic_po`; `dynamic_po_frac` has 5 distinct velocities almost surely → K=8 still ≥ 5); mp4 compression noise in frames.

## 3. Phase 2: Track A campaign on released splits

Variants (ours): `base` (K hypotheses, bilinear, no refine, order 1), `newton`, `spectral`, `order2`, `full` (newton+spectral+order2), `grid25` (fixed integer hypotheses control), `max_readout` (FloWM-style readout ablation). Baselines: released FloWM; retrained FloWM ×3 seeds. Splits: `dynamic_po` (train `dynamic_training`; eval `dynamic_validation` 20 frames and `dynamic_validation_200` 150 frames, all 8000 clips each), `static_po` (train `static_training`; eval both static validation splits). Metrics: MSE, PSNR, SSIM via the shared evaluator; MSE-vs-frame-index curves; per-episode paired differences.

## 4. Phase 3: Track B generator and splits

1. `experiments/flowm_bench/mnist_world_gen.py`: re-implementation of the generator (MNIST test digits placed on a 50×50 torus at integer positions, 5 digits, wrap-around, window = camera centre − 16, actions uniform integers in [-10,10]², 70/200 frames). **Verification**: for every one of the 8000 `dynamic_validation` episodes, re-render from the stored labels, initial positions, velocities and actions; require per-frame mean absolute error against the decoded mp4 below the H.264 noise floor (threshold set from the frame-to-frame quantization noise, expected < 0.01) and identical digit bounding boxes. Which MNIST split (train/test) and which digit index the labels refer to is resolved by matching; if no exact digit match is possible, the generator uses the same labels with matched digits and this is declared.
2. New splits written in the same `.mp4 + .pt` layout (mp4 at the same codec settings, so compression effects are shared; plus lossless `.npy` frames kept for a sensitivity check): `dynamic_po_frac` (per-digit velocity uniform in the disc |v| ≤ 2, components continuous, bilinear sub-pixel rendering; training excludes speed bands [0.35,0.4] ∪ [0.6,0.66] ∪ [1.2,1.26] which form the held-out evaluation bands), `dynamic_po_accel` (initial velocity as `frac`, per-digit acceleration 0 w.p. 0.3 else log-uniform magnitude in [2e-4, 2e-2] with |v(t)| ≤ 2.4 over 200 frames). 180k training / 8000 validation episodes each, 70 and 200 frames.
3. Baselines on Track B: FloWM-25 and FloWM-81 (half-grid velocity set; `v_list` spacing 0.5 requires fractional rolls: implemented as accumulated-offset nearest-integer transport exactly as the parent study's `densify.py`, which matched the integer implementation with max abs diff 0.0), both trained from scratch with the official pipeline and the same budget; ours: `base`, `newton`, `spectral`, `order2`, `full`.

## 5. Statistics and reporting

- Unit of replication for superiority claims: training seed (paired t-test across 3 seeds, one-sided where the direction is prespecified; with 3 seeds the minimum attainable p is 0.125 one-sided, so **the primary evidence is the paired-by-episode bootstrap CI of the mean MSE difference pooled over seeds, with the seed-level test reported alongside**; if all three seeds agree in sign and the episode-bootstrap CI excludes zero, the claim is "supported").
- Holm correction within each track's primary family (A: 2 claims; B: 2 claims).
- Non-inferiority (A2) at margin 1e-4 MSE, one-sided upper 95% bound.
- PSNR and SSIM reported for every contrast, never selected on.
- Every evaluation records model and evaluator source hashes; the report cross-checks them.
- Negative results are reported as such; the write-up states explicitly that spectral and Newton are identities on integer motion and that Track B is our generalization of their benchmark, not their released data.

## 6. Order of work and checkpoints for review

1. Phase 0 (reproduction + evaluator) — starts now; result gates everything.
2. Opus adversarial review of this plan; revise.
3. Phase 1 model written by gpt-6-astra against a spec and the kinetic test suite; Opus review of the code; CPU tests: K=1 equals the parent model bit-for-bit; grid25 control; order-invariance of hypotheses; camera-origin invariance; synthetic 2-digit scene where both velocities are recovered.
4. Phase 2 training (ours + FloWM retrain) on GPU 1; interim report after the first seed.
5. Phase 3 generator (gpt-6-astra) with verification against released episodes; Opus review; splits generated; Track B training.
6. Final report (`experiments/flowm_bench/RESULTS.md`, figures) and commit/push.

## 7. Explicit non-claims

No claim about DFoT/DFoT-SSM (not retrained; their released checkpoints are absent and their numbers are two orders of magnitude worse than FloWM in the paper). No Blockworld. No claim that spectral or Newton help on integer motion. Track B results are on our generalization of their generator.
