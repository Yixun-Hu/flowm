# Plan v2: a Galilean world model with acceleration, built on kinetic-world-models

Supersedes PLAN_FLOWM_BENCH.md (v1) after its adversarial review (REVIEW_plan_v1_opus.md). Project goal as stated by the user: finish a Galilean world model with acceleration and raise world-model performance on continuous 2D and 3D environments. Base: `/home/yixunhu/codespace/kinetic-world-models` (KWM, upstream 302d8f8; 69 tests pass with the flowm env). Our components come from `/home/yixunhu/codespace/flowm/experiments/continuous_velocity/kinetic/`. Delivery: a fork under Yixun-Hu, one pull request per component against upstream main, each with tests and an evaluation table; coding by gpt-6-astra (`codex exec`, workspace-write), adversarial review by Opus of this plan and of every PR before it is opened. GPU 1 only (shared; ~22 GB free). Phase 0 of v1 (released FloWM reproduced on its benchmark) stands and is referenced.

## 0. Why KWM is the base, and what it leaves open

KWM already carries the structure the Galilean group asks for: a memory with a fiber over velocities (boost subgroup), exact real-valued translation and boost transport through the Fourier phase, online operation, five independently moving objects, FloWM re-implemented to 1e-6, the Kinetic MNIST World benchmark (`kwm/env.py`: real-valued digit velocities and camera motion, 50×50 torus, 32 window, 5 digits, 30 or 50 observed frames, up to 150 predicted), boost-equivariance measurement (`kwm/equiv.py`). Its own results locate the open problems (docs/RESULTS.md, all at 32 channels, batch 16, one or two seeds):

| Gap | Evidence (MSE, real velocities + real camera) | Component that targets it |
|---|---|---|
| Node velocity estimation | adaptive 0.036 vs oracle nodes 0.006 at 20 frames; node error median 0.054 px/frame; "digits in the same lattice cell compete for one node"; "some digits not locked by the end of observation" | PR1 measurement-driven node update |
| No acceleration | nodes carry constant velocities; `env.py` has no accelerated motion | PR2 second-order node state + accelerated environment |
| Long-horizon drift | adaptive 0.089 at 150 vs oracle 0.043; spectral model saturates to white; error not monotone in training | PR3 freeze-and-render prediction |
| Observation writing | `lift` overwrites the window in every channel with the encoded frame; no coverage mask; partial-visibility handling is learned | PR4 visibility-aware fusion |
| Evaluation | one or two seeds, 300 sequences, no source hashes, no interventions | PR5 methodology |
| No rotations | translation + boost only (also FloWM, also ours) | out of scope for this project; stated |
| 3D | none | Stage 2, after 2D components are proven |

## 1. Components, in PR order (each PR is independently switchable; defaults unchanged)

### PR1 — Measurement-driven node velocities (`--node_update corr`)
KWM's `AdaptiveNodeCell` estimates each node's offset ρ_i by a Lucas-Kanade (LK) step between the transported imprint and the new imprint, pooled over the window with evidence weights, then an evidence-weighted running mean with one learned gain. LK is a single-step, small-displacement estimator (gate |d| ≤ 0.6 px), so two digits in one lattice cell average, and a digit whose velocity is 0.75 px/frame from its node needs several steps to lock.

Our replacement for steps 2–3 of KWM's update, per node i and per step:
1. **Masked normalized similarity on the lattice**, as in our `kinetic_observations`: between the transported imprint m_i (restricted to the window, with the window mask) and the new imprint P(U f_t), over integer displacements within an admissible radius (|δ| ≤ r, r = 2.5·Δt + 1 px as in the parent), via masked FFT. This replaces LK's local linearization with a global search on the window: a digit 0.75 px off its node is found at the first observation, and two digits in one cell give two peaks instead of one average.
2. **Sub-pixel refinement**: Newton steps on the trigonometric interpolant of S = 2N/D (our `refine_peak`, with the denominator-floor guard), accepted under the same guards.
3. **Multi-lag evidence**: keep the last L imprints per node (L = 3 lags: 1, 2, 4 steps, transported exactly by the spectral shift, so no extra estimation error) and fit the node velocity by the robust weighted least squares of our `fit_motion` on the L displacement measurements (Cauchy reweighting, scale .15). Longer lags give the velocity precision that single-step LK cannot (the v1 review: lag·Δv must reach 2–4 px to resolve; lag 4 resolves Δv ≈ 0.75).
4. **Update rule**: ρ_i ← ρ_i + s·g_i·(v̂_i − ν_i), gain g_i = w_i/W_i as in KWM (evidence-weighted running mean, learned scalar s), with w_i the robust fit's total weight; multi-peak case: when the lattice similarity has a second peak above 0.6 of the first inside the cell, the node keeps the peak nearest its current velocity (the other digit is left to the neighbouring node; KWM's overlapping cells make this a gauge choice).
5. **Diagnostics**: node velocity error against true digit velocities (KWM's `node_velocity_error_median`), lock time per digit, fraction of digits with no node within 0.1 px/frame.

Tests: synthetic two-digit cell with Δv = 0.8 resolves both; a digit 0.75 off its node locks in one step; identical to KWM's LK path when `--node_update lk`; boost equivariance of the update (shift scene and camera by u → node velocities shift by u, test to 1e-3).
Expected: node error median well below 0.054; 20-frame MSE moving from 0.036 toward the oracle 0.006. Primary hypothesis H1: corr < lk in MSE@20 on Kinetic MNIST World (real velocities + real camera), 5 seeds, paired by seed and by validation sequence.

### PR2 — Second-order node state and accelerated environment (`--order 2`, `env --accel`)
Node state (ν_i, α_i). Transport by ν_i Δt + ½ α_i Δt² (the spectral shift takes any real displacement; the composition over a step with changing velocity is exact for piecewise-constant acceleration). The multi-lag displacement measurements of PR1 fit (v, a) jointly by our gated second-order WLS (`fit_motion(order=2)`: sandwich variance, n_eff inflation, gate g = clip(1 − κ Var(a)/a², 0, 1), κ = 9 default, κ ∈ {0, 4} ablations). With L = 3 lags the design matrix is 3×2; we extend the imprint bank to L = 4 lags (1, 2, 4, 8) when `--order 2`, which fits within 30 observed frames. The acceleration is frozen during prediction (no measurements), velocity is advanced by α per step.

Environment: `env.py` gains `--accel` (per-digit constant acceleration: zero w.p. 0.3, else signed log-uniform magnitude in [2e-4, 2e-2] px/frame², rejection-sampled so |v(t)| ≤ vmax over the sequence length; the realised (a, v0) joint distribution is logged) and reuses KWM's existing `--walls` for velocity changes out of view (not in primary claims). Lattice coverage: `--vmax` already sets the node lattice; accelerated runs use vmax = 2.4 to cover the speed bound.

Tests: a digit with a = 4e-3 is tracked with |v error| < 0.01 px/frame after 30 frames; zero-acceleration data gives gate-open rate below 2% (null calibration); order 2 equals order 1 bit-for-bit when the gate is closed; prediction positions follow p0 + vτ + ½aτ² on a synthetic scene.
Primary hypothesis H2: order 2 < order 1 on the accelerated benchmark in MSE@150; non-inferiority of order 2 on the constant-velocity benchmark at margin 0.002 MSE@20 (KWM's CI width; justified from seed variance in PR5).
Baselines that can represent acceleration: none exist in FloWM/KWM; we add the kinetic study's learned-offset control (`relaxed`: a small MLP on the node state summary predicting a residual displacement per step) as the flexible baseline, so the gain is not a win by construction.

### PR3 — Freeze-and-render prediction (`--dec_input frozen`)
KWM predicts autoregressively (`prev` fed back, or zeros with `dec_input='zero'`), running the cell every step. Our mode: after the last observation, stop updating memory content; for prediction step τ, transport the frozen memory by the node motions (ν_i τ + ½ α_i τ², real-valued, one spectral shift from the frozen state, no composition of τ steps) and decode. The ego motion is applied as the cumulative action sum. This removes error accumulation and the white-out failure of the spectral model; it also makes long-horizon cost O(1) per frame instead of a recurrence.
Ablations: `zero` (KWM), `flowm` (KWM default), `frozen`; and `frozen` with node updates on (hybrid: update nodes from predictions, as KWM allows).
Tests: at τ = 1 frozen equals one cell step with zero input up to the mix/activation (documented difference); linear growth of the frozen error with velocity error (slope test against the v1 review's drift table); permutation of nodes leaves output unchanged.
Primary hypothesis H3: frozen < flowm in MSE@150 for the adaptive model, 5 seeds.

### PR4 — Visibility-aware fusion (`--lift ncv`)
KWM's `lift` writes the encoded frame into the window of every channel additively before the ReLU (`act(lift + mix(stream(h)))`), with no notion of coverage; the model must learn how to merge new observations with transported content. Our normalized-convolution writing: keep a coverage channel c_i(x) ∈ [0,1] per node transported with the memory; write the observation as h ← (c·h_streamed + ρ·u)/(c + ρ) on the window with ρ a learned per-node write weight (sigmoid of a confidence from the correspondence quality of PR1), c ← c + ρ. This is the parent's `memory_attention` fusion with the parent's visibility mask, adapted to the recurrent setting.
Tests: a static scene observed twice yields the mean; an unobserved pixel keeps its content; boost equivariance preserved.
Hypothesis H4 (secondary): ncv ≤ KWM lift in MSE@20 and @150; its main expected effect is sharper long-horizon memory with PR3.

### PR5 — Evaluation methodology
Source hashes of `kwm/` recorded in every checkpoint and eval json, cross-checked by a report script; ≥ 5 seeds per primary arm; validation 1000 sequences (KWM uses 300; CI ≈ 0.002 at 300); exact permutation test on seed means (5 vs 5, min p 1/252), Welch t-test, hierarchical bootstrap over seeds × sequences; non-inferiority bounds from seed variance; Holm within each PR's primary family; oracle-node and zero-acceleration interventions as diagnostics; localization success (template detector within 2 px, from the parent's `paper/metrics.py`, adapted to 5 digits via per-digit templates and masks); per-frame MSE curves with seed bands; cost table (parameters, FLOPs/step, latency). A `scripts/campaign.py` runs the full matrix shardable on one GPU; a `docs/RESULTS_v2.md` is regenerated from the json.

## 2. Benchmarks (all in KWM's `env.py`, same 50/32 geometry, 5 MNIST-train digits, additive compositing)

| Name | Digit motion | Camera | Observed / predicted | Role |
|---|---|---|---|---|
| K-int | integer {-2..2}² (FloWM's) | integer | 50 / 20, 150 | sanity: all models must match FloWM-level 0.0007/0.0020 (published weights) |
| K-real | real velocities, |v| ≤ 2 | real | 30 / 20, 150 | KWM's main table; H1, H3, H4 |
| K-accel | real v0 + constant a (above) | real | 30 / 20, 150 | H2 |
| K-real-nearest | as K-real, digits rendered at rounded positions | real | 30 / 20, 150 | renderer robustness (the parent's known weakness; reported, not primary) |
| FloWM dynamic_po (released data) | integer {-1,0,1}² | integer ±10 | 50 / 20, 150 | external anchor: KWM `benchmark_official.py` path; all models evaluated through our Phase 0 evaluator |

Training: KWM defaults (32 channels, 25 nodes, batch 16, lr 1e-3, bf16 autocast, 20k steps) for the main matrix; one run per arm at FloWM's sizes (`--preset flowm`: 64 channels, batch 32, 50 observed) for the external anchor. Budget per the KWM reproduction doc on an A6000: flowm ≈ 3.6 h, spectral ≈ 5 h, adaptive ≈ 10 h per 20k-step run; PR1's correlation search adds ≈ 2 FFT pairs per node per step (≈ +30% over adaptive, measured before committing). Matrix: {lk, corr} × {order 1, 2} × {flowm, frozen} on K-real and K-accel, 5 seeds = 40 runs × ≈ 12 h ≈ 480 GPU-h, plus FloWM and spectral baselines 2 × 2 × 5 = 20 runs × 4.5 h = 90 GPU-h. On a shared GPU this is 4–6 weeks; the plan therefore runs seeds incrementally (2 seeds for every arm first, then 3 more for the arms in primary contrasts) and stops arms that are dominated after 2 seeds (prespecified: an arm is dropped if both seeds are worse than both seeds of its comparator by more than 2 CI widths).

## 3. Claims and non-claims
Claims are H1–H4 above, each on Kinetic MNIST World (KWM's benchmark, credited), against unmodified KWM and retrained FloWM at matched budget, plus the external anchor on FloWM's released split (where we expect parity at best, and say so). Newton refinement and spectral transport are KWM's components; our contribution is the measurement-driven multi-lag estimator, the gated second-order state, the frozen analytic prediction, the visibility-aware fusion and the methodology. No rotation equivariance. No 3D claim in Stage 1.

## 4. Stage 2 (3D), scoped after Stage 1
FloWM-3D is a ViT over first-person 128×128 views with a `v_range 1` velocity set in a top-down latent map. The components that transfer are PR2 (second-order node state), PR3 (frozen analytic prediction of the latent map) and PR5. PR1 needs a correspondence on the latent map rather than on pixels. A plan for Stage 2 is written only after H1–H3 are settled; it will need the Blockworld data (not on disk) and a model an order of magnitude larger.

## 5. Order of work
1. Fork; branch `pr1-corr-nodes`; gpt-6-astra implements PR1 against this spec and KWM's tests; Opus review; 2-seed pilot on K-real (lk vs corr); open PR1.
2. PR5 scaffolding early (hashes, seeds, campaign script) since every later table depends on it; can be PR1's sibling.
3. PR2 (env + order 2), PR3, PR4 in that order, each with its pilot.
4. Campaign per Section 2; RESULTS_v2; final PRs updated with tables.
