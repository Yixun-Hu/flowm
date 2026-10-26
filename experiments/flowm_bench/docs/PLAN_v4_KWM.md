# Plan v4: translation-boost equivariant world model with acceleration, built on kinetic-world-models

Supersedes PLAN_v3_KWM.md after its adversarial review (REVIEW_plan_v3_opus.md). Changes from v3: PR3 re-scoped (freeze-and-render fails an evaluation-time check; zero-input prediction is the new comparator and candidate), K-accel law re-specified from its realised statistics with an oracle ceiling, PR2 estimator rewritten as centred-time cumulative weighted least squares on top of the unchanged order-1 update, primary evaluation changed to a checkpoint average on a held-out seed, statistics and budget fixed, rotation stage rewritten (exact C4 first), issue draft corrected.

User decisions (2026-10-25): pilot runs to 20k; branches on the fork, upstream issue first, stacked PRs, AI disclosure, explicit approval before every public action; 2D first, short 3D feasibility step, rotations later toward the full Galilean group; 3 seeds screening / 5 seeds primaries; "translation-boost equivariant" wording until a rotation PR exists.

## 0. State of evidence (2026-10-25)

| Item | Status |
|---|---|
| Released FloWM reproduced on its benchmark | done (`experiments/flowm_bench/README.md`) |
| Bit-exact MNIST World generator | done (`mnist_world_gen.py`) |
| Estimator gate: lock-then-refine vs KWM's LK, randomly initialised encoder, 3 × 100 sequences | done: matched median 0.082 → 0.024 px/frame, misses 42% → 18% (fork `docs/GATE_estimator.md`) |
| PR1 `--node_update lock` | done: branch `pr1-lock-nodes` (28ba51e), lk path bit-identical to upstream, 95 tests |
| PR5a evaluation hygiene | done: branch `pr5a-eval-hygiene` (ad7f881 + cache self-description addendum), 120 tests; integration branch `int/pr1-pr5a` |
| Pilot lk vs lock, seed 0, 20k steps, K-real | running (lk ≈ 12k, lock ≈ 9k steps). Selection-set MSE@20 at matched checkpoints: lock 0.028–0.031 vs lk 0.033–0.036. MSE@150: lock 0.077–0.087 at every checkpoint; lk ≈ 0.093 typically with spikes to ≈ 0.2 at 3 of 11 checkpoints. One seed, selection set, mid-training: not evidence yet |
| Evaluation-time freeze-and-render on the pilot checkpoints (review) | 2.2× worse than the default rollout at MSE@20 and 1.7× at MSE@150 (ghosting: the lift writes every frame into all 25 nodes). Zero-input prediction is ≈ 10% better than the default at every horizon for both rules |
| Anything beating KWM or FloWM end to end with ≥ 3 seeds | not yet |

## 1. Benchmarks

KWM's `kwm/env.py` (50×50 torus, 32 window, 5 MNIST digits, MAX compositing; FloWM's released data is additive with clip, a documented domain shift). 30 observed frames; 20, 50 and 150 predicted.

| Name | Digit motion | Camera | Role |
|---|---|---|---|
| K-real | real velocities, ‖v‖∞ ≤ 2 | real | H1, H3, order-2 non-inferiority |
| K-accel (PR2) | real v0 + constant per-digit acceleration, law below | real | H2 |
| K-int | integer velocities and camera | integer | sanity, 1 seed per arm, no claim |

**K-accel law (to be finalised from realised statistics, not the proposal).** Per digit, a = 0 with probability 0.3; otherwise direction uniform and |a| log-uniform in [2e-3, 2e-2] px/frame² (the lower decade of v3's range is undetectable in 30 frames and only dilutes the effect). (v0, a) are rejection-sampled jointly so that ‖v(t)‖∞ ≤ 2 over a fixed horizon of 180 frames in time units (so the dt = 0.5 cache uses the same law); `--accel` with `--walls` is rejected. Positions use the closed form p(t) = p0 + v0 t + ½ a t²; `vel[t]` is the mean velocity over [t, t+1] so `node_velocity_error` stays consistent. Before any GPU run, `scripts/accel_law.py` reports the realised acceptance, the a = 0 fraction, |a| quantiles, the per-|a|-bin corr(a, v0) (large accelerations are mostly decelerations under joint rejection: a shortcut the controls below address), and the median ½|a|τ² at τ = 20/50/150. These realised numbers, not the proposal, go into the PR doc. `EnvConfig` gains `accel`, included in `tag()` and `env_cfg_from_args`, with a test that K-real and K-accel tags differ.

**Oracle ceiling for H2 (CPU, run for the final law and placed in the PR2 doc).** MSE of a perfect order-1 extrapolation against the true accelerated future, using (a) the mid-observation velocity (what a running mean converges to), (b) the instantaneous velocity at t = 29 (what a forgetting-factor filter can reach at best). For the v3 law the review measured (a) 0.0168 / 0.0450 and (b) 0.0043 / 0.0279 at MSE@20 / @50: about 75% of the order-1 loss at τ = 20 is averaging lag, which the forgetting comparator removes without an acceleration state. The order-2 headroom over that comparator is therefore at MSE@50, which is H2's primary endpoint.

**Evaluation protocol (every table).** Held-out test seed 20261025 (`kwm.eval --test_seed`, PR5a; cache built, 1000 sequences); the selection seed 12345 is reported only as a secondary column. Primary number per run = mean over the last five checkpoints (16k, 17k, 18k, 19k, 20k; `--keep_ckpts`), because training uses a constant learning rate and `last.pt` is a noisy endpoint (lk's MSE@150 spikes). Checkpoint-to-checkpoint SD is reported next to seed SD. `best.pt` is not used for claims. Evaluation GPU time is budgeted (5 checkpoints × arms × 1000 sequences × 180 frames). All tuning of any knob (forgetting factor, κ, zero-input choice) happens on the selection seed with equal effort per arm, before the test-seed evaluation.

## 2. PRs, stacked, in order

### PR5a — Evaluation hygiene (done; small; may go first)
`--test_seed`, self-describing validation cache (seed, n, T, env hash, env.py hash, MNIST hash in the payload; verified on load), per-sequence MSE, provenance (source hashes, software, git, resume history), `scripts/report.py` (paired t-test df = n − 1 primary, exact sign-flip, hierarchical bootstrap resampling seeds then sequences jointly across the paired arms, Holm; validates test seed, n, env and checkpoint per pair). The upstream grad-checkpoint bug (`--node_update lk --grad_ckpt --node_update_pred off`: gradients differ from the non-checkpointed run by 0.4–3% relative depending on configuration, no error raised; affects only an experimental flag not used in the published results) is filed as a separate issue with the reviewer's minimal reproduction, not as part of this PR.

### PR1 — Lock-then-refine node velocities (`--node_update lock`) — implemented
Spec and gate evidence in the fork's `docs/PR1_lock_nodes.md`. Before opening: H1 table on 5 seeds, the 25-node oracle row added to the gate (so the remaining 18% misses are attributable to the estimator rather than to node assignment), Opus pre-PR review of the final diff.
**H1 (primary):** lock < lk in MSE@20 on K-real, checkpoint-averaged, test seed, 5 seeds, paired t-test. Five seeds always run; the 3-seed look only gates when PR2 screening starts. **Secondary:** MSE@50, MSE@150 (reported as a mean and as stability: fraction of checkpoints above 1.5× the arm's median), node velocity error on the full test set (raise `node_velocity_error` from n = 64 to the test-set size), boost-equivariance error, `--node_update_pred off` at evaluation time, step time and memory.

### PR2 — Second-order node state and accelerated environment (`--order 2`, `--accel`)
**Estimator.** The order-1 update (running mean with gain evidence/W, learned scalar `n_s`, step clip 0.3, clamp ±0.75) is kept with exactly its existing arithmetic and produces ν̄. On top of it each node accumulates centred-time weighted sums over its per-step velocity measurements z_t = ν + δ_t/dt (δ_t from `lock_measurement`, weight w_t = selected evidence): S_w, S_wt, S_wtt, S_wz, S_wtz (fp32, in `LockState.nodes`, which grows from 4 to 9 columns; `bounce`, `boost_transform_nodes`, diagnostics and `node_velocity_error` are updated and tested). The slope α̂ = (S_wtz − t̄ S_wz)/(S_wtt − t̄ S_wt) with t̄ = S_wt/S_w is the acceleration estimate; the node velocity used for transport is ν = ν̄ + g α̂ (t − t̄). With g = 0 every floating-point operation on ν is the order-1 one (test: bit for bit). The gate g = clip(1 − κ Var(α̂)/α̂², 0, 1) uses an empirical variance: consecutive one-step displacements share frames, so their errors are MA(1) and a white-noise plug-in variance is wrong; Var(α̂) is estimated from the residual scatter of the fit with the effective-sample-size correction calibrated on the null (a = 0) rate measured in the gate harness, target gate-open rate < 2% per node-step among nodes with evidence at t = 30 on a = 0 data. κ = 9 default; ablations κ ∈ {4, ∞}. The gate and the sums are detached (same policy as the other discrete decisions); `n_s` does not scale α. On a mode switch or lock loss the sums are reset (a mode switch is a velocity jump, not an acceleration). Transport per step by ν dt + ½ α dt², then ν ← ν + α dt (composes exactly to ντ + ½ατ²; test). Node bound: clamp to the ±0.75 cell in both phases with counters; H2 bins are reported next to the clamp rate.
**Prediction phase.** With `--node_update_pred off`: ν advances by α each step, α frozen. With `on` (KWM default): measurements from the model's own frames still enter the order-1 running mean, α frozen; both settings are evaluated, and `off` is the primary H2 setting because `on` lets the order-2 model confirm its own extrapolation.
**Comparators** (all tuned equally on the selection seed): order 1; order 1 with a forgetting factor on the running mean; order 1 with a learned shrinkage of ν toward zero during prediction ("deceleration prior": the control for the corr(a, v0) shortcut). Order 2 must beat all three at MSE@50 to claim that it uses measured acceleration.
**Gate harness before GPU** (`scripts/estimator_gate.py --accel`): null gate-open rate on a = 0 data; detection curve (fraction of nodes with gate open vs |a|; with 30 frames and κ = 9 roughly |a| ≥ 3e-3 is detectable); single-digit synthetic scene with a = 4e-3: |v error at t = 29| threshold set from this run, not from v2; readout OOD check (the moments readout uses ν as a feature; report its drift under order 2).
**Tests:** bit-for-bit order 1 at g = 0; null gate rate; synthetic constant acceleration; exact composition; α unchanged under a comoving boost; fractional dt; `LockState` helpers under the 9-column layout; K-real/K-accel cache tags differ.
**H2 (primary):** order 2 < order 1 with forgetting factor in MSE@50 on K-accel, `--node_update_pred off`, checkpoint-averaged, test seed, 5 seeds for all three arms (15 runs; the deceleration-prior control at 3 seeds, secondary). **Secondary:** MSE@20 vs plain order 1; per-|a| bins with clamp rates; `pred on`. **Non-inferiority on K-real:** order 2 vs lock order 1 at MSE@20, margin fixed in advance at 5% of the lock arm's mean, 5 paired seeds (same seeds as PR1), one-sided 95% TOST. The oracle model is not run on K-accel (its constant-v0 oracle is undefined there); stated.

### PR3 — Prediction-phase input: zero-input vs autoregressive, and a gated freeze
The review measured that the zero-input rollout (`--dec_input zero`, already in KWM) beats the default autoregressive rollout by ≈ 10% at every horizon on both pilot checkpoints, while freeze-and-render is 2× worse because the node-agnostic lift leaves 25 copies of the last frame that frozen transport spreads apart.
**Re-scoped PR3.** (i) H3 becomes: zero-input < flowm in MSE@150 on K-real with the lock model, evaluation-time switch on the PR1 checkpoints (free) and then trained with `--dec_input zero` (3 seeds). (ii) Freeze-and-render stays out of the PR unless it passes an evaluation-time gate on the pilot checkpoints: frozen ≤ min(flowm, zero) at MSE@20 and @150 on the selection set. Two candidate fixes to try at evaluation time, in order: freeze the pre-lift memory `mix(stream(h_{T−1}))` (and k-step variants) instead of the post-lift state; node-selective lift weighted by the lock measurement's per-node pixel weights (a model change, needs training). If neither passes the gate, frozen is dropped and documented as a negative result. Hybrid mode and the trivial node-permutation test are dropped; the ghosting node-concentration diagnostic stays.

### PR4 — Visibility-aware fusion: deferred (reasons in REVIEW_plan_v2_opus.md MAJOR 8).

### PR5b — Campaign script and results
`scripts/campaign.py` (matrix, resume, shardable) and `docs/RESULTS_v2.md` from `scripts/report.py`.

### Stage 2 — 3D feasibility (one week, no claims)
Obtain FloWM Blockworld data, run the released FloWM-3D evaluation once, write a note on whether PR2 (second-order state) and PR3 transfer to the ViT latent map. Go/no-go criterion: the FloWM-3D latent map must have an explicit velocity-node axis on which analytic transport and a node state can act; if it does not, Stage 2 needs a different design and is re-planned.

### Stage 3 — Rotation equivariance (toward the full Galilean group)
Two separate things, planned separately:
- **Symmetry (the claim):** global rotation of scene, camera, actions and velocities, x → Rx, v → Rv, a → Ra. On the 50×50 torus only rotations by multiples of 90° are maps of the world to itself, and the square velocity lattice is closed only under C4. First target: **exact C4 equivariance**: p4-equivariant `conv_u`, `conv_h`, decoder and readout projection (pure torch, weight tying under rot90, regular-representation channels, no new dependency), node permutation under rot90 (spectral transport stays exact since rot90 ∘ shift(ν) = shift(Rν) ∘ rot90 on a square torus), rotated actions. Estimator compatibility: the ±1 search grid and Gaussian are C4-symmetric; the two-mode "left wins ties" rule and the fixed random projection P are not and must be made equivariant. Test: exact rot90 equivariance of memory and predictions with an interior mask, as `kwm/equiv.py` does for boosts; plus a composition test with boosts (rotation acts on boost vectors). Cost: p4 convolutions are 4× the channels at equal width; measured before training. Second target: approximate C8/SO(2) on a non-periodic padded world with a disc-shaped window.
- **Dynamics (optional benchmark):** per-digit spin is new dynamics (an angular-velocity state analogous to α), not part of the symmetry; added only if the symmetry work lands.
- **3D-relevant case:** camera yaw needs a world-frame memory or a non-periodic ego memory and a yaw action component; a larger change, scoped in Stage 2's note.

## 3. Budget (measured: lock ≈ 4.4 s/step alone ≈ 24 h per 20k run; lk ≈ 3.5 s/step; two concurrent runs share throughput)

| Block | Runs | Days |
|---|---:|---:|
| Pilot remaining | — | 1.5 |
| PR1 primary: lk vs lock × seeds 1–4 | 8 | 8 |
| PR2 gate harness, debug and 2 short pilots | 2 short | 2 |
| PR2 K-accel: {order 1, order 1 forgetting, order 2} × 5 seeds | 15 | 15 |
| PR2 deceleration-prior control × 3 seeds | 3 | 3 |
| PR2 non-inferiority on K-real: order 2 × 5 seeds | 5 | 5 |
| PR3 trained zero-input × 3 seeds | 3 | 3 |
| FloWM (KWM-sized, 32 channels, matched steps) on K-real and K-accel × 3 seeds | 6 | 2 |
| K-int sanity: lk, lock, flowm × 1 seed | 3 | 2.5 |
| Checkpoint-averaged test-seed evaluations (5 ckpts × ~45 runs × 1000 seq × 180 frames) | — | 3 |
| Total | ~45 | ~45 |

Seven to nine weeks of wall clock at the current pace, two runs concurrently when memory allows. Gating: PR2 screening starts after the H1 3-seed look; all H1 seeds run regardless. **Cut list, in order, if week 6 arrives:** K-int sanity; FloWM on K-accel; the deceleration-prior control to 3 → 2 seeds is not allowed (it is a control), instead PR3 trained runs drop to evaluation-time only; FloWM-64/FloWM-81 larger-lattice baselines are explicitly deferred.

## 4. Statistics
Paired t-test on seed means (df = seeds − 1) primary; exact sign-flip and hierarchical bootstrap (seeds, then sequences jointly across arms) as sensitivity; Holm within each primary family (H1: one test; H2: one primary test, the rest secondary; H3: one). Stopping rule (screening arms only, never primaries or comparators): dropped after 3 seeds if all paired differences have the wrong sign and exceed one SD of the paired differences. Non-inferiority margin fixed numerically before the runs (5% relative). Checkpoint averaging and SDs as in Section 1.

## 5. Public process
1. Fork branches pushed (`gate/estimator`, `pr1-lock-nodes`, `pr5a-eval-hygiene`, `int/pr1-pr5a`).
2. Upstream issue (draft v2 in `UPSTREAM_ISSUE_draft.md`) after user approval; the grad-checkpoint bug as a second, separate issue with a reproduction.
3. PR5a first as a small independent PR if the maintainer agrees; PR1 after the H1 table and the Opus pre-PR review; PR2, PR3 after their tables. Each PR: per-seed table on the test seed, AI-assistance disclosure, MIT headers, no scipy, no imports from flowm.
4. Nothing public without the user's explicit OK.

## 6. Order of work
1. PR5a addendum (cache self-description) → review → done. Correct the pilot claims in all docs (M3).
2. Pilot finishes → evaluate both arms: last 5 checkpoints × {test seed, selection seed} × {flowm, zero} × {pred on, off}; frozen gate with the pre-lift variant; fill the PR1 doc; launch seeds 1–2 of lk and lock (both arms concurrently).
3. PR2: `accel_law.py` and the oracle ceiling for the final law; env `--accel` with tag; gate harness; estimator per spec; tests; Opus review; 2 short pilots.
4. H1 3-seed look → PR2 screening starts; seeds 3–4 of H1 run regardless.
5. PR3 evaluation-time results on all checkpoints; trained zero-input runs.
6. FloWM baselines, K-int, campaign script, RESULTS_v2, PRs per Section 5; 3D feasibility note; Stage 3 plan.

## Amendment 2026-10-25 (after the PR2 implementation, its Opus review REVIEW_pr2_opus.md, and the corrected gate)

**PR2 gate result: NO-GO for order-2 training.** With the variance units corrected (weight-scale invariant sandwich), a velocity-jump reset, the correction anchored on the fit intercept and κ = 9: null gate-open rate 0.40% (target < 2%, met), but detection of nonzero acceleration in five-digit K-accel scenes is 0–2% per |a| bin, and the displacement proxy (mean |Δp̂ − Δp| at τ = 20/50 from the final node state) is 0.3% worse than the best forgetting comparator (F = 0.9), against a required ≥ 10% improvement. In a clean single-digit scene the same estimator detects a = 4e-3 in 94–100% of 50 seeds with 0/50 false positives, so the estimator is correct; the per-step velocity measurement noise in five-digit scenes (σ_z ≈ 0.14 px/frame vs 0.012 in the clean scene, from association and mode switches) makes a 30-frame slope undetectable below |a| ≈ 1e-2. Full tables: fork `docs/GATE_order2.md`.

**Consequences.**
- PR2 is re-scoped to what passed: the accelerated environment (`--accel`, realised law and oracle ceiling via `scripts/accel_law.py`), the forgetting comparator (`--nu_forget`), and `--order 2` as an experimental opt-in flag with the negative gate documented. The 23-run H2 training campaign is cancelled (saves ≈ 3 weeks of GPU).
- H2 survives only as an evaluation-time check (order 2 has no trainable parameters): on the order-1 lock checkpoints trained on K-accel, if and when those are trained, evaluate `--order 2` and `--nu_forget F` overrides with prediction updates off. No GPU is allocated to it now.
- The real lever for acceleration is the measurement noise of the node update (association across digits, mode switches), not the motion order. A future order-2 attempt needs a better measurement first; the oracle ceiling (perfect instantaneous velocity 0.009/0.051 vs mid-observation 0.033/0.077 at MSE@20/@50) shows the headroom exists.
- Budget reallocated: PR1 H1 seeds (8 runs), PR3 zero-input trained runs (3), FloWM baselines on K-real (3), K-int sanity (3), evaluations. ≈ 20 runs, ≈ 3–4 weeks.

## Amendment 2026-10-25 (PR3 gate and review, REVIEW_pr3_opus.md)
Freeze-and-render fails its evaluation-time gate on both pilot arms in every variant tried (naive, pre-lift, zero-input warm-up K ∈ {0..4}, winner-take-all per pixel, oracle node mask, locked-only, mix-once): at τ = 1 the gap to the zero-input rollout is the missing recurrence step, at long horizons it is ghosting (energy spread over 23–24.5 of 25 nodes; de-ghosting fixes MSE@150 but not MSE@20). It ships only as a documented negative result with evaluation-time switches; no training support. **H3 is re-specified:** zero-input prediction < autoregressive (flowm) at **MSE@20** (primary; zero wins on all four checkpoint/rule combinations in the gate), MSE@150 secondary; lock model, checkpoint-averaged, test seed, 5 seeds at evaluation time on the PR1 runs, plus 3 trained zero-input seeds.
