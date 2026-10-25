# Plan v3: toward a Galilean world model with acceleration, built on kinetic-world-models

Supersedes PLAN_v2_KWM.md after its adversarial review (REVIEW_plan_v2_opus.md), the estimator gate, the PR1 implementation and its review. Decisions taken by the user on 2026-10-25: let the running PR1 pilot finish; stacked PRs with an upstream issue first and explicit approval before every public action; 2D first, then a short 3D feasibility step; translation-boost wording until rotations exist; rotation equivariance is a later stage toward the full Galilean group (user request, 2026-10-25); 3 seeds for screening, 5 for primary claims.

Base: KWM fork `Yixun-Hu/kinetic-world-models` (upstream `denzelzheng/kinetic-world-models` at 302d8f8). Coding by gpt-6-astra via `codex exec`, adversarial review by Opus of every PR before it is opened. GPU 1 only, shared.

## 0. State of evidence (2026-10-25)

| Item | Status |
|---|---|
| Released FloWM reproduced on its benchmark (Phase 0) | done, `experiments/flowm_bench/README.md` |
| Bit-exact MNIST World generator with release action and label laws | done, `mnist_world_gen.py` |
| Estimator gate: lock-then-refine beats KWM's LK without learning | done, 3 seeds, 1500 digits: matched median 0.082 to 0.024 px/frame, misses 42% to 18%, `docs/GATE_estimator.md` (fork) |
| PR1 `--node_update lock` implemented, lk path bit-identical to upstream, boost equivariance error ~3e-7 | done, branch `pr1-lock-nodes` at 28ba51e, 95 tests |
| End-to-end pilot lk vs lock, seed 0, 20k steps, K-real | running; at matched 6k steps lock 0.0286 / 0.083 vs lk 0.0361 / 0.215 (MSE@20 / @150, selection set) |
| Anything beating FloWM or KWM end to end with ≥ 3 seeds | not yet |

What changed from v2: window-level masked-FFT correlation and Newton refinement (the parent model's estimator) lost the gate and are not in PR1; the gain comes from local search, two-mode pooling and lock/refine switching. Multi-lag robust WLS stays an optional variant in the gate harness.

## 1. Benchmarks

All in KWM's `kwm/env.py` (50×50 torus, 32 window, 5 MNIST-train digits, MAX compositing; FloWM's released data is additive with clip, a documented domain shift), 30 observed frames, 20 and 150 predicted.

| Name | Digit motion | Camera | Used for |
|---|---|---|---|
| K-real | real velocities, ‖v‖∞ ≤ 2 | real | H1, H3, non-inferiority of order 2 |
| K-accel (new, PR2) | real v0 + constant per-digit acceleration | real | H2 |
| K-int | integer velocities and camera | integer | sanity only, 1 seed per arm, no claim |

K-accel law: per digit, a = 0 with probability 0.3, else direction uniform and magnitude log-uniform in [2e-4, 2e-2] px/frame²; the speed bound ‖v(t)‖∞ ≤ 2 is enforced over a fixed horizon of 180 frames regardless of the generated length (so train and eval distributions match; v2 review MAJOR 5); positions integrate exactly, p(t) = p0 + v0 t + ½ a t². The realised (a, v0) joint statistics and the sign(a)·v0 correlation are logged by the generator test. Median ½|a|τ² is 0.7 px at τ = 20 and tens of pixels at 150, so H2's endpoint is MSE@20 and MSE@50; MSE@150 is reported with per-|a| bins only.

Evaluation sets: KWM selects `best.pt` on the same 300-sequence set that `kwm.eval` scores. Rule for every table here: primary checkpoint is `last.pt`; evaluation on a fresh test seed (`--test_seed`, PR5a) with 1000 sequences; the selection set is reported only as a secondary column.

## 2. PRs, stacked, in order

### PR5a — Evaluation hygiene (small, first; no GPU)
`kwm.eval --test_seed` (fresh generator seed, default keeps 12345 for backward compatibility), `--n_val 1000`, source hashes of `kwm/` and of the environment config written into every checkpoint and eval json, a `scripts/report.py` that aggregates eval jsons into a markdown table with per-seed values, paired t-test (df = seeds − 1) on seed means as the primary test, sign-flip and hierarchical bootstrap as sensitivity. Every later PR's table is produced by this script. Also files the pre-existing upstream bug (lk + `--grad_ckpt` + `--node_update_pred off` gives gradients off by 3%) as a separate issue with a minimal reproduction.

### PR1 — Lock-then-refine node velocities (`--node_update lock`) — implemented
Spec and evidence in the fork's `docs/PR1_lock_nodes.md`. Remaining before opening: the end-to-end comparison (H1), then the Opus pre-PR review on the final diff.
H1: lock < lk in MSE@20 on K-real, `last.pt`, test seed, 5 seeds, paired t-test. Secondary: MSE@150, node velocity error median, boost-equivariance error, `--node_update_pred off` at evaluation time (free), step time and memory. Seed 0 is the running pilot (configuration identical to the campaign configuration: `--model adaptive --real_velocity --real_camera --bs 16 --grad_ckpt --steps 20000`).

### PR2 — Second-order node state and accelerated environment (`--order 2`, `--accel`)
Node state (ν_i, α_i), transport by ν_i dt + ½ α_i dt² per step (one real-valued spectral shift). Estimation is a per-node constant-acceleration α-β-γ filter over the per-step displacement measurements that PR1 already produces (locked LK residual or search result): state (ν, α), prediction ν ← ν + α dt, correction with gains from the accumulated evidence W (the KWM running-mean gain in the limit α ≡ 0), and the acceleration channel gated by its own innovation variance (g = clip(1 − κ Var(α)/α², 0, 1), κ = 9, ablations κ ∈ {0, 4}). With the gate closed and κ → ∞ the filter reduces to PR1's order-1 update exactly (test). The prediction phase advances ν by α per step and freezes α. Node bound: when ν_i + α_i τ leaves the ±0.75 cell during prediction the node is clamped and the clamp is counted (diagnostic); no handoff between nodes in this PR.
Comparators (so the gain is not a win by construction): order 1 as is; order 1 with a forgetting factor on the running mean (tracks a drifting velocity without an acceleration state); order 2.
Tests: zero-acceleration data gives a gate-open rate below 2% over 30 frames; a digit with a = 4e-3 px/frame² is tracked to |v error| < 0.01 px/frame after 30 frames with the frozen encoder; order 2 with κ → ∞ equals order 1 bit for bit; prediction positions follow p0 + vτ + ½aτ² on a synthetic scene; α is unchanged under a comoving boost; fractional dt.
H2: order 2 < order 1 (both comparators) in MSE@20 and MSE@50 on K-accel. Non-inferiority of order 2 on K-real at a margin set from the observed seed standard deviation of PR1 (not the 0.002 of v2).

### PR3 — Freeze-and-render prediction (`--dec_input frozen`)
After the last observation the memory content stops updating; prediction frame τ is the frozen memory transported by ν_i τ + ½ α_i τ² (one spectral shift from the frozen state) plus the cumulative camera action, then decoded. Ablations: KWM `flowm` (default, autoregressive), `zero`, `frozen`; each with `--node_update_pred on/off`, since H3 is otherwise confounded by prediction-phase node updates (v2 review MAJOR 7). Hybrid mode dropped.
Tests: frozen equals the zero-input recurrence when the content convolution is an identity tap (up to the activation); node-concentration diagnostic for ghosting (lift writes to all nodes, frozen transport separates copies at |ν_i − ν_j| px/frame); node permutation invariance.
H3: frozen < flowm in MSE@150 on K-real with the lock model, `last.pt`, test seed, 5 seeds.
Freeze-and-render is also an evaluation-time switch on existing checkpoints; the evaluation-time result is reported first (free), the trained-with-frozen result second.

### PR4 — Visibility-aware fusion: deferred
Not budgeted in this plan. If revisited: separate normalised-convolution observation memory feeding the mix, per-pixel write weights from coherence, decayed coverage with a floor, bilinear-tapered coverage (v2 review MAJOR 8).

### PR5b — Campaign script and results
`scripts/campaign.py` runs the matrix below shardable on one GPU with resume; `docs/RESULTS_v2.md` regenerated from the jsons by `scripts/report.py`.

### Stage 2 — 3D feasibility (after PR1–PR3 tables exist)
A one-week step, no claims: obtain the FloWM Blockworld data (not on disk), run the released FloWM-3D evaluation once, and write a one-page note on which of PR2 (second-order state) and PR3 (frozen analytic prediction of the latent map) transfer to the ViT latent map, with a cost estimate. Full 3D runs are a separate plan.

### Stage 3 — Rotation equivariance (toward the full Galilean group)
The Galilean group adds rotations to translations and boosts. Neither MNIST World nor Kinetic MNIST World rotates anything, so this stage needs both an environment and a model change:
- Environment: per-digit angular velocity (spin) and, for the 3D case, camera yaw; the 2D version is a `--spin` option in `kwm/env.py` with bilinear rotation about the digit centre.
- Model: a rotation is not a Fourier phase shift, so spectral transport does not extend directly. Candidates to evaluate: (a) a rotation-augmented node state (ν_i, ω_i) with transport by rotation about the node's tracked centre in the spatial domain (bilinear, loses the exactness of the spectral shift); (b) a steerable or polar-Fourier memory in which rotation becomes a phase shift along the angular coordinate; (c) data augmentation with a rotation-equivariance test only (weakest, cheapest).
- Tests: rotate scene and camera together by θ → memory rotates by θ (interior mask), as `kwm/equiv.py` does for boosts.
This is planned after the 2D PRs and the 3D feasibility step; the docs use "translation-boost equivariant" until a rotation PR exists.

## 3. Budget (measured on the shared GPU 1, 2026-10-24/25)

Adaptive arm, bs 16, `--grad_ckpt`: 3.46 s/step alone (19 h / 20k steps); two concurrent runs 7.9–10 s/step each, i.e. about one 20k-step adaptive run per day of wall clock either way. FloWM arm 1.27 s/step (7 h).

| Block | Runs | Days |
|---|---:|---:|
| PR1 primary: lk vs lock × 5 seeds (seed 0 running) | 9 | 9 |
| PR2 on K-accel: {order 1, order 1 forgetting, order 2} × 3 seeds screening, +2 seeds for the best two arms | 13 | 13 |
| PR2 non-inferiority on K-real: order 2 × 3 seeds (comparator = PR1 lock seeds) | 3 | 3 |
| PR3 trained-with-frozen: 3 seeds on K-real (evaluation-time frozen on all existing checkpoints is free) | 3 | 3 |
| FloWM retrained on K-real and K-accel, 3 seeds each | 6 | 2 |
| K-int sanity, 1 seed per arm | 3 | 3 |
| Total | 37 | ~33 |

About five weeks of wall clock at the current pace, sequentially gated: PR2 screening does not start until H1 is decided on 3 seeds. Stopping rule, prespecified: an arm is dropped after 3 seeds if its paired seed differences against its comparator are all of the wrong sign and larger than one seed standard deviation; primary and comparator arms are never dropped. If the GPU frees up, runs are stacked two at a time.

## 4. Statistics and reporting
Paired t-test on seed means (df = seeds − 1) primary; sign-flip and hierarchical bootstrap (seeds × sequences) as sensitivity; Holm within each PR's primary family (H1: one test; H2: two comparators at two horizons; H3: one). Everything on `last.pt` and the fresh test seed; selection-set numbers in a secondary column. Per-frame MSE curves with seed bands; node velocity error; boost-equivariance error; step time and memory. No claim on FloWM's released split beyond Phase 0 (parity at best; domain shift documented).

## 5. Public process
1. Push branches to the fork (done 2026-10-25 for `gate/estimator`, `pr1-lock-nodes`).
2. Open one upstream issue describing the series (draft: `UPSTREAM_ISSUE_draft.md`), after user approval.
3. PR1 after H1 on 5 seeds and the Opus pre-PR review; description discloses AI-assisted code and the reviewers; MIT headers; no imports from flowm; no scipy.
4. PR5a may go first as a small independent PR (test seed, hashes, report script) if the maintainer prefers; otherwise stacked under PR1.
5. Each later PR after its table exists. Nothing public without the user's explicit OK.

## 6. Order of work
1. Now: PR5a (eval test seed, hashes, report script) via codex, reviewed; pilot continues.
2. Pilot done: evaluate `last` and `best150` on selection set and test seed, fill the PR1 doc, launch seeds 1–2 of lk and lock.
3. PR2 code (env `--accel`, α-β-γ node state) while seeds run; gate tests pass before any GPU run.
4. H1 decided on 3 seeds → seeds 3–4; open the upstream issue; PR2 screening.
5. PR3, FloWM baselines, K-int sanity, RESULTS_v2, PR openings per Section 5.
6. 3D feasibility note; rotation stage plan.
