# Opus adversarial review of PLAN_v3_KWM.md and UPSTREAM_ISSUE_draft.md (2026-10-25)

**Verdict: v3 is a large improvement on v2, and PR5a and PR1 can go ahead. PR2 and PR3 are not ready to code as specified.** Most of the v2 review is addressed. Three new problems are serious, and each one would invalidate a headline claim:

1. Evaluation-time freeze-and-render on the pilot lock checkpoint is 2.2–2.3x worse at MSE@20 and 1.7x worse at MSE@150 than the default rollout (both pilots, 32 sequences), and the zero-input rollout beats the default. H3 is likely false as specified.
2. Under the plan's own 180-frame bound, the K-accel numbers in the plan are wrong by about 8x. An oracle shows that most of the possible H2 gain at MSE@20 comes from removing running-mean lag, not from curvature, so the forgetting-factor comparator takes most of it.
3. Neither `--test_seed` nor `--accel` is trivial. `val_cache` keys its cache file only by `cfg.tag()`, n and T, so both changes would silently load the old cache.

Scratch scripts are in `/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/review_v3/`: `accel_law.py`, `h2_ceiling.py`, `frozen_eval.py` (output in `frozen_eval_32.txt`) and `ckbug.py`. Everything ran on CPU and only read the repos; the pilot checkpoints were loaded read-only.

---

## BLOCKERS

### B1. H3 / PR3: as an evaluation-time switch, freeze-and-render is much worse than the autoregressive rollout. The plan's design has a structural ghosting source.

**Evidence.** `frozen_eval.py` takes `runs/pilot_lock/last.pt` (step 9000). It runs the observation phase, then transports the final feature memory analytically by `ν_i τ − Σa` and decodes. The test set is the first N sequences of the selection cache, so this is a diagnostic only.

N=32 sequences. These numbers replace the earlier N=4 run and have the same ordering.

| Mode | lock (step 9000): MSE@20 / @50 / @150 | lk (step 11000): MSE@20 / @50 / @150 |
|---|---|---|
| flowm (default) | 0.0243 / 0.0388 / 0.0749 | 0.0314 / 0.0523 / 0.0924 |
| flowm, `--node_update_pred off` | 0.0242 / 0.0383 / 0.0731 | 0.0313 / 0.0520 / 0.0916 |
| zero input | **0.0218 / 0.0343 / 0.0689** | **0.0284 / 0.0475 / 0.0833** |
| frozen | 0.0566 / 0.0843 / 0.1275 | 0.0694 / 0.1042 / 0.1532 |
| frozen + ReLU | 0.0557 / 0.0828 / 0.1253 | 0.0665 / 0.1001 / 0.1475 |

Frozen is 2.3x (lock) and 2.2x (lk) worse than flowm at MSE@20, and still 1.7x worse at MSE@150, which is the H3 endpoint. Zero input beats flowm by about 10% at every horizon for both rules. Prediction-phase node updates barely matter (pred off is about 2% better at 150).

**Cause.** `_Base.lift` (`kwm/cells.py:48-55`) writes `conv_u(frame)` into every one of the V=25 nodes. So the final state `h_T = act(lift(obs_T) + mix(stream(h_{T-1})))` holds 25 equal copies of the last observation. Freeze-and-render then moves those copies apart at the 25 node velocities: this is the v2 review's ghosting concern, now measured. In the recurrent rollout, `conv_h` (mix) together with the ReLU and repeated re-lifting suppresses wrong-velocity copies over time. A frozen state has no such mechanism. Training with `--dec_input frozen` cannot remove the copies either, because the lift is node-agnostic.

**Consequences.**

- H3 as stated ("frozen < flowm in MSE@150", `PLAN_v3_KWM.md:52`) is very likely false as an evaluation-time switch.
- The trained variant can only work if the model learns per-node suppression, and nothing in the architecture provides that.
- The zero-input rollout (`model.py:284`), not flowm, is the strongest comparator. H3 should be tested against the best of {flowm, zero}.

**Fix (any one).**

- (a) Freeze the *pre-lift* memory, `mix(stream(h_{T-1}))` before adding `lift(obs_T)`, and/or freeze several steps' worth of the mix recurrence.
- (b) Give the lift a node-selective write: weight `lift` per node by a coherence or ownership weight from the lock measurement (`lock_measurement` already computes `selected` pixel weights per node). This is a model change, not an evaluation switch.
- (c) Re-scope PR3 as "zero-input vs flowm", which is already a measurable and apparently positive effect, and drop "frozen" until (a) or (b) passes a no-learning or evaluation-time check.

Add a mandatory evaluation-time gate before any PR3 training run: frozen ≤ min(flowm, zero) on the selection set for the pilot checkpoints.

### B2. K-accel: the plan's displacement numbers are wrong under its own law, and the H2 effect is mostly an averaging-lag effect.

**Realised law** (`accel_law.py`; 400k digits, joint rejection of (v0, a) over 180 frames, ∞-norm ≤ 2; resampling a only gives the same numbers to ±0.01):

- Acceptance is 0.85.
- 35% of digits have a = 0 (the target was 30%).
- For nonzero a, the median |a| is 1.28e-3 against the proposal's 2.0e-3. The 90th percentile is 7.7e-3 against 1.26e-2. The upper decade is heavily truncated, so "log-uniform in [2e-4, 2e-2]" (`PLAN_v3_KWM.md:30`) does not describe the data.

**Displacement.** The median ½|a|τ² over all digits is 0.09 px at τ=20, 0.57 px at τ=50 and 5.1 px at τ=150. Over nonzero-a digits it is 0.26, 1.6 and 14 px. The plan's "0.7 px at τ=20, tens of pixels at 150" (`:30`) is about 8x too high at τ=20.

**Shortcut correlation.** corr(a_k, v0_k) is −0.05 for |a| < 2e-3, −0.22 for 2e-3 to 6e-3, and −0.63 for |a| > 6e-3. Using the same 180-frame horizon for train and eval fixes the train/eval mismatch from v2 MAJOR 5. It does not fix the shortcut: large accelerations are mostly decelerations.

**Oracle ceiling** (`h2_ceiling.py`; 300 sequences × 5 digits, MNIST-test, real camera, MAX compositing). This is the MSE of a perfect order-1 extrapolation against the true accelerated future:

| Order-1 velocity used for extrapolation | MSE@20 | MSE@50 |
|---|---:|---:|
| Mid-observation (the running-mean limit) | 0.0168 | 0.0450 |
| Instantaneous at t=29 (the best a forgetting filter can reach) | 0.0043 | 0.0279 |
| Last finite difference | 0.0039 | 0.0273 |

So of the 0.017 MSE@20 that an order-1 running mean loses to perfect knowledge, about 75% is lag, which "order 1 with forgetting factor" recovers without any acceleration state. Order 2 beyond the forgetting comparator has a ceiling of about 0.004 at MSE@20 and 0.028 at MSE@50, before estimation noise. The models' K-real MSE@20 is already about 0.025. A realistic order-2 advantage over the forgetting comparator at MSE@20 is a fraction of 0.004. That is comparable to the seed-to-seed spread you can expect, so with 5 seeds it is underpowered.

**Fix.**

- State the realised law, not the proposal.
- Make MSE@50 the primary H2 endpoint vs the forgetting comparator, and keep MSE@20 only vs plain order 1.
- Or reshape the law, e.g. nonzero |a| log-uniform in [2e-3, 2e-2] with joint rejection, and report the realised quantiles.
- Run this oracle table (it costs nothing) for the final law, and put it in the plan as the stated ceiling for H2.
- Report per-|a| bins together with the correlation bins, and add a "deceleration-prior" control: order 1 plus a learned shrinkage of ν toward 0 during prediction. Order 2 must beat it to show that it uses measured acceleration rather than the prior.

### B3. `--test_seed` and `--accel` both collide with the validation cache (silent wrong evaluation).

`val_cache` (`kwm/train.py:53-73`) writes `results/val_cache_{cfg.tag()}_n{n}_T{T_obs}+{T_pred}.pt` and loads that file whenever it exists. The seed is the module constant `VAL_SEED = 12345` (`train.py:27`, used at `:59`), and neither the seed nor any new env field is in the key.

- A `--test_seed` that only changes the generator seed loads the old 12345 cache, and evaluation silently reuses the selection set.
- `EnvConfig.tag()` (`env.py:57-71`) has no accel term, so K-accel evaluation would load the K-real cache.
- Every caller has to be threaded: `env_cfg_from_args` (`train.py:38-50`) copies a fixed key list, so `--accel` must be added there or evaluating an accel checkpoint rebuilds a K-real config. The eval main uses `val_cache` at `eval.py:307`, and the dt=0.5 sweep at `eval.py:434-436`.
- The caches are also prefix-consistent (chunk 50, one generator): the n=300 set is exactly the first 300 of an n=1000 cache at the same seed. That is fine with a fresh seed, but it shows that "`--n_val 1000`" on seed 12345 would still contain the selection set.
- Note that `--n_val` already exists in `kwm.eval` (`eval.py:260`). PR5a only needs to change the default and the documentation.

**Fix.** PR5a puts the seed in the cache filename and in the cache dict. On load it asserts that the stored cfg, seed, n and T equal the requested ones, and it asserts that the test seed is not VAL_SEED. PR2 adds `accel` to `EnvConfig`, `tag()` and `env_cfg_from_args`, and adds a test that K-real and K-accel tags differ. Store the source hash and the generator version inside the cache. This is the cheapest place to make PR5a's hashes meaningful.

---

## MAJOR

### M1. PR2's estimator spec does not compose with the PR1 measurement as written (`PLAN_v3_KWM.md:44`).

**(a) It is not an α-β-γ filter.** `lock_measurement` (`lock_nodes.py:188-244`) returns a one-step residual displacement `delta` against the snapshot transported by ν·dt. The cell turns it into a measured velocity `z_t = ν + delta/dt` and applies a running mean with gain `evidence/W`, a learned scale `n_s`, `step_clip` 0.3 and a ±0.75 clamp (`cells.py:246-251`). A filter with state (ν, α) and velocity measurements is a 2-state α-β (linear-trend) filter. "α-β-γ" means (position, velocity, acceleration) with position measurements.

**(b) The noise model matters.** Consecutive one-step displacements share frames, so per-frame localisation error enters `z_t` as e_t − e_{t−1}, which is MA(1) and negatively correlated. A slope fit that assumes white noise misstates Var(α) by a large factor. The cumulative-displacement (position) form is far more efficient for α, and it is the natural setting for the α-β-γ name. Either fit positions, using cumulative `Σ delta + Σ ν dt` per node, or estimate Var(α) empirically (the null-rate test) and calibrate κ on it. Do not use a plug-in sandwich variance under white noise.

**(c) "κ → ∞ equals order 1 bit for bit"** (`:46`) holds only if order 1's ν is computed by exactly the existing arithmetic and α enters as an additive, gated correction. The clean construction is weighted least squares in centred time: keep the existing running mean ν̄ (including `n_s`, clip and clamp) and add the sums Σw, Σwt, Σwt², Σwtz and Σwz² per node. Then α̂ = S_tz/S_tt, and ν = ν̄ + g·α̂·(t − t̄_W). With g = 0, every floating-point operation on ν is the order-1 one.

**(d) Unspecified details.**

- The node state grows from `[B,V,4]` to about `[B,V,9]`, so `LockState` changes and every helper that reads it must be updated: `bounce` (`cells.py:159-166`), `equiv.boost_transform_nodes`, and the `node_velocity_error` and diagnostics paths.
- Does `n_s` scale the α gain?
- Is the gate differentiable, or detached like the other discrete decisions in `lock_measurement`?
- What happens to α when the lock bit flips or two_modes switches mode, since a mode switch is a velocity jump that the trend fit reads as huge acceleration? Specify a reset rule for the sums on a mode switch, or a robust weight.

**(e) Prediction phase.** "Advances ν by α and freezes α" is ambiguous with `--node_update_pred on`, the default and the pilot configuration: are ν measurements from the model's own frames still applied? State both cases. With `on`, the order-2 model also confirms its own extrapolation, which inflates H2 relative to `off`.

**(f) Two tests cannot pass against current numbers.**

- "|v error| < 0.01 px/frame after 30 frames with a = 4e-3 and the frozen encoder" (`:46`): the order-1 lock estimator's matched median on constant-velocity 5-digit scenes is 0.024 px/frame with 18% misses above 0.1 (`docs/PR1_lock_nodes.md`, estimator table). The test must specify a single-digit synthetic scene and get its threshold from a gate-harness run, not from v2.
- The 2% gate-open test needs a unit (per node-step? per node with evidence > 0? at t = 30?).

**(g) Detectability.** With 30 frames, a slope standard error of roughly σ_z·√(12/N³) ≈ 0.02·σ_z, and κ = 9 (|α̂| > 3 s.e.), only |a| of about 3e-3 and above opens the gate for typical σ_z of 0.05 px/frame. Under the realised law (B2) that is about 25% of nonzero digits. Estimate this with the gate harness before any GPU run.

### M2. PR2 node bound during observation (`:44`).

The plan clamps only during prediction. Under the realised law, Δv over the 30 observed frames reaches 0.6 px/frame at the upper tail, and Δv over τ = 50 reaches 1 px/frame. Both cross the ±0.75 cell, and the two_modes `inside` restriction (`lock_nodes.py:94`) then discards exactly those pixels. The clamp counter must also report observation-phase saturation. Report H2 per-|a| bins with the clamp rate next to them, otherwise a reviewer will attribute the upper-bin result to clamping. The v2 review asked for a K-accel definition of the oracle model (`oracle_nodes_from(vel0)`, `model.py:302`, uses constant v0). It is still undefined, so either define it or say that it is not run on K-accel.

### M3. `last.pt` under a constant learning rate is a noisy endpoint, and lk's MSE@150 is bimodal.

`train.py:207` uses Adam at a constant 1e-3 with no schedule. Pilot `val.csv` MSE@150 (64 sequences):

- lk: 0.095 (5k), **0.215** (6k), 0.162 (7k), 0.092 (8k), 0.096 (9k), **0.203** (10k), 0.093 (11k).
- lock: 0.077 to 0.087 throughout.

**The headline number is an lk spike.** The plan's "at matched 6k steps … lk 0.0361 / 0.215" (`PLAN_v3_KWM.md:15`) picks lk's spike. At 5k and 8k lk is about 0.093 to 0.095, so a fair statement is "lock 0.08 vs lk about 0.093, with lk spiking to about 0.2 at 3 of 11 checkpoints". This is an overclaim in the state-of-evidence table. Correct it before it reaches the PR doc or the issue.

**Fix.** Make the primary evaluation the mean over the last 5 checkpoints (16k to 20k, 1k apart; `keep_ckpts` is already true) on the test seed, or add a fixed LR decay or EMA to *all* arms. Report checkpoint-to-checkpoint SD next to seed SD. Lock's stability at 150 frames is a real secondary result; claim it as stability (fraction of checkpoints above 1.5× median), not as a mean.

### M4. Statistics design inconsistencies (`:47`, `:78`, `:85`, `:88`, `:101`).

- **The H2 family conflicts with the budget.** H2 has two comparators at two horizons, which is four tests, and the stopping rule says comparators are never dropped. But the budget gives "+2 seeds for the best two arms" (13 runs). One comparator would stay at 3 seeds, and a paired t-test with df = 2 has t_crit = 4.30. Either budget all three arms to 5 seeds (15 runs) or make one comparator secondary.
- **The non-inferiority design is weak.** Order 2 on K-real gets 3 seeds against the PR1 lock seeds, so it is paired on 3 of 5 with df = 2, and the margin is "set from the observed seed SD of PR1". That margin depends on the data and is circular, the horizon is unspecified, and with df = 2 the TOST lower bound is roughly the mean difference − 2.9·SE (one-sided 95%). That is not a non-inferiority claim. Fix the margin numerically *before* the order-2 K-real runs, as a relative margin (e.g. 5% of lock MSE@20), give it 5 seeds, and use the same seeds as PR1 for pairing.
- **"H1 decided on 3 seeds → seeds 3–4"** (`:101`): if seeds 3 and 4 depend on the 3-seed result, that is optional stopping. State that 5 seeds always run for H1, and that the 3-seed look only gates when PR2 starts.
- **The stopping rule** (`:85`) does not say which SD it means: the SD of paired differences or the arm's seed SD. Use the SD of paired differences.
- **The hierarchical bootstrap** must resample sequences *jointly* across the paired arms, since all arms see the same 1000 test sequences. Otherwise it ignores the pairing.
- **Tuning leakage.** The forgetting factor of the comparator, κ, and any PR2/PR3 knob must be tuned on the selection seed with equal effort for each arm. The plan does not say this, and an untuned forgetting factor will be called a strawman.

### M5. The budget is optimistic by roughly 30–60% (`:71-85`).

**Measured from the pilot logs** (`runs/pilot_*/ckpt_step*.pt` mtimes):

- Concurrently, lk reached 11k steps in 24.3 h (about 7.9 s/step) and lock reached 9k in 25.3 h (about 10.1 s/step). Lock is about 27% slower than lk.
- Lock alone is therefore about 4.4 s/step, i.e. about 24 h per 20k run, not 19 h. Order 2 adds more.
- The pilot itself needs about 31 more hours.

**Not budgeted.**

- PR2 debug and pilot runs (at least 1–2 short runs).
- The gate harness for order 2.
- The M3 checkpoint-averaged evaluations: 5 checkpoints × arms × modes × 1000 sequences × 180 frames. On CPU this is about 8 s per 180-frame lock rollout (measured), which is infeasible, so it needs GPU time.
- Any rerun after a failed test.
- The B1 fix for PR3.

The sequential gate also idles the GPU, because PR2 screening waits for 4 more runs after the pilot. A realistic estimate is 7–9 weeks, not "about five". State a fixed cut list in advance: what is dropped if week 6 arrives, e.g. K-int sanity first, then FloWM on K-accel.

### M6. The v2 review items that are neither addressed nor explicitly deferred.

- **B1, 25-node oracle.** Not in the gate (`docs/GATE_estimator.md` has no oracle row). This matters for interpreting the remaining 18% misses as estimator failures rather than assignment failures.
- **MAJOR 11, FloWM-64 / FloWM-81 baselines.** Not mentioned. "FloWM retrained on K-real and K-accel" (`:81`) does not say which size or preset. Specify that it is the KWM-sized FloWM, matched in parameters or compute, and say explicitly that the larger-lattice variants are deferred.
- **MAJOR 4, readout OOD.** The moments readout takes ν as an input feature (`model.py:171-178`), and under order 2 ν drifts during prediction. This is not addressed; at least report it.
- **MINOR, the trivial node-permutation test.** It is still in PR3's test list (`:51`). Drop it or replace it with a non-trivial one.

Everything else from the v2 review is addressed. The checklist is at the end.

### M7. Rotation stage (`:64-69`): the sketch mixes up two different things, and its test is ill-posed on the torus.

- **Spin is dynamics, not symmetry.** Per-digit spin (`--spin`) is new *dynamics*: an angular-velocity state, analogous to adding α. It is not part of the Galilean group. Galilean rotation equivariance means a *global* rotation x → Rx, v → Rv, a → Ra of the whole scene, the camera and the actions. Plan the two separately. The symmetry is the claim; spin is an optional benchmark.
- **The torus breaks SO(2).** KWM's world is a 50×50 torus with a square window in the top-left corner (`cells.py:1-2`, `env.py:5-6`). A rotation by θ ∉ {90°, 180°, 270°} is not a map of the torus to itself, so "rotate scene and camera by θ → memory rotates by θ" (`:68`) is undefined except for C4. The square velocity lattice {−2..2}² (`collision.lattice_nodes`) is also closed only under C4. Continuous SO(2) needs a non-periodic (padded) world and a window shaped like a disc, or it can only be approximate.
- **(a) Rotation about the node's tracked centre is ill-defined.** A node is a velocity group spanning the whole map. It has no centre, and it holds several digits. Rotating learned `conv_u` features is also not the same as encoding the rotated image, unless the encoder is steerable.
- **(b) Polar-Fourier memory** makes rotation about one fixed origin a phase shift but destroys the exact translation and ego-motion transport. SE(2) has no common basis in which both act as phases. With a translating camera this is not credible.
- **(c) Augmentation** gives only approximate invariance. On the torus it is not even a valid transform of the data except for multiples of 90°.
- **Most credible: exact C4 first, then approximate C8/SO(2).** Make conv_u, conv_h, the decoder and the moments readout's 1×1 projection p4-equivariant. This can be written in pure torch with weight tying under rot90 and regular-representation channels, with no escnn dependency. The square lattice is C4-closed, so permute the nodes under rot90, rotate the action and ν vectors, and keep spectral transport exact, since rot90 ∘ shift(ν) = shift(R ν) ∘ rot90 on a square torus.
- **Estimator compatibility under C4.** The lock estimator is nearly C4-compatible: the ±1 grid at 0.5 px is C4-symmetric and the Gaussian is isotropic. But two_modes' "left wins ties" (`lock_nodes.py:118`) and the fixed random projection P (`cells.py:20-27`) are not. P must act on the fibres equivariantly (or be invariant).
- **Test.** Use exact rot90 equivariance of the memory and the predictions (interior mask), as `kwm/equiv.py` does for boosts.
- **Missing pieces.**
  - Ego-rotation (camera yaw) is the practically relevant rotation for 3D. It needs a world-frame memory or a non-periodic ego memory and an action with a yaw component. This is a larger change than "a `--spin` option".
  - The interaction with boosts (Galilean composition: rotation acts on boost vectors) needs its own test.
  - A cost estimate (p4 convs are 4x the channels at equal width).

---

## MINOR

1. **Accel law time units.** `PLAN_v3_KWM.md:30` gives the bound in "180 frames" and a in px/frame². KWM supports dt = 0.5 (`env.py:20-22`, eval `:432-436`), so define the bound and a in time units, otherwise the dt = 0.5 cache uses a different law. Exact integration means replacing the incremental loop at `env.py:241-256` with a closed form in `tt`. Define `vel[t]` as the mean velocity over [t, t+1] so that `node_velocity_error` (`eval.py:248`) stays consistent. Also specify `--accel` together with `--walls`: reject it, or reflect a as well as v.
2. **Lower bound of the law.** a = 2e-4 means Δv = 0.006 px/frame over 30 frames, which is undetectable. About 60% of nonzero draws are below 2e-3, so the effective K-accel population is mostly order 1, which dilutes the mean effect (see B2).
3. **Node-velocity error uses only n = 64** (`eval.py:238`) even with `--n_val 1000`. Raise it to the test-set size for the secondary endpoint.
4. **PR5a "source hashes of the environment config".** `cfg.to_dict()` is already stored. What is missing is a hash of `kwm/env.py`, plus the MNIST file checksum.
5. **The draft says PR5a also "files the pre-existing upstream bug".** That is an issue, not a PR. Keep it separate in the plan text (`:37`).
6. **K-int arms** (`:82`) are not named.
7. **"Holm within each PR's primary family; H1: one test"** is fine. Say that MSE@150 for H1 is explicitly secondary, given M3.
8. **Order-2 transport per step** (ν dt + ½α dt², then ν ← ν + α dt) composes exactly to ντ + ½ατ². Good; keep the test.
9. **Stage 2 (3D) is reasonable** as a one-week, no-claims note. Add one go/no-go criterion, e.g. whether the FloWM-3D latent map has an explicit velocity-node axis at all.

---

## Upstream issue draft (`UPSTREAM_ISSUE_draft.md`)

The tone is good: polite and scoped, it asks before opening PRs, it discloses AI assistance, and the defaults are unchanged. Accuracy fixes:

1. **Line 11, "gradients off by about 3%".** Confirmed silently wrong, with no error raised. Upstream 302d8f8 `KineticWM.step` (`model.py`) does not capture `update_nodes`, and the `finally` at upstream `model.py:235-240` resets it before the backward recompute. My minimal CPU repro (`ckbug.py`: C=8, B=2, 6+4 frames) gives a relative gradient difference of **0.36%**, not 3%. The magnitude depends on the configuration. Write "gradients differ from the non-checkpointed run (0.4–3% relative in my tests)" and attach the repro. Also say that it affects only the experimental `--node_update_pred off`, which `docs/REPRODUCE.md:42` says is not used in the published results. Without that, a maintainer may read it as a claim that the results are wrong.
2. **Line 11, "best.pt is currently selected on the same 300-sequence set that eval scores".** This is accurate (`train.py:59` and `eval.py:303-307`). `docs/RESULTS.md:71` reports the best checkpoint, so phrase it neutrally ("eval defaults to the validation cache used for checkpoint selection; a separate test seed would make the tables unbiased") and do not imply that the numbers are inflated without having measured it. You could measure best vs last on a fresh seed for one existing run first and quote that.
3. **Line 12, "locked nodes use your LK step".** Not quite: lock uses LK with **per-node** damping and texture normalisation (`per_node_damping=True`, `lock_nodes.py:201`), while the upstream LK path is unchanged. Say "a per-node-normalised variant of your LK step".
4. **Line 12, "No new parameters, same state keys".** The parameters and checkpoint keys are the same, but the recurrent state is a different structure (`LockState`). Say "same parameter set and checkpoint keys; lock uses a compact separate recurrent state".
5. **Line 12, "No-learning evidence on 1500 digits"** should say "with a randomly initialised encoder" (`GATE_estimator.md:13`). Matched errors are filtered by visibility in the last ten frames, so the effective count is below 1500; say "3 × 100 sequences".
6. **Line 12, "End-to-end training comparison on 5 seeds in progress"** is an overclaim: only one seed pair is running. Write "a one-seed pilot is running; five seeds are planned before the PR".
7. **Line 14, PR4 `--dec_input frozen`.** Given B1, do not announce it. Replace it with "prediction-phase input ablations (zero-input vs autoregressive)", or leave it out until it passes.
8. **Line 13, PR3 "α-β-γ filter".** Call it "a gated constant-acceleration (trend) filter" until the spec is settled (M1a).
9. **Line 7, "with the eventual goal of the full Galilean group".** Fine as a goal, but the first contact with a single maintainer is better without it. Keep "translation-boost". Note that exact boost equivariance holds only for on-lattice comoving boosts and interior nodes (`docs/PR1_lock_nodes.md` cost-and-equivariance section). If equivariance is mentioned, say "on-lattice comoving boosts".
10. **Missing.** Say up front that each PR comes with a per-seed table on a fresh test seed, and that the PRs are stacked (PR2 needs PR1). Ask the maintainer whether they want the report or campaign scripts upstream at all.

---

## Prioritised fix list

1. **(B3)** In PR5a, put the seed in the `val_cache` key and in the payload, assert cfg/seed/n/T on load, and assert test_seed ≠ VAL_SEED. In PR2, add `accel` to `EnvConfig`, `tag()` and `env_cfg_from_args`, with a tag-collision test.
2. **(M3)** Correct the "lk 0.215" claim in `PLAN_v3_KWM.md:15`. Change the primary evaluation to the mean of the last 5 checkpoints (or add LR decay/EMA for every arm), and report checkpoint SD.
3. **(B1)** Before any PR3 training, add an evaluation-time gate: frozen ≤ min(flowm, zero) on pilot checkpoints. Redesign the freeze (pre-lift state, or node-selective lift), or re-scope PR3 to zero vs flowm. Make zero a required comparator for H3.
4. **(B2)** Re-specify K-accel from the realised law. Run the oracle-ceiling table for the final law. Make H2 primary at MSE@50 vs the forgetting comparator. Add the deceleration-prior control. Correct the 0.7 px figure.
5. **(M1)** Rewrite the PR2 estimator as centred-time cumulative WLS (or a position-level α-β-γ) on top of the unchanged order-1 running mean. Define the noise and variance model, n_s, gradient and mode-switch handling, the prediction-phase rule under pred on/off, and the LockState extension. Re-derive the two tests' thresholds from the gate harness on a specified scene.
6. **(M4)** Make the H2 family consistent with the budget (5 seeds for all three arms, or one primary comparator). Fix the non-inferiority margin numerically before the runs, with 5 paired seeds. Always run 5 seeds for H1. Use paired-difference SD in the stopping rule, a paired bootstrap, and tuning on the selection seed for every arm.
7. **(M5)** Re-budget at the measured lock cost (about 24 h per run alone), add PR2 pilots and the evaluation GPU time, and add a cut list. Estimate 7–9 weeks.
8. **(M2)** Clamp counters in both phases, and H2 bins with the clamp rate. Define or exclude the oracle on K-accel.
9. **(Issue)** Apply fixes 1–10 above. The most important are the 3% figure plus the scope of the bug, "5 seeds in progress", "your LK step", and dropping frozen.
10. **(M7)** Rewrite the rotation stage: separate the symmetry (global rotation) from the dynamics (spin), make the first target exact C4 with p4 convolutions, a permuted lattice and the rot90 test, make continuous SO(2) a second target that needs a non-periodic world, and name camera yaw as the 3D-relevant case.
11. **(M6)** Add the 25-node oracle row to the gate, or defer it explicitly. Specify the FloWM baseline size and defer FloWM-64/81 explicitly.

## v2 review checklist

| v2 item | Status in v3 |
|---|---|
| B1 estimator redesign and no-learning gate | Done. The 25-node oracle is missing (M6). |
| B2 budget | Measured and the matrix is cut. Still optimistic (M5). |
| M1 PR1 spec | Done in the code. |
| M2 PR1 tests | Done. |
| M3 PR2 filter | Named only. The spec does not compose yet (M1). |
| M4 PR2 node bound | Partial: prediction-phase clamp only, no oracle, readout OOD open (M2, M6). |
| M5 K-accel | Horizon fixed. Numbers wrong and shortcut remains (B2). |
| M6 comparators | Forgetting factor added. Add the deceleration-prior control. |
| M7 PR3 | Tests addressed. The premise now fails empirically (B1). |
| M8 PR4 | Deferred with reasons. |
| M9 selection | Rule chosen. The implementation trap is B3, the noisy endpoint is M3. |
| M10 statistics | Mostly done. Margin and H2 seed counts open (M4). |
| M11 benchmarks | K-int and the anchor fixed, domain shift documented. FloWM-64/81 silent (M6). |
| M12 process | Done. |
| M13 goal and wording | Done. Rotation stage needs work (M7). |
| Minors | Mostly done. Node-permutation test kept, bootstrap kept as sensitivity only (acceptable). |
