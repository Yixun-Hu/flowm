# Adversarial review: PR2 `ecf5c03` (branch `pr2-order2`, base `0c1198a`)

Reviewer: Opus, 2026-10-25. CPU only. I did not modify the worktree. Scratch scripts and raw outputs are in
`/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/pr2_review/`
(`record.py`, `analyze.py`, `analyze2.py`, `noreset.py`, `synth.py`, `synan.py`, `scale.py`, `intercept.py`, `modelbit.py`).
All K-accel/K-real numbers below come from the production cell on the harness's own data (100 sequences, seed 777,
frozen C=32/K=8 encoder, seed 0). My replay reproduces the harness exactly: order 1 median 0.060391, order 2
median 0.058931.

## VERDICT

**Do not merge as is, and do not spend GPU on H2 as planned.** The environment half (`--accel`, the law script, the
ceiling) is sound and can ship. The order-1 path is bit-identical to the pre-change code. That holds at cell level in
the PR's tests, and at model level in my own check: lk/lock × pred on/off × grad_ckpt, forward and all parameter
gradients.

The order-2 estimator has a **units bug in its variance formula** that explains most of the negative gate result:

- Null false positives come almost entirely from strong-evidence nodes.
- Weak-evidence nodes can never open.
- The single-digit "recovery" test passes on a scene where the gate also opens 44% of the time with a = 0.

With the variance fixed, calibration passes easily (0.29% null at κ=9, inflation 1). Detection then collapses to
about 0, because the mode-switch reset leaves a median final n_eff of 1.8. Remove that reset and order 2 reaches 1.2%
null, opens on 30% of the top-|a| bin, and gives a matched median of 0.054. That is still worse than order 1 with
F = 0.8 (0.043) on the PR's own metric. It is about 6% better than order 1 on a long-horizon displacement proxy, and
mixed against forgetting. The binding constraint is association noise in 5-digit random-camera scenes: robust σ_z is
0.14 px/frame there against 0.012 in a clean single-digit scene. A second-order state cannot fix that.

## BLOCKERS

**B1. The slope variance has the wrong units: it scales as 1/(weight scale).** `kwm/lock_nodes.py:274-275`

```
dof = w * (1 - 2 / neff)               # = S_w (n_eff-2)/n_eff
variance = inflation * rss / dof / ctt
```

- s² = RSS/dof is a weighted mean squared residual, so its units are velocity². C_tt = Σ w (t − t̄)² carries the
  weight unit. So Var(α̂) ∝ 1/w̄.
- For equal weights w the correct value is s²/Σ(t − t̄)²; the code gives s²/(w Σ(t − t̄)²).
- Evidence is pixel mass with no normalisation. Per measured node-step it has median 0.39, mean 11.4, p10 0.02 and
  p90 37.6.
- Demonstration (`scale.py`, white noise, a = 0, 30 steps, κ=9, inflation 1). Reported Var / true Var is 10.0, 1.0,
  0.10 and 0.025 for w = 0.1, 1, 10 and 40. Null gate-open rate at t=29 is 0, 0, **41%** and **78%**.
- Attribution on the harness null set (orig formula, inflation 4, κ=9):

  | Effective weight Σw²/Σw | Null open rate |
  |---|---|
  | < 0.5 | 0.0% |
  | 0.5–2 | 0.03% |
  | 2–10 | 4.0% |
  | ≥ 10 | **51.9%** |

  So the 2.16% null rate is this bug, not MA(1) noise.
- Clean single-digit scene, a = 0, 80 seeds (`synth.py`/`synan.py`). The PR gate opens on **43.8%** of seeds. The
  reported √(Var/2) is 5.0e-5; the empirical s.e. across seeds is 2.0e-4, 4× larger even with inflation 4.
- Fix: multiply by Σw²/Σw. In code, `variance *= state[..., SW2] / state[..., SW]`. This is exact for equal weights,
  and it is the Kish approximation of the sandwich Σw²(t − t̄)²/C_tt². Alternatively, keep Σw²(t − t̄)² as one more
  Welford column and use the sandwich exactly.
- With the fix, the null node-step rate at κ=9 is 0.29% at inflation 1 and 0.97% at inflation 0.5. Calibration passes
  at inflation 1, and inflation is not needed (see M3).

**B2. The tests are tautological or use unit weights, so they cannot catch B1.**

- `tests/test_order2.py:158-178` re-implements the same formula, with weights U[0.5, 1.5], mean ≈ 1.
- The "MA(1) null" test, `tests/test_order2.py:307-321`, uses `torch.ones` weights. MA(1) noise also makes the
  white-noise plug-in conservative (see M3), so passing says nothing.
- `tests/test_order2.py:324-337` checks gate > 0.9 on one seed of a scene where a = 0 opens the gate 44% of the time.
  It picks the node by `error.argmin()` against the truth. This is not a detection test.
- Needed instead:
  - a weight-scale invariance test (scale w by 10: slope and Var unchanged);
  - a calibration test with realistic heterogeneous weights;
  - a null counterpart of the single-digit scene over at least 50 seeds.

## MAJOR

**M1. The reset rule starves the fit.** `kwm/lock_nodes.py:294-295`

- Matched nodes on K-accel see **7.1 mode-switch resets per digit-sequence** in 30 frames. The clean single-digit scene
  sees 0.47.
- Final n_eff of active null nodes has median **1.8**. Only 26% reach ≥ 3, and 0.7% reach ≥ 10.
- The principal-axis left/right labels flip with the other digits in the window. They are not velocity jumps.
- Replay without mode/unlock resets (`noreset.py`, fixed variance, κ=9, inflation 1):
  - median n_eff rises to 8.1;
  - null rate is 1.17%;
  - gate-open rates by |a| bin are 2.1% / 7.0% / 30.4%, against 0 / 0 / 1.3% with resets;
  - matched median is 0.0542 (order 1: 0.0604).
- Fix: reset on a statistically large jump in z (|z − ẑ| beyond k·s), not on a label change. Or match the selected
  mode by velocity continuity, not by canonical side.

**M2. The correction is anchored on the wrong intercept.** `kwm/cells.py:286`, `kwm/lock_nodes.py:315-332`

- ν = ν̄ + g α̂ (t − t̄) assumes ν̄ ≈ v(t̄_fit). But ν̄ is the PR1 all-time running mean. It is never reset, and it is
  shaped by n_s, the 0.3 step clip and the rho clamp.
- t̄_fit is post-reset and recent (median 26.3 at gated nodes).
- At gated nodes, |ν̄ − z̄_fit| has median 0.024 and mean 0.058, which exceeds the correction itself (median 0.019) in
  53% of cases (`intercept.py`).
- In a trained model n_s ≠ 1 makes this worse.
- Fix: ν = ν̄ + g·(z̄ + α̂(t − t̄) − ν̄), using `torch.where` on g == 0. This is still bit-identical at g = 0.

**M3. The MA(1) rationale for inflation points the wrong way; the inflation sweep is the wrong fix.**
`docs/PR2_order2.md:66-70`

- For z_t = v + (e_t − e_{t−1}), the slope numerator telescopes. Var(α̂) ≈ 72σ_e²/N⁴, against the white-noise
  plug-in 24σ_e²/N³. So the plug-in **over**states the variance by about N/3 ≈ 10 at N = 30.
- Measured lag-1 correlation:
  - clean single digit, a = 0: −0.11, so MA(1) is weak;
  - matched K-accel inliers: +0.11; all residuals: +0.45.
  The positive values come from persistent association errors and lagging bias, not shared frames.
- The excess null rate is B1 plus small-dof t-tails at n_eff ≈ 3–5, where most fitted node-steps live. It is not
  MA(1).
- Use an F/t threshold that depends on dof, or require n_eff ≥ 6, instead of a global scalar.
- Also, κ's meaning: the gate uses the 2-D **trace**. κ=9 therefore means ‖α̂‖² > 18 σ_k², i.e. ≈ 4.24 s.e. per
  component, not the plan's "3 s.e.". Under a correct variance the χ²₂ null rate is e^{−κ}: 1.2e-4 at κ=9 and 1.8% at
  κ=4. κ=4 matches the plan's intent.

**M4. `--node_update_pred on` (the default) double-counts acceleration in prediction.** `kwm/cells.py:259-264, 289-291`

- The PR1 measurement is centred on the uncorrected ν̄ (`nu = nodes + state[..., :2]`).
- Generated frames move with ν̄ + CORR + ½gα̂dt, so z ≈ ν̄ + CORR and ν̄ absorbs CORR at gain e/W. CORR keeps growing
  at the same time.
- W is about 2 at the end of observation (median final S_w 2.28), so the absorption is fast.
- This is a positive-feedback extrapolation, beyond the "confirms its own extrapolation" caveat the plan already makes.
  It is not documented.
- Fix: in prediction under order 2, measure around the corrected velocity and subtract CORR from z. Or skip the ν̄
  update for gated nodes.

**M5. The harness metric favours short-memory estimators, and the arms are not equally tuned.**
`scripts/order2_gate.py:179`

- Matched median |v̂ − v(29.5)| is a velocity-at-anchor metric. H2 is MSE@50.
- On a displacement proxy, mean/median of |D̂(τ) − D(τ)|² over τ ≤ 50 in px², with the same order-1 node assignment
  for all arms:

  | Arm | Mean | Median |
  |---|---|---|
  | order 1 | 49.9 | 9.88 |
  | F = 0.9 | 51.5 | 8.97 |
  | F = 0.8 | 58.6 | 9.08 |
  | PR order 2 | 65.8 | 9.86 |
  | order 2 ungated | 157.6 | 14.1 |
  | order 2 fixed variance, no reset, κ=9 | **46.8** | 9.54 |
  | order 2 fixed variance, no reset, κ=4 | 48.1 | 8.71 |

- So "forgetting wins" holds for medians, but on means forgetting is worse than plain order 1.
- F was tried only at {0.9, 0.95}. F = 0.8 gives 0.0434 on the PR metric, better than both. Order 2 was tried only at
  the most conservative (κ=9, inflation 4).
- The forgetting arms are not inflated by a harness bug. They rerun the production estimator, which is correct
  because ν̄ moves the search centre. If anything they are under-tuned.
- **No harness bug hides an order-2 gain.** The gain is hidden by B1 and M1 in the estimator.

**M6. The default `--accel_var_inflation 4` ships a failed calibration.** `kwm/train.py:178`, `kwm/cells.py:126`

- The default bakes B1 into checkpoints.
- After B1 the right default is 1, or the flag can be removed.

## MINOR

1. Harness assignment (`scripts/order2_gate.py:81-88`):
   - Hungarian matching is on velocity only, per arm.
   - It can match a digit to an empty lattice node with zero gate. That deflates the detection bins.
   - The detection bins use order 2's own assignment.
   - Restrict candidates to nodes with evidence, or assign by spatial ownership.
2. The null denominator (`scripts/order2_gate.py:55-69`) counts all 30 steps, including steps before n_eff ≥ 3. Report
   the n_eff distribution next to the rate.
3. The single-digit scene (`scripts/order2_gate.py:90-111`) uses a digit-tracking camera, which is unlike K-accel. Its
   σ_z (robust 0.012, raw 0.031) is about 12× better than K-accel matched nodes (robust 0.14; 22% of residuals > 0.5).
   It is one seed and has no a = 0 twin.
4. Clamp counters (`kwm/model.py:288-290, 321`):
   - The denominator includes empty nodes.
   - The ½gα̂dt term is ignored.
   - Nothing is reported for order-1 arms, whose rho also saturates at ±0.75, so the H2 "bins with clamp rates" are
     not comparable across arms.
5. `boost_transform_nodes` (`kwm/equiv.py:85-95`) averages the MODE column under off-lattice merges. Non-integer modes
   then trigger spurious resets (`lock_nodes.py:294`).
6. `--pred_shrink learned` (`kwm/cells.py:294-299`) shrinks ν̄ and CORR but not α̂, so acceleration keeps accruing.
   This is a weak deceleration control.
7. Env (`kwm/env.py:260`): accel silently forces continuous v0 while the tag still says `vel0`. Either require
   `--real_velocity` or put `vel1` in the tag.
   - Closed form, float64, `vel[t] = v0 + a(t+dt/2)`, subsample adjustment (`env.py:424-427`), joint redraw,
     endpoint bound, walls/ring/scene_boost rejection, tag only when nonzero, non-accel bit identity (test
     `:84-95`): all correct.
   - The `val_cache` legacy-hash shim (`train.py:80-85`) is acceptable.
   - `env_cfg_from_args` threads all four fields.
   - `dt = 0.5` uses the same time-unit horizon.
   - Eval override args propagate to the JSON (`eval.py:73-85, 331-334`).
8. Tests: there is no model-level order-1 identity test against the frozen pre-change `model.py`. It holds; add my
   `modelbit.py` as a test. The cell-level test (`tests/test_order2.py:33-58`) does compare against faithful frozen
   copies (diffed: imports only) and covers gradients, dt=0.5 and the prediction flag. It is non-vacuous: W > 0 and
   locks occur.
9. 17-column layout (`lock_nodes.py:250-257`):
   - Sound. Centred Welford is the right fp32 choice, and all readers were updated: bounce, boost, offsets,
     diagnostics, panorama, eval node trajectory.
   - Storing a per-node clock (TIME) duplicates a global quantity.
   - CORR duplicates g·α̂·(t − t̄). It needs to be carried only because the gate is frozen; acceptable.
10. Units are inconsistent: README `:146` says px/frame², the doc says px/time-unit². The commit trailer
    "Claude Fable 5.1" follows the spec.

## Docs

- "Calibration passed: False" is reported prominently and honestly (`docs/PR2_order2.md:5-7, 243-245`,
  `README.md:152`). The explanation is wrong: it attributes the failure to MA(1) and association noise (`:66-70`,
  `:181-183`) when it is B1.
- Overclaims:
  - "order-2 matched median improves slightly over order 1" (`:268`): 0.0604 → 0.0589 with no uncertainty, from mostly
    false-positive gates.
  - The single-digit scene "tests recoverability" (`:185`, `:261-264`): it does not, given the 44% null opening.
- Style a maintainer would flag:
  - The doc is 292 lines.
  - It mixes provenance from the planning stage ("planning reference", "proposal", `:225-236`).
  - It contains test-tolerance derivations (`:263`) and validation run times.
  - Trim to method, law, ceiling and results. Move provenance to the commit message.

## Answers to the specific questions (short)

1. **Harness correctness.** The truth v(29.5), the anchor at TIME − dt = 29, the visibility filter after assignment,
   the camera and the bins are correct.
   - The forgetting arms correctly rerun the production estimator. They are fair, and under-tuned rather than
     inflated.
   - Weak points: assignment by velocity only (Minor 1), a median-only velocity metric (M5), and the synthetic scene
     (Minor 3).
   - None of these hides an order-2 gain. B1 and M1, in the estimator, do.
2. **Implementation against spec.**
   - Centred weighted fit, detachment and the reset rule are as specified.
   - Order-1 bit identity is verified, including grad_ckpt and pred on/off at model level.
   - The variance is **not** a valid empirical variance (B1).
   - The reset rule is specified but harmful in practice (M1).
   - The anchor is inconsistent (M2).
   - All layout readers are updated.
3. **Why detection is weak.** σ_z and slope s.e.:

   | Scene | σ_z (robust) | Slope s.e. over 30 frames | |a| for 50% open at κ=9 |
   |---|---|---|---|
   | Clean single digit | 0.012 | empirical 2.0e-4 (theory σ√(12/N³) = 2.5e-4) | ≈ 9e-4 |
   | K-accel matched | 0.14 (raw 0.28, heavy tails) | 2.9e-3 at N = 30; ≈ 0.02 at the realised n_eff ≈ 2–8 | ≈ 1.2e-2 at N = 30 |

   - Position-level fit: no material gain. In the clean scene its s.e. is 1.84e-4 vs 1.85e-4 for the weighted velocity
     fit, because the noise is not strongly MA(1). With measurement gaps it is biased.
   - Last-15-frame or EW(0.9) fits: s.e. 5e-4, 2.5× worse. They are less biased at a = 4e-3, but detection is worse.
   - The null rate is dominated by B1. Remaining contributors are the reset-starved small dof and association
     outliers. MA(1) is not a contributor.
4. **Headroom.** The ceiling allows up to 0.048 MSE@50 over perfect forgetting. But the estimator's own velocity error
   (median 0.06, mean 0.11 px/frame) already means 3–5 px of displacement error at τ = 50. The displacement proxy
   shows the oracle-velocity order-1 at 10.3 px² against 47–50 px² for every real estimator. Realistic order-2 headroom
   over a tuned forgetting comparator is about 0–6% on the proxy, after fixes. That is below the 5-seed resolution.
5. **Environment.** Correct (Minor 7).
6. **Docs.** As above.

## RECOMMENDATION (re-scoping)

Re-scope PR2:

1. **Ship:**
   - `--accel` env, `accel_law.py` and the ceiling;
   - `--nu_forget` as the K-accel comparator;
   - order 2 as an **experimental opt-in** flag after the B1/M1/M2/M4 fixes, with the gate result documented as
     negative or neutral.
   - Drop the inflation flag, or default it to 1.
2. **Do not run H2 as a 15+3+5-run training campaign.** The order-2 path has no trainable parameters: fit and gate are
   detached, and n_s does not scale α. So H2 can be tested almost for free at **evaluation time**:
   - train order-1 lock on K-accel (3–5 seeds; these runs are needed anyway as the order-1 arm);
   - evaluate {F ∈ {0.6, 0.7, 0.8, 0.9, 0.95}, order 2 fixed, order 2 + F} through the existing `kwm.eval --order
     --nu_forget` overrides, with pred off.
   - Spend GPU on order-2-trained runs only if the evaluation-time order 2 beats the best F at MSE@50 by more than the
     seed spread.
3. **CPU go/no-go before any of that.** Rerun the fixed harness. Report the displacement proxy and n_eff. Require a null
   rate < 2%, and order 2 better than the best F by ≥ 10% on the proxy mean. My offline estimate (+6% vs order 1, mixed
   vs F) says this will likely fail. In that case document it as a negative result and move the GPU to H1/PR3.
4. The real lever for K-accel is the order-1 association noise (σ_z 0.14 vs 0.012 in the clean scene), not the motion
   order. Any order-2 follow-up should come after a better association, measurement or robust-weight estimator.

## Prioritized fix list

1. `kwm/lock_nodes.py:274-275`: variance × Σw²/Σw (or the exact sandwich column). Add a weight-scale invariance test
   and a heterogeneous-weight null test (B1, B2).
2. `kwm/lock_nodes.py:294-295`: replace the label-switch reset with a jump test on z, or continuity matching (M1).
3. `kwm/cells.py:286` and `lock_nodes.py:315-332`: anchor the correction on the fit intercept z̄, blended by g (M2).
4. `kwm/cells.py:259-264, 289-291`: remove the prediction double count under pred on (M4).
5. `kwm/lock_nodes.py:270-283`: dof-aware threshold or an n_eff floor. Document the trace-κ meaning, or use κ=4 (M3).
6. `scripts/order2_gate.py:179`: sweep F ∈ {0.6…0.95} and κ ∈ {4, 9}. Add the displacement proxy and the n_eff
   distribution. Add an a = 0 twin of the synthetic scene over ≥ 50 seeds. Assign among evidence-bearing nodes (M5,
   Minor 1–3).
7. `tests/test_order2.py:324-337`: replace the single-seed argmin test with a seed-ensemble detection-vs-null test
   (B2).
8. `kwm/train.py:178` and `cells.py:126`: inflation default 1, or remove the flag (M6).
9. Add a model-level order-1 identity test against frozen `model.py` (Minor 8). Tag `vel1` for accel (Minor 7). Fix
   the clamp-counter denominator (Minor 4) and the mode averaging in the boost (Minor 5).
10. Docs: correct the MA(1) attribution, remove the "improves slightly" and "recoverability" claims, and trim the
    provenance (Docs).
