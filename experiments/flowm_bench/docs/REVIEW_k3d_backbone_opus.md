# Adversarial review: kinetic 3D backbone (`kwm3d`) and KWM k3d primitives

Reviewed 2026-11-04, CPU only. Probe scripts are in `scratchpad/review/probes/` (p1–p8); test logs are `flowm_tests.log` and `kwm_tests.log` in the same folder.

**Scope and moving target.** KWM worktree `k3d-base` at 0190d25 + 94803f5. FloWM main at 7134424, fd2b157 and **21baf74**. 21baf74 ("use kwm's `_measure_lock` hook") landed while this review was running. Every finding below is against HEAD = 21baf74 unless it says otherwise.

**Test runs.**
- KWM: `pytest tests`, 369 passed and 2 skipped (the slow dataset regressions, which need `KWM_SLOW`), 141 s.
- FloWM, first run: `pytest tests/kwm3d` gave 1 failed and 39 passed. The tree was at fd2b157 for that run, so this failure is the arm-A crash described in F1.
- FloWM at 21baf74: 40 passed in 38.8 s.

## Verdict

At 21baf74, arms C, C-pw, B and A-oracle (the configured `incoming_object` mode) are wired correctly. I checked the ordering, signs, turns, oracle index arithmetic, `load_released` and checkpointing numerically, and found no mismatch (details under "Verified correct"). Arm C is **bit-identical** to `ViTBlockworldModel`, including training mode with the teacher-forcing coin and checkpointing. Its gradients are bit-identical too.

Four things need fixing before results are trusted:
1. The README is stale. It describes an adapter that no longer exists and pins a KWM commit under which HEAD crashes.
2. The Python default `oracle_velocity="raw"` is the convention measured to be wrong, by 1.3–2.4 cells/step.
3. The static-residual estimator gives evidence to most wrong nodes, even on the test's own synthetic scene, and loses the true node's evidence on mildly textured backgrounds.
4. Arm A throws away the valid observation pair after every turn.

---

## 1. Factual errors (claims in README / REPORT / docs that are wrong)

**F1. The README describes a dependency and adapter that HEAD no longer uses.** README §"Dependency and source reuse":
- The README pins KWM at `0190d250…`.
- It says the fork "has no injectable measurement hook", and that `_LockArithmetic` "reuses that installed function's code object with a private globals dictionary".

At 21baf74, `_LockArithmetic` overrides `_measure_lock` (kwm3d_models.py:44-46). That hook only exists from KWM **94803f5**. Under the pinned 0190d25, `_forward_lock` calls the module-global `lock_measurement` directly. That call gets no box, so the 19×17 snapshot hits `ValueError("matching square snapshots required without box")` (lock_nodes.py:309).

Evidence that the mismatch is real and loud, not silent: at fd2b157 with KWM 94803f5 (the opposite mismatch), my first test run failed exactly this way (`flowm_tests.log`: `test_lock_arithmetic_box_and_gradient` → `cells.py:283 _measure_lock` → `ValueError` at lock_nodes.py:309). Arm A was therefore unusable between the two commits. It is fixed at 21baf74, but the README pin must become 94803f5 and the adapter paragraph must be rewritten. Item 3 of the README's "Differences" list is also stale.

**F2. The README says two contradictory things about the oracle default, and the code agrees with the wrong one.**
- README says "The default **`oracle_velocity: raw`** preserves …", then a few lines later says incoming_object is "the configured default".
- `configurations/algorithm/backbone/kwm3d.yaml` sets `incoming_object`.
- `KineticBlockworldModel.__init__` still defaults to `oracle_velocity="raw"` (kwm3d_models.py:57). So does every non-Hydra construction: `scripts/kwm3d_bench.py:47`, all of `tests/kwm3d`, and any notebook or stage-1 script. See M1 for why this matters.

**F3. README: "Turn steps … invalidate the snapshot; the following observation seeds a new pair" is presented as necessary. It is not.** The imprint at a turn step is computed from that step's post-turn frame, in the post-turn map frame (`observation_imprint(obs)` depends only on the frame). So the pair (turn step t, step t+1) is valid without any rotation. The README rationale ("not rotated into a fictitious full cone") applies to the *pre*-turn snapshot only. See M3.

**F4. kwm `docs/K3D_primitives.md` / test (b): "the synthetic blob … velocity error below 0.1" is true, but it hides that most other nodes also get evidence** (see M2). The REPORT's statement that "background-only … cases have exactly zero evidence" is correct only for exactly ego-consistent synthetic backgrounds.

**Correct claims** (checked, not errors): the parameter counts (c / c_pw 12,830,976 = FloWM; a_oracle / b 12,774,144; a +1); "40 tests passed"; the C-vs-FloWM parity on 2 real clips; the 177-cell cone at rows 7–25 × cols 16–32; the mask median IoU (my independent probe: 0.883 on 70 single-block frames).

## 2. Blockers

None at 21baf74 for the configured Hydra path. F1 *was* a blocker for arm A at fd2b157 + KWM 94803f5. M1 becomes a blocker for any A-oracle number produced outside Hydra (bench, ad-hoc eval, stage-1 reuse of the class).

## 3. Major issues

**M1. The `oracle_velocity` Python default is `"raw"`, which is measured to be wrong.**
- **Kinematics alone (p3_chain.py, 40 K3D-v clips, 5,600 transitions incl. 537 turns).** I predicted each block's map position at t from t−1 with the model's own rule, `p_t = rot_vec(p_{t−1} + rot_vec(inc_t, −k) − ego, k)`:
  - `incoming_object`: max error 0.0000 cells, both non-turn and turn steps.
  - `raw` used as the model uses it (`block_vel_agent[t]`, outgoing, ego already subtracted): mean 0.90 / max 2.91 cells on non-turn steps, mean 0.50 / max 2.63 on turn steps.
- **End-to-end through the real dataset loader and `KineticBlockworldModel._move` (p4_oracle_e2e.py, 6 clips, start offset 30, 2,239 block-steps).** I placed a Gaussian blob at each block, ran `_move` with the `_oracle` nodes, and measured the circular centroid:
  - `incoming_object`: max error 4.5e-4 cells (turn) and 4.7e-4 (non-turn).
  - `raw`: 1.27 (turn) and 2.38 (non-turn) cells.

  Fix: make `"incoming_object"` the constructor default (keep `raw` opt-in), and have the bench script pass real metadata or `incoming_object`.

**M2. The static-residual estimator (`_lock_box`, `static_residual=True`) puts evidence on wrong nodes and fails on textured backgrounds.** Probe p8_residual.py uses the kwm test scene (`tests/test_lock_box.py::cone_scene`), v17 nodes, σ = 1 and true velocity (0.3, −0.6).

*(a) The test's own scene.* 13 of 16 moving nodes get non-zero evidence. Several of their estimates are nowhere near the truth:

| node | estimate | evidence |
|---|---|---|
| (−0.75, −0.75) | (−1.14, −0.54) | 0.36 |
| (0.75, 0.75) | (0.72, 1.28) | 0.31 |
| (−0.75, 0.75) | (−1.01, 1.34) | 0.10 |
| (−0.25, −0.75) | (−0.48, −0.63) | 0.88 |
| true node | — | 1.37 |

With the cumulative-gain update (`gain = e/W`), the wrong nodes still take a full clipped step (≤ 0.3) toward these values after one measurement. Even blob-only input (no background) gives 0.92 evidence to (−0.25, −0.75). The residual pair (m_i − m_0, current − m_0) is a dipole-vs-dipole match with false minima.

*(b) Background texture.* With white-noise background texture, the true node's evidence collapses and its error exceeds the 0.1 gate:

| background | true-node error | true-node evidence | locked |
|---|---|---|---|
| test's smooth, amplitude 0.1 | 0.072 | 1.37 | yes |
| same, amplitude ×3 | 0.12 | 0.085 | no |
| white noise, 10 % of blob | 0.17 | 0.000 | no |
| white noise, 30 % of blob | 0.16 | 0.000 | no |
| white noise, 100 % of blob | 0.16 | 0.000 | no |

This is the weakness the docs name (the hypothesis side keeps B(x − ν_i) − B(x)), but it is far worse than "may bias". On a ViT imprint of a textured floor, A could fail kill test 3 because of this residual construction rather than because the idea is wrong. Test (b) only asserts the nearest node, so the suite cannot see either effect.

Suggestions:
- Use the target residual |current − m_0| only as a *support/weight mask*, and match m_i against the current image (not m_i − m_0) inside it.
- Or subtract m_0 on the target side only.
- Report all 16 nodes' evidence in the stage-1 gate, not just Hungarian-matched ones.

**M3. Arm A also discards the step after every turn.** kwm3d_models.py `_step`:

```python
snapshot = torch.where(turning[:, None, None, None], torch.zeros_like(current), current)
snapshot_valid = ~turning
```

The turn-step imprint is in the post-turn frame and is a valid "previous" for t+1. On the first 200 K3D-v validation clips (70-frame contexts), 912 of 12,498 measurable context pairs (**7.3 %**) are lost this way, on top of the turn steps themselves. Fix: in context, always set `snapshot = current; snapshot_valid = True`; skip only the measurement at turn steps (`usable = snapshot_valid & ~turning` already does that). The test `test_prediction_freezes_estimator_and_turn_invalidates` encodes the current behaviour (`len(calls) == 1`) and would need updating (to 2).

**M4. The tests do not test the riskiest claims.** My probes do, and they pass. They should become tests:
- *C parity* (`test_c_released_parity_two_real_clips`) uses only 3 + 2 frames, in eval mode under `no_grad`. Turns are not guaranteed, and the checkpoint path is inactive. p7_c_parity.py extends it: 24 + 8 frames on 2 real clips with 3 turns each gives max |Δpred| = 0 and max |Δh| = 0. Training mode with TF = 0.5, checkpointing on and a non-trivial mixing kernel gives max |Δy| = 0 and max relative grad diff 0 versus `ViTBlockworldModel`.
- *Turn tests* (`test_turn_and_stream`) check only that `turn_state(+1)` then `(−1)` inverts, plus rotation commutation. No test checks that action 0 in arms b/a turns the same way as FloWM's `roll_map`. p2_transport.py does: B's `_move` on the v5 subset of v9 vs FloWM `shift_v_channels`+`roll_map` gives max diff 9.5e-7 for actions 0, 1, 4 and 1.7e-6 for forward; `turn_scalar_state(v5)` vs `roll_map` is exactly 0 for both directions.
- *Oracle*: no test checks the convention end to end. `test_oracle_masked_slots_inert` only perturbs `block_vel_agent`, the raw key. p4 should be a test.
- *`test_cone_translating_blob`* asserts only the nearest node (see M2).

**M5. Memory under per-step checkpointing is dominated by the saved step inputs, which grow with V.** Non-reentrant checkpointing saves each step's `h` (B, V, 33, 33, 256 fp32) = 1.115 MB × V per sample-step:

| arm | V | per sample-step | at batch 8 × 140 steps | at batch 8 × 90 steps (pilot) |
|---|---|---|---|---|
| A | 17 | 19.0 MB | 21 GB | 13.6 GB |
| A-oracle | 11 | 12.3 MB | 13.8 GB | — |
| C | 5 | 5.6 MB | 6.2 GB | — |

All of this comes before the recompute peak of one step. With the 14 GB H1/C4 chains on the 49 GB A6000, A at 140 frames and batch 8 is near the limit. The README's CPU RSS table cannot show this, and it says so. The GPU micro-benchmark should report it.

## 4. Minor issues

1. **Possible checkpoint recompute divergence on GPU.** Arm A's step contains discrete estimator decisions (argmin in `LocalSearch`, `two_modes`, lock bits) recomputed inside the checkpoint. On GPU, nondeterministic FFT or conv reductions could flip a decision between forward and recompute, giving gradients for a different forward. CPU is deterministic (the test passes). Consider computing the measurement under `no_grad` outside the recompute, or check `torch.use_deterministic_algorithms` on the GPU benchmark.
2. **Host syncs per step.** `forward` and `observation_imprint` call `mask.nonzero().tolist()` every step. `LocalSearch` builds `mask_key` via `.cpu().tolist()` per call. Cache the box and key once.
3. **Missing-key collate hazard.** `load_block_metadata` adds `block_speed` / `block_vel_object_incoming` only when present in the .pt file. Mixing k3d and non-k3d splits in one batch would make `default_collate` fail on missing keys. This doesn't affect current configs, which use a single split.
4. **Global test environment override.** `tests/conftest.py` applies globally to `tests/` (including the stage-1 `tests/k3d`). It forces `OMP_NUM_THREADS=8`, `torch.set_num_threads(8)`, `TMPDIR` and `basetemp` into the kwm3d scratch, overriding the caller's thread setting. Scope it to `tests/kwm3d/conftest.py`.
5. **Unused cfg key.** `metric_mixin` silently drops `block_mse` from every algorithm's metric list. Harmless, but `cfg.logging.metrics` then lists a metric that only FloWMVideo implements.
6. **Turn-split test is vacuous.** `block_region_splits` uses the incoming target action for the turn split, which is correct, but `test_region_mse_splits` has turn and non-turn regions both equal to 1, so it cannot catch a swap.
7. **Approximate block footprint.** The block projection ignores block yaw (if K3D blocks are rotated, the footprint is not axis-aligned) and occlusion. The documented IoU of 0.876 makes this acceptable.
8. **Lost early return.** The `k3d_v_training` early return in `build_metadata` prevents caching an empty index, but a run on that split before it is generated will fail later with a less clear error.

## 5. Verified correct (with evidence)

- **(a) Ordering.** Every arm runs transport (+ ego for kinetic) → turn → mix → (decode, in prediction) → write, the same order as `ViTBlockworldModel.forward`. The first action is discarded (forced to 4). C is bit-identical (p7).
- **(b) Signs and frames.**
  - Forward: `stream` with ego (1, 0) equals `roll(−1, W)`, within 1.7e-6 (p2).
  - Left turn: `rot90(k=−1)` plus the channel shift, exactly equal to FloWM (p2).
  - `RIGHT_ROW_SIGN = +1`, from three independent lines of evidence:
    - Kinematics: SIGN = +1 reproduces 537 turn transitions exactly, while SIGN = −1 gives a mean error of 27 cells (p3).
    - Released model: on 4 K3D-int clips × 30 frames (192 block-cells), the mean centred gate activity is 0.527 at row 16 + rel_z vs 0.393 at row 16 − rel_z, with 4/4 clips favouring + (p6). This agrees with the stage-1 `geometry.csv` (20/20 clips).
    - RGB projection: a block with rel_z > 0 lands on the right half of the image in 64 of 70 single-saturated-block frames, median IoU 0.883 (p5).
- **(c) Oracle index arithmetic.**
  - `block_vel_object_incoming[t] = R(agent_dir[t])·block_vel_world[t−1]`, zero at global frame 0. The rotation matches `_to_agent_frame`.
  - The clip start offset and `cond[t] = action[t−1]` (t−1→t) alignment are correct through the real loader (p4, start 30), including turns.
- **(d) Lock arithmetic.** `_LockArithmetic` now calls KWM's own `_forward_lock`: gain e/W, clip 0.3, bound 0.75, static pinning. `test_lock_arithmetic_box_and_gradient` verifies a step of exactly 0.2/3 and W = 3. Box (7, 16), the 177-cell support and the 19×17 crop are correct. A spy confirms the box, mask, σ and `static_residual` reach `lock_measurement` (p1b).
- **(e) `load_released` and arm C.** Missing and unexpected key sets are exact per arm (test). `rebuild_mixing` makes `_mix` the exact identity (`dirac_(groups=256)`). The C bit-identity claim holds, and more strongly than tested (p7).
- **(f) Per-step checkpointing.** The teacher-forcing coin is drawn outside the checkpoint. Gradients are bit-identical to FloWM for C (p7), and within 1e-5 relative for c_pw and a (test).
- **(g) KWM default path.** `tests/reference/lock_nodes_pre_k3d.py` is byte-identical to b98ead2's `lock_nodes.py` (empty diff). The legacy-parity test passes, and all 334 existing KWM tests pass unchanged. Masked values cannot affect outputs or gradients (test). Support is detached.
- **(h) Node sets and the symmetric conv.** v5, v9 and v17 are closed under `rotation_perms` and `reflection_perms`, with node 0 static. `C4SymmetricDepthwise`'s kernel is exactly rot90-invariant (`torch.equal`), equivariance holds to ≤ 1e-6, the identity init is exact, and it has 3·C parameters.
- **(i) Dataset with `load_block_info: false`.** Byte-identical (`test_opt_out_unchanged`). The pad and cond-processing changes are gated by the flag. Metadata has a fixed padded shape, so it collates.
- **Oracle masking.** Invalid oracle slots are inert, with zero features and nodes (test).

## 6. Ranked fix list

1. Change the `KineticBlockworldModel` default to `oracle_velocity="incoming_object"`, and make the bench script use it (M1, F2).
2. Fix the README: pin KWM at **94803f5**, rewrite the adapter paragraph and "Differences" item 3, and remove the contradictory "default raw" sentence (F1, F2).
3. Redesign or at least instrument the static residual before kill test 3. Report evidence on all nodes and on textured backgrounds; consider using the residual as a mask rather than subtracting m_0 from the hypotheses (M2, F4).
4. Stop invalidating the pair after a turn: keep the turn-step imprint as the snapshot (M3, a 7.3 % evidence gain). Update the corresponding test.
5. Turn probes p2, p4 and p7 into tests: turn direction vs FloWM, oracle end-to-end on real metadata, and long-horizon training-mode C parity with gradients (M4).
6. Report saved-input memory per V in the GPU micro-benchmark, and plan batch and clip length for A at V = 17 (M5).
7. Minor items: scope conftest to `tests/kwm3d`; cache the box and mask key; decide on GPU determinism for checkpointed estimator decisions; guard collate keys.
