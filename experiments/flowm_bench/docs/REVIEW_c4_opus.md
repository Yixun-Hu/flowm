# PR6 `--rot c4` review (Opus)

Scope: branch `pr6-c4-rotation`. During the review the change was committed as **cdfe0b2** (on top of 4212949). I checked that the committed tree matches what I reviewed. CPU only, fp32 unless stated. My scripts and raw outputs are in `/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/c4_review/` (`t1*`: equivariance/sign, `t2*`: bit-identity, `t3_gate.py` + `gate/`: estimator, `t4*`: long horizon, `t5`: training, `t7`: profile, `mut/`: mutation tests).

## Verdict

**Fine as committed; make the doc/eval fixes below before the GPU campaign.** No BLOCKER. The group action, the p4 layers, `rot_state`, the bit-identity of `--rot none` and training/checkpointing are all correct. The overclaims are in how the numbers are worded: "exact ≤1e-5" for the lock estimator, a "32% estimator regression", and "unstable long rollouts". Two MAJOR items are about interpretation, and both touch the GPU comparison.

## Independent measurements

### 1. Equivariance: env episode, my own helpers, C=32, 10 observed + 10 predicted, fp32

Setup: `kwm.env` with real_velocity, real_camera, 5 digits, B=2. Frames are rotated with `torch.rot90(x,-k)`. Vectors are rotated with `v @ (R^k)^T`. I wrote my own lattice permutation. kwm.rot is not used.

| model | k | pred nL2 (all steps) | worst step | node vel max abs | node rel | wrong-sign control (`rot90(+k)`) |
|---|---|---|---|---|---|---|
| lk c4 | 1 | 2.1e-7 | 3.5e-7 | 1.0e-6 | 1.1e-7 | 0.46 |
| lk c4 | 2 | 2.2e-7 | 4.0e-7 | 8.3e-7 | 1.1e-7 | (same as k=2) |
| lk c4 | 3 | 2.1e-7 | 3.4e-7 | 7.2e-7 | 9.5e-8 | 0.46 |
| lock c4 | 1 | **8.5e-5** | 1.1e-4 | **2.2e-3** | 1.0e-4 | 0.13 |
| lock c4 | 2 | 9.9e-7 | 3.2e-6 | 1.7e-5 | 2.1e-6 | (same) |
| lock c4 | 3 | **8.5e-5** | 1.1e-4 | **2.2e-3** | 1.0e-4 | 0.13 |
| lock none | 1 | 0.28 | | | | |

**Sign convention vs the environment.** I used single-digit env episodes with the digit fully in view and compared ink centroids. With `rot90(-1)`, the rotated position matches `R(p-15.5)+15.5` to 0.0 px. The rotated ego motion matches `R(v-a)` to within 0.05–0.22 px, the residual coming from the Fourier-rendering edge. With `rot90(+1)` the errors are 5–21 px. On ego centres, `rot_episode` agrees with my rotation to 0.0 px. The convention R(dx,dy)=(−dy,dx) ⇔ `rot90(-k)` is correct and consistent with `kwm.env`.

**Lock errors above 1e-5: cause.** I perturbed the unrotated input by 1.2e-7 relative noise (about 1 ulp), with no rotation at all:

| run (C, seed) | node diff after 1st measurement | node diff late | pred diff max (10 steps) |
|---|---|---|---|
| c4 lock C32 s5 / s6 | 1e-5 / 2e-5 | 1e-5 / 1e-3 | 1.3e-6 / 5.1e-5 |
| **none** lock C32 s5 / s6 | 5e-5 / 1e-5 | 2e-3 / 2e-5 | 4.3e-5 / 2.5e-6 |
| c4 lock C8 s5 / s6 | 2e-5 / 4e-5 | 1e-1 / 5e-2 | **0.24** / 3.3e-2 |
| none lock C8 s5 / s6 | 1e-5 / 6e-6 | 4e-3 / 4e-4 | 2.5e-3 / 2.5e-3 |
| c4 / none lk C32 | ≤8e-7 | ≤1e-6 | ≤9e-7 |

The lock estimator turns a 1-ulp input change into a node-velocity change of about 1e-5 in a single step: roughly 100× amplification, from hard gates, argmin and parabolic refinement. Later threshold flips push it to 1e-3–1e-1. This happens with `--rot none` too, so it is pre-existing and not a C4 defect. A rotation reorders fp32 reductions, so for lock it can never be more exact than this floor. The rotation error I measured with C=8, seed 5 over 20 steps reached 0.10 for the same reason.

### 2. `--rot none` bit-identity vs `kwm-wt-pr3` (separate processes, C=32, world 50, env data)

- **14 adaptive configs**, all `torch.equal`: lk/lock × grad_ckpt on/off × bf16 autocast on/off (lk with ckpt+bf16 omitted) × readout_rel on/off. Each config compares the state dict, predictions, loss, node trajectories, memory states at t=2 and t=5, and all parameter gradients.
- **Trained `pilot_lock/best.pt`** on 8 selection-cache sequences (30 observed + 150 predicted, fp32): predictions and node trajectories are `torch.equal` between the two worktrees.
- **`kwm.eval` main** on the same checkpoint (`--n_val 8 --skip_ft --skip_tp`, CPU, cache built in scratch): all 49 result keys are identical apart from the provenance hashes.
- **Gating.** I grepped every `spectral_shift`/`stream`/`_phase`/`shift_snapshot`/`LocalSearch`/`lock_measurement`/`velocity_weights` call. All three numerical changes depend on `rot == "c4"`: transport.py:19, lock_nodes.py:169/186/233, model.py:175, cells.py:107/223/264/271, model.py:273. The `step()` update-flag capture (model.py:255) is also c4-only. Ungated legacy calls remain in `make_wall0`/wall transport (unreachable: c4 rejects collisions) and in `kwm/equiv.py:46` (boost helper, see MINOR).

### 3. Estimator gate (seed-2024 12-sequence slice, D_lock…heavier_pernode): encoder-seed variance

| encoder (cell seed) | s0 | s1 | s2 | s3 | mean median / mean miss |
|---|---|---|---|---|---|
| none C32 K8 | **0.01566 / 15.0%** | 0.01695 / 16.7% | 0.02086 / 20.0% | 0.02151 / 18.3% | 0.0187 / 17.5% |
| c4 C32 K8 | **0.02072 / 18.3%** | 0.01735 / 18.3% | 0.02372 / 23.3% | 0.01668 / 15.0% | 0.0196 / 18.8% |
| c4 **C64** K8 | 0.02211 / 21.7% | 0.02037 / 16.7% | 0.01727 / 13.3% | – | 0.0199 / 17.2% |
| c4 C32 **K32** (Kf=8) | 0.01827 / 20.0% | 0.01974 / 20.0% | 0.02137 / 21.7% | – | 0.0198 / 20.6% |
| none C32 K32 | 0.01571 / 20.0% | 0.01874 / 16.7% | 0.02004 / 16.7% | – | 0.0182 / 17.8% |
| none encoder + C4 numerics | 0.01789 / 16.7% | 0.01894 / 18.3% | – | – | 0.0184 / 17.5% |
| c4 encoder, dense P (non-equivariant) | 0.01910 / 21.7% | 0.01827 / 20.0% | – | – | 0.0187 / 20.8% |
| c4 encoder, legacy numerics | 0.02304 / 16.7% (reproduces the implementer's ablation) | | | | |

Each run takes about 40 s on one thread. The s0 rows reproduce the implementer's numbers exactly.

**Diagnosis.**
- The "32% regression" compares one draw of each encoder, and the none seed-0 encoder is the best of four none draws. Across seeds the ranges overlap almost completely: none spans 0.0157–0.0215 and c4 spans 0.0167–0.0237. The difference in means is +5% median and +1.3 pp misses. That is well inside the seed spread: the standard deviation over seeds is about 0.0028.
- Changing only the numerics on the none encoder moves seed 0 by +14% (0.0157→0.0179). So this slice cannot resolve effects of this size.
- **C=64 does not recover it.** Analytically, the gate only sees `P·conv_u(frame)`, a linear map. In c4 this map consists of Kf=K/4=2 independent random 3×3 filters × 4 rotations, for any C. In none it is 8 independent filters. Raising K to 32 (Kf=8) does not help either.
- Conclusion: no regression can be resolved at this sample size. I find no evidence that it is inherent to tying. The cause is draw-to-draw variance of a random metric.

### 4. Long horizon (untrained, seq 22 of seed 777, 6 observed + 150 predicted, lock, readout_rel; max |pred| at t=150)

| variant | C8 seed 9 | C32 seed 9 |
|---|---|---|
| none | 0.15 | 0.05 |
| c4 | **2.1e13** | 0.13 |
| c4 with floor 1e-6 | 2.1e13 | 0.13 |
| c4 with legacy Nyquist + norm-dot SSD | 2.1e13 | 0.14 |
| c4 C8, seeds 0/1/2/3 | 0.05 / 0.22 / 0.22 / 0.54 | – |
| none C8, seeds 0/1/2/3 | 0.40 / 0.77 / 0.10 / 0.34 | – |
| **trained pilot_lock (none), 8 seqs × 150** | max \|pred\| 1.19 → 1.00 → 0.92 → 0.87 → 0.88 → 0.67 (every 25 steps) | |

The blow-up needs a single untrained C=8 c4 draw (seed 9). It does not occur for c4 C8 seeds 0–3 or for c4 C32, and it is unchanged when the Nyquist factor or the floor is reverted. It is an artefact of the initialisation, not an instability introduced by C4. The trained none model does not grow. The 9e-4 relative error at 2e13 magnitude is at the lock roundoff floor from §1. **Not a concern for the PR**, but the doc misattributes it (see MAJOR-1).

### 5. Training readiness

- `kwm.train --rot c4 --node_update lock --grad_ckpt --bs 2 --steps 3 --real_velocity --real_camera --world 50 --device cpu`, run with ROOT redirected to scratch: losses 0.132 / 0.137 / 0.154, all finite; about 30–37 s/step on CPU (30+20 frames).
- `last.pt` stores args `rot=c4`. `load_model` rebuilds `GConvLift` with `cell.rot=c4`. A fresh `build_model(ck["args"])` gives an identical state dict and bit-identical predictions. `rotation_check` on the trained model (4 sequences, 6+10) gives 3.6e-7.
- Profile, one fwd+bwd (lock, C32, bs 2, 6+4, grad_ckpt, 4 threads): none 5.26 s, c4 5.59 s (+6%). There are 0 `aten::item`/`_local_scalar_dense`/`nonzero` calls in either mode. The extra transients come from the direct-SSD `sub`/`pow` (about 1 GB cumulative per step on CPU at bs 2, chunked). At bs 16 on GPU that is about 92 MB per chunk, which is fine.

### 7. Tests: mutation check (test_rot.py on a scratch copy)

| mutation | result |
|---|---|
| clean | 29 passed (108 s with the machine heavily loaded, load avg ~45; the implementer measured 66 s) |
| GConvRegular orientation shift `roll(r)`→`roll(-r)` | 15 failed |
| lift kernel `rot90(-r)`→`rot90(+r)` | 15 failed |
| invariant-out `rot90(-r)`→`rot90(+r)` | 15 failed |
| moment directions `[vx,vy,-vx,-vy]`→`[vx,-vy,-vx,vy]` | 15 failed |
| Nyquist fix on x only | 15 failed |
| P ordering `krfs`→`rkfs` | 7 failed |
| spread weights per orientation | 16 failed |
| floor 1e-2→1e-6 | 3 failed (only `test_weak_memory_readout`) |
| direct SSD→norm-dot | 1 failed (only `test_lock_search_roundoff_on_digits`) |

The tests are meaningful and catch every structural mutation. The last two changes are protected only by tests that were tuned to specific seeds and sequences.

## Findings

**MAJOR-1: "exact ≤1e-5" overstates what the lock estimator achieves** (docs/PR6_c4_rotation.md:126-130, 175-195; kwm/eval.py:284-300).

- *Evidence:* §1. With real 5-digit episodes and C=32, the lock rotation error is 8.5e-5 in predictions and 2.2e-3 px/frame in node velocities after 10+10 steps. With C=8 it reaches 0.10 within 20 steps. The 6.9e-7 held-out figure depends on a 4-step horizon. The unrotated 1-ulp control gives the same magnitudes with `--rot none`.
- *What the doc gets wrong:* it attributes the 150-step 9e-4 to an "unstable long rollout". In fact the C4 symmetry is exact in exact arithmetic and lock cannot be more exact than its own ulp sensitivity.
- *Fix:*
  - (a) In the doc, state the lock guarantee as "rotation error ≤ the model's own 1-ulp perturbation sensitivity". Drop the 2e13 explanation, or say that it comes from one untrained C8 init.
  - (b) In `rotation_check`, also compute the same normalized error between unrotated runs on `obs` and on `obs*(1+ε·noise)`, and report it as `rot_equiv_floor`. Report the max over sequences next to the mean. Without that floor, a GPU-trained c4 lock model at 150 steps may show `rot_equiv_error` of about 1e-3, which could be misread as broken equivariance.
  - (c) Optionally add a test that compares rotation error against the ulp floor (ratio ≲ 10) instead of a fixed 1e-5 on long horizons.

**MAJOR-2: the "estimator regresses 32%" headline is a single-draw artefact** (docs/PR6_c4_rotation.md:233-262; commit message).

- *Evidence:* §3. Across four encoder seeds, none and c4 overlap. The seed-0 none encoder is the best none draw. Numerics alone move one seed by 14%. C64 and K32 do not change the picture.
- *Fix:* replace the paragraph with the multi-seed table (or rerun on the 300-sequence set with ≥4 seeds and a bootstrap CI). Conclude "no resolvable difference at n=12".
- The doc currently explains the regression by the eight filters vs 32. That explanation is not supported, and C is irrelevant: the effective count is Kf=2 vs 8 independent imprint filters, and K=32 does not help either.

**MINOR-1: the C4 arm changes four things at once** (model.py:175, transport.py:19-26, lock_nodes.py:186-194, cells.py:145-150).

- The GPU comparison will confound weight tying with the readout floor, the Nyquist factor and direct SSD.
- I measured the numerics alone on the none encoder (0.0179/0.0189 vs 0.0157/0.0170 on two seeds): in the noise.
- The floor 1e-2 is an absolute, scale-dependent constant that only `test_weak_memory_readout` depends on.
- *Fix:* add an internal switch, for example `--c4_numerics on` for `--rot none`, default off so bit-identity is preserved. That allows a "none + C4 numerics" control arm if budget permits. At minimum, list the four differences in the comparison write-up.

**MINOR-2: misleading comment on the test bound** (tests/test_galilean.py:133-139).

- The beta=20 override is commented as if it were needed for the "new tied-readout case".
- I measured the relative-readout boost residual at beta=10: none 6.8e-4 / 1.6e-3 / 4.1e-4 (seeds 0/1/2) and c4 1.3e-3 / 1.1e-3 / 8.6e-4. With the floor at 1e-6, c4 gives the same values. So the c4 case fails the 1e-3 bound only because of the seed draw, exactly like none would with seed 1. At beta=20 both modes are ≤1.1e-6.
- *Fix:* say that the bound is marginal for both modes at beta 10, or use beta 20 for both.

**MINOR-3: unused buffer** (kwm/model.py:124-125). The `perm_rot` buffer is never read; `rot_state` recomputes `rotation_perms`. Either use it in `rot_state(..., perm_rot=model.perm_rot)` or remove it.

**MINOR-4: boost helper uses legacy transport for c4 models** (kwm/equiv.py:46). `boost_transform` calls the legacy `spectral_shift` even for c4 models, which stream with the c4 phase. The difference sits at the Nyquist bins of fractional boosts, and the acceptance tests still pass. *Fix:* pass `c4=model.rot=="c4"`, or document it.

**MINOR-5: inconsistent update-flag semantics** (kwm/model.py:255). The fix for capturing the update flag under checkpointing applies to lk only under c4. As a result, `none` lk with `grad_ckpt` and `node_update_pred off` keeps its pre-existing recompute mismatch, while c4 lk does not. This does not affect the planned lock comparison with prediction updates on. *Fix:* fix it in none in a separate PR (it changes gradients).

**MINOR-6: `rot_check` gaps** (kwm/eval.py:284-300). It tests only k=1 and reports only the mean. k=2 behaves differently for lock (≈1e-6 vs 8.5e-5 for k=1), so k=1 is the stricter choice, which is good. Add the max and the floor (MAJOR-1). On GPU, `kwm.eval` does not disable TF32, so state whether `rot_check` ran with TF32 on.

**MINOR-7: doc sign wording.** The convention is correct: `rot90(-k)` is clockwise on screen. The doc says the task's two requirements "cannot both hold"; it would be clearer to say that the task's "counter-clockwise" assumes a y-up display.

Items I checked and found correct:
- **gconv.py:** lift `R^r base`; regular `R^r base[(r'−r)]` via `roll(r,2)`; invariant-out `R^{r'} base`. I checked the algebra by hand, and the mutations confirm it.
- **MomentProjection:** `[vx,vy,−vx,−vy]` is the regular field of projections on `e_r`; spread is lifted invariantly, and the fan-in scaling is right.
- **Imprint P:** P = P0⊗I4 in `k*4+r` order.
- **`rot_state`:** handles LK metadata (rho rotated, W carried) and the LockState snapshot correctly.
- **Nyquist cos factor:** it is the correct real band-limited interpolation and is exact for integer shifts. It is not unit-modulus, so the Nyquist component is damped by |cos πd| each step. The doc already notes this.

## Recommendation for item 6 (GPU: C4 lock vs lock)

- **Primary comparison: C4 C=32 vs lock C=32.** Both have the same feature/memory tensor shapes, the same K=8 imprint, the same estimator cost and about the same wall time (+6% CPU). The only intended difference is the symmetry constraint (5.3k vs 21.2k parameters). The existing H1 lock seeds at C=32 can serve as the baseline if the protocol is identical (same steps, amp, grad_ckpt, held-out seeds), so only the C4 arm needs new runs. This answers whether imposing the symmetry helps at fixed architecture. Because the dataset contains only upright digits, any gain is a sample-efficiency or inductive-bias effect, not a symmetry of the data. Rotation does not restrict this transport/reconstruction task, since identity reconstruction is equivariant.
- **Secondary comparison, run if budget allows: C4 C=64** (20.9k parameters ≈ matched). It doubles the memory state and has about 4× the conv FLOPs, so it is not compute-fair. Note that it does not increase the imprint diversity (Kf stays 2). Use it as a capacity control: if C4-32 loses to lock-32 but C4-64 matches it, the cost is capacity, not symmetry.
- Run both with ≥3 seeds and use the paired seed-level statistics already used in H1.
- Add the "none + C4 numerics" arm (MINOR-1) only if C4-32 differs from lock-32 by more than the seed spread.
- Report `rot_equiv_error` together with the ulp floor (MAJOR-1).
- Do not use the 12-sequence estimator gate to choose between the arms (MAJOR-2).
