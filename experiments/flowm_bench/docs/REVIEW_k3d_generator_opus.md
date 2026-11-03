# Adversarial review: k3d block law for the Blockworld generator (Miniworld-k3d, 4c6eeed + b4aee98 on b28988d)

## Verdict

The law, the env integration and the metadata are correct. I re-derived each of them independently and found no correctness bug in `k3d_dynamics.py` or `moving_blockworld.py`. The grid and continuous env paths are bit-identical to the base commit. The problems are all in how the generator gets driven at scale: two process hazards would silently produce wrong splits, and one change alters the default-path mp4 bytes.

- **BLOCKER (process):** B1, the seed passed after `--` gives N identical clips.
- **MAJOR:** M1, `K3D_COMMANDS.md` uses `--steps 60`, but the data is 500 steps. M2, the ffmpeg thread cap changes the default-path RGB mp4s.
- **MINOR:** m1 to m6.

(Note: REPORT.txt says no commits were made and rendering could not run. Both are stale. Commits 4c6eeed and b4aee98 exist, and every rendering test passes here under xvfb: `10 passed, 1 xfailed in 10.6 s`.)

## Findings

### B1 (BLOCKER for split generation): a `--seed` after `--` now makes every batch clip identical
- **Where:** `scripts/generate_videos_batch.py:121-156` (the forwarded `--seed` passes through verbatim) and `scripts/K3D_COMMANDS.md:70`, which documents this as "intentionally repeats the same clip".
- **Failure scenario:** the existing K3D-int script `scratchpad/k3d/gen/k3d_int_val.sh` passes `--seed 20261103` *after* `--`. On base code, single-run `main` ignored `--seed`, so the 1000 clips came out distinct (I checked: 1000 unique action sequences) but unseeded. Copy that template for K3D-v or K3D-a on HEAD and every subprocess gets `--seed 20261103`. HEAD now honours it, so you get 1000 identical clips. I reproduced this with a 2-clip batch run: seeds 100 and 100, `block_pos_world` identical.
- **Corollary:** the memory note "K3D-int … seeds 20261103+idx" is false. K3D-int is unseeded, has no `seed` key in `_block_info.pt`, and cannot be reproduced.
- **Fix:**
  - In the batch launcher, treat a forwarded `--seed S` as the base seed: strip it from `fwd` and append `--seed S+idx`. Alternatively, `parser.error` whenever `--seed` appears after `--` and `--num-videos > 1`.
  - Change the K3D_COMMANDS text to match.
  - Regenerate K3D-int with HEAD (grid mode is verified bit-identical in dynamics) and `--seed` before `--`, so all three splits are seeded the same way (see also M2).

### M1 (MAJOR): split commands in K3D_COMMANDS.md use `--steps 60`; the released and K3D-int clips are 500 steps
- **Where:** `scripts/K3D_COMMANDS.md:14,25,36,47,79` ("These are the released 60-step … settings").
- **Failure scenario:** `data/blockworld/sunday_v2_validation/*/_actions.pt` and `k3d_int_validation` both have `actions (500,)`, and the eval protocol is 70 context + 70 prediction frames (≥140). Following the doc gives 60-frame K3D-v/K3D-a clips that the 140-frame eval cannot use. The spec's "released command" had `--steps 60` only as a smoke command.
- **Fix:** use `--steps 500` in the K3D-int, K3D-v, K3D-a and batch commands. Keep 20 to 60 steps only for the smoke command, and remove the "60-step released settings" sentence.

### M2 (MAJOR for the "default path bit-identical" claim, MINOR scientifically): `-threads 2 -filter_threads 2` changes the encoded RGB mp4s of the released path
- **Where:** `scripts/generate_videos.py:1784`.
- **Failure scenario:** I ran the same grid clip (seed 5, `--num-blocks 7`, 100 steps) on base and on HEAD.
  - Every tensor is identical: actions, agent_pos, agent_dir, deltas, block_pos_world, depth.
  - `_rgb.mp4` differs in bytes, and its decoded frames differ: max |Δ| = 4/255, 0.04 % of pixels.
  - Encoding one fixed synthetic array with base `write_mp4_rgb` versus HEAD `write_mp4_rgb` gives different md5s. Each writer is deterministic on its own. So the cause is x264 threading, not rendering.
  - Consequence: K3D-int (base code, default threads on this 48-core box) and K3D-v/a (HEAD, 2 threads) would carry systematically different codec noise.
- **Fix:** do one of the following.
  - Make the thread cap opt-in (`--ffmpeg-threads N`, default unset, which reproduces base bytes).
  - Generate all three splits with the same HEAD encoder settings (preferred together with B1: regenerate K3D-int on HEAD).

  Either way, the test should compare decoded mp4 frames from `main()`, not only the in-memory `run_rollout` arrays. That gap is why `test_released_generator_bit_identical` missed this.

### m1 (MINOR, cost): k3d renders the observation twice per step (about 20 % slower)
- **Where:** `miniworld/envs/moving_blockworld.py:521` (and `:359` in reset).
- **Failure scenario:** `MiniWorldEnv.step` already calls `render_obs`, and the k3d branch calls it again. I counted 2.0 `render_obs` calls per step in k3d versus 1.0 in grid. 500-step `--no-depth` clips took 16.6 s (K3D-v and K3D-a) versus 13.9 s (grid), with 4 concurrent runs at 1 thread each.
- **Fix:** return the `obs` from `MiniWorldEnv.step`, which keeps the grid-mode convention of a stale obs that the generator never uses, since it calls `env.render()`. Or document the extra render. This would save about 2.7 s per clip, roughly 45 min per 1000-clip split at 1 process.

### m2 (MINOR): agent handling is not *exactly* grid mode
- **Heading quantised at reset:** `moving_blockworld.py:348` calls `_quantize_heading()` in reset, which grid mode does not do. With `--heading-zero` this has no effect, because the generator sets dir = 0. Without it, k3d frame 0 has a quantised heading where grid frame 0 has a random one.
- **No early return on termination:** `:504` skips the early return on `termination or truncation`, so blocks still move on the truncating step, where grid mode freezes them. This does not matter with `--no-time-limit` and the dropped terminal frame.
- **Carried block:** `:512` does not exclude a carried block from `step_blocks`. The policy never emits pickup: actions seen are {0, 1, 2, 4}.
- **Fix:** document these, or mirror grid mode exactly (drop the reset quantisation, keep the early return, skip carried entities).

### m3 (MINOR): `blocks_static` leaves nonzero velocity on the entities and in `k3d_block_state()`
- **Where:** `moving_blockworld.py:403,514`.
- **Detail:** base static mode sets `ent.velocity = 0`. k3d samples a velocity and exposes it, although the blocks never move. The generator correctly records zero `block_vel_k3d`.
- **Fix:** zero `k3d_vel` when `blocks_static`, or document.

### m4 (MINOR): `vmax` is not a bound and the rejection is inactive at the K3D-a defaults
- **Where:** `k3d_dynamics.py:81-83`.
- **Rejection never fires:** with speed ≤ 1.0, |a| ≤ 0.01 and hi − lo = 14, the envelope is ≤ sqrt(1 + 0.28) = 1.130 < 1.2, so nothing is ever rejected.
- **The spec's envelope is not the true worst case:** the potential drop along a diagonal a is up to 14·(|a_x| + |a_z|) ≤ 14√2·|a|. That gives 1.179 at the defaults, still < 1.2.
- **Bounce error lets speeds drift past `vmax`:** measured over 4000 blocks:

  | Horizon | Max outgoing speed |
  |---|---|
  | 500 steps | 1.148 |
  | 5000 steps | 1.209 |
  | 50000 steps | 1.249 |

  The true invariant E = |v + a/2|²/2 − a·p shows no secular drift: mean ≈ 0, std 0.006 / 0.009 / 0.012.
- **Fix:** doc only, or use the L1 envelope (|a_x| + |a_z|)·(hi − lo). Fine for 500-step clips.

### m5 (MINOR): `--k3d-accel-max` below 2e-3 with `accel_min = 0` raises at reset
- **Where:** `moving_blockworld.py:96` and `k3d_dynamics.py:58`.
- **Detail:** the implicit minimum of 2e-3 is larger than the maximum, so reset fails with "accel_min <= accel_max". Not a silent bug.
- **Fix:** use `min(2e-3, accel_max)`, or document.

### m6 (MINOR, tests)
- **Bit-identical test is incomplete:** `test_released_generator_bit_identical` goes through `run_rollout` with the global `random.randint` monkeypatched. It never exercises `main()`, `_generate_one` or mp4 encoding, which is why M2 went undetected.
- **Energy test checks code, not physics:** the identity in `test_accelerated_energy_identity_and_speed_envelope` is the algebra of `step_blocks` re-derived. It is a valid regression check that the code implements the law, but not an independent physics check. A useful physics check: |v + a/2|²/2 − a·p is exactly conserved on free steps, and |ΔE| ≤ |a|²/2 + 2|a|·|v + a| at a bounce.
- **No test for the batch seed hazard (B1).**
- **The xfail is justified.** At p = (15, 8), v = (1, 0), a = (0.01, 0): the kick gives v = 1.01, the drift gives 16.01, and the mirror gives p = 13.99, v = −1.01. E goes from 0.5 − 0.15 = 0.35 to 0.51005 − 0.1399 = 0.37015, so ΔE = 0.02015 > 1.5·|a|·|v| = 0.015. In general a bounce changes a·p by 2a_x·(overshoot), which is ≈ 2|a||v|. On free steps ΔE = −|a|²/2 exactly (verified: −5.0e-5). The spec's bound is wrong; replace the strict xfail with the correct bound above.
- **Behaviour change in dataset mode:** `_generate_one` (`generate_videos.py:1817`) now resets once with `item_seed`, where base did `reset(seed)` followed by an unseeded `reset()`. The same `--seed` therefore gives different clips than base in `--num-videos` dataset mode. This is intended and documented as deviation 6, but it is not bit-identical to base. Not used by the batch launcher.

## Verified (all independently, CPU, xvfb + Mesa)

Scratch scripts are in `/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/k3d_review/`. The base worktree was created at `…/k3d_review/base` (b28988d) and removed at the end.

### 1. Default path
- **Env level (`env_compare.py`):** I loaded the base `moving_blockworld.py` via importlib next to HEAD's class, with the same kwargs, `reset(seed)` and actions. Over 6 modes × 6 seeds × 60 steps, everything was IDENTICAL: obs, reward/term/trunc, agent pos/dir, every block's pos/velocity/dir, `env.render()` frames, and the final `np_random` state. The modes were grid cardinal, grid non-cardinal, continuous, continuous + domain_rand, torus, and grid without overlap.
- **CLI level (`cli_driver.py` + `compare_out.py`):**
  - Setup: the released command with `--num-blocks 7` (the min/max flags would draw from unseeded `random` on base), 100 steps, `--store-block-info`. Base was run with `reset` patched to inject seed 5, because base ignores `--seed`; HEAD was run natively with `--seed 5`.
  - Identical: every `_actions.pt` key, every `_block_info.pt` tensor, `_depth.pt`, and the decoded `_map_2d.mp4`.
  - `_block_info.pt` gains `seed` and `num_blocks`; `block_colors` become `str` instead of `np.str_` (equal values, and now loadable with `weights_only=True`).
  - The RGB mp4 differs (M2).
- **Not comparable:** base CLI clips with `--num-blocks-min/max` or without the seed patch are nondeterministic, so they cannot be compared at all.
- **HEAD reproducibility:** HEAD grid-mode with min/max and `--seed 5` gives N = `random.Random(5).randint(6, 10)` = 10. Run twice (once with and once without `--no-depth`), it produced byte-identical rgb and map mp4s and identical tensors, so `--no-depth` changes nothing but the depth file.

### 2. The law (`law_check.py`, `grid_vs_law.py`)
- **Order is correct:** `step_blocks` does v += a, then p += v, then a per-axis mirror with the velocity component flipped. Acceleration is never modified. A hand case gives p = (14.8, 7.8), v = (−0.7, −0.2) as expected.
- **Integer exactness, own scalar reference** (sequential X then Z, step back from the old position): 6000 states including corners × 100 steps, 0 mismatches.
- **Integer exactness, against the actual released env code:** grid-mode `MovingBlockWorld`, 40 seeds × 150 steps × 10 blocks, compared with `step_blocks` started from the env's initial state. 0 mismatches.
- **Rejection:** applied per pending block; both v and a are redrawn. With vmax = 1.05 the max envelope is 1.04998, and corr(speed, |a|) of accepted pairs is −0.07.
- **Sampling source:** sampling uses `self.np_random` after `super().reset`, so `reset(seed)` reproduces the state. The block count comes from `random.Random(seed)` in both `main` and `_generate_one`. Verified: the batch item with seed 101 equals a single run with `--seed 101`.
- **Integer mode distribution:** positions are uniform over {1..15} (counts 26.2k to 27.0k per value); velocities split 4 × 25 % across the cardinals; a = 0. The released snapping histogram over cells 1..15 shows no edge deficit.
- **Continuous mode distribution:** θ, speed and log|a| histograms are flat; speed lies in [0.25, 1.0] and |a| in [2e-3, 1e-2].

### 3. Environment
- The agent gets forward_drift 0, snapping in reset and step, and heading quantisation (plus the reset quantisation in m2).
- `ent.pos[[0, 2]]` is synced from `k3d_pos` every step, and y stays at its placed value (0, constant).
- `blocks_static` freezes positions.
- `grid_mode + k3d_mode` raises `ValueError`.
- The EzPickle kwargs are added.
- Off-mode behaviour is unchanged (see 1).
- Existing `tests/test_miniworld.py` gives 7 failed / 70 passed on both base and HEAD, with the same failing set (including the pre-existing `test_pickle_env[MovingBlockWorld]`).

### 4. Generator outputs (`check_k3d_out.py`, `montage.py`, `track.py`)
- **Runs checked:** K3D-v and K3D-a, both with seed 3, 120 steps, `--no-depth --store-block-info`, N = 7.
- **Files and keys:**
  - `_actions.pt` keys are exactly the released six.
  - No `_depth.pt` is written.
  - RGB and map are each 120 × 128 × 128 frames.
  - `_block_info.pt` has the new keys with shape (T, N, 2) / (T, N), the dtypes in the spec, and the `k3d` dict (K3D-a: accel_min 0.002, accel_max 0.01).
- **Physics in the stored arrays:**
  - Free steps: `pos[t+1] = pos[t] + block_vel_k3d[t]`, max error 9.5e-7 (float32), over 1610 steps.
  - The 56 bounce steps match the mirror reflection to 7.7e-7. Bounce flags equal "candidate out of bounds" exactly.
  - The velocity recursion `v[t+1] = ±v[t] + a` holds to 5e-8.
  - Acceleration is constant per block (ptp 0).
  - Speeds stay ≤ 1.002 < vmax.
  - Positions stay within [1, 15], and 89 to 90 % of coordinates are more than 0.05 from an integer.
  - The agent stays on integer cells with heading a multiple of π/2.
- **Visual:** I made montages (`…/k3d_review/out/k3da_montage.png`, `…/k3dv_montage.png`, `*_mapseq.png`) and tracked colour blobs in the map against the stored positions. Blue and yellow fit with a scale of 7.11 px/unit and residual RMS of 0.3 px, about 0.045 cells, while the positions have a fractional-part std of 0.29. So the renderer shows the continuous positions, and motion is smooth. The red-blob fit fails only because the red agent marker contaminates the mask.
- **Batch launcher:** a 2-clip `--steps 10` run with `--seed 100` before `--` and the k3d/accel/no-depth/store-block-info flags after it produced seeds 100 and 101, forwarded flags, and the right N. With the seed after `--`, both clips were identical (B1).
- **Timing** (500 steps, 4 concurrent at 1 thread each):

  | Run | Time |
  |---|---|
  | grid with depth | 16.5 s |
  | grid `--no-depth` | 13.9 s |
  | K3D-v / K3D-a `--no-depth` | 16.6 s |

### 5. Tests
- Under xvfb: 10 passed, 1 xfailed in 10.6 s.
- The tests are meaningful. The integer reference is independent of the module. The env test cross-checks the env against the pure law step by step. The output test checks alignment and reflection.
- Gaps are listed in m6. The xfail arithmetic was verified (ΔE = 0.02015 > 0.015).

## Checklist before the split generation at scale
1. [ ] **B1:** the batch launcher must derive per-item seeds from a forwarded `--seed`, or reject it; fix the K3D_COMMANDS text. Never reuse the `k3d_int_val.sh` pattern (seed after `--`) on HEAD.
2. [ ] **M1:** use `--steps 500` in all split and batch commands in K3D_COMMANDS.md.
3. [ ] **M2:** decide the encoder settings and use the same ones for all three splits. Preferred: regenerate K3D-int on HEAD (grid mode, seeded, same `-threads 2`) so it is reproducible and codec-matched; correct the "seeds 20261103+idx" note for the existing K3D-int. Or make the ffmpeg thread cap opt-in.
4. [ ] Optional **m1:** drop the second `render_obs` per k3d step (about 20 % of generation time).
5. [ ] Optional: replace the strict xfail with the correct bounce bound, add a `main()`-level decoded-mp4 regression test, and add a batch seed-hazard test.
6. [ ] Re-run `tests/test_k3d.py` plus one 500-step K3D-v and one K3D-a smoke clip through the batch launcher with the corrected commands, and check that the `_block_info.pt` seeds are distinct and N matches `random.Random(seed)`.
