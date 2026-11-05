# PR1 pre-submission review: `--node_update lock` (branch pr1-lock-nodes @ 28ba51e vs upstream main 302d8f8)

Probes, logs and scripts are in this directory: `stats.py`, `probe_step.py`, `compare.py`, `probe_ckpt.py`, `analyze_ck.py`, `probe_sign.py`, `pytest_default.log`, `pytest_slow.log`, and `upstream/` (an archive of 302d8f8, with data symlinked in).

**Verdict.** The code is sound and the default path is unchanged bit-for-bit. Nothing in the code blocks submission. The blockers are in the documents. The PR's own doc still says "End-to-end comparison: Pending". The gate doc links to 9.5 MB of result files that are not committed. The planned headline relies on zero prediction input, which PR1 cannot produce by itself. Inside PR1, zero input does nothing to the nodes. The honest PR1 headline is the native-input result: −22 % MSE@20, which is solid. At 150 frames, all five seeds point the same way, but the result is not significant.

---

## Blockers (fix before opening)

**B1. The PR's own doc contradicts the claim it is meant to support.** `docs/PR1_lock_nodes.md`, section "End-to-end training comparison", reads "**Pending.** Compare LK and the final lock rule …". `docs/GATE_estimator.md` says "The final-rule training comparison remains pending." The five-seed result exists, so these sections must be replaced with it before the PR is opened (see the draft below).

**B2. Broken evidence links.** `docs/GATE_estimator.md` links to `../results/lock_nodes_gate/{seed2024,seed777,seed31337,pooled}.json`. `git ls-files results` on the branch lists only `results/figs/*`. Those JSONs are untracked (`git status`: `?? results/lock_nodes_gate/`, 9.5 MB; `?? results/gate_estimator.json`, 18 MB). Upstream tracks no result JSONs (`git ls-tree 302d8f8` lists only figs). Do not commit 9.5 MB. Use one of these options:
- drop the links and give the summary table plus the reproduction commands;
- commit a compact `pooled` summary of a few KB;
- attach the files to a release or gist.

**B3. Headline framing: the zero-input numbers are not a PR1 capability.** PR1's `kwm/eval.py` has no prediction-input override: `grep dec_input kwm/eval.py` returns nothing on this branch. Every checkpoint was trained with `dec_input='flowm'`, as the checkpoint `args` of all 10 runs show. So "−16 % at 150 frames under zero prediction input" needs PR3's `kwm.eval --dec_input zero`.

I also measured that **under zero input the node update is inert**, for both rules:
- On h1_lock_s1 at step 20k (16 sequences, generator seed 4242), lock node drift over the 150 prediction steps was **exactly 0.0** with zero input. With feedback input it was mean 0.18 and max 0.53 px/frame. LK drifted by ≤4.3e-7.
- In the campaign JSONs, `zero_predon` and `zero_predoff` have **bit-identical per-sequence MSE in 50/50** checkpoint×run pairs.

So the zero-input result means "lock-estimated nodes, frozen after observation, with zero input". It is a joint PR1+PR3 effect. Lead with the native-input numbers in PR1. Put the zero-input numbers in a clearly labelled supplementary line that names the other branch, or move them to PR3. Never present "lk (published) → lock + zero = −29.5 %/−39.1 %" as PR1's effect, because it combines two changes.

## Major

**M1. "Identical compute" is false. Say "identical optimization budget".** Arms do match on steps, batch size, seed, `grad_ckpt`, lr, tf, t_obs/t_pred and data stream: I diffed all 10 checkpoint `args`, and only `node_update`, `seed` and `out` differ. Wall-clock does not match. Median s/step from `runs/*/train.csv` on the shared A6000:

| Seed | lk s/step | lock s/step | Ratio |
| --- | ---: | ---: | ---: |
| 0 | 6.93 | 8.22 | 1.19× |
| 1 | 6.61 | 8.43 | 1.28× |
| 2 | 4.47 | 5.73 | 1.28× |
| 3 | 4.47 | 5.76 | 1.29× |
| 4 | 4.47 | 5.74 | 1.28× |

Peak memory is 6433 MB for lk and 5239–5241 MB for lock, about 19 % lower. The PR doc's "GPU timing" table (4.52 vs 5.34 s, 15.2 vs 13.2 GB) comes from a different 60-step configuration (`accum 2`, no grad_ckpt). Replace it with these campaign medians, or report both and label each.

**M2. The 150-frame claim under the native protocol is not significant. Several tests were run, so say so.** I recomputed from the held-out JSONs. For each run I averaged checkpoints 16k–20k, asserted `test_seed==20261025`, `n_val==1000` and the arm's `node_update`, then paired over 5 seeds:

| Prediction input (pred updates) | MSE@20 lk → lock | p (t) | MSE@150 lk → lock | Per-seed change @150 | p (t) |
|---|---|---|---|---|---|
| feedback (on) — **PR1 alone** | 0.0284 → 0.0220, −22.2 % | 2e-4 | 0.1298 → 0.0918, −26.1 % | −22.4, −37.2, −19.8, −50.6, **−0.6** | 0.065 |
| feedback (off) — PR1 alone | 0.0283 → 0.0219, −22.4 % | 2e-4 | 0.1298 → 0.0867, −29.9 % | −27.4, −42.8, −21.9, −54.3, −3.0 | 0.054 |
| zero (needs PR3) | 0.0256 → 0.0200, −22.1 % | 1e-4 | 0.0904 → 0.0754, −16.2 % | −21.5, −30.4, −4.8, −20.2, −4.2 | 0.040 |

These match CAMPAIGN_H1_progress.md. All 5/5 seeds improve in every cell. With n=5 the exact sign-flip or Wilcoxon p cannot go below 0.0625, so state "5/5 seeds" next to the paired t.

After a Holm correction over the four primary cells (2 horizons × 2 inputs), only the 20-frame results survive. The zero-input 150-frame cell would need p ≤ 0.025 and has 0.040. Recommended wording: "MSE@20 improves by 22 % (5/5 seeds, p < 0.001). MSE@150 improves on 5/5 seeds, by 26 % on average, but the spread is large (−0.6 % to −51 %) and the result is not significant at n = 5."

The large lk spread at 150 frames comes from prediction-feedback spikes. For example, the lk seed-3 checkpoints are 0.253, 0.100, 0.152, 0.114 and 0.244. Say this, because it means part of the gap at 150 frames is lk instability, not lock accuracy.

**M3. Lock nodes drift more on their own predictions. Document this.** In the same probe (h1_lock_s1 vs h1_lk_s1, feedback input, pred on), mean node drift during prediction was 0.18 px/frame for lock and 0.04 px/frame for lk. Turning prediction updates off improves lock at 150 frames (−29.9 % vs −26.1 % against lk; lock MSE@150 0.0867 vs 0.0918). Add this to Limitations. Optionally say that `--node_update_pred off` is the better setting for lock under feedback, but stop short of changing the default.

**M4. `scripts/estimator_gate.py` (1,086 lines) does not belong in this PR in its current form.**
- It re-implements `gaussian`, `coherence`, `pool`, `two_modes`, `LocalSearch` and `shifted_window_mask` in parallel with `kwm/lock_nodes.py`, so the two can drift apart.
- It keeps about 20 historical variants (`Spec("A_LK" …)` to `E_window_*`, lines 367–382).
- It hard-sets `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS=1` and defaults `CUDA_VISIBLE_DEVICES=""` at import time (lines 27–29).
- It imports SciPy, which is not in `requirements.txt`.

`GATE_estimator.md` also has about 60 lines of historical ablations ("earlier cross-node rule", "historical prediction-input proxy", "former bit-parity test retired"). Options: (a) move the script and doc to a follow-up "analysis tools" PR, or keep them fork-only; or (b) reduce the script to `A_LK` vs the final rule, importing `kwm.lock_nodes`, and cut the historical sections. A maintainer will very likely ask for one of these.

**M5. The docs read as a work log, not upstream documentation.** Examples:
- `docs/PR1_lock_nodes.md`: "These supplied integration measurements were not rerun during CPU validation", "A review diagnostic allowing TF32…", "all 30 node updates matched the pre-optimization implementation", "LK reference is frozen from main@302d8f8".
- The file name itself encodes the fork's PR numbering. Rename it to `docs/LOCK_NODES.md` and update the README and REPRODUCE links.
- Cut the narrative to: method, state, gradients, cost, results, limitations.

## Minor

- **m1.** `--node_update lock` is silently accepted for non-adaptive models. `build_model(dict(model=k, node_update='lock'))` succeeds for `spectral`, `flowm` and `oracle` with no warning. `kwm.eval` then writes `node_update: "lock"` into a FloWM JSON (`eval.py`, `res = dict(... node_update=model.node_update ...)`). Raise in `KineticWM.__init__` when `node_update != "lk"` and `kind != "adaptive"`, and record `node_update` only for adaptive models.
- **m2.** `cells.py`, `_forward_lock`: `residual = (nu + delta / dt) - nu` and `rho_new = (nu - nodes + step)` are round-about forms of `delta/dt` and `rho + step`, and they add fp32 rounding. If this is deliberate, to stay bit-compatible with the harness, the comment should say so. Otherwise simplify. Changing it now would perturb the trained-model reproducibility claims, so a comment is the cheaper fix.
- **m3.** No unit test checks the search route's sign or accuracy for an unlocked node. `test_displacement_gradient_and_scale` with v=0.5 checks only the lock bit and the zero gradient. Only the slow dataset tests would catch a sign error. My probe (`probe_sign.py`) shows the route is correct: v=(0.5,0) gives (0.499,0); v=(−0.5,0.3) gives (−0.519,0.301); v=(0.7,−0.6) gives (0.588,−0.539) at low evidence; v=(0,0.9) is rejected with evidence 0. Add a fast assertion such as `abs(delta - v) < 0.05` for v=0.5 so the default suite covers it.
- **m4.** Inconsistent signatures. `KineticCell.forward(..., *, update_nodes=None)` is keyword-only, while `AdaptiveNodeCell.forward(..., collision=None, update_nodes=None)` is positional.
- **m5.** README: the lock line is inserted at the top of "Quick start", before the install commands. Put it in the command table or under the adaptive model description.
- **m6.** Train/eval precision differs. Training uses bf16 autocast and TF32 matmul on CUDA. `kwm.eval` defaults to fp32 with no TF32. Lock's discrete gates (lock bit, mode split, edge rejection) are more sensitive to this than LK. The doc's TF32 diagnostic covers only the SSD contraction on one seed. The difference applies to both arms and the results were measured in fp32 eval, so this is not a validity problem. State it in Limitations.
- **m7.** `tests/reference/adaptive_cell_pre_lock.py` is a 152-line frozen copy of the upstream cell, and it still imports `kwm.transport.lk_displacement` from the live tree. So it is not fully frozen: a change to `lk_displacement` would change both sides. The bit-identity probe below covers this gap. Either note it in the file header or freeze `_lk_fp32` too.
- **m8.** Commit hygiene: four commits, two of them "Estimator gate" iterations. Squash to one or two commits: estimator plus model integration, and the optional analysis tool if kept.
- **m9.** Evidence doc, outside the PR: `CAMPAIGN_H1_progress.md` "Reading so far" is stale. It quotes four seeds, "−19 %, p = 0.045", "−33 %, p = 0.053", and "lk 0.0283 / 0.136", while the table above it gives 0.0284 / 0.1298. Fix it before quoting from it.
- **m10.** Provenance gaps to disclose:
  - The seed-0 pilot checkpoints have no `provenance` (trained on 28ba51e before PR5a existed).
  - Seeds 1–4 were trained on the PR3 tree (4212949).
  - Seed 3 of both arms was resumed from `last.pt` (train.csv has 405 and 410 rows instead of 401). The pairing is preserved.
- **m11.** The estimator variant was selected on MNIST *test-split* sequences (seeds 2024, 777, 31337). The held-out set (seed 20261025) uses the same digit pool with different sequences. This is not leakage, but state it in one line.

## Verified correct (with evidence)

1. **The default LK path is bit-identical to upstream.**
   - I ran 3 optimizer steps of the real training loop (make_batch → forward with tf=0.5 → MSE → backward → clip → Adam), with `torch.use_deterministic_algorithms(True)` and 1 thread.
   - Configurations: `base`, `--grad_ckpt`, `--node_update_pred off`, `--dec_input zero`. PR branch with and without explicit `--node_update lk`, against `git archive 302d8f8`.
   - **BITIDENTICAL** in all 8 comparisons for losses, predictions, node trajectories, every parameter gradient at every step, and the final `state_dict`. The only arg-key difference is the new `node_update`.
   - The PR3 tree's lk path is also bit-identical to upstream (base and ckpt).
2. **Test suite.**
   - Default: `95 passed, 2 skipped` in 27.7 s at 6 threads (34.9 s wall). Upstream: 19.2 s wall.
   - `KWM_SLOW=1`: `97 passed` in 59 s. The slowest tests are the two slow dataset regressions (14.9 s and 13.7 s), the on-lattice boost (2 × 3.9 s) and the lock memory equivariance (3 × 2.0 s).
   - The doc's 1-thread numbers (53 s and 129 s) are consistent with this.
3. **The forward value equals the detached estimator, and the gradient goes only through locked LK** (`lock_nodes.py` 239–243). `delta + (differentiable - differentiable.detach())`: in both the split and unsplit branches of `two_modes`, `selected` sums to the returned `wp` = `evidence`, so `differentiable` equals `primary` algebraically on locked nodes. `torch.where(locked, …, 0)` zeroes the gradient on searched nodes. The test asserts `count_nonzero(grad)==0` for an unlocked node and a nonzero gradient for a locked one.
4. **Detachment.** Search, gates, `two_modes` and `support` run under `no_grad`. `n_s` gets its gradient through `step` on both routes, and the tests assert `n_s.grad > 0`.
5. **The lock bit is evaluated fresh each step and never fed back.** `h_prev.nodes[..., 3]` is never read in `_forward_lock`. It is only re-thresholded in `boost_transform_nodes`. This matches the doc ("diagnostic, without hysteresis").
6. **Evidence accumulation and gain.** `W_new = W + evidence` and `gain = evidence / max(W_new, 1e-6)`. The step is clipped to ±0.3 and the result clamped to ±0.75. This matches the doc and the LK running-mean form.
7. **Doc constants match the code:**
   - `lam_rel` 0.01, `eps` 1e-3, `tr_rel` 0.1, gate 0.6, `coh_rel` 0.5;
   - lock thresholds: agreement ≥ 0.1 and |pooled LK| ≤ 0.3;
   - 25 search shifts on ±1 at 0.5 spacing;
   - Gaussian σ = 2, radius 6; mass < 0.5 rejected; support > 0.99 required; boundary minima rejected;
   - two modes: 5 iterations, 10 % minimum mass, 0.1 separation, left wins ties (`sl >= sh`);
   - state size: 33,168 B/sample (400 + 32,768).
8. **The checkpoint closure captures the phase flag for lock** (`model.step`). On CPU, lock with `--grad_ckpt` is bit-identical to lock without it over 3 training steps (`pr1lock_base` vs `pr1lock_ckpt`).
9. **The PR3 tree used for training and evaluation is numerically the PR1 lock.**
   - `lock_nodes.py` and `transport.py` are identical between 28ba51e and 4212949.
   - 3 lock training steps are bit-identical between the PR1 and PR3 trees, with and without grad_ckpt.
   - Loading `runs/h1_lock_s1/ckpt_step20000.pt` with PR1 code and with PR3 code gives bit-identical predictions in all 4 cases (input feedback or zero, prediction updates on or off). The headline JSONs therefore describe PR1's estimator.
10. **Held-out hygiene.**
    - Every JSON used has `test_seed=20261025`, `n_val=1000`, MNIST `test` split and role `heldout`.
    - Training draws from the MNIST `train` split with generators seeded 0–4. The selection cache is seed 12345.
    - The 5-checkpoint average (16k–20k) removes checkpoint selection. It was fixed in plan v4 on 2026-10-25, before the pilot held-out evaluations of 2026-10-28.
    - Parameter counts are identical (21,154 in both arms).
11. **Search-route sign and accuracy** are correct on synthetic shifts (see m3). The lock fraction on a trained model is 16 % of nodes at the end of observation (h1_lock_s1), so most nodes use the gradient-free search route.

---

## Recommended PR title

**Add opt-in lock-then-refine node velocity update for the adaptive model (`--node_update lock`)**

## Draft PR description

> ### What
> Adds `--node_update lock`, an opt-in alternative to the adaptive model's Lucas–Kanade (LK) node-velocity update. The default `lk` is unchanged: forward values and gradients are bit-identical to `main` (see Validation). `kwm.eval --node_update` can override the rule saved in a checkpoint. Older checkpoints load as `lk`.
>
> ### Why
> In 5-digit scenes, LK's per-node running mean often locks onto the wrong velocity: 42 % of digits end more than 0.1 px/frame from their assigned node in a no-learning test. Same-cell digit pairs are rarely resolved.
>
> ### How
> - Each step, each node checks whether its LK residual is trustworthy ("locked": enough coherent textured evidence and a small residual).
> - Locked nodes use LK. Unlocked nodes use a dense ±1 px SSD search with sub-pixel refinement against a per-sample observation snapshot.
> - The per-pixel residuals are pooled with the existing gates into up to two modes. The node takes the heavier mode.
> - The offset update is the existing evidence-weighted running mean with the learned scale `n_s`. No parameters are added.
> - Gradients pass only through the locked LK displacement. Search and mode selection are detached.
> - Node state is an explicit fp32 `LockState`: features, plus `[B,V,4]` node metadata, plus one `[B,K,32,32]` snapshot.
>
> See `docs/LOCK_NODES.md`.
>
> ### How it was evaluated
> - **Data and training.** Kinetic MNIST World with real velocities and camera. Both rules trained with identical settings: 20k steps, batch 16, `--grad_ckpt`, same seed and data stream per pair, 5 seeds.
> - **Scoring.** Mean over checkpoints 16k–20k (no checkpoint selection) on 1000 held-out sequences (seed 20261025, MNIST test digits), with the repository's default prediction protocol (model feedback, node updates on).
> - **Statistics.** Paired over seeds.
>
> | | LK | Lock | Change | Seeds improved | Paired t |
> |---|---:|---:|---:|---:|---:|
> | MSE@20 | 0.0284 | 0.0220 | −22 % | 5/5 | p < 0.001 |
> | MSE@150 | 0.1298 | 0.0918 | −26 % (per-seed −0.6 % to −51 %) | 5/5 | p = 0.065 |
>
> The 20-frame gain is consistent across seeds (−18 % to −26 %). The 150-frame gain is in the same direction on every seed but is not significant with 5 seeds. Part of that gap is LK's prediction-feedback instability (late-checkpoint spikes up to 0.25).
>
> Supplementary, using an evaluation-time zero prediction input from a separate branch (not in this PR): MSE@20 is −22 % (p < 0.001) and MSE@150 is −16 % (0.0904 → 0.0754, 5/5 seeds, p = 0.04, not significant after correcting for the four comparisons). Under zero input the node update has no effect during prediction for either rule.
>
> No-learning estimator test (frozen random encoder, 300 sequences, 3 seeds): matched median node error drops from 0.082 to 0.024 px/frame, and misses from 42 % to 18 %.
>
> **Cost.** Same optimization budget, but lock takes 1.2–1.3× longer per step on a shared A6000 (4.47 → 5.74 s/step at batch 16 with grad_ckpt). Peak memory is 19 % lower (6.4 → 5.2 GB).
>
> ### Validation
> - `pytest -q`: 95 passed, 2 slow skipped. `KWM_SLOW=1 pytest -q`: 97 passed.
> - LK forward values and gradients are bit-identical to `main@302d8f8`. Checked over several training steps (base, `--grad_ckpt`, `--node_update_pred off`, `--dec_input zero`) and against a frozen reference cell.
> - Comoving on-lattice boost equivariance of the node memory is ≤ 1e-6 on synthetic tests and ≤ 1e-5 on MNIST.
>
> ### Limitations
> - Search radius, support rejection and the offset clip limit the residual motion that can be recovered per step (about 0.6 px/frame).
> - The heavier-mode rule can discard a smaller object that shares a velocity cell.
> - Discrete gates can turn small floating-point differences into different decisions. Training uses bf16/TF32 on GPU and evaluation runs in fp32.
> - With feedback input and prediction updates on, lock nodes drift more on generated frames than LK nodes (about 0.18 vs 0.04 px/frame on average over 150 steps in one probe). `--node_update_pred off` scores slightly better for lock at 150 frames.
> - Search runs for every node, so cost does not fall as more nodes lock.
> - Only Kinetic MNIST World with real velocities and camera was evaluated. Integer-velocity data, walls and odometry noise were not tested.
>
> ### AI assistance
> Most of the code, tests and documentation were written with AI coding assistants (OpenAI Codex and Anthropic Claude), directed and reviewed by me. All experiments were run and checked by me, and I take responsibility for the change.

## Ranked fix list

1. Replace the "Pending" sections in both docs with the native-input five-seed result. Use the wording and statistics of the draft (B1, B3, M2).
2. Change "identical compute" to "identical optimization budget" everywhere, and report wall-clock (1.2–1.3×) and memory (−19 %) from the campaign `train.csv` (M1).
3. Move the zero-input numbers into a labelled supplementary line, or into PR3. State that node updates are inert under zero input. Drop the "lk published → lock zero" combined figure from PR1 (B3).
4. Fix the GATE doc links: remove them, commit a compact summary, or host the files externally. Do not commit the 9.5 MB / 18 MB JSONs (B2).
5. Remove `scripts/estimator_gate.py` and `docs/GATE_estimator.md` from PR1, or slim them to LK vs the final rule built on `kwm.lock_nodes`. Remove the import-time env overrides, and declare SciPy as optional (M4).
6. Rewrite `docs/PR1_lock_nodes.md` as `docs/LOCK_NODES.md` with the work-log phrasing removed, and update the links (M5).
7. Add the lock-node prediction drift and the predoff observation to Limitations (M3).
8. Reject `node_update='lock'` for non-adaptive models, and record `node_update` only for adaptive runs (m1).
9. Add a fast search-route accuracy assertion for an unlocked node (m3).
10. Comment or simplify the `residual` and `rho_new` arithmetic (m2). Make the `update_nodes` signatures consistent (m4). Move the README line (m5). Note the precision differences (m6) and the frozen-reference caveat (m7).
11. Squash the commits (m8). Fix the stale "Reading so far" in the campaign doc (m9). Disclose provenance in the PR, or in a linked doc: seed-0 checkpoints without provenance, seeds 1–4 trained on the PR3 tree, seed 3 resumed (m10, m11).
