# PR1 text for the user's approval (2026-11-05). Not posted. Target: Denzel Zheng's kinetic-world-models, from
# branch `pr1-submit` (single commit, diff identical to `pr1-lock-nodes` at 161a568) on Yixun-Hu/kinetic-world-models.

## Title

Add opt-in lock-then-refine node velocity update for the adaptive model (`--node_update lock`)

## Description

### What
Adds `--node_update lock`, an opt-in alternative to the adaptive model's Lucas–Kanade (LK) node-velocity update.
The default `lk` is unchanged: forward values and gradients are bit-identical to `main` (see Validation).
`kwm.eval --node_update {lk,lock}` can override the rule saved in a checkpoint; older checkpoints load as `lk`.
Lock is accepted only for the adaptive model.

### Why
In five-digit scenes the LK running mean often settles on the wrong velocity when a node's residual is large or
two digits share a velocity cell. Lock-then-refine uses LK only where the current node velocity already explains
the observation, searches locally elsewhere, and pools the evidence into the heavier of two motion modes.

### How
- Each step, each node checks whether its LK residual is trustworthy ("locked": enough coherent textured evidence
  and a residual velocity ≤ 0.3 px/frame).
- Locked nodes use LK; unlocked nodes use a dense ±1 px SSD search with sub-pixel refinement against a per-sample
  observation snapshot.
- Per-pixel residuals are pooled with the existing gates into up to two modes; the node takes the heavier one.
- The offset update is the existing evidence-weighted running mean with the learned scale `n_s`; no parameters
  are added and checkpoint weight keys are unchanged.
- Gradients pass only through the locked LK displacement; search and mode selection are detached.
- Node state is an explicit fp32 `LockState`: features, `[B,V,4]` node metadata, one `[B,K,32,32]` snapshot
  (33 kB per sample at the defaults).

Method, thresholds and gradient routing: `docs/LOCK_NODES.md`.

### How it was evaluated
- Kinetic MNIST World with real velocities and camera. Both rules trained with identical settings: 20k steps,
  batch 16, `--grad_ckpt`, the same seed, initialisation and data stream per pair, five seeds.
- Scoring: mean over checkpoints 16k–20k (no checkpoint selection) on 1000 held-out sequences (seed 20261025,
  MNIST test digits), with the repository's default prediction protocol (model feedback, node updates on).
- Statistics: two-sided paired t-test over seeds (df = 4); percentages are the mean of per-seed relative changes.

| | LK | Lock | Change | Seeds improved | Paired t |
|---|---:|---:|---:|---:|---:|
| MSE@20 | 0.0284 | 0.0220 | −22 % | 5/5 | p = 2e-4 |
| MSE@150 | 0.1298 | 0.0918 | −26 % (per seed −0.6 % to −51 %) | 5/5 | p = 0.065 |

The 20-frame gain is consistent across seeds (−18 % to −26 %). The 150-frame gain has the same sign on every
seed but is not significant with five seeds; part of the spread is LK's prediction-feedback instability
(late-checkpoint spikes up to 0.25 on one seed).

Supplementary, using an evaluation-time zero prediction input from a separate branch (not in this PR): MSE@20
−22 % (p = 1e-4), MSE@150 −16 % (0.0904 → 0.0754, 5/5, p = 0.040; not significant after a Holm correction over
the four comparisons). Under zero input the node update has no effect during prediction for either rule.

Cost: same optimisation budget, but lock takes 1.2–1.3 × longer per step on a shared A6000 (4.47 → 5.74 s/step
at batch 16 with `--grad_ckpt`) and uses 19 % less peak memory (6.3 → 5.1 GiB).

### Validation
- `pytest -q`: 106 passed, 2 slow skipped (28 s); `KWM_SLOW=1`: 108 passed (55 s).
- LK forward values and gradients are bit-identical to `main@302d8f8` over three training steps in four
  configurations (base, `--grad_ckpt`, `--node_update_pred off`, `--dec_input zero`), with and without an explicit
  `--node_update lk`, and against a frozen reference cell in the test suite.
- Lock results with and without `--grad_ckpt` are bit-identical.

### Limitations
- Search radius, support rejection, the 0.6 px/frame component gate and the offset clip limit the motion that can
  be recovered per step; the heavier-mode rule can discard a smaller object sharing a velocity cell.
- With feedback input and node updates on, lock nodes drift more on generated frames than LK nodes (0.18 vs
  0.04 px/frame of velocity change over 150 prediction steps in one probe). `--node_update_pred off` scores
  slightly better for lock at 150 frames (−30 % vs −26 % against LK); the default stays on.
- Training uses bf16/TF32 on GPU and evaluation runs in fp32; the discrete gates can amplify small differences.
- Only Kinetic MNIST World with real velocities and camera was evaluated; integer-velocity data, walls and
  odometry noise were not. The estimator rule was chosen in a separate no-learning study that is not part of
  this PR.
- Seed-0 checkpoints carry no provenance record; seeds 1–4 were trained on a later tree whose lock computation is
  bit-identical to this one; seed 3 of both arms was resumed from its last checkpoint.

### AI assistance
Most of the code, tests and documentation were written with AI coding assistants (OpenAI Codex and Anthropic
Claude), directed and reviewed by me, with an adversarial review pass by Claude. All experiments were run and
checked by me, and I take responsibility for the change.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
