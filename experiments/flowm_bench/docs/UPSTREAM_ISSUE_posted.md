Hi Denzel,

I have been building on kinetic-world-models toward a translation-boost equivariant world model, and before opening pull requests I would like to check that the direction fits the repo. Everything is on my fork (`Yixun-Hu/kinetic-world-models`) as stacked branches. Defaults are unchanged, and the existing `--node_update lk` path is bit-identical to `main@302d8f8` in forward values and gradients. Each PR comes with per-seed tables on a fresh test seed.

Planned PRs, smallest first:

1. **Evaluation hygiene** (PR opened alongside this issue). `kwm.eval --test_seed`: eval currently defaults to the validation cache that `kwm.train` uses to select `best.pt`, so a separate seed makes the tables an independent estimate. Also a self-describing validation cache (seed and config verified on load, legacy caches still load with a warning), source hashes in checkpoints and result files, per-sequence errors, and a small report script (paired t-test, exact sign-flip, bootstrap, Holm). Tell me if you would rather not have the report script upstream.

2. **`--node_update lock`: lock-then-refine node velocity update.** Unlocked nodes run a local dense search (±1 px, 0.5 px grid, Gaussian-aggregated SSD, parabolic sub-pixel refinement) on a per-sample observation snapshot; locked nodes use a per-node-normalised variant of your LK step; two-mode pooling with a heavier-mode rule handles two digits in one velocity cell. Same parameter set and checkpoint keys; lock keeps a compact separate recurrent state. Evidence so far:
   - No learning, randomly initialised encoder, 3 × 100 sequences: matched median node velocity error 0.082 → 0.024 px/frame, misses (> 0.1 px/frame) 42% → 18%.
   - One training seed, K-real (real velocities and camera), your adaptive configuration, 20k steps, scored as the mean over checkpoints 16k–20k on 1000 sequences from a held-out seed: lk 0.0288 / 0.1144 → lock 0.0218 / 0.0888 (MSE@20 / MSE@150), about −22% at both horizons, paired over sequences with 95% bootstrap intervals clear of zero. A KWM-sized FloWM trained identically scores 0.0683 / 0.1584 on the same set, and your published lk and FloWM numbers reproduce within checkpoint noise.
   - Five seeds are running before the PR; I will only open it with the five-seed table.

3. **Prediction-phase input.** Evaluation-time options for what the memory sees during prediction. Feeding predictions back (the current behaviour) is unstable at some checkpoints over 150 frames (one checkpoint scores 0.21 instead of 0.09); a zero input removes the instability and improves both lk and lock by roughly 9% at 20 frames and 22% at 150 on the seed above. A freeze-and-render variant is a documented negative result. Nothing changes unless the flag is set.

4. **Acceleration**, re-scoped after a negative result. An `--accel` option in `env.py` (constant per-digit acceleration, realised law documented) and an opt-in second-order node estimator with a variance gate. In five-digit scenes the per-step velocity measurement noise (≈ 0.14 px/frame) is an order of magnitude above the acceleration signal, so the gate never opens usefully; on a single clean digit it detects acceleration 94–100% of the time with no false positives. I would propose this as environment plus harness plus the negative result, unless you prefer it kept on the fork.

Separately I will file a small issue with a reproduction for a pre-existing behaviour: `--node_update lk --grad_ckpt --node_update_pred off` gives gradients that differ from the non-checkpointed run (0.4–3% relative in my tests, no error raised), because the forward's `finally` resets the update flag before the backward recomputation. It affects only that experimental flag, which REPRODUCE.md says the published results do not use.

Code is written with AI assistance (OpenAI Codex for implementation, Anthropic Claude for review and these write-ups) under my direction and review; each PR says so. Happy to adjust scope, naming, or the split.

Yixun Hu
