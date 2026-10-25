# Draft upstream issue (NOT posted; needs user approval)

Title: Proposed series of PRs: lock-then-refine node velocities, second-order node state, frozen prediction, evaluation hygiene

Hi Denzel,

I have been building on kinetic-world-models toward a translation-boost equivariant world model with acceleration, with the eventual goal of the full Galilean group. Before opening pull requests I would like to check that the direction fits the repo. Everything is on my fork (`Yixun-Hu/kinetic-world-models`) as stacked branches, defaults unchanged, lk path bit-identical to main@302d8f8.

Planned PRs, smallest first:

1. Evaluation hygiene: `kwm.eval --test_seed` (fresh generator seed; `best.pt` is currently selected on the same 300-sequence set that eval scores), source hashes in checkpoints and eval jsons, a report script with per-seed tables and paired tests. Also a separate issue for a pre-existing bug: `--node_update lk --grad_ckpt --node_update_pred off` gives gradients off by about 3% because the forward's `finally` resets the update flag before recomputation.
2. `--node_update lock`: lock-then-refine node velocity update. Unlocked nodes run a local dense search (±1 px, 0.5 px grid, Gaussian-aggregated SSD, parabolic sub-pixel) on a per-sample observation snapshot; locked nodes use your LK step; two-mode pooling with a heavier-mode rule handles two digits in one cell. No new parameters, same state keys. No-learning evidence on 1500 digits across 3 seeds: matched median node error 0.082 → 0.024 px/frame, misses 42% → 18%. End-to-end training comparison on 5 seeds in progress; I will attach it to the PR.
3. `--order 2`: second-order node state (ν, α) with an α-β-γ filter over the existing displacement measurements and ½ατ² transport, plus an `--accel` option in `env.py` for constant per-digit acceleration.
4. `--dec_input frozen`: freeze-and-render prediction (frozen memory transported analytically by ντ + ½ατ², no autoregressive feedback).

Code is written with AI assistance (OpenAI Codex for implementation, Anthropic Claude for review) under my direction and review; I will say so in each PR. Happy to adjust scope, naming or split differently if you prefer.

Yixun Hu
