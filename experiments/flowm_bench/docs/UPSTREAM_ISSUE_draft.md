# Draft upstream issue v2 (NOT posted; needs user approval)

Title: Proposed series of PRs: evaluation hygiene, lock-then-refine node velocities, second-order node state

Hi Denzel,

I have been building on kinetic-world-models toward a translation-boost equivariant world model with acceleration. Before opening pull requests I would like to check that the direction fits the repo. Everything is on my fork (`Yixun-Hu/kinetic-world-models`) as stacked branches; defaults are unchanged and the existing `--node_update lk` path is bit-identical to main@302d8f8 in forward values and gradients. Each PR would come with a per-seed table on a fresh test seed, and PR2 depends on PR1.

Planned PRs, smallest first:

1. **Evaluation hygiene.** `kwm.eval --test_seed`: eval currently defaults to the validation cache that `kwm.train` uses to select `best.pt`, so a separate seed would make the tables unbiased (I have not measured how much it changes anything; I can quote best-vs-last on a fresh seed for an existing run if useful). Also a self-describing validation cache (seed and config verified on load), source hashes in checkpoints and result files, per-sequence errors, and a small report script (paired t-test, exact sign-flip, bootstrap, Holm). Tell me if you would rather not have the report script upstream.
2. **`--node_update lock`: lock-then-refine node velocity update.** Unlocked nodes run a local dense search (±1 px, 0.5 px grid, Gaussian-aggregated SSD, parabolic sub-pixel refinement) on a per-sample observation snapshot; locked nodes use a per-node-normalised variant of your LK step; two-mode pooling with a heavier-mode rule handles two digits in one velocity cell. Same parameter set and checkpoint keys; lock uses a compact separate recurrent state. No-learning evidence with a randomly initialised encoder on 3 × 100 sequences: matched median node velocity error 0.082 → 0.024 px/frame, misses (> 0.1 px/frame) 42% → 18%. A one-seed end-to-end pilot is running; five seeds are planned before the PR, with checkpoint-averaged numbers on the test seed.
3. **`--order 2`:** a gated constant-acceleration (trend) estimate per node on top of the unchanged order-1 update, with ½ατ² transport, plus an `--accel` option in `env.py` for constant per-digit acceleration with the realised law documented.

Separately I will file a small issue with a reproduction for a pre-existing behaviour: `--node_update lk --grad_ckpt --node_update_pred off` gives gradients that differ from the non-checkpointed run (0.4–3% relative in my tests, no error raised), because the forward's `finally` resets the update flag before the backward recomputation. It affects only that experimental flag, which REPRODUCE.md says is not used in the published results.

Code is written with AI assistance (OpenAI Codex for implementation, Anthropic Claude for review) under my direction and review; I will say so in each PR. Happy to adjust scope, naming or split differently if you prefer.

Yixun Hu
