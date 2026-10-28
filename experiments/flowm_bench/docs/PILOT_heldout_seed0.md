# Pilot (seed 0) held-out evaluation, 2026-10-28

Kinetic MNIST World, real velocities and camera, 30 observed frames. Both arms trained identically
(`--model adaptive --bs 16 --grad_ckpt --steps 20000 --seed 0`, KWM pr1 branch). Evaluation: PR3 worktree
(`kwm.eval`, 1000 held-out sequences, `--test_seed 20261025`), **mean over checkpoints 16k–20k** as pre-registered
in plan v4. Selection seed 12345 agrees to the third decimal (not shown). Raw JSON:
`kwm-wt-pr3/results/pilot_{lk,lock}/eval_seed20261025_ckpt_step*_<tag>.json` (per-sequence arrays included).

| Arm | Prediction input | MSE@20 | MSE@150 |
|---|---|---:|---:|
| lk (KWM adaptive as published) | flowm (predictions fed back) | 0.0288 | 0.1144 |
| lk | zero | 0.0262 | 0.0892 |
| lk | zero, then freeze after 30 | 0.0262 | 0.0896 |
| lock (PR1) | flowm | 0.0218 | 0.0888 |
| lock | zero | 0.0200 | 0.0700 |
| lock | zero, then freeze after 30 | 0.0200 | **0.0675** |

Paired over the 1000 sequences (each sequence averaged over the five checkpoints), 95% bootstrap intervals, 5000 replicates:

| Comparison | Δ MSE@20 | Δ MSE@150 | sequences improved @20 / @150 |
|---|---|---|---|
| lk → lock, flowm input | −24.3% [−0.0076, −0.0064] | −22.4% [−0.0270, −0.0243] | 0.82 / 0.89 |
| lk → lock, zero input | −23.4% [−0.0067, −0.0056] | −21.5% [−0.0198, −0.0185] | 0.79 / 0.97 |
| flowm → zero input, lock | −8.0% | −21.2% | 0.93 / 0.96 |
| flowm → zero input, lk | −9.0% | −22.1% | 0.94 / 0.99 |
| zero → zero-then-freeze 30, lock | 0.0% | −3.6% [−0.0029, −0.0021] | – / 0.68 |
| lk flowm → lock zero-then-freeze 30 (best vs published) | −30.3% | −41.0% | 0.86 / 0.99 |

Node updates during prediction (`--node_update_pred on/off`) make no difference under zero input (identical to
four decimals) and a small difference under flowm input (lock @150 0.0888 on vs 0.0827 off).

## Observations

- **Checkpoint 19k spike is a prediction-feedback instability, not a bad checkpoint.** With flowm input the
  19k checkpoint gives MSE@150 0.211 (lk) and 0.152 (lock) on the held-out set (1.93 for lock on the 300-sequence
  selection set); with zero input the same checkpoint gives 0.095 and 0.063. Feeding predictions back amplifies
  errors over the 150-frame horizon; zero input removes the spike on both arms.
- Zero input helps both arms by about the same amount; lock's gain over lk is therefore additive with it
  (≈ −22% at both horizons, both inputs).
- Zero-then-freeze adds a small but consistent 3.6% at 150 frames on lock, nothing at 20 frames; on lk it is neutral.
- Single seed. The five-seed H1 table (seeds 0–4, same protocol) is the pre-registered primary evidence;
  seed 1 is training (2026-10-28, step ~10k lk / ~8k lock).

Reference points (KWM RESULTS §4, Denzel's one-seed runs, selection set, best checkpoint): FloWM 0.067 / 0.145,
adaptive 0.027 / 0.089, oracle nodes 0.006 / 0.043.
