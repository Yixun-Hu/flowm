# H1 campaign progress (updated 2026-11-04, 22:45 EST; five adaptive seeds complete)

Protocol: Kinetic MNIST World, real velocities and camera; every arm trained with the same seed, schedule and batch
(`--bs 16 --steps 20000`, adaptive arms with `--grad_ckpt`); scored as the **mean over checkpoints 16k–20k on 1000
sequences from held-out seed 20261025**. Raw JSON with per-sequence arrays under `kwm-wt-pr3/results/<run>/`.
Seed 0 adaptive arms are the pilot runs (`pilot_lk`, `pilot_lock`); later seeds are `h1_<arm>_s<seed>`.

## Per-seed table (MSE@20 / MSE@150)

| Arm | Prediction input | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| FloWM (KWM-sized) | native | 0.0683 / 0.1584 | 0.0677 / 0.1750 | 0.0682 / 0.2024 | – | – | 0.0681 / 0.1786 (3 seeds) |
| FloWM | zero | 0.0811 / 0.1581 | 0.0726 / 0.1325 | 0.0853 / 0.1336 | – | – | 0.0797 / 0.1414 (3 seeds) |
| lk (KWM adaptive, published) | flowm | 0.0288 / 0.1144 | 0.0293 / 0.1463 | 0.0269 / 0.1096 | 0.0283 / 0.1724 | 0.0284 / 0.1061 | 0.0284 / 0.1298 |
| lk | zero | 0.0262 / 0.0892 | 0.0262 / 0.0995 | 0.0245 / 0.0864 | 0.0257 / 0.0877 | 0.0256 / 0.0893 | 0.0256 / 0.0904 |
| lock (PR1) | flowm | 0.0218 / 0.0888 | 0.0217 / 0.0918 | 0.0222 / 0.0879 | 0.0219 / 0.0851 | 0.0226 / 0.1055 | 0.0220 / 0.0918 |
| lock | zero | 0.0200 / 0.0700 | 0.0196 / 0.0692 | 0.0202 / 0.0823 | 0.0198 / 0.0699 | 0.0202 / 0.0855 | **0.0200 / 0.0754** |
| lock | zero, freeze after 30 | 0.0200 / 0.0675 | 0.0196 / 0.0709 | 0.0202 / 0.0733 | 0.0198 / 0.0720 | 0.0202 / 0.0728 | 0.0200 / 0.0713 |

All planned runs are complete: FloWM 3 seeds, adaptive arms 5 seeds (the standing rule for primary comparisons).
Seed 3 was resumed from its last checkpoint after the tmux session was killed on 2026-10-31 (resume boundary recorded
in the checkpoint provenance). Seed 4 trained 2026-11-03 09:26 → 2026-11-04 18:42 alongside the 3D data rendering
on the same GPU. Seed 4's late checkpoints are the weakest of the five at 150 frames for both arms (lk zero
0.088, 0.083, 0.092, 0.090, 0.093; lock zero 0.082, 0.074, 0.096, 0.086, 0.090), like seed 2's lock run.

## Seed-level paired comparisons (5 seeds; paired t-test df = 4)

| Comparison | @20: mean, per seed, p | @150: mean, per seed, p |
|---|---|---|
| lk → lock, flowm input | −22.2% [−24.3, −26.1, −17.6, −22.7, −20.4], p < 0.001 | −26.1% [−22.4, −37.2, −19.8, −50.6, −0.6], p = 0.065 |
| lk → lock, zero input | −22.1% [−23.4, −25.3, −17.9, −22.7, −21.1], p < 0.001 | −16.2% [−21.5, −30.4, −4.8, −20.2, −4.2], p = 0.040 |
| lk (published default) → lock zero | −29.5% [−30.3, −33.3, −25.1, −30.0, −28.9], p < 0.001 | −39.1% [−38.8, −52.7, −24.9, −59.4, −19.4], p = 0.025 |
| flowm → zero input, lock | −9.4% [−8.0, −9.7, −9.1, −9.4, −10.7], p < 0.001 | −17.8% [−21.2, −24.6, −6.4, −17.8, −18.9], p = 0.005 |
| zero → zero-then-freeze 30, lock | 0.0% (identical by construction) | −4.8% [−3.6, +2.5, −10.9, +3.0, −14.9], p = 0.24 |

FloWM (3 seeds) → lock zero (5 seeds), unpaired means: −70.6% at 20 frames, −57.8% at 150 frames.
Statistics script: scratchpad `campaign/h1_stats5.py` (reads the held-out JSONs under `kwm-wt-pr3/results/`).

## Reading (five seeds, final)

- **lock over lk** holds on all five seeds at 20 frames under both inputs (−18 to −26 % per seed, p < 0.001). At
  150 frames the direction holds on 5/5 seeds but the size varies: −16 % under zero input (p = 0.040; does not
  survive a Holm correction over the four primary comparisons) and −26 % under feedback (p = 0.065), where seed 4
  adds a near-zero difference (−0.6 %) and lk's feedback runs carry late-checkpoint spikes (seeds 1, 3). The
  honest headline for PR1 is the native-input result: −22 % at 20 frames (5/5, p = 2e-4), −26 % at 150 frames
  (5/5, p = 0.065, not significant).
- **Zero prediction input over feedback** holds on every arm and seed (lock: −9 % at 20 frames, −18 % at 150
  frames, p = 0.005). It is a PR3 option; under zero input the node update is inactive during prediction for both
  rules (predon and predoff give identical per-sequence MSE), so the combined "lk → lock + zero" figure mixes two
  changes and is not reported as PR1's effect.
- **Zero-then-freeze 30**: sign flips (−3.6, +2.5, −10.9, +3.0, −14.9 %), p = 0.24. Not a claim; stays an option.
- **Cost**: lock takes 1.19–1.29 × lk's time per step (median 4.47 vs 5.74 s/step at batch 16 with gradient
  checkpointing on the shared A6000) and 19 % less peak memory (6.4 vs 5.2 GB). "Identical optimization budget",
  not identical compute.
- **Denzel's published numbers reproduce**: lk 0.0284 / 0.130 and FloWM 0.0681 / 0.179 bracket his one-seed
  0.027 / 0.089 and 0.067 / 0.145 (selection-set best checkpoints flatter the 150-frame column).
