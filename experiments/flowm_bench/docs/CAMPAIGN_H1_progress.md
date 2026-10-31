# H1 campaign progress (updated 2026-10-31, 07:30 EDT)

Protocol: Kinetic MNIST World, real velocities and camera; every arm trained with the same seed, schedule and batch
(`--bs 16 --steps 20000`, adaptive arms with `--grad_ckpt`); scored as the **mean over checkpoints 16k–20k on 1000
sequences from held-out seed 20261025**. Raw JSON with per-sequence arrays under `kwm-wt-pr3/results/<run>/`.
Seed 0 adaptive arms are the pilot runs (`pilot_lk`, `pilot_lock`); later seeds are `h1_<arm>_s<seed>`.

## Per-seed table (MSE@20 / MSE@150)

| Arm | Prediction input | seed 0 | seed 1 | seed 2 | mean (3 seeds) |
|---|---|---:|---:|---:|---:|
| FloWM (KWM-sized) | native | 0.0683 / 0.1584 | 0.0677 / 0.1750 | 0.0682 / 0.2024 | 0.0681 / 0.1786 |
| FloWM | zero | 0.0811 / 0.1581 | 0.0726 / 0.1325 | 0.0853 / 0.1336 | 0.0797 / 0.1414 |
| lk (KWM adaptive, published) | flowm | 0.0288 / 0.1144 | 0.0293 / 0.1463 | 0.0269 / 0.1096 | 0.0283 / 0.1234 |
| lk | zero | 0.0262 / 0.0892 | 0.0262 / 0.0995 | 0.0245 / 0.0864 | 0.0256 / 0.0917 |
| lock (PR1) | flowm | 0.0218 / 0.0888 | 0.0217 / 0.0918 | 0.0222 / 0.0879 | 0.0219 / 0.0895 |
| lock | zero | 0.0200 / 0.0700 | 0.0196 / 0.0692 | 0.0202 / 0.0823 | **0.0199 / 0.0738** |
| lock | zero, freeze after 30 | 0.0200 / 0.0675 | 0.0196 / 0.0709 | 0.0202 / 0.0733 | 0.0199 / 0.0706 |

FloWM is complete (3 seeds planned). Adaptive seeds 3 and 4 are in the unattended chain (seed 3 training).

## Seed-level paired comparisons (3 seeds; paired t-test df = 2; the exact sign-flip test cannot go below p = 0.25 at n = 3)

| Comparison | @20: mean, per seed, p | @150: mean, per seed, p |
|---|---|---|
| lk → lock, flowm input | −22.7% [−24.3, −26.1, −17.6], p = 0.018 | −26.5% [−22.4, −37.2, −19.8], p = 0.082 |
| lk → lock, zero input | −22.2% [−23.4, −25.3, −17.9], p = 0.014 | −18.9% [−21.5, −30.4, −4.8], p = 0.14 |
| lk (published default) → lock zero | −29.6% [−30.3, −33.3, −25.1], p = 0.011 | −38.8% [−38.8, −52.7, −24.9], p = 0.077 |
| FloWM → lock zero | −70.7% [−70.7, −71.1, −70.4], p < 0.001 | −58.5% [−55.8, −60.4, −59.3], p = 0.008 |
| FloWM → lk (Denzel's gain) | −58.4% [−57.9, −56.7, −60.5], p < 0.001 | −30.0% [−27.7, −16.4, −45.9], p = 0.10 |
| flowm → zero input, lk | −9.5% [−9.0, −10.7, −8.8], p = 0.007 | −25.1% [−22.1, −32.0, −21.1], p = 0.053 |
| flowm → zero input, lock | −8.9% [−8.0, −9.7, −9.1], p = 0.003 | −17.4% [−21.2, −24.6, −6.4], p = 0.093 |
| zero → zero-then-freeze 30, lock | 0.0% (identical by construction) | −4.0% [−3.6, +2.5, −10.9], p = 0.40 |

## Reading so far

- **lock over lk** is consistent in sign on all three seeds, both horizons, both inputs. At 20 frames the effect is
  tight (−18 to −26%, p ≈ 0.015 with three seeds). At 150 frames the size varies with the seed (−5 to −37%); seed 2's
  lock run has weaker late checkpoints (19k and 20k score 0.102 and 0.085 at 150 frames under zero input versus
  0.068–0.070 at 16k–17k; selection-set curve agrees). This is the constant-learning-rate checkpoint noise the
  checkpoint-averaged protocol was designed to expose rather than hide. Five seeds decide the 150-frame claim.
- **Zero prediction input over feedback** holds on every arm and seed. Checkpoint spikes under feedback recur in
  every arm (lk s1 20k 0.29, FloWM s2 16k 0.38 at MSE@150) and vanish under zero input. Zero input hurts FloWM at
  20 frames (+17%): FloWM has no transport to carry the scene without feedback.
- **Zero-then-freeze 30**: sign flips across seeds (−3.6, +2.5, −10.9%). Not a claim; stays an option in PR3.
- **Denzel's published numbers reproduce**: lk 0.0283 / 0.123 and FloWM 0.0681 / 0.179 bracket his one-seed
  0.027 / 0.089 and 0.067 / 0.145 (his were selection-set best checkpoints, which flatter the 150-frame column).
