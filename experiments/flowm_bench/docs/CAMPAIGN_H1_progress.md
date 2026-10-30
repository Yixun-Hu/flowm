# H1 campaign progress (updated 2026-10-29, 23:45 EDT)

Protocol: Kinetic MNIST World, real velocities and camera; every arm trained with the same seed, schedule and batch
(`--bs 16 --steps 20000`, adaptive arms with `--grad_ckpt`); scored as the **mean over checkpoints 16k–20k on 1000
sequences from held-out seed 20261025**. Raw JSON with per-sequence arrays under `kwm-wt-pr3/results/<run>/`.
Seed 0 adaptive arms are the pilot runs (`pilot_lk`, `pilot_lock`); later seeds are `h1_<arm>_s<seed>`.

## Per-seed table (MSE@20 / MSE@150)

| Arm | Prediction input | seed 0 | seed 1 | seed 2 | mean so far |
|---|---|---:|---:|---:|---:|
| FloWM (KWM-sized) | native | 0.0683 / 0.1584 | 0.0677 / 0.1750 | 0.0682 / 0.2024 | 0.0681 / 0.1786 |
| FloWM | zero | 0.0811 / 0.1581 | 0.0726 / 0.1325 | 0.0853 / 0.1336 | 0.0797 / 0.1414 |
| lk (KWM adaptive, published) | flowm | 0.0288 / 0.1144 | 0.0293 / 0.1463 | training | 0.0291 / 0.1304 |
| lk | zero | 0.0262 / 0.0892 | 0.0262 / 0.0995 | | 0.0262 / 0.0943 |
| lock (PR1) | flowm | 0.0218 / 0.0888 | 0.0217 / 0.0918 | | 0.0217 / 0.0903 |
| lock | zero | 0.0200 / 0.0700 | 0.0196 / 0.0692 | | **0.0198 / 0.0696** |
| lock | zero, freeze after 30 | 0.0200 / 0.0675 | 0.0196 / 0.0709 | | 0.0198 / 0.0692 |

FloWM's three seeds are complete. Adaptive seeds 2–4 are in the unattended chain.

## Paired deltas within seed (sequence-paired, 1000 sequences)

| Comparison | @20 seed 0 / seed 1 | @150 seed 0 / seed 1 |
|---|---|---|
| lk → lock, flowm input | −24.3% / −26.1% | −22.4% / −37.2% |
| lk → lock, zero input | −23.4% / −25.3% | −21.5% / −30.4% |
| FloWM → lock zero | −70.7% / −71.1% | −55.8% / −60.4% |
| flowm → zero input, lk | −9.0% / −10.7% | −22.1% / −32.0% |
| flowm → zero input, lock | −8.0% / −9.7% | −21.2% / −24.6% |
| zero → zero-then-freeze 30, lock | 0.0% / 0.0% | −3.6% / **+2.5%** |

## Reading so far

- lock over lk: consistent in sign and size across both seeds, at both horizons and under both inputs.
- Zero prediction input over feedback: consistent for lk, lock and (at 150 frames) FloWM. The checkpoint spikes
  under feedback recur in every arm and seed (lk s1 20k 0.29, FloWM s2 16k 0.38 at MSE@150) and vanish under zero
  input. Zero input hurts FloWM at 20 frames (+17%): FloWM has no transport to carry the scene without feedback.
- Zero-then-freeze 30: sign flips between seeds (−3.6%, +2.5%). Not a claim; stays an option in PR3.
- Seed-level statistics (paired t-test, sign-flip) start at three seeds; the pre-registered primary is five.
