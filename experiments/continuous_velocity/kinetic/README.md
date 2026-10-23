# Kinetic extension: spectral transport, sub-pixel refinement, and second-order motion

This package extends the continuous-velocity model of `../model.py` (the
"velocity density" estimator: a soft attention density over integer
displacements, a robust weighted constant-velocity fit, a fixed canonical
memory and analytic transport) with three independent, switchable changes.
Two come from [kinetic-world-models](https://github.com/denzelzheng/kinetic-world-models)
(KWM); the third adds acceleration. Nothing in the parent study is modified;
the parent's hash guards and completed artifacts remain valid.

| Switch | Values | Source |
|---|---|---|
| `align` | `bilinear` (parent) / `spectral` | KWM §2, Fourier-phase shift of the zero-padded world field |
| `render` | `bilinear` (parent) / `spectral` | KWM §2, phase shift of the canonical image, then crop |
| `refine` | `none` (parent) / `newton` | KWM `phase_correlation` step 3, adapted to the masked similarity |
| `order` | `1` (parent) / `2` | new: joint velocity + acceleration fit with an evidence gate |

`KineticMemory(align='bilinear', render='bilinear', refine='none', order=1)`
reproduces `ContinuousMemory('physical')` exactly (same parameters, same
outputs; tested bit-for-bit). Formal model inputs are unchanged: context
images, integer camera origins, timestamps, future camera origins and target
times. Velocity, acceleration, positions and templates never enter the
forward path except through explicitly labeled oracle overrides.

## Method

### Spectral alignment and rendering

For a periodic map `F` and a real displacement `d`, `spectral_shift(F, d)` is
`irfft2(rfft2(F) · exp(-2πi k·d))`, moving content from `x` to `x+d`; integer
`d` equals `torch.roll`. Nyquist bins use the symmetric cosine convention so the
interpolant is real. The parent reads each context frame at `u + (t_s − t_0) v`
by periodic bilinear sampling, which blurs every observation once before
fusion and the canonical image once more at render time. Spectral alignment
shifts the world field by `−offset_s` instead. Because a hard camera window
rings under a fractional phase shift, the field is multiplied by a 3-pixel
raised-cosine taper before scattering (matching still uses the untapered
field, as the parent). The coverage support `μ_s` is the *bilinear*-sampled
tapered mask: smooth, in `[0, 1]`, exactly zero outside the window. A
phase-shifted mask would ring over the whole zero-padded region and dividing
by it amplifies ringing by orders of magnitude (found in review), so aligned
values and support are zeroed where the support is below 0.05. Fusion keeps the
parent's normalized convolution, `M = Σ_s ρ_s A_s / Σ_s ρ_s μ_s`, with
`ρ_s = μ_s e^{h(A_s/μ_s)}`, so the taper is divided out where the frame is
visible. Spectral rendering shifts the canonical image by `offset − c` and
crops the 32×32 window.

Our data is windowed and bilinear- or nearest-rendered, not band-limited, so
no "exact transport" claim is made; which operator is better is the question
the experiment answers (a no-learning oracle check during plan review favoured
spectral alignment and rendering on both renderers).

### Newton refinement of each pair displacement

The parent's displacement for pair `k` is the softmax expectation of the masked
normalized similarity `S_k(δ) = 2N_k(δ)/D_k(δ)` on the integer lattice. Both
`N_k` and `D_k` are available as rfft2 spectra, so `S_k` has a trigonometric
interpolant with analytic gradient and Hessian (quotient rule). Three damped
Newton steps (±0.5 px each) start at the expectation; the Hessian is detached
and used as a preconditioner, gradients flow through the Newton residual, the
spectra and the start. The refined point is accepted only if the Hessian is
negative definite with a relative margin, the point moved at most 1 px, the
denominator kept at least half its initial value, and the similarity improved
without exceeding 1; otherwise the expectation is kept. Confidence features
(match, entropy, support) are unchanged.

### Second-order motion

Relative to the last context time `t_0`, `p(t_0+τ) = p(t_0) + v τ + ½ a τ²`.
A pair `(i, j)` then satisfies `d_k = v Δt_k + a s_k` with
`s_k = ½[(t_j − t_0)² − (t_i − t_0)²]`, linear in `(v, a)`. The weighted
least-squares normal equations are 2×2 and shared across the two axes. Two
Cauchy reweighting steps (scale 0.15 px/frame, residual `(d_k − vΔt_k − a s_k)/Δt_k`)
use the unshrunk solution. The acceleration is then gated per axis by its own
standard error from the scale-invariant sandwich estimate
`Var(a) = σ² [A⁻¹ B A⁻¹]₂₂ · n_eff/(n_eff − 2)`, `σ² = Σ w r²/Σ w`,
`B = Σ w² x xᵀ`, `n_eff = (Σ w)²/Σ w²` (the lag-1/2/4 pairs share frames, so
the nominal 36 observations overstate the evidence):
`g = clip(1 − κ Var(a)/a², 0, 1)`, `κ = 9` (a 3σ rule, prespecified; `κ = 4`
and `κ = 0` are ablations), and the velocity is re-solved with the gated
acceleration fixed. A (nearly) singular normal matrix falls back to the
order-1 fit exactly. The gate does not depend on the overall scale of the
pair weights, which the image loss leaves unanchored; the rate at which the
gate opens on zero-acceleration validation sequences is logged during training
as a null calibration. Alignment and future transport use `v τ + ½ a τ²`.

Two inference-time interventions isolate what the acceleration buys:
`zero_acceleration='both'` replaces the motion by the constant-velocity profile
of the same robust normal equations (alignment and transport), and
`'transport'` keeps the second-order alignment and `v(t_0)` but drops the
future curvature. The learned-offset controls (`relaxed`, `offset`) receive a
15-dimensional summary (per lag group: weighted mean velocity of the earlier
and of the later six pairs, and the mean weight), which resolves curvature;
the parent's 9-dimensional summary averages each lag group over the same mean
midpoint and carries no curvature information.

The admissible-displacement bound `2.5Δt + 1` and the metrics' continuity bound
assume `|v(t)| ≤ 2.5`; all accelerated data keep `|v(t)| ≤ 2.4` over the whole
sequence.

### Data

`KineticTrainingData(motion='accelerated')` draws digits, camera episodes,
starts and initial velocities exactly as the paper sampler (same RNG order) and
the acceleration from a separate generator: zero with probability 0.3,
otherwise a signed log-uniform magnitude in `[2e-4, 2e-2]`, rejection-sampled
to respect the speed bound over the 90-frame training sequence. Logged
`true_velocity` is the velocity at the last context time. `make_kinetic_case`
re-renders a paper test scene (same digit, start, camera path, template) along
the accelerated law and provides `true_offsets` (unwrapped displacement from the
last context frame), valid for any motion.

## Protocol (`campaign.py`, tag `kinetic_v1`)

Group A trains on the paper's constant-velocity stream (bilinear rendering,
horizontal motion): `base`, `alignS`, `renderS`, `spectral`, `newton`,
`spectral_newton`, `order2` (κ=9), `order2_k4`, `order2_k0`. Group B trains on
the accelerated stream: `base` (order 1), `order2`, `order2_k4`, `order2_k0`,
and the flexible controls `relaxed` and `offset`. Each variant selects its learning rate in
{3e-4, 1e-3} on seed 2026 by validation loss, then trains five fresh seeds
2027–2031 for 5,000 updates (batch 8, Adam, image-only loss). The Group A
transport/refinement package with the lowest mean validation loss is also
trained with order 2 in both groups (`combine`). Evaluation suites:

| Suite | Scenes | Content |
|---|---|---|
| `core` | 32 | paper core speeds, both renderers, horizon 40 |
| `time` | 16 | held-out speeds, horizon 160 |
| `accel` | 16 | held-out initial speeds × a ∈ {0, ±5e-4, ±1e-3, ±2e-3, ±4e-3, ±1.2e-2}, horizon 40 |
| `accel_long` | 16 | a ∈ {0, ±1e-3, ±4e-3}, horizon 80 |

Primary hypotheses, each with one metric (bilinear MSE at horizon 40; core
restricted to the held-out trio), paired across training seeds, Holm-corrected:
P1 `spectral` vs `base`; P2 `newton` vs `base`; P3 `order2` non-inferior to
`base` on constant velocity (margin 1e-4); P4 Group B `order2` vs `base` on
`accel`. The statistic is a paired t-test across the five training seeds on
per-model pooled scores; conditional scene-cluster bootstrap intervals are
descriptive. Oracle-motion (true `v(t_0)` and `a`) and zero-acceleration
diagnostics, in-place 20k-update extensions of the seed-2026 selection runs
(evaluated at both the 5k and the 20k budget), and a confirmatory evaluation on
unused test scenes 32–63 complete the campaign. Group B speeds are not held out.

## Running

```bash
cd /home/yixunhu/codespace/flowm
PY=/home/yixunhu/miniconda3/envs/flowm/bin/python
$PY experiments/continuous_velocity/kinetic/tests.py                  # CPU, ~5 s
for stage in select train combine evaluate extended confirm; do
  $PY experiments/continuous_velocity/kinetic/campaign.py --stage $stage --device cuda:1   # shardable: --shard i --shards n
done
$PY experiments/continuous_velocity/kinetic/report.py --root outputs/continuous_velocity/kinetic_v1
```

Outputs go to `outputs/continuous_velocity/kinetic_v1/` and checkpoints to
`downloaded_checkpoints/continuous_velocity/kinetic_v1/` (both ignored by git).

## Results

See `RESULTS.md` in this directory (written from `report.py` output after the
campaign completed).

## Known approximations (from review, documented rather than changed)

- Spectral alignment divides a spectrally shifted, tapered field by a
  bilinear-sampled taper. Inside the 3-pixel ramp the two differ (ratios of
  roughly 0.8–1.05 at quarter-pixel offsets; exactly 1 at integer offsets);
  the oracle-motion MSE difference against dividing by the shifted taper is
  below 1%. Isolated ramp pixels can reach raw values near 1.9 (fraction above
  1.5 below 1e-5).
- The acceleration gate's "3σ" is nominal. With the parent study's trained
  features, the empirical spread of the raw acceleration estimate is 1.3–1.8
  times the sandwich standard error, so the measured null gate-open rate on
  the x axis at κ=9 is about 1% (bilinear) and 5% (nearest), against 7% and
  24% at κ=4. Detection power at κ=9 on bilinear data: about 30% of gates open
  at |a|=1e-3, 70% at 2e-3, 90% at 4e-3, with shrinkage of the gated estimate
  of about −83%, −50% and −21%. The training log records the null gate-open
  rate on validation sequences with zero acceleration.
- The image-velocity truth is a least-squares slope over the frames the image
  tracker could use (visible and detected); the tracker itself fits a constant
  velocity to its detections, so a small floor remains for accelerated cases.

## Limits and non-claims

Fourier shifts, phase correlation and sub-pixel Newton refinement are standard;
the contribution is their controlled combination with the parent estimator and
the gated second-order fit. The model still estimates one rigid motion for the
whole scene, holds the memory fixed during forecasting, and does not address
the parent study's top open problem (renderer-robust estimation on nearest
rendering). Frozen-FloWM velocity densification with spectral transport (the
literal KWM × densification combination) was not implemented.
