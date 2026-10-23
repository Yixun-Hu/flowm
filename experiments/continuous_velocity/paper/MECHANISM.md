# Mechanism controls

`mechanism.py` adds controls without modifying the pilot. Put
`experiments/continuous_velocity` on the Python import path and import
`paper.mechanism`. Importing the repository's top-level `experiments` package
also imports unrelated training dependencies and is unnecessary.

## Model factory

`build_model(kind, channels=4)` returns a model with the pilot's forward
signature and `prediction, auxiliary` outputs:

| kind | Matching / velocity | Memory and rendering | Training |
| --- | --- | --- | --- |
| `physical` | Original learned matching + robust physical fit | Original learned fusion and residual decoder | Train normally |
| `offset`, `relaxed` | Original pilot controls | Original | Train normally |
| `raw_correlation` | Fixed raw-intensity correlation, fixed temperature/confidence, robust physical fit | Same trainable feature encoder, fusion and decoder as physical | Train normally; matching itself is not learned |
| `mlp_velocity` | Same learned matching observations; pairwise MLP and confidence-weighted pooling regress velocity | Same memory and decoder as physical | Train normally |
| `mean_memory` | Original learned matching + physical fit | Visibility-weighted mean replaces learned fusion; same decoder | Train normally |
| `raw_untrained` | Fixed raw-intensity correlation + robust physical fit | Visibility mean of raw appearance; identity residual decoder | No parameters and no training |

`raw_untrained` is template-free and wholly nonlearned. Its memory has one raw
intensity channel. It uses the same alignment, coverage normalization and
periodic sampling operations as the learned models, but it cannot retain their
learned feature channels or decoder. Do not describe `raw_correlation` as a
wholly nonlearned model: only its matching and velocity fitting are nonlearned.

The MLP receives 36 observed pair correspondences: two displacement coordinates,
elapsed time, correspondence weight, and three confidence statistics. It uses
the same learned descriptor network and candidate displacement attention as the
physical model. No fitted velocity, displacement/time ratio, GT velocity, GT
position, or digit template is supplied to the MLP. It must learn the temporal
scaling; short-budget underperformance alone does not demonstrate inferior
expressivity. Pair weights also remain trainable through the correspondence
model. Additional MLP parameters must be included in parameter/runtime reports.

## Fixed-weight interventions

`mean_memory_copy(model)` replaces fusion by a visibility-weighted mean while
preserving all other learned parameters. `raw_matching_copy(model)` replaces
matching with fixed raw correlation while preserving learned memory features,
fusion and decoder. These complement retrained controls: inference-only swaps
may introduce distribution shifts, whereas retrained controls can adapt.

`VelocityIntervention(model, grid_spacing=None, scope='both')` evaluates one
loaded physical model with the same input signature. Use `grid_spacing=1.` or
`.5` for integer or half-integer quantization. With no override, velocity is
estimated from context. For a separately labeled oracle diagnostic, pass
`velocity_override=true_velocity`, shape `[batch, 2]`.

- `scope='both'`: quantize velocity for both context alignment and future
  transport. This measures the combined architectural impact of quantization.
- `scope='transport'`: build canonical memory with the unquantized selected
  velocity, then quantize only future transport. This isolates forecast
  quantization while holding canonical memory fixed. In an oracle run, memory
  alignment also uses the true continuous velocity.

The auxiliary output includes `estimated_velocity`, used `velocity`,
`alignment_velocity`, `oracle_velocity_used`, `grid_spacing` and
`intervention_scope`. Rounding uses PyTorch ties-to-even; there is no range
clipping. At exactly 0.5 and 1.5, integer rounding therefore picks 0 and 2. This
is a single-trajectory discretization diagnosis, not a simulation of FloWM's
multiple velocity hypotheses or channel-wise max readout.

## Validity limits inherited from the pilot

Matching searches pixel displacement on a 50x50 periodic lattice and uses a
soft expectation for subpixel displacement. The velocity representation is
continuous even though observation pixels and displacement candidates are
discrete. Search is bounded by `2.5 * elapsed_time + 1` per axis. For supported
short gaps and speeds below 2 px/frame, motion fits within half the world width;
large or irregular gaps that permit a displacement of 25 pixels or more need
explicit winding disambiguation before this estimator is reliable.

For frames with no valid foreground overlap, matching receives zero weight.
The physical/raw fit returns zero when all weights vanish; this represents
unobservability, not evidence that the object is stationary. Report visibility
and correspondence coverage for any such cases.

## Verification

```sh
/home/yixunhu/miniconda3/envs/flowm/bin/python experiments/continuous_velocity/paper/test_mechanism.py -v
```

Tests compare the factored correspondence computation against the original
pilot, compare the no-op velocity wrapper bit-for-bit, check that quantization
affects the declared stages, verify zero parameters in the raw baseline, check
camera-compensated boundary-crossing motion on synthetic pixels, establish raw
matching's independence from learned parameters, and verify gradients through
the MLP without calling the physical fit.
