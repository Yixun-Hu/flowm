# Continuous-motion memory: controlled evidence study

For a complete handoff covering the earlier FloWM stress test, pilot, this campaign, direct comparison figures, resource constraints, and continuation commands, read [HANDOFF.md](../../../HANDOFF.md).

This directory extends the completed pilot without modifying its code or saved
results. The study separates motion representation, analytic fitting, learned
matching, memory fusion, initialization, and optimization. It is a single-object,
rigid, constant-velocity study with known integer camera motion on a periodic
50-pixel world. It does not establish a general-purpose world model.

The completed `paper_v1` campaign contains 44 training runs and 106 evaluation
jobs, covering 41,408 model-sequences. Start with the
[results and discussion](../../../outputs/continuous_velocity/paper_v1/research_findings.md),
[browser-viewable report](../../../outputs/continuous_velocity/paper_v1/report.html),
and [independent review](../../../outputs/continuous_velocity/paper_v1/independent_review.md).
The [completion audit](../../../outputs/continuous_velocity/paper_v1/completion.json)
verifies all planned artifacts, checkpoint hashes, source hashes and metric rows.

## Protocol and evidence

The executable protocol is `campaign.py`; the recorded protocol, selected learning
rates, source snapshots, and raw measurements live in
`outputs/continuous_velocity/paper_v1/`. Checkpoints are written through the
repository symlink to `/media/diskstation/yixunhu/flowm/downloaded_checkpoints/`.

All learned models receive context images, camera coordinates, and timestamps.
Training uses future image loss only. True positions, velocities, digit identity
templates, and visibility are excluded from model inputs and training losses.
They are used by evaluation or explicitly labeled oracle interventions.

Core controls use 5,000 optimizer updates, batches of eight, and the same paired
training streams per seed. Each model receives a two-value learning-rate search
(`0.0003`, `0.001`) using 64 heldout validation sequences and tuning seed 2026.
Validation pixel loss selects the checkpoint and learning rate. Fresh seeds
2027/2028/2029 replicate every core control; physical and relaxed also use 2030.
The tuning seed is reported separately from post-selection training variability.

Four model types additionally train to 20,000 updates using seed 2026. Their
5,000-update checkpoint artifacts remain fixed. The extended run answers a
limited optimization question; it is not a convergence guarantee or an
independent multi-seed result.

| Control | Changed component |
|---|---|
| `physical` | Analytic constant-velocity fit and transport |
| `relaxed` | Same physical initialization plus learned time-dependent residual |
| `offset` | Learned time-dependent offsets from correspondence summary |
| `mlp_velocity` | MLP regresses one continuous velocity from the same pair evidence |
| `raw_correlation` | Fixed raw-pixel matching; learned memory/decoder retained |
| `mean_memory` | Visibility mean instead of learned temporal fusion |
| `raw_untrained` | Parameter-free raw matching, fit, visibility mean, intensity transport |
| Fixed-checkpoint interventions | Raw matching, mean fusion, or velocity quantization |

Quantization is tested both before alignment and transport, and only during
future transport. The latter keeps the canonical memory exactly fixed. Oracle
continuous and quantized velocities are diagnostic controls, not formal model
results. These are interventions on a single motion estimate, not simulations
of FloWM's multi-channel maximum readout.

## Generalization

- Training rendering: bilinear, nearest, or a paired 50/50 mixture. Test both
  rendering modes. Changing the rendering condition preserves the underlying
  digits, starts, velocities, and camera actions for a fixed seed.
- Train prediction horizons through 40 frames; evaluate 20/40/80/160. The final
  ten actions needed for a 160-frame forecast replay the start of the original
  200-action episode. The camera coordinate never resets at the seam.
- Compare all 50 observations with irregular25 and approximately uniform
  sparse25. The latter separates sampling irregularity from observation count.
- Remove the final 10/20/30 context observations as a controlled observation
  blackout. This is not a geometric out-of-view intervention.
- Signed 2-D training excludes angular wedges within 10 degrees of 45 and 225
  degrees. Test seen and withheld directions with matched physical/MLP controls.
- Unfamiliar procedural rigid shapes are an exploratory appearance transfer
  check. They do not establish broad visual generalization.

Core testing uses 32 paired scene seeds and eight velocities in each renderer.
Generalization suites use 16 scene seeds. Heldout horizontal speeds are
0.37, 0.63, and 1.23; their ±0.03 neighborhoods are excluded from horizontal
training. These are interpolation tests inside the training speed range.

## Evaluation

Version 2 of `metrics.py` makes unconditional image MSE and visible-frame
localization success the primary image outcomes. Frame localization uses a
two-pixel **periodic image-position** tolerance; it does not measure world winding.
Missing or detected duplicate objects fail this frame score. A separate bounded
continuity tracker measures unwrapped trajectory error only while winding remains
identifiable. It starts from the last accepted context detection, uses GT once
to select the global coordinate origin, and never resets or wraps future errors
against GT. Its ambiguity rate and coverage accompany every trajectory score.

The old constant-velocity fitted ADE, velocity estimate, coverage, and joint
score remain explicitly labeled **CV-certified diagnostics**. They can reject
nonconstant predictions and even correct images under occlusion. Perfect-GT
prediction ceilings are therefore calibrated across every suite and horizon.
Fitting uses only the prediction prefix belonging to the scored horizon.

All compared models also expose their actual renderer offsets. Internal
transport-law error compares these unwrapped offsets to elapsed-time times true
velocity, without a detector or fitted motion assumption. This is a motion-head
diagnostic, not object trajectory error: canonical-memory location errors can
reinforce or compensate for motion errors. Blackout cases use the full elapsed
time since the last observation.

The detector uses an oracle appearance template exclusively for evaluation.
Its constant-velocity winding fit may reject nonconstant predictions; coverage,
ambiguity, frame localization, and unconditional MSE must therefore accompany ADE.
`calibration.py` measures sensitivity to blur, fragments, faint appearances, and
duplicates. In particular, overlapping ghosts can be missed; zero measured ghost
rate means "no detected ghosts."

The independent review that motivated this correction is preserved in the output
directory. Pre-correction results are archived under `evaluation_pre_review_v1`
and excluded from current tables. Training uses image loss and was unaffected.

Aggregate metrics and bootstrap estimates use the same pooled numerators and
denominators. `statistics.py` keeps training-seed mean/SD separate from paired
test-sequence uncertainty. Test intervals conditional on the fitted models are
not confidence intervals for future training runs.

## Run

Use the existing `flowm` environment. GPU 1 sharing was authorized; these runs
are not controlled hardware speed benchmarks.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
PYTHON=/home/yixunhu/miniconda3/envs/flowm/bin/python
$PYTHON experiments/continuous_velocity/paper/campaign.py --stage all
$PYTHON experiments/continuous_velocity/paper/additional_seeds.py
$PYTHON experiments/continuous_velocity/paper/evaluate_campaign.py
$PYTHON experiments/continuous_velocity/paper/report.py
```

Evaluation can also run with `--watch` while training artifacts arrive. It
resumes by completed test seed and rejects changed checkpoint/source signatures.
To split CPU-heavy measurement across two independent workers, run the evaluator
twice with `--watch --shard-count 2 --shard-index 0` and `--shard-index 1`.
Stable job-name hashes assign disjoint work; shard 0 declares global completion
only after all planned artifacts and both training workers are complete.
The additional-seed worker can likewise run while the main campaign selects
learning rates. For an interrupted extended run beyond 5,000 updates, resume
with `campaign.py --stage extended`; the trainer intentionally refuses to treat
that partial run as a smaller-budget training job. After the extension finishes,
rerun `campaign.py --stage all` to validate/reuse completed artifacts, finish any
remaining stages, and publish the full-campaign training completion marker.

Run CPU checks directly to avoid unrelated package-level optional dependencies:

```bash
$PYTHON experiments/continuous_velocity/paper/test_mechanism.py
$PYTHON experiments/continuous_velocity/paper/test_data.py
$PYTHON experiments/continuous_velocity/paper/test_metrics.py
$PYTHON experiments/continuous_velocity/paper/test_statistics.py
```

See `method_notes.md`, `MECHANISM.md`, `protocol_audit.md`, and `literature.md`
for equations, intervention semantics, implementation review, and claim limits.
Frozen FloWM remains a reference in the earlier pilot. No claim of architecture
superiority to a comparably trained FloWM is made by this campaign.
