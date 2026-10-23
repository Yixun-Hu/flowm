# FloWM continuous velocity research handoff

Prepared on 2026-10-21 for the next agent working with Yixun Hu in this repository. The main experiment campaign completed on 2026-10-13. This handoff records the work, results, implementation, operational constraints, and remaining research questions. Creating this document did not launch new training or inference.

**Current state:** the frozen FloWM stress test, continuous-velocity pilot, multi-seed evidence campaign, and direct comparison figures are available. The main campaign passed its artifact audit and 43 tests. The 289-channel FloWM exploration is incomplete. There is no unfinished training task to resume automatically. A process inspection during handoff found no matching experiment Python jobs running.

**Scientific position:** continuous velocity estimation and fixed canonical appearance transport work well for single-object constant motion with bilinear observations. Fixed-memory interventions demonstrate a cost of quantizing transport velocity. Learned memory fusion and strict motion enforcement have not shown independent advantages over their strongest controls. Nearest-rendering transfer remains a substantial failure. Publication readiness, architectural novelty, and superiority to equally trained FloWM are not established.

## Start here

Read these in order before changing code or launching another campaign:

1. [Final results and discussion](outputs/continuous_velocity/paper_v1/research_findings.md): the authoritative interpretation of the completed follow-up campaign.
2. [Independent review](outputs/continuous_velocity/paper_v1/independent_review.md): original defects, their corrections, adversarial checks, and remaining scientific limitations. Its early verdict is historical; the top disposition and appended v2 recheck describe the final state.
3. [Method notes](experiments/continuous_velocity/paper/method_notes.md): equations and the actual implementation, including differences from the original proposal.
4. [Study README](experiments/continuous_velocity/paper/README.md), [protocol audit](experiments/continuous_velocity/paper/protocol_audit.md), and [mechanism controls](experiments/continuous_velocity/paper/MECHANISM.md).
5. [Completed campaign report](outputs/continuous_velocity/paper_v1/report.html) and [direct FloWM comparison](outputs/continuous_velocity/flowm_comparison/report.html).
6. [Completion audit](outputs/continuous_velocity/paper_v1/completion.json), [test record](outputs/continuous_velocity/paper_v1/verification.json), and [selected runs](outputs/continuous_velocity/paper_v1/selected_runs.json).

The recommended next research target is renderer-robust correspondence/velocity estimation with controlled initialization and temporal-baseline comparisons. That is a recommendation, not a result or a queued experiment. Do not simply launch all scripts again.

## User intent and decisions

The user first requested a frozen-checkpoint fractional-velocity stress test to establish an actual FloWM failure before changing the architecture. They then approved a pilot based on:

> Learned correspondence attention → differentiable constant-velocity fit → trajectory-constrained memory attention → analytic continuous transport.

The user prefers a physical inductive bias for continuous velocity, ideally expressed through attention and motion structure. They agreed to start with one object and a fixed canonical feature map. The implicit appearance field, Moving Velocity Frames, and Continuous Velocity Memory were design directions; a learned implicit appearance field and multi-object extension have not been implemented.

The later request was to use subagents to strengthen the work toward a method-paper contribution. The requested priorities were mechanism controls, multiple training seeds, temporal/rendering generalization, neutral evaluation, and honest novelty boundaries. The implementation and evidence campaign followed those priorities. The user subsequently requested visual comparisons with FloWM, a complete experiment summary, and this handoff.

The user requested an Opus 5.5 xhigh code/plan review. Opus was not callable in the available environment. This was disclosed, and an independent available-model reviewer ran at xhigh. Its exact backend identifier was not exposed. **Do not describe this as an Opus review.**

Existing subagent roles in the session were `evaluation`, `mechanism`, `generalization`, and `independent_review`. Their lasting outputs are files listed here; continuation must not depend on their conversational memory.

## Workspace and resource constraints

| Item | Value or state |
|---|---|
| Repository | `/home/yixunhu/codespace/flowm` |
| Shell | Bash |
| Python | `/home/yixunhu/miniconda3/envs/flowm/bin/python` |
| Git branch at handoff | `main` |
| Git HEAD at handoff | `c909c54a3d58ae240de03f5ebbec222d3e6b1264` |
| Data symlink | `data` → `/media/diskstation/yixunhu/flowm/data` |
| Checkpoint symlink | `downloaded_checkpoints` → `/media/diskstation/yixunhu/flowm/downloaded_checkpoints` |
| Result storage | Repository-local `outputs/`; currently ignored by Git |
| GPU authorization | User explicitly authorized sharing GPU 1 for the pilot; this preference carried into the follow-up campaign |
| Resource sensitivity | User previously relayed a request to avoid disturbing a speed benchmark on another internal server; later GPU 1 sharing was explicitly approved |
| Current handoff action | Documentation only; no new GPU work |

Preserve the NAS destinations and symlinks. The user explicitly requested moving current data/checkpoints there and downloading new assets there. The environment itself remains local. Some NAS operations do not preserve Unix metadata; byte copies and atomic replacements were used successfully. Do not remove a symlink and silently duplicate large assets on local storage.

Use CPU for inspection, rescoring saved outputs, and plotting when possible. Before future compute, inspect current occupancy; shared-GPU timings are not speed benchmarks. Do not kill, stop, or inspect other users' jobs beyond ordinary resource visibility. GPU 0 is not the approved default for continuation, even though original historical scripts/manifests include GPU 0 runs. The original fractional script still defaults to `cuda:0`; override explicitly for any authorized new run.

Verified installed versions at handoff: PyTorch `2.9.0+cu128`, Lightning `2.6.6`, flash-attn `2.7.4.post1`, setuptools `80.10.2`, NumPy `1.26.4`, Matplotlib `3.11.2`. Use the working environment rather than reinstalling dependencies. Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` or `2` to keep CPU load modest.

Before this handoff, `git status --short` showed untracked `data`, `experiments/continuous_velocity/`, and `experiments/fractional_velocity/`. The research code was not committed. This handoff adds another local file and README navigation. No commit, push, deployment, or external sharing was requested or performed. Preserve these untracked directories; a clean/reset operation could destroy the work. No applicable ancestor or repository `AGENTS.md` was found during the handoff inspection.

## Assets and source map

Official MNIST train/test IDX gzip files and the released camera episodes are under [data/fractional_velocity](data/fractional_velocity). The original checkpoint is:

```text
downloaded_checkpoints/hf_models/mnistworld-models/mnistworld/flowm-fernn/dynamic_po_flowm.pt
```

Released checkpoint SHA-256: `4f84d5c172e29b99e899400464a4fa0570a2dd2aa4c6306ac3289e074cf47e37`. The original run recorded Hugging Face revision `72affe592a5b184d51b37b87ad5358362bbcaa1e`. [Pilot provenance](outputs/continuous_velocity/pilot/provenance.json) records dataset hashes. The MNIST source mirror used was `https://ossci-datasets.s3.amazonaws.com/mnist/`.

| Code | Purpose |
|---|---|
| [Original FloWM implementation](algorithms/mem_wm/backbones/flowm/flowm_models_2d.py) | Released 2D model loaded without unrelated framework imports |
| [fractional_velocity/run.py](experiments/fractional_velocity/run.py) | Frozen checkpoint loading, released-data reproduction, paired sequence generation, prediction, original evaluation |
| [fractional_velocity/densify.py](experiments/fractional_velocity/densify.py) | Test-time velocity densification with unchanged learned weights |
| [fractional_velocity/measurement.py](experiments/fractional_velocity/measurement.py) | Periodic renderer, template detector, legacy tracking primitives |
| [fractional_velocity/analyze.py](experiments/fractional_velocity/analyze.py) | Historical sweep plots/statistics; uses legacy measurements |
| [fractional_velocity/validate.py](experiments/fractional_velocity/validate.py) | Original detector and tracker calibration |
| [fractional_velocity/visualize_saved.py](experiments/fractional_velocity/visualize_saved.py) | Original CPU-only PNG/PDF/GIF comparisons |
| [continuous_velocity/model.py](experiments/continuous_velocity/model.py) | Shared implemented neural model; remains a dependency of the paper study |
| [continuous_velocity/data.py](experiments/continuous_velocity/data.py) | Pilot data/renderer utilities; also imported by paper data |
| [continuous_velocity/train.py](experiments/continuous_velocity/train.py), [evaluate.py](experiments/continuous_velocity/evaluate.py), [report.py](experiments/continuous_velocity/report.py), [diagnostics.py](experiments/continuous_velocity/diagnostics.py) | Historical one-seed pilot |
| [paper/mechanism.py](experiments/continuous_velocity/paper/mechanism.py) | MLP/raw/mean controls and fixed-checkpoint interventions |
| [paper/data.py](experiments/continuous_velocity/paper/data.py) | Paired train/validation/test scenarios and generalization suites |
| [paper/train.py](experiments/continuous_velocity/paper/train.py) | Image-only training, validation selection, source snapshots, resumable checkpoints |
| [paper/campaign.py](experiments/continuous_velocity/paper/campaign.py) | Main training protocol, learning-rate selection, domain runs, budget extensions |
| [paper/additional_seeds.py](experiments/continuous_velocity/paper/additional_seeds.py) | Additional seed 2029 for four core controls |
| [paper/evaluate.py](experiments/continuous_velocity/paper/evaluate.py) | Suite definitions, model loading, prediction, current metrics, examples, manifests |
| [paper/evaluate_campaign.py](experiments/continuous_velocity/paper/evaluate_campaign.py) | Job matrix, selection merge, resumable/sharded workers, completion detection |
| [paper/metrics.py](experiments/continuous_velocity/paper/metrics.py) | Audited schema-2 evaluation |
| [paper/statistics.py](experiments/continuous_velocity/paper/statistics.py) | Pooled metrics, training variability, paired scene-cluster bootstrap |
| [paper/report.py](experiments/continuous_velocity/paper/report.py), [qualitative.py](experiments/continuous_velocity/paper/qualitative.py) | Current tables, scientific figures, static HTML, fixed-case examples |
| [paper/calibration.py](experiments/continuous_velocity/paper/calibration.py), [measurement_ceiling.py](experiments/continuous_velocity/paper/measurement_ceiling.py) | Detector corruptions and perfect-prediction ceilings |
| [paper/lag_diagnostic.py](experiments/continuous_velocity/paper/lag_diagnostic.py) | Explicitly post-hoc raw-matching lag study |
| [paper/validate_campaign.py](experiments/continuous_velocity/paper/validate_campaign.py) | Independent artifact/source/checkpoint audit and completion publication |
| [paper/compare_flowm.py](experiments/continuous_velocity/paper/compare_flowm.py) | CPU-only exact-pair verification, FloWM v2 rescoring, and latest direct comparison |

The original upstream model was not edited to train the new architecture. The research lives in the experiment directories. **Do not modify the shared parent model/data files in place for a new experiment and expect old source-hash guards to keep passing.** Snapshot the completed implementation or create an isolated version before architecture changes.

## Implemented method

The current model is a batch-context, single-motion predictor. It receives only context images, camera-window origins, context timestamps, future camera origins, and target timestamps. Future camera motion is known input. It does not receive true position, true velocity, digit identity, or a template. Oracle experiments are separately labeled.

1. A shared encoder maps grayscale images through convolutions `1→12→4` with SiLU. Concatenating raw intensity gives five feature channels.
2. Known integer camera origins scatter these features and observation masks into a periodic 50×50 world field.
3. Masked FFT correlations compare 36 frame pairs: 12 pairs at each index lag 1, 2, and 4. Softmax over admissible integer spatial displacements gives a continuous displacement expectation. Actual elapsed time controls the search bound and velocity fit.
4. Learned confidence and matching support weight a differentiable least-squares estimate, followed by two robust Cauchy reweighting steps with residual scale 0.15:

   `v = sum_k(w_k * dt_k * displacement_k) / sum_k(w_k * dt_k^2)`.

5. The estimated velocity aligns original observed fields to the last context time. Learned temporal fusion or visibility-weighted averaging builds a canonical field.
6. A decoder with an intensity skip produces the appearance field. Future queries use accumulated displacement `(target_time - last_context_time) * v`. Generated frames/hidden maps are not recursively resampled at every forecast step.

The canonical map is **50×50 and world-aligned at the last observed time**, not an object-centered crop. Object position is implicit in this field; there is no separately predicted object center or object slot. The inferred velocity does not feed back into the initial correspondence attention. Correspondence is global displacement matching, not independent per-token query/key attention. Continuous output velocity does not make the spatial correspondence search or feature lattice continuous.

The memory is built from the context batch and held fixed during forecasting. It is not an indefinitely streaming recurrent memory update, and no cost/capacity comparison to FloWM's online memory was performed. There is one velocity for all scene content, so multiple independently moving objects require an architectural change.

The physical model has 1,540 parameters. The MLP velocity control has 10,502; relaxed/free-offset controls have 6,534; mean-memory has 1,483. Parameter count alone is not a runtime benchmark.

| Variant | Meaning and interpretation |
|---|---|
| `physical` | Robust analytic velocity estimation and strict linear-in-time transport |
| `relaxed` | Same initial physical predictions plus a learned time-dependent residual initialized to zero |
| `offset` | MLP produces time-dependent offsets; starts near no motion and changes initialization/optimization |
| `mlp_velocity` | MLP regresses one continuous velocity from the same pair evidence; still uses strict `dt*v` transport, so this is an estimator comparison |
| `raw_correlation` | Fixed raw-pixel matching with trained memory/decoder |
| `raw_untrained` | No learned parameters; raw matching, robust fit, visibility mean, intensity transport; no oracle digit template |
| `mean_memory` | Replace learned temporal fusion with visibility-weighted mean |
| `physical_untrained` | Physical network at initialization; multiple seeds are initialization variation, not training variation |
| Matcher/fusion swaps | Replace one component in a fixed trained checkpoint |
| `scope='transport'` quantization | Keep canonical memory exactly fixed and quantize only future transport |
| `scope='both'` quantization | Quantize alignment and future transport; changes canonical memory |

Quantization uses `torch.round` (ties to even), without clipping. Oracle pairs have oracle-aligned memory, which differs from estimated-velocity memory; memory is identical only within the specified intervention pair. Oracle image scores are diagnostic references, not guaranteed bounds.

## Data and training protocol

| Split | Digit identities | Camera episodes |
|---|---|---|
| Training | Official MNIST train indices 0–54999 | 64–223 |
| Validation | Official MNIST train indices 55000–59999 | 224–255 |
| Test | Official MNIST test set | 0–63 |

Horizontal training samples velocities in `[0,2]`, excludes ±0.03 neighborhoods around 0.37, 0.63, and 1.23, and replaces 15% of samples with integer speeds. These held-out speeds are interpolation holes, not extrapolation outside the speed envelope. Note that 1.25 also falls inside an excluded neighborhood.

The core paper test suite uses velocities `[0, .37, .5, .63, 1, 1.23, 1.5, 2]`, 32 scene seeds, both renderers, and horizon 40. Primary tables restrict to `.37/.63/1.23`, hence 96 sequences per fitted model/renderer. Generalization usually uses 16 scene seeds. A scene seed fixes digit, start, and camera; changing velocity/renderer/model preserves the pairing.

Nearest rendering rounds each absolute position to the nearest pixel; bilinear rendering samples the original sprite at each absolute position. Ground truth does not accumulate interpolation blur. Coordinates are `(x,y)`; world grids must preserve that ordering. The world wraps, but underlying trajectories stay unwrapped.

The 2D task samples signed components within `[-2,2]` and excludes ±10° wedges around 45° and 225°. Test velocities probe wedge centers at three magnitudes and both signs. Seen and held-out direction sets have different speed norms; an across-group difference is not a direction-only causal test.

Training uses 50 observations and targets at future offsets `[1,5,10,20,30,40]`, batch size 8, Adam, gradient clipping at norm 1, validation every 250 updates, and image-only loss weighted by `1 + 5*(target > .05)`. True velocity is logged diagnostically but never enters the loss. All formal models use the same permitted input keys.

The pilot used one seed 2026, 5,000 updates, and learning rate `3e-4`. The paper campaign compared `3e-4` and `1e-3` using seed 2026 and 64 validation examples. All six core selections chose `1e-3`. Primary results exclude tuning seed 2026:

- Physical and relaxed: fresh seeds 2027–2030, four fits each; five including tuning.
- Offset, MLP velocity, mean memory, trained raw matching: fresh seeds 2027–2029, three each; four including tuning.
- Nearest/mixed horizontal and mixed 2D domain fits: fresh seeds 2027–2028, two each; three including tuning. Learning rates transfer from horizontal bilinear tuning.
- Four 20,000-update extensions: physical, relaxed, offset, MLP, seed 2026 only.

`budget_5000.pt` means the best validation checkpoint available within 5,000 updates, not necessarily step 5,000. Likewise for `budget_20000.pt`. Read its saved `step`. Fixed-budget comparisons must not use a later extension's `best.pt`. For example, the fixed qualitative physical seed 2027 checkpoint was selected at step 4,250.

For horizon 160, full sequence length is 210. The released camera bank has 200 actions, so the final ten replay the beginning of the same episode before cumulative camera integration. The position never resets. This is a disclosed continuation convention, not newly sampled policy actions.

## Experiment history and results

Success below denotes schema-2 visible-frame localization unless explicitly called historical. MSE includes all pixels and frames in the scored forecast prefix. Fresh-seed means are not directly interchangeable with tuning-seed or pilot results.

### Frozen checkpoint reproduction and stress test

- Original smoke pilot: four sweep seeds and eight reproduction episodes, overlapping the later main run; not extra independent replication.
- Released reproduction: 32 episodes; MSE20 `0.000530707`, MSE40 `0.000720405`.
- Main frozen 25-channel sweep: 928 sequences = 32 seeds × (13 nearest speeds + 13 bilinear speeds + 3 integer half-offset controls).
- Original 13 speeds: `0,.1,.25,.5,.75,.9,1,1.1,1.25,1.5,1.75,1.9,2`; vertical velocity zero.
- Context 50, autoregressive prediction 40, measurements at 20/40; original learned weights verified unchanged.

| Nearest rendering, horizon 40 | Missing detections | Image MSE |
|---|---:|---:|
| Integer speeds 0,1,2 | 0% | 0.000339 |
| Near-grid .1,.9,1.1,1.9 | 53.68% | 0.037193 |
| Half-grid .5,1.5 | 99.77% | 0.039724 |
| Quarter-grid .25,.75,1.25,1.75 | 94.27% | 0.041463 |

Half-pixel appearance offsets at integer velocities had zero missing detections and MSE `0.000888`, versus integer/no-offset MSE `0.000339`. Appearance change alone did not reproduce catastrophic off-grid missing. Bilinear off-grid rendering also failed; the phenomenon was not confined to nearest rendering.

A context constant-velocity tracker gave nearest velocity MAE approximately `0.00587 px/frame` and valid-case trajectory ADE `0.4846 px` in the original calibration. It used an oracle appearance template and legacy winding assumptions. Do not confuse it with the later parameter-free raw-correlation baseline. The original half-offset perfect-GT ADE was itself about `0.709 px`, demonstrating detector bias in that control.

### Frozen velocity densification

- 81 channels, spacing 0.5: complete nearest sweep, 416 sequences, 32 seeds, same learned weights.
- Fractional channels use accumulated-offset nearest-integer transport rather than repeatedly interpolating the hidden state. The modified spacing-1 implementation matched the original integer implementation with max absolute difference `0.0`.
- At speeds .5/1.5, missing detections fell from `99.77%` to `0%`; MSE from `0.039724` to `0.001381`.
- Quarter-grid missing detections remained `91.98%`, versus `94.27%` originally. Finer channels helped speeds newly covered by the grid, not arbitrary motion.
- Later v2 rescoring at nearest .5/1.5 gives localization success `0%` for 25 channels and `96.46%` for 81. Old CV-certified success is a different metric.
- 289 channels, spacing 0.25: **incomplete**. Current CSV has 234 horizon rows = 117 sequences across nine complete seeds; 127 NPZ files include ten additional partial-seed cases. No final unchanged-weights completion flag. Original aggregate/statistics files for this variant reflect only two earlier seeds and are stale. Do not publish them as complete results.

Sources: [original main outputs](outputs/fractional_velocity/main), [81-channel outputs](outputs/fractional_velocity/dense05), [partial 289 outputs](outputs/fractional_velocity/dense025), [original calibration](outputs/fractional_velocity/calibration/summary.json).

### One seed continuous velocity pilot

The pilot trained physical, relaxed, and offset models for 5,000 updates each. It evaluated 64 scene seeds and 16 velocities (original 13 plus .37/.63/1.23). Neural controls have both renderers; the pilot FloWM25/81 directories contain bilinear predictions only. The completion record lists 12,288 model-sequences across seven conditions.

| Held-out .37/.63/1.23, bilinear, horizon 40 | Pilot MSE |
|---|---:|
| Physical | 0.002011 |
| Relaxed | 0.002015 |
| Untrained physical | 0.004292 |
| Free offsets | 0.043230 |
| Oracle velocity | 0.000174 |

This established feasibility, not multi-seed stability or strictness superiority. On nearest heldouts, trained physical MSE `0.027964` was worse than untrained `0.017754`; oracle velocity reduced it to `0.001398`. Pilot trajectory scores are historical v1 metrics and should not be pooled with paper v2 metrics.

Temporal subsampling/composition diagnostics were also run. Physical displacement composition error was at most `1.9e-6`, versus about `0.00204` for relaxed. This is primarily a structural correctness check. Sparse context did not destroy prediction and motivated later sampling controls.

### Main mechanism controls and training stability

The completed `paper_v1` campaign has 44 distinct training runs, 280,000 optimizer updates, 106 evaluation jobs, 41,408 model-sequences, and 101,152 horizon-specific metric rows. Counts include tuning and repeated evaluations; do not treat them as independent samples.

| Bilinear heldouts, horizon 40 | Fresh seeds | MSE mean ± training SD | Success |
|---|---:|---:|---:|
| Physical | 4 | 0.001094 ± 0.000095 | 99.46% |
| Relaxed | 4 | 0.001056 ± 0.000037 | 99.46% |
| Mean memory | 3 | 0.001072 ± 0.000062 | 99.48% |
| Raw matching with trained memory/decoder | 3 | 0.005407 ± 0.000148 | 90.89% |
| Free offsets | 3 | 0.003892 ± 0.003968 | 96.75% |
| MLP velocity | 3 | 0.054709 ± 0.000015 | 3.65% |
| Fully nonlearned raw reference | Not trained | 0.004718 | 91.88% |
| Untrained physical | 3 initializations | 0.004511 ± 0.000059 | 92.29% |

Learned matching improved the bilinear result over the specified raw-matching control: on common fresh seeds, MSE difference `-0.00428047`, conditional paired 95% CI `[-0.00628019,-0.00245165]`; success improvement `8.57 pp`. Fixed-checkpoint raw-matcher swaps also hurt. This tests the learned matching package, not descriptors, temperature, and confidence separately or the best possible classical registration method.

Mean memory was competitive or slightly better. Fixed-checkpoint mean-fusion swaps also slightly improved MSE while preserving motion. There is no independent necessity claim for learned memory attention.

Physical minus relaxed MSE was `+0.00003855`, CI `[-0.00000416,+0.00009255]`. No clear strict-constraint advantage. Nonsignificance is not equivalence.

Offset fresh-seed MSEs were `0.00846329, 0.00134018, 0.00187222`; successful seeds exist. MLP image-only gradients were independently verified finite and nonzero. Its poor outcome is not a disconnected-gradient bug, but initialization and optimization remain confounded with estimator choice.

### Velocity quantization interventions

One physical checkpoint, tuning seed 2026, with 96 held-out scenes:

| Future transport, fixed memory within each pair | MSE | Success | Internal transport ADE |
|---|---:|---:|---:|
| Estimated continuous | 0.001256 | 99.27% | 0.1379 px |
| Estimated quantized to 0.5 | 0.033959 | 30.05% | 3.3483 px |
| Estimated quantized to 1 | 0.049474 | 13.14% | 6.6283 px |
| Oracle continuous | 0.000138 | 100% | Approximately 0 |
| Oracle quantized to 0.5 | 0.034121 | 29.16% | 3.3483 px |
| Oracle quantized to 1 | 0.049834 | 13.14% | 6.6283 px |

Estimated half-grid MSE increase was `0.03270282`, paired scene CI `[0.02871449,0.03684582]`; integer-grid increase `0.04821722`, CI `[0.04156655,0.05513728]`. Internal velocity staircase/drift is partly prescribed by the intervention; changed image loss and detection are the empirical outcomes. Both-alignment-and-transport interventions have three checkpoint seeds, but transport-only intervention does not. This is causal evidence inside the new single-velocity pipeline, not a faithful simulation of FloWM's multiple hypotheses and channel max readout.

### Rendering train and test matrix

| Training renderer | Test renderer | Fresh seeds | MSE | Success | Velocity L2 error |
|---|---|---:|---:|---:|---:|
| Bilinear | Bilinear | 4 | 0.001094 | 99.46% | 0.00607 |
| Bilinear | Nearest | 4 | 0.034384 | 46.27% | 0.07090 |
| Nearest | Bilinear | 2 | 0.007133 | 87.09% | 0.02289 |
| Nearest | Nearest | 2 | 0.016787 | 69.93% | 0.04363 |
| Mixed | Bilinear | 2 | 0.001902 | 99.11% | 0.00792 |
| Mixed | Nearest | 2 | 0.020231 | 64.71% | 0.04899 |

The raw nonlearned reference obtains nearest MSE `0.017127` and success `61.64%`, better than the bilinear-trained neural model under transfer. Oracle velocity in the paper seed-2026 checkpoint gives nearest MSE `0.001592` and success `100%`, implicating estimation/alignment without uniquely separating them. Nearest/mixed training helps but does not solve the problem.

### Temporal extrapolation and sampling

Train through horizon 40; test 20/40/80/160. At horizon 160 with regular context:

| Model | Bilinear MSE | Bilinear success | Nearest success |
|---|---:|---:|---:|
| Physical | 0.002785 | 94.11% | 29.79% |
| Relaxed | 0.002722 | 94.18% | 29.66% |
| Free offsets | 0.027963 | 47.14% | 24.88% |
| MLP velocity | 0.053501 | 3.79% | 3.98% |

Physical internal transport ADE is `0.513 px` on bilinear and `5.173 px` on nearest. Conservative image-derived continuity coverage is only about 35% and 23%, respectively. Do not report conditional unwrapped ADE without coverage.

| Physical context, same elapsed span | Bilinear success at 160 | Nearest success at 160 |
|---|---:|---:|
| 50 regular observations | 94.11% | 29.79% |
| 25 irregular observations | 92.42% | 62.50% |
| 25 approximately evenly spaced observations | 94.52% | 69.11% |

Actual timestamps are used. The uniform-count control means improvement cannot be uniquely attributed to irregular sampling; subsampling changes matching time lags. Mixed-trained models show the same sensitivity: nearest success `39.06%` regular, `74.23%` irregular25, `78.01%` sparse25.

### Blackouts and appearance transfer

Blackout and appearance results use physical models with two fresh checkpoints per training-renderer condition, seeds 2027–2028. Remove the last 10/20/30 context observations while leaving nominal future timestamps fixed. For the bilinear-trained physical models, bilinear horizon-40 success is `99.95% / 99.31% / 96.66%`; nearest is `38.36% / 32.28% / 40.48%`. These are observation blackouts, not controlled geometric occlusion. Forecast elapsed time from the last observation increases; conservative winding coverage becomes zero for these unresolved gaps.

Unfamiliar shapes consist of three procedural ellipses, not another natural-image dataset. At horizon 80:

| Train renderer | Shape test renderer | MSE | Success |
|---|---|---:|---:|
| Bilinear | Bilinear | 0.001989 | 97.68% |
| Bilinear | Nearest | 0.032017 | 34.04% |
| Mixed | Bilinear | 0.002903 | 97.02% |
| Mixed | Nearest | 0.021463 | 50.03% |

### Two dimensional motion and held out directions

Two fresh fits per model, 16 scene seeds, horizon 80, six held-out and six seen vectors per renderer. Physical held-out success is `95.12%` bilinear and `36.80%` nearest; corresponding MSE `0.003900` and `0.034420`. Physical seen-direction success is `94.78%` bilinear and `49.77%` nearest. MLP held-out success is about `0.97% / 1.00%`. Both estimators still use constant-velocity transport. Do not infer a pure direction effect across unmatched speed norms or generalize to every withheld angle.

### Longer training

One tuning seed 2026; validation selection within each budget, not a convergence proof:

| Model | Bilinear MSE40 at 5k → 20k | Bilinear success160 at 5k → 20k |
|---|---:|---:|
| Physical | 0.001256 → 0.000733 | 93.03% → 97.28% |
| Relaxed | 0.001221 → 0.000423 | 90.60% → 92.74% |
| Free offsets | 0.001525 → 0.000846 | 68.75% → 79.89% |
| MLP velocity | 0.054045 → 0.050558 | 3.80% → 4.70% |

Free offsets improve substantially. Relaxed is better at the trained horizon and physical better at 160 for this single pair. This does not override the multi-seed finding of no clear strictness advantage. Nearest long-horizon failure remains.

### Exploratory lag diagnostic

Raw-pixel matching only, 16 test scene seeds × three heldout speeds × two renderers; compare lags `[1,2,4]`, `[1,4,8]`, `[2,4,8]`, `[1,4,12]` and weighted/unweighted, robust/nonrobust fitting. No formal checkpoint or selection was changed.

With robust weighted fitting, nearest velocity L2 error falls from `0.040235` for `[1,2,4]` to `0.015059` for `[1,4,8]`, about 62.6%. Bilinear error changes from `0.009724` to `0.013507`. The nearest paired interval for the change is `[-0.03746,-0.01378]`. This is post-hoc velocity-only evidence; larger time gaps introduce aliasing risks and still need a prospectively specified prediction experiment.

### Direct comparison figures with FloWM

The later [comparison script](experiments/continuous_velocity/paper/compare_flowm.py) performs no inference. It checks exact equality of frames, positions, cameras, actions, template, and visibility against the current generator, then rescored 832 saved FloWM sequences using metrics v2. It combines them with four fresh paper-model seeds.

| Bilinear .37/.63/1.23, 32 scenes each, horizon 40 | MSE | Success |
|---|---:|---:|
| Frozen FloWM 25 | 0.038101 | 0% |
| Frozen FloWM 81 | 0.040522 | 0.47% |
| Ours | 0.001094 | 99.46% |

This is 97.1% lower MSE than frozen FloWM25 on these sequences. **Training is unmatched:** ours trained on continuous velocities; FloWM uses frozen released weights. There is no matched-trained FloWM experiment.

At nearest .5/1.5, FloWM81 MSE/success are `0.001381 / 96.46%`; ours `0.001887 / 99.58%`. These speeds lie on the half-grid. They exclude the difficult nearest heldout trio, on which ours achieves only 46.27%; saved matched FloWM predictions for that nearest trio are unavailable.

Comparison coverage: bilinear common speeds are all eight core speeds; nearest intersection is only `[0,.5,1,1.5,2]`. The older pilot FloWM directories are bilinear-only. Do not infer a complete nearest comparison from the overlap.

Qualitative figures use fixed test scene 0 and first fresh training seed 2027, not a model-quality-selected example. Snapshot times maximize ground-truth visibility inside three predetermined time bins. The GIF shows all 40 future frames. Every panel shares intensity range `[0,1]`. The nearest v=1.5 example favors FloWM81 in image MSE and is retained as a balanced example.

## Evaluation corrections and statistical interpretation

The first evaluator fitted a global constant-velocity path, rejected outliers, and required identifiable context winding. It could reject correct frames because other frames did not fit a constant-velocity line. Independent review reproduced only 92.45% success for some perfect-prediction aggregates and failures on an unambiguous smooth nonlinear forecast.

Schema 2 makes unconditional MSE and visible-frame periodic localization primary. It preserves legacy outputs explicitly as CV-certified diagnostics and adds bounded continuity unwrapping plus model-internal transport measurements.

| Field or concept | Correct use |
|---|---|
| `mse` | Unconditional squared pixel error over the forecast prefix |
| `frame_joint_success` | Visible target, present detection, no detected duplicate, periodic position error ≤2 px |
| `frame_success_frames / visible_frames` | Numerator/denominator for pooled frame success |
| `continuity_ade`, `continuity_coverage` | Conditional unwrapped image trajectory error with its identifiable fraction |
| `transport_ade` | Unwrapped error of model displacement outputs against `elapsed_time*true_velocity`; not rendered object trajectory error |
| `ade`, `coverage`, `joint_success`, `certified_*` | Legacy global CV-fit diagnostics; not neutral headline image scores |
| `missing_rate`, `ghost_rate` | Report alongside MSE; ghost rate means detected duplicates only |

Continuity starts at the last accepted context detection, uses GT once to choose global winding origin, bounds motion by `2.5*elapsed + 1` per axis, and marks winding permanently unknown if ambiguous. It never wraps future errors back toward GT. Long gaps can therefore remove coverage even for perfect predictions. Internal transport error excludes canonical-memory position/appearance errors and cannot substitute for object tracking.

Perfect-GT calibration covers 1,952 sequences, 4,960 rows, and 62 groups. Core perfect bilinear frame success is 100% at horizon 20 and 99.98% at 40; nearest is 100% at both. Perfect bilinear continuity coverage is only 83.72%/72.86% at 20/40. Duplicate detection recall is 100% at tested separations 8/12/16 px but only 12.5% at 4 px. Strong blur also reduces detection. Frame localization can accept degraded appearance, so always include image MSE.

Aggregate ratios pool numerator/denominator within each fitted model, then summarize across training seeds. Paired bootstrap resamples scene-seed clusters, retaining velocities and model pairing. These intervals are conditional on fitted models and do not estimate uncertainty over future training runs. The many intervals are descriptive and unadjusted for multiple comparisons. Tuning seed results are not fresh-seed stability evidence.

Independent review also corrected missing seen/heldout direction grouping, loaded-code/disk-hash provenance, report completion checks, atomic metadata handling, and oracle wording. Adversarial checks include nonlinear forecasts, exact GT, missing/duplicate detections, ambiguous gaps, and 160-pixel unwrapped drift.

All 106 final paper evaluations use schema 2 and a compatible source-hash set. Old results are preserved under [evaluation_pre_review_v1](outputs/continuous_velocity/paper_v1/evaluation_pre_review_v1) and [pre_review_v1_report](outputs/continuous_velocity/paper_v1/pre_review_v1_report); never pool them into current results. The original pilot and old fractional analysis remain historical. Updated FloWM comparisons use their saved predictions but fresh v2 rescoring.

Frozen current metrics SHA-256: `4cd962176d112c93b6d714be64fd750ed8c00b18e5b069e3cbdc0fdfea05377d`. Evaluation source SHA-256: `ba79b3d1c79cc4684eb61897be9b0acfd02a6568e4cb7b9dcb079743c0dadb2b`. All seven evaluation-source hashes in the completion record matched disk at handoff. Read manifests as the authority rather than copying these values into a new implementation.

## Artifact map and provenance

| Location | Contents and use |
|---|---|
| [outputs/fractional_velocity](outputs/fractional_velocity) | Original stress test, calibration, density variants, historical analysis, original figures |
| [outputs/continuous_velocity/pilot](outputs/continuous_velocity/pilot) | Completed single-seed pilot, all saved neural/FloWM predictions, diagnostics, legacy report |
| [outputs/continuous_velocity/paper_v1](outputs/continuous_velocity/paper_v1) | Authoritative final follow-up artifacts |
| [outputs/continuous_velocity/flowm_comparison](outputs/continuous_velocity/flowm_comparison) | Post-hoc direct comparison, new v2 CSV, source lineage, figures/GIF |
| [downloaded_checkpoints/continuous_velocity/pilot](downloaded_checkpoints/continuous_velocity/pilot) | Pilot `best.pt`/`last.pt` by model |
| [downloaded_checkpoints/continuous_velocity/paper_v1](downloaded_checkpoints/continuous_velocity/paper_v1) | Run-specific budget checkpoints and resume state |

Important files inside `paper_v1`:

- `completion.json`: independently checked 44 runs/280k updates/106 jobs/41,408 sequences/101,152 rows, with checkpoint/source hashes.
- `verification.json`: 43 distinct passing tests, zero failures. Counts: mechanism 8, data 8, metrics 9, statistics 5, report 6, campaign 7. Later repeat checks are not extra distinct tests.
- `protocol.json`, `additional_protocol.json`, `selected_runs.json`, `additional_runs.json`, `evaluation_jobs.json`: actual protocol, run selection, additions, job matrix. The original main protocol's seed list alone omits later extra seeds.
- `training_all_complete.json`, `additional_training_complete.json`, `evaluation_status.json`: required completion markers. `early_extension_complete.json` records a helper used for two extensions; those are not extra training runs.
- `training/<run>/manifest.json`, `training.csv`, `source/`: configuration, learning curves, preserved training sources. `aborted_*` directories are historical and excluded from complete counts.
- `evaluation/<job>/<suite>/manifest.json`, `metrics.csv`, `examples/*.npz`: source/checkpoint lineage, full measurements, saved qualitative predictions. Paper examples are saved for test scene 0 only, not every evaluated scene.
- `summary.json`, `group_summary.csv`, `per_training_seed.csv`, `image_velocity_by_speed.csv`: statistical summaries and breakdowns.
- `research_findings.md`, `research_findings_evidence.json`, `research_findings_extended_pairs.json`: final discussion and numerical evidence. `*_initial_snapshot` files preserve an incomplete earlier draft and must not replace final results.
- `measurement_ceiling.json/csv`, `calibration/summary.json`, `lag_diagnostic.json/csv`: evaluation validity and exploratory lag results.
- `independent_review.md`, `independent_review_*.json`, `independent_review_reproduce.py`, `metrics_revision.json`: review findings, reproductions, and correction record.
- `figures/`: PNG/PDF mechanism, rendering, time, quantization, learning curves, and qualitative sheets. Final qualitative provenance contains 14 figures, 66 examples, zero missing requests. Quantization figures show per-scene dots as well as means because averaging integer snaps can misleadingly look fractional.
- `logs/`: copied main, additional, early-extension, and evaluation/shard logs. `/tmp` logs are convenience copies, not authoritative durable storage.

The direct comparison's [summary.json](outputs/continuous_velocity/flowm_comparison/summary.json) includes input file hashes, baseline manifests, scoring versus figure-generation source hashes, scene-conditional paired intervals, and snapshot selection. Its [paired_metrics_v2.csv](outputs/continuous_velocity/flowm_comparison/paired_metrics_v2.csv) is separate from the 106-job main campaign. Replotting does not add training/evaluation jobs to the campaign count.

## Commands for the next agent

Run scripts directly from the repository root. `python -m experiments...` can trigger unrelated optional framework imports. The following variable is only a convenient local shell alias:

```bash
cd /home/yixunhu/codespace/flowm
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
FLOWM_PY=/home/yixunhu/miniconda3/envs/flowm/bin/python
```

### Inspect completed state without compute

```bash
git status --short
readlink -f data
readlink -f downloaded_checkpoints
$FLOWM_PY - <<'PY'
import json
from pathlib import Path
p = Path('outputs/continuous_velocity/paper_v1')
j = json.loads((p/'completion.json').read_text())
for key in ('complete', 'completed', 'completed_training_runs', 'optimizer_updates',
            'completed_evaluation_jobs', 'completed_model_sequences', 'metric_rows'):
    print(key, j[key])
print('tests', json.loads((p/'verification.json').read_text())['passed'])
PY
```

### Regenerate plots from saved evidence

These are CPU work and overwrite derived report/figure files; they do not train a model. Preserve a report snapshot first if changing presentation logic.

```bash
$FLOWM_PY experiments/continuous_velocity/paper/report.py --root outputs/continuous_velocity/paper_v1
$FLOWM_PY experiments/continuous_velocity/paper/qualitative.py --root outputs/continuous_velocity/paper_v1 --train-seed 2027
$FLOWM_PY experiments/continuous_velocity/paper/compare_flowm.py --figures-only
```

Without `--figures-only`, `compare_flowm.py` verifies/rescores 832 saved baseline cases on CPU and regenerates its CSV/figures. It does not use GPU or perform model inference. Its paths and comparison cohort are fixed to the completed pilot/paper_v1 artifacts; this is not a generic new-tag evaluator.

### Run relevant tests

Tests already passed. Re-run those affected by new changes, not the full campaign automatically:

```bash
$FLOWM_PY experiments/continuous_velocity/paper/test_mechanism.py
$FLOWM_PY experiments/continuous_velocity/paper/test_data.py
$FLOWM_PY experiments/continuous_velocity/paper/test_metrics.py
$FLOWM_PY experiments/continuous_velocity/paper/test_statistics.py
$FLOWM_PY experiments/continuous_velocity/paper/test_report.py
$FLOWM_PY experiments/continuous_velocity/paper/test_campaign.py
```

`validate_campaign.py --tag paper_v1` rechecks complete manifests, CSV bounds/counts, sources, checkpoints and budget artifacts on CPU. **It writes a fresh completion timestamp and copies any matching `/tmp` logs.** Read existing audit records if no new verification is needed; do not call it assuming it is purely read-only.

### Training and resume reference

The completed run does not need these commands. They document how it was orchestrated and what a deliberately resumed run would use:

```bash
$FLOWM_PY experiments/continuous_velocity/paper/campaign.py --tag paper_v1 --device cuda:1 --stage all
$FLOWM_PY experiments/continuous_velocity/paper/additional_seeds.py
$FLOWM_PY experiments/continuous_velocity/paper/evaluate_campaign.py --tag paper_v1 --device cuda:1
```

Before a new scientific experiment, choose a new tag/run ID and isolate changed sources. **A new tag alone does not make every script generic:** `additional_seeds.py` hardcodes `paper_v1` and defaults to GPU 1; its configuration must be adapted for a new campaign. Inspect job planning and report assumptions too. Do not invoke it for a new tag unchanged.

Training refuses run-ID reuse with changed immutable configuration/source hashes. It resumes optimizer and sampler RNG from `last.pt`. If an interrupted extended run has advanced beyond 5,000 updates, resume `campaign.py --stage extended`; requesting the smaller budget correctly fails. After finishing, `--stage all` validates/reuses completed artifacts and publishes the complete main marker. Avoid multiple workers training or overwriting the same run ID.

Evaluation can watch checkpoints with `--watch`. Two workers may use `--shard-count 2 --shard-index 0` and `--shard-index 1`; stable SHA-256 partitioning assigns disjoint `(job,suite)` pairs. Shard 0 publishes global completion only after all planned artifacts and both training completion markers exist. File-lock presence alone does not mean a worker is active; process-held locks do. Do not bypass a live lock.

Evaluation resumes by completed test scene seed and rejects changed source/checkpoint signatures. The signature also includes the device string and seed count: changing `cuda:1` to `cpu`, `cuda:0`, or a renumbered visible device under the same job can invalidate reuse. Even a cached evaluation startup constructs the model on its requested device before validation; use manifests for status checks instead of relaunching the evaluator. Training resume starts at the last saved validation checkpoint, not necessarily the last printed training step.

Loaded-module hashes are captured at process start and checked against disk. **Never edit evaluation/training dependencies while workers are running.** Restart processes and use a separate output namespace after changing semantics. Do not disable guards to make incompatible artifacts appear complete.

When resuming old frozen FloWM work, note that `fractional_velocity/run.py` defaults to `cuda:0` and the existing main output directory. Supply an approved device and a new output path; do not casually overwrite the original sweep. The 289-channel partial run should only be resumed if it answers an explicitly chosen research question.

## Kinetic extension (campaign kinetic_v1, 2026-10-22)

A follow-up study integrated two kinetic-world-models components (spectral
Fourier-phase transport; Newton sub-pixel refinement of each pair displacement)
and added a gated second-order (acceleration) motion fit, as an isolated package
`experiments/continuous_velocity/kinetic/` that subclasses the parent model
without editing it. See its `README.md` (method, protocol, approximations) and
`RESULTS.md` (all numbers). Artifacts: `outputs/continuous_velocity/kinetic_v1`
(protocol, selected runs, 742 evaluation manifests, report, figures) and
`downloaded_checkpoints/continuous_velocity/kinetic_v1` (119 training runs).
Headline: Newton refinement halves bilinear MSE (0.00108 to 0.00054, five fresh
seeds, Holm p 3e-4, confirmed on unused scenes 32-63); spectral transport alone
does nothing after training; the gated order-2 fit costs about +0.0003 MSE on
constant velocity and cuts MSE five-fold under acceleration (0.0205 to 0.0040,
success 58% to 94%); nothing helps nearest rendering. Reviews (two adversarial
passes) changed the design before training: no fixed ridge on acceleration,
evidence-based gate; spectral alignment uses a bilinear-sampled tapered support.
The campaign driver script lived in a session scratchpad; the stages are
re-runnable from `kinetic/campaign.py` (resumable; it refuses changed sources).

## Research boundaries and suggested continuation

The building blocks have prior art. [literature.md](experiments/continuous_velocity/paper/literature.md) contains primary-source notes covering FloWM, GMFlow, Motionformer, DNA/CDNA, SNA, STN, LIIF, D-NeRF, continuous-time/physical video prediction, and PhysStream. The prior review was targeted, not exhaustive. FloWM itself names continuous velocity families as future work. Do not claim invention of continuous velocity, soft correspondence, warping, or a new physical theorem; recheck literature before a novelty statement.

The most defensible current claim is a specific observation-conditioned estimator/alignment/transport design and its controlled empirical characterization. Negative attention/strictness ablations and rendering failures are part of that characterization.

Useful next experiments, in priority order, are:

1. **Renderer-robust estimation:** predefine a train/validation protocol for temporal lag selection or multi-lag attention, preserve exact pairing, and test nearest/bilinear/mixed conditions. Diagnose velocity and image outcomes together; do not tune against the existing heldout test results.
2. **Stronger estimator controls:** compare initialization-matched learned regression, better-tuned classical subpixel registration, and robust fitting with matched downstream modules. Current MLP weakness does not establish an expressivity limit.
3. **Replicate the clean interventions:** transport-only quantization and 20k optimization findings over fresh training seeds. Keep fixed-memory identity assertions and budget selection.
4. **Matched-trained FloWM:** train or fine-tune on the same fractional data if claiming architectural superiority. Report the difference in batch-context versus streaming memory settings and actual computational cost.
5. **Broader generalization when justified:** norm-matched heldout directions, controlled geometric visibility, dataset-size curves, then multiple independently moving objects. Implicit appearance fields remain a later ablation rather than the first unresolved bottleneck.

Not performed: a matched-trained FloWM baseline; a full completed 289-channel study; fresh-seed convergence evidence at 20k; training-data-size sweeps; isolated hardware benchmarking; natural-scene testing; acceleration/deformation; unknown-camera robustness; multiple independent motion components; or online recurrent assimilation into the new canonical memory.

Preserve the completed evidence, make the next hypothesis explicit, and carry the renderer failures and neutral metrics forward. A narrower supported result is preferable to attributing all improvements to attention or to velocity discretization alone.
