# Independent protocol audit

Audit date: 2026-10-13. Scope: `paper/train.py`, `paper/campaign.py`, the data
sampler, mechanism controls, and the metric interfaces available at review.
This document records the initial code/protocol review and its final disposition.
It incorporates `additional_protocol.json`, `additional_seeds.py`, and metric
schema 2. The campaign is now complete and independently artifact-validated:
44 training runs, 280,000 optimizer updates, 106 evaluation jobs, 41,408 model
sequences, and 101,152 metric rows. The full 43-test suite passed. Training-run
counts include tuning alternatives and extended-budget runs; they are not a
count of independent post-HPO seeds.

See the [validated completion record](../../../outputs/continuous_velocity/paper_v1/completion.json)
and [final research findings](../../../outputs/continuous_velocity/paper_v1/research_findings.md).
The reviewer did not edit frozen training, model or evaluation sources or launch
GPU training. Historical issues below are labeled as corrected or retained
limitations; the MLP 20k extension and all planned evaluations have completed.

## What this campaign can establish

The completed campaign supplies evidence for the following questions. A
completed comparison does not imply that its proposed improvement was confirmed:

1. Whether template-free, continuous-motion prediction is reliable across
   multiple optimization seeds in the specified single-object, rigid,
   constant-velocity, periodic-world environment.
2. Whether learned correspondence helps beyond fixed raw-pixel matching, using
   both a fully nonlearned baseline and a fixed-checkpoint matcher swap that
   retains the learned memory and decoder.
3. Whether an analytic weighted fit is more effective at a stated optimization
   budget than a learned velocity regressor receiving the same correspondence
   observations.
4. Whether learned memory fusion improves over visibility-weighted averaging.
5. Whether quantizing the same inferred velocity degrades the same decoder's
   forecasts. The `scope='transport'` intervention keeps canonical memory
   unchanged; `scope='both'` also measures damage to context alignment. Oracle
   velocity versions are separately labeled mechanism diagnostics.
6. Whether strict physical transport differs from physically initialized
   transport with learned corrections. A null difference is an informative
   outcome, not evidence for strict-constraint superiority.
7. Whether the method transfers across rendering rules, sampled timestamps,
   larger forecast horizons, held-out motion directions, and the explicitly
   tested new shapes. Conclusions must be separated by training setting.

The core fixed-budget comparison uses 5,000 updates, batch size 8, and two
learning rates selected using 64 validation sequences. Report that definition
of budget, parameter counts, and checkpoint selection rules. Updates and
examples seen are controlled; FLOPs and elapsed shared-GPU time are not equal.

The complete core design includes tuning seed 2026 plus fresh post-selection
seeds 2027/2028/2029 for every model. Physical and relaxed additionally include
2030, giving five total fits and four post-HPO fits each. The extra 2029 fits for
offset, MLP velocity, mean memory and raw correlation are specified in
`additional_protocol.json`; the original `protocol.json` alone does not describe
the full amended design. Primary image outcomes use metric schema 2: image MSE
and per-frame periodic localization success. Unwrapped continuity and internal
transport are distinct diagnostics, not interchangeable versions of that score.

## Reviewer-facing claim matrix

| Claim level | Defensible statement | Evidence needed or boundary |
| --- | --- | --- |
| Design property | One continuous velocity controls context alignment and direct future queries; no fixed velocity-channel bank is instantiated | Implemented equations and unit checks; correspondence still searches a finite pixel-displacement lattice |
| Design property | The physical transition composes additively in elapsed time and keeps canonical memory fixed | Algebraic property; neither exact rasterization composition nor prediction accuracy follows automatically |
| Empirical question; completed evidence available | Which learned components add value beyond their controls? | Mean fusion matches the main model on the core image outcomes, so memory attention is not established as necessary; interpret matching/fitting controls separately |
| Empirical question; completed evidence available | Does velocity quantization degrade forecasts? | The controlled interventions show clear loss at held-out fractional speeds; transport-only comparisons hold canonical memory fixed |
| Empirical question; completed evidence available | Does strict constant velocity improve over learned corrections or guarantee transfer? | Strict versus relaxed has no clear primary image advantage; bilinear-to-nearest transfer remains a failure case |
| Conditional optimization observation | Offset-only fitting is sensitive to learning rate at the fixed budget | Completed HPO validation records; this is not a claim about final test ranking or convergence |
| Unsupported extension | Analytic fitting is universally superior to MLP regression, or the MLP has reached its best attainable performance | The 20k MLP diagnostic is complete but uses one tuning seed and a learning rate selected at 5k; broader optimization and architecture choices remain untested |
| Unsupported extension | The method establishes matched-trained superiority to FloWM or a general multi-object world model | Requires trained FloWM controls and additional state structure/tasks |

No empirical superiority claim is implied merely by the listed design
properties. Oracle velocity is an intervention diagnostic, not an image-error
bound; perfect-ground-truth image predictions are a separate evaluator-ceiling
control.

## Claims this campaign cannot establish by itself

- Architecture superiority to FloWM after comparable fractional-motion
  training: current FloWM references are frozen checkpoints. A matched-trained
  or fine-tuned FloWM baseline is still required for that stronger claim.
- Discretization as FloWM's sole failure cause. Single-trajectory quantization
  interventions do not reproduce channel competition, recurrent dynamics, or
  channel-wise max readout.
- That attention is essential merely because the full system works. The
  parameter-free baseline, learned/raw matcher comparison and mean-fusion
  control must determine the contribution actually supported by results.
- That 20,000 updates establish convergence. The extension uses one optimization
  seed and a learning rate chosen at the 5,000-update budget. It diagnoses
  longer-budget behavior, without exhaustive long-budget hyperparameter tuning.
- Data efficiency across training-set sizes. This campaign keeps the available
  MNIST identity pool at 55,000; learning curves establish budget dependence,
  not the effect of smaller training datasets.
- General multi-object world modeling, acceleration, interaction, deformation,
  unknown camera motion, or unrestricted natural appearances.
- Benefit from an implicit appearance field, which is absent from this model.
- Novelty of the entire combination without a sufficiently broad related-work
  comparison. Prior motion-compensation, video prediction, state estimation and
  structured-memory methods remain relevant.

## Verified separation of information

`network_input` forwards exactly context images, context cameras, context
timestamps, target cameras and target timestamps. Velocity labels appear only
in diagnostic logging and validation diagnostics; checkpoint/HPO selection uses
weighted pixel loss. No target position, digit template or true velocity enters
the formal model call or training objective.

The data generator necessarily uses motion/appearance parameters to render
images. That is simulator construction, not model label leakage. Future camera
positions are supplied controls and must be explicitly stated as known inputs.

Training uses official MNIST training identities 0:55,000 and released camera
episodes 64:224. Validation uses training identities 55,000:60,000 and cameras
224:256. Test uses official MNIST test identities and cameras 0:64. Horizontal
training and validation exclude the declared intervals around held-out speeds;
the 2-D sampler excludes the declared angular wedges. The 2-D held-out-direction
claim is different from horizontal held-out-speed interpolation.

For a fixed optimization seed and batch size, model variants receive the same
NumPy-generated training stream. Rendering-mode choice consumes a draw in all
rendering settings, preserving paired scene parameters when changing rendering.
Validation has a separate fixed RNG seed and its generation does not consume
the continuing training RNG stream. Within each test seed, appearances,
positions and camera paths are paired across velocity/rendering conditions.

## Review findings and disposition

### Fixed-budget HPO after an extension — corrected in campaign code

Initially, rerunning the core stage after a selected run was extended to 20,000
updates read that run's new `best_validation_loss` and compared it against the
other learning rate's 5,000-update loss. This mixed budgets in HPO. The reviewed
correction explicitly reads `budget_5000.pt['best']` for each tuning candidate.
Core evaluation must likewise load `budget_5000.pt`, not the mutable `best.pt`.

`budget_5000.pt` means the best validation-selected checkpoint available *within*
5,000 updates, potentially including initialization; it is not necessarily the
model at exactly update 5,000. Report both the allowed budget and selected step.

### Completed-cache and resume validation — corrected and retested

At the first audit, a completed manifest returned before config or source checks.
A CPU-only reproduction with temporary mocked output paths confirmed that a
manifest for `physical`, seed 2026, was returned for an `offset`, seed 2030 request
using the same run ID. This could silently reuse the wrong run.

The corrected trainer compares immutable configuration and source hashes before
any early return and includes batch size and validation size in resume checks.
A repeated CPU-only temporary-path test verified rejection of changes to kind,
seed, batch size, validation size and source hash. It now explicitly rejects an
incomplete restored run whose step exceeds the requested budget, avoiding the
empty-loop bug. To resume such an extension, use the extended campaign stage;
read the existing 5,000-update artifact for fixed-budget evaluation.

The initial partial training run was archived separately as a pre-audit aborted
run, and the campaign restarted from the reviewed implementation. That pre-audit
partial run is excluded from the validated primary evidence.

Operational limitation: completed manifests do not themselves guarantee every
checkpoint file still exists. The campaign explicitly loads the required budget
file and therefore fails visibly if it is missing, rather than silently using a
different checkpoint.

### Source identity and persistence

The trainer snapshots model, sampler, mechanism and trainer files and rejects
changes on resume. Also snapshot the campaign and evaluation protocols. Freeze
training sources before starting the long-lived campaign: imported Python code
can otherwise differ from a later on-disk file hashed for a new run. A snapshot
is provenance, not proof that future edits are harmless.

The corrected trainer writes its main checkpoints and manifest JSON to a
temporary file followed by atomic replacement. Campaign selection JSON now also
uses atomic replacement. Budget checkpoint copies are not atomic; an
interruption may require regenerating those derived artifacts. NumPy
sampler state is saved; the current networks have no stochastic train-time
dropout. If stochastic PyTorch operations are introduced, save/restore their RNG
states too. GPU determinism is not guaranteed merely by fixing initialization
and data seeds.

### HPO seed versus independent replicates

Seed 2026 chooses each model's learning rate and is included among reported
optimization seeds. This is not test-data leakage, but it makes that fit
conditional on a tuning decision. Identify it as the tuning replicate and
report it separately. Primary core stability claims should use post-selection
seeds 2027/2028/2029, plus 2030 for physical and relaxed. A supplementary all-seed
summary may include 2026 with that qualification. Rendering/2-D training
protocols currently retain seeds 2026/2027/2028 and therefore have two fresh
post-selection seeds, not three.
Do not describe a test-sample bootstrap as uncertainty over training seeds.

Physical and relaxed completed five selected fits (four post-HPO); other core
controls completed four selected fits (three post-HPO), including the
additional-seed worker.
Compare models using their common optimization seeds for paired differences.
The extra physical/relaxed fits improve that particular comparison, not the
precision of every ablation. Tables count completed evaluated fits, separately
from the discarded alternative learning-rate runs.

### Optimizer sensitivity and the completed longer-budget diagnostic

The completed tuning-seed records in `selected_runs.json` show offset-only best
weighted validation loss of 0.078321 at learning rate 0.0003 and 0.004436 at 0.001,
both within 5,000 updates: approximately a 17.7-fold reduction. This is a
validation-only optimization observation. It invalidates treating the weak
original pilot offset result as evidence that unconstrained offsets cannot solve
the task, and requires using the tuned baseline in subsequent comparisons.

The selected MLP-velocity tuning run completed its 20,000-update diagnostic;
its final manifest records that budget. The extension uses seed 2026 and the
learning rate selected at 5,000 updates. Neither a convergence claim nor a claim
that analytic fitting is indispensable follows from this finite optimization
budget. Its completed longer-budget result is reported separately from the
fixed-budget table, alongside the corresponding physical, relaxed and offset
extensions.

### Generalization interpretation

Rendering and 2-D variants reuse learning rates selected in horizontal bilinear
training. This is a valid fixed transfer protocol if disclosed; it is not equal
per-domain hyperparameter optimization. The 2-D MLP control uses paired scenes
and the same update budget, but any negative result remains conditional on this
tuning procedure.

The released camera action stream has 200 actions. Forecasts through time 209
replay the beginning of that episode's actions after time 199 before cumulative
integration. This preserves integer actions and pairing, but introduces a seam
that is not a newly sampled continuation of the original policy.

The final-context observation blackout removes observations. It is not the same
as a digit geometrically leaving and re-entering the camera field of view. Name
these experiments separately. The estimator uses a bounded periodic displacement
search; timestamp gaps permitting displacement at least half the world width
require winding disambiguation and fall outside its current guaranteed regime.

### Metric schema 2 and evaluator validity

The original CV-fitted joint score could reject correct frames and preferentially
reject nonlinear offset predictions. `metrics_revision.json` documents the
revision; old artifacts are archived under `evaluation_pre_review_v1` and
`pre_review_v1_report`. Training is unchanged. All models must be reevaluated
with schema 2 and compatible evaluation-source hashes; do not merge old and new
rows in a primary comparison.

- **Image MSE** scores every forecast pixel and requires no detector or motion
  fit.
- **Frame-localization success** (`frame_joint_success`) counts visible target
  frames with a present, non-ghost detection whose periodic position error is at
  most 2 pixels. It uses no velocity fit. Its denominator includes missing and
  duplicated detections as failures. It does not identify unwrapped motion:
  whole-world winding errors can have zero periodic localization error.
- **Continuity ADE and coverage** (`continuity_*`) unwrap accepted detections
  from one context anchor using a per-axis 2.5 px/time speed bound and 1-pixel
  localization slack. They use no constant-velocity fit. A gap permitting
  half-world travel, or an inconsistent/ambiguous increment, permanently marks
  winding unknown. There is no later GT reanchoring. Low coverage can therefore
  be an identifiability/evaluator limitation; show ambiguity and the perfect-frame
  ceiling alongside conditional errors.
- **Internal transport ADE** compares the model's displacement output against
  true accumulated displacement. It measures unwrapped motion-law agreement,
  not generated-image object-position accuracy; it omits appearance and initial
  position error. It cannot replace the two primary image outcomes.
- **CV-certified diagnostics** (`certified_*`, plus legacy `ade`, `coverage`,
  `joint_success`) retain the old constant-velocity fitting assumption. They are
  secondary diagnostics and must not silently be relabeled model-neutral joint
  success.

Giving true velocity to a trained checkpoint is not guaranteed to minimize its
image error. The learned memory/decoder can compensate for systematic estimated
velocity bias, and an oracle swap changes their input distribution. Use oracle
runs to investigate error sources, not as a guaranteed upper bound on image
quality. Evaluating exact GT frames instead measures detector/evaluator ceilings
and should be labeled separately.

## Reporting and decision rules

1. Finish the predeclared training budgets and select checkpoints using
   validation only. Keep fixed-budget and extended-budget tables separate.
2. Report per-training-seed results, post-selection cross-seed mean and SD, and
   paired test-case uncertainty separately. Show the tuning seed separately.
3. Lead with unconditional image MSE and schema-2 `frame_joint_success`. Also
   report missing/detected-ghost rates, frame coverage, continuity error and
   coverage/ambiguity, and internal transport error. Explicitly name periodic
   localization versus unwrapped motion; neither substitutes for the other.
4. Preserve the same ratios-of-sums estimator in summary tables and bootstraps.
   Cluster paired test sequences by seed so shared digit/camera paths across
   speeds are not treated as independent samples.
5. Distinguish bounded-continuity unwrapping from the secondary CV-certified
   diagnostic. Only the latter fits a constant-velocity line. Both need coverage
   and evaluator ceilings; neither should selectively define the primary image
   success rate.
6. The detector uses the true appearance template for evaluation, including for
   all template-free predictors. Calibration has not shown perfect sensitivity
   to overlapping or fragmented ghosts. Write “no ghosts detected” when the
   measured rate is zero, rather than “no ghosts.”
7. Report negative findings. If physical and relaxed remain equivalent, claim
   benefit of the shared continuous-motion structure/initialization only to the
   extent supported. If fixed raw matching is competitive, center the argument
   on architecture and conditions where learning adds measurable value.

## Small supplementary diagnosis, not a campaign result

A read-only CPU check on 16 paired test seeds and speeds 0.37, 0.63 and 1.23
compared the existing pilot checkpoint's velocity estimates. Bilinear vx MAE
was 0.00505 for learned matching and 0.00762 for fixed raw matching. On nearest
rendering the corresponding errors were 0.05175 and 0.03927. Learned matching
with ordinary rather than robust fitting yielded 0.04871 on nearest.

These initial results motivated the renderer-by-matcher interventions that have
now completed.
The swap changes matching descriptors, temperature and confidence together, so
it does not identify which learned component specializes. Both matchers remain
affected by nearest-rendering displacement quantization. This historical small
diagnostic does not replace the completed multi-seed rendering experiment.

## Reviewed source identity

The final source check accepted the following SHA-256 values before the restart:

| File | SHA-256 |
| --- | --- |
| `model.py` | `08caa42abd1fae0f47fc1c97dbea96bef2a1bbb0236c7d4f6359e70393666266` |
| `data.py` | `b5cd971595312e29dd150cfea3c78f6301d2caf45ce4e1e24629ee7a4045cf58` |
| `paper/data.py` | `a4646c5d08eb58907ef6a7f1f46ed261531da289d0bef3d69848d567de0ac78c` |
| `paper/mechanism.py` | `ad7212315989215ce2900e02c8c103f7ff55cef9a25f455474532f39298cabb0` |
| `paper/train.py` | `587cdebee0f5b454a856d91bf585ccdff21ad4e0333c33aebcfbabf6b6becbb6` |
