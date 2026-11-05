# Released FloWM-3D: kinetic stage-one CPU tests

Run from the repository root with the supplied environment. The wrapper uses
`/home/yixunhu/miniconda3/envs/flowm/bin/python`, hides CUDA, sets OMP/MKL/OpenBLAS
to eight threads and W&B offline, and puts temporary/cache files in `.scratch/`.
It imports `k3d` directly to avoid the `experiments` package's eager application
imports before CPU/cache setup. It never constructs a writable NAS dataset index.

`bash experiments/flowm_bench/k3d/run_cpu.sh run_stage1` runs the complete sequence
and records command wall times. The individual commands are:

```bash
bash experiments/flowm_bench/k3d/run_cpu.sh released_sanity
bash experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_int_validation --clips 20
bash experiments/flowm_bench/k3d/run_cpu.sh geometry --clips 20
bash experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_v_validation --clips 200
bash experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_null_noblocks --clips 50
bash experiments/flowm_bench/k3d/run_cpu.sh lift --split k3d_null_static --clips 50
bash experiments/flowm_bench/k3d/run_cpu.sh probe_subcell --clips 200
bash experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_null_noblocks --clips 50
bash experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_null_static --clips 50
bash experiments/flowm_bench/k3d/run_cpu.sh null_test --split k3d_int_validation --clips 20
bash experiments/flowm_bench/k3d/run_cpu.sh estimator_gate --clips 200
bash experiments/flowm_bench/k3d/run_cpu.sh report
```

The default probe and estimator runs include `tokens` (written/gated) and
`h_new` (pre-gate). The gate runs σ=0.5,1,2 and K=8,32; nulls use K=8.
`--variants tokens`, `--ks 8`, `--sigmas 1`, and `--clips N` support explicit
smaller runs. Clips always retain 140 frames. `--score-only` recomputes CSVs
from trajectories. A smaller scoring run overwrites that task's aggregate CSVs;
rerun the full scoring command to restore full results.

Lifts and trajectories are written atomically per clip and reused on restart.
The cache is specific to this checkpoint, geometry, implementation and precision;
remove the relevant files under `.scratch/lifts` or `.scratch/trajectories` when
changing these. `.scratch` should be gitignored by the maintainer. No checkpoint,
dataset, dependency or application configuration is modified by these programs.

## Protocol details

- Source clips are sorted numerically by shard and index; all start at frame 0.
  `ReadOnlyClips` uses the original dataset decoding, transformation and aligned
  integer actions. Stored action t leads from frame t to t+1; aligned cond t is
  stored action t−1. Codes 0/1 turn left/right, 2 moves forward, 4 is no-op.
- Observation lift calls the unmodified map processor with a zero map at every
  frame. A temporary gate hook captures g and h_new. Equality across all five
  velocity channels is asserted before retaining one copy. B=32, fp32 compute,
  fp16 storage. The fixed KWM projection uses its default seed 1234.
- Geometry: forward is +column, right is +row, origin (16,16), box (7,16), shape
  19×17, 177-cell support. `geometry.csv` compares actual write/gate activity at
  both candidate signs across 20 integer clips, including per-cell centering.
- Probe uses five clip-grouped outer folds, an inner grouped 20% validation split,
  train-only feature standardization, and ridge α∈{1,100,10000}. Positive controls
  use integer (column,row) relative to the box corner and one-hot colour targets.
  All fold membership and predictions are retained in scratch NPZs.
  Ridge solves use float64 to avoid conditioning warnings from correlated features;
  lift and estimator computations remain fp32.
- The v17 tracker imports the production masked-box `measure_nodes`. It carries
  rho and evidence exactly as the 2D lock cell at s=1 (gain=e/W, component step
  bound .3, rho bound .75), pins static metadata and rotates node identities and
  offsets on turns. It never estimates motion from the nearly disjoint turn cones.
- `agreement_diagnostic` observes the first production coherence call and obtains
  the lock's normalized evidence (LK weight divided by textured transported mass).
  Agreement≥.1 is the lock-evidence threshold. Full lock additionally requires
  pooled LK residual norm≤.3. The production API does not expose the agreement;
  `.scratch/kwm_patch.diff` proposes an optional return field. It is not applied.
  The local diagnostic wrapper is the workaround used by this run.
- Null alarms use immediate measured velocities; gate matches updated recurrent
  velocities of locked, evidenced nodes. Null denominators include all 17 nodes;
  stricter 16-node rates are also exported. The evidence-weighted speed uses raw
  selected evidence, without discarding unlocked nodes.
- The gate uses SciPy Hungarian matching, all jointly visible blocks before speed
  filtering, Euclidean distances, lower median, and the requested miss threshold
  .5. Unassigned truths count as misses; assigned errors above .5 still enter
  matched statistics. False alarms are active nodes without any truth within .5.
  The overall and criterion-band FA rates count all active nodes. The three
  narrow speed-bin FA rates bin by node speed. Duplicates near one truth are not false
  alarms under this definition, but cannot both match that truth.
- The supplied frame range 1..139 gives **139** pairs, not 138. The delayed score
  uses 138 pairs (1..138), comparing estimate t against labels t+1. Turn scores
  use carried, rotated estimates and both-time visibility, separately from criteria.
  Delayed rows are labelled turn if either the estimate or target transition turns.
- The generator's `block_vel_agent[t]` is outgoing and camera-relative. It differs
  from KWM's intrinsic incoming estimate. Literal requested results remain primary;
  the separate `intrinsic_incoming` diagnostic transforms incoming world block
  displacement into current heading, without camera subtraction. Literal speed
  bins use literal target norm; corrected bins use incoming nominal block speed.
  Reflections are retained. Labels only enter metrics/probes and geometry checks.

## Results and runtime

See `../docs/K3D_stage1_results.md` for verdicts, every σ/K/speed/turn/delay table,
geometry evidence, commands and deviations. `results/commands.csv` records actual
wall times; `supplemental_commands.csv` covers initial runs and checks with explicit
timing bases. Sanity took 95 s initially; the repeated command took 117 s including
imports. Lifting took 41.8 minutes for all 320 clips. Null scoring/measurement
took 14.4 minutes; the full gate took 63.7 minutes including scoring; the final
probe took 60 s. Every lift and estimator batch also stores its runtime.
The local `.gitignore` makes the requested result CSVs visible despite the root
repository's broad results/CSV exclusions. All durable results are CSV:

- `released_sanity*.csv`, `geometry.csv`, `lift_*.csv`: loading, sign and lift times.
- `probe_*_{summary,folds,clips}.csv`: nested held-out probe and control results.
- `null_*_{summary,clips,distribution}.csv`: rates, counts and velocity quantiles;
  `null_static_raw_velocities.csv` retains every nonturn moving-node measurement.
- `gate_{summary,clips}.csv`: both target conventions, both delays, turns, all speeds
  and bins, all lift/K/σ choices; nearest-integer appears in the summary.
- `verdicts.csv`, `provenance.csv`, `execution_notes.csv`: report audit trail.
- `kwm_revision_parity.csv`: audit of the external KWM checkout update during this
  run. All 144 real pair/configuration checks were bitwise equal on the used path;
  the full measurements used revision `2e73903d7ef5e6268cf6bca3a9703a3b9971ccf9`.

## Fast tests

```bash
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/experiments/flowm_bench/k3d/.scratch"
export XDG_CACHE_HOME="$TMPDIR/cache" MPLCONFIGDIR="$TMPDIR/mpl"
/home/yixunhu/miniconda3/envs/flowm/bin/python -m pytest tests/k3d/test_stage1.py -q \
  -o cache_dir=experiments/flowm_bench/k3d/.scratch/pytest_cache
```

Seven tests cover geometry signs/turns, empirical geometry evidence, dataset
alignment, lift shape and five-channel identity, box placement/projection,
Hungarian corner cases, static pinning, rotation of recurrent state, and bitwise
parity of the agreement observer with the production estimator.
The empirical evidence test skips if `geometry.csv` has not yet been generated.
The final suite passed all seven tests in 11.53 s, including released-weight loading.
The log is `.scratch/pytest_final.log`.

## Stage-one follow-up (2026-11-04)

`bash experiments/flowm_bench/k3d/run_cpu.sh run_followup` runs the mask rerun,
fast tests, alarm breakdown, direct patch search and memory diagnostic sequentially.
It records status and wall times in `results/followup_commands.csv`; successful
stages and atomically saved per-clip caches are reused on restart. The initial
campaign started `followup mask` separately and used `run_followup --after-mask`
to wait for that stage before continuing. Remove a stage's successful command row
if intentionally repeating it. Source/configuration changes require new cache
versions or removal of the affected follow-up cache; original caches are retained.

Individual commands, in delivery order:

```bash
bash experiments/flowm_bench/k3d/run_cpu.sh followup mask
bash experiments/flowm_bench/k3d/run_cpu.sh alarm_breakdown
bash experiments/flowm_bench/k3d/run_cpu.sh patch_motion
bash experiments/flowm_bench/k3d/run_cpu.sh memory_lift
```

The results document is **append-only** for this follow-up. Each completed
split/configuration appends its numbers immediately, guarded by a unique HTML
marker to avoid duplicate sections on resume. All new CSVs start with `followup_`.
The original report generator is not run because it would replace the document.

- `followup.py`: K=8 observation-only `tokens`, mask residual, sigma .5/1;
  50+50 null clips, 20 integer contrast clips, 200 continuous-velocity clips.
  Nonturn `intrinsic_incoming` is primary and `literal` secondary. Keeps the
  original recurrent update, Hungarian matching and null alarm definition.
  Calls production diagnostics on every measured pair and saves all 17 nodes'
  pre-selection and selected evidence, immediate estimates, prior and recurrent
  velocities, locks and actions under `.scratch/followup_trajectories/`.
  The all-node CSV tables include fixed-lattice and pre-update adaptive nearest
  nodes. A wrong node is farther than .5 from every visible truth. Infinite
  ratios from zero true-node evidence and undefined 0/0 cases are counted;
  no evidence floor is silently added to make those ratios finite.
- `alarm_breakdown.py`: empty-room alarms and K3D-v locked/evidenced nodes,
  subtract/mask, sigma 1, full/near/far support. Reuses original full-support
  subtract trajectories. Near/far rerun the recurrence with cone cells at x<=6
  or x>=7. Nonturn action groups are exclusive; turn-adjacent means either
  neighboring pair is a turn, and takes precedence over forward/no-op. Actual
  turns have no fresh measurement. Includes per-clip counts, full 2D velocity
  histograms, radius-.25 mass near (+1,0), and an independent same-cell versus
  ego-transported imprint SSD check on empty-room forward pairs.
- `patch_motion.py`: oracle previous block location, uniform 3x3 SSD on a .5-cell
  grid with axiswise parabolic refinement. Uses exact Fourier interpolation on
  the same zero-filled 33x33 torus as LocalSearch, evaluated efficiently on a
  half-cell grid. Search is ±1.5 **intrinsic** cells around the ego-transported
  previous centre; output therefore already includes compensation for camera
  translation. Shared bilinear cone support normalizes SSD; at least 4.5 of 9
  cells must be valid. Boundary minima remain in the bounded search results.
  Invalid pairs remain in denominators with infinite error. All jointly visible
  blocks/nonturn pairs in 200 clips are retained, ignoring occlusion as before.
  Tokens/h_new at K=8/32 and raw tokens at K=256 use the same pairs. Pair and
  per-clip CSVs permit alternative aggregation; diagnostics are not a universal
  mathematical upper bound on learned or temporally extended estimators.
- `memory_lift.py`: mirrors the released forward's context loop, including
  velocity flow, ego roll, learned cross-velocity mixing and the gated write.
  Saves **post-blend map cone tokens**, max over V=5, before decoder positional
  encoding; no decoder or prediction call. The public model forward cannot
  return a zero-length prediction sequence, so this local adapter is checked
  against hooks on its actual context loop. Caches 140 frames per clip under
  `.scratch/lifts_mem/`, then runs mask sigma=1/K=8 on 50+50 null and 100 K3D-v
  clips, appending each result as soon as that split finishes. This is a memory
  diagnostic, explicitly distinct from the design's observation-only lift.

The follow-up also includes `stationary_attribution.py`, added after whole-region
SSD failed to establish global far-field stationarity. It replays every empty-room
forward pair at the cached subtraction prior and observes the production selected
pixel weights without changing the estimator return. Same-cell/ego SSD is then
compared at the cells carrying alarm evidence. Reconstructed estimates are checked
against the original cache. This runs once at the start of `memory_lift` (before
loading the model), sequentially with the other eight-thread stages; its runtime
has a separate CSV. It can also be run with `run_cpu.sh stationary_attribution`.

The extended tests additionally check fractional single-patch motion with and
without ego translation, Fourier interpolation, empty-support handling, alarm
count/denominator partitions, multi-object evidence ratios, and exact fp16 parity
of the context-only memory adapter with the released model's context writes.
Use the existing fast-test command above. The follow-up test log is
`.scratch/followup_tests.log`; final measured runtime and verdicts are in the
dated Follow-up section of `../docs/K3D_stage1_results.md`.

Completed follow-up: all requested samples, 1,840 measured trajectories,
200 memory clips, and five runs over the same 44,990 block/pair examples.
Measured command wall time was 71.98 minutes; final tests passed 11/11 in
26.76 seconds. Both mask widths and the memory variant fail the null/gate
criteria. Raw-token patch motion also remains weak. Exact results and the
evidence-localized cone-stationarity diagnosis are in the appended report.

The external KWM checkout changed from `2e73903` to `c7067d3` during execution.
`run_cpu.sh followup_dependency_audit` compares scratch-loaded git snapshots on
observation and memory pairs: all outputs and diagnostics were bitwise identical
in 216 pair/configuration cases for the used default mask-key path. Its results
and final provenance are retained in `followup_kwm_*.csv` and
`followup_provenance_final.csv`. No KWM file was edited by this experiment.
