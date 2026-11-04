"""Render the stage-one report from CSV measurements (never hand-enter scores)."""
import csv
from pathlib import Path
from collections import defaultdict
import math
from .common import ROOT, RESULTS, clips, write_csv


def read(name):
    with (RESULTS/name).open() as f:
        return list(csv.DictReader(f))


def number(value, percent=False):
    try:
        x = float(value)
        if not math.isfinite(x):
            return "—"
        return f"{100*x:.2f}%" if percent else f"{x:.5f}"
    except (ValueError, TypeError):
        return str(value)


def table(rows, columns):
    lines = ["| " + " | ".join(title for _,title in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "—")
            if key.endswith("rate") or key in ("color_accuracy", "color_balanced_accuracy"):
                value = number(value, True)
            elif key in ("median", "p90", "r2", "shuffled_r2", "integer_r2", "mse", "raw_mse",
                         "turn_mse", "nonturn_mse", "evidence_weighted_speed"):
                value = number(value)
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    gate = read("gate_summary.csv")
    primary = [r for r in gate if r["variant"] == "tokens" and r["truth_mode"] == "literal"
               and r["delay"] == "0" and r["step_type"] == "nonturn" and r["speed_bin"] == "0.25-1.0"]
    default = [r for r in primary if r["k"] == "8"]
    def rank(row):
        med = float(row["median"])
        return row["passed"] != "True", med if math.isfinite(med) else math.inf, float(row["miss_rate"])
    best = min(default, key=rank)
    best_sigma = float(best["sigma"])
    probe = read("probe_tokens_summary.csv") + read("probe_h_new_summary.csv")
    pmain = [r for r in probe if r["variant"] == "tokens" and r["distance"] == "all" and r["step_type"] == "all"]
    null = sum((read(f"null_{split}_summary.csv") for split in
                ("k3d_null_noblocks", "k3d_null_static", "k3d_int_validation")), [])
    nmain = [r for r in null if r["variant"] == "tokens" and float(r["sigma"]) == best_sigma and r["split"].startswith("k3d_null")]
    sanity = read("released_sanity_summary.csv")[0]
    verdicts = [dict(test="sub-cell probe", verdict="PASS" if all(r["passed"] == "True" for r in pmain) else "FAIL",
                     criterion="R² ≥ .25 on both axes and ≥ 5 × shuffled R²",
                     measurement="; ".join(f"{r['axis']} R²={number(r['r2'])}, shuffled={number(r['shuffled_r2'])}" for r in pmain)),
                dict(test="background null", verdict="PASS" if all(r["passed"] == "True" for r in nmain) else "FAIL",
                     criterion="Alarm ≤ 2% and evidence-weighted speed ≤ .05 on both null splits",
                     measurement=f"σ={best_sigma:g}; " + "; ".join(f"{r['split']}: {number(r['alarm_rate'],True)}, {number(r['evidence_weighted_speed'])}" for r in nmain)),
                dict(test="estimator gate (K=8)", verdict="PASS" if best["passed"] == "True" else "FAIL",
                     criterion="Matched median ≤ .1 and misses ≤ 30%, speeds .25–1, best σ",
                     measurement=f"σ={best_sigma:g}, median={number(best['median'])}, misses={number(best['miss_rate'],True)}")]
    write_csv("verdicts.csv", verdicts)
    chunks = ["# Kinetic 3D stage 1: released-weight CPU kill tests\n",
              "Measured 2026-11-04. No training and no GPU execution. Results below concern the released observation-only lift; they do not select a subsequent model arm.\n",
              table(verdicts, [("test","Test"),("criterion","Pre-registered criterion"),("measurement","Measurement"),("verdict","Verdict")]),
              "\n## Protocol, conventions and deviations\n",
              "- Checkpoint loaded strictly: 240 `main_model.` tensors, V=5, 256 channels, six map and decoder layers, eight heads, 16-pixel patches, max-pool decode, flattened 2D velocity mixing, no velocity identity embedding.\n"
              "- Numeric shard/index clip order; frame zero start. Read-only dataset adapter calls the existing dataset's `__getitem__`, `/255` decoder, resize, `action_int` loader and `t-1->t` alignment; it bypasses writable NAS metadata indexing. First action is discarded by the model. No NAS data or KWM package files were edited.\n"
              "- Actions in stored data: 0 left, 1 right, 2 forward, 4 no-op. Aligned action at t is stored action t−1. Forward shifts the map by −1 column; left rotates clockwise, right counterclockwise.\n"
              "- The requested range t=1..139 contains **139**, not 138, measurement pairs. All clips retain 140 frames. Delay diagnostics use t=1..138, so truth t+1 remains inside those 140 frames; a delayed pair is labelled turn if either its estimate or target transition turns.\n"
              "- **Velocity metadata mismatch:** generator `block_vel_agent[t]` is outgoing block displacement minus outgoing camera displacement in heading t. KWM returns intrinsic velocity because camera displacement is already subtracted in transport. The literal requested comparison remains primary. `intrinsic_incoming` is a separately labelled diagnostic: `block_vel_world[t−1]` rotated into heading t, without camera subtraction. Its speed bins use incoming nominal `block_speed`; the literal comparison bins by its literal target norm. Reflections can make finite-step displacement differ from instantaneous pre-bounce speed. No labels enter the estimator.\n"
              "- Turns provide no cone-pair measurement. Rotate and permute rho, accumulated evidence and lock metadata using the named lattice's C4 permutation; carry the rotated node estimates into the separately reported turn diagnostic. Both-time visibility is retained, so turn truth counts can be very small. Null turns are skipped and counted.\n"
              "- `measure_nodes` is imported from KWM, as are named nodes, fixed projections and rotation primitives. Hungarian matching uses SciPy, following the 2D harness's Euclidean assignment and lower-median convention. Its fixed five-digit final-frame metric cannot be imported unchanged. The 0.5 miss cutoff and speed bins are this task's overrides, not the 2D gate's 0.1 cutoff.\n"
              "- Lock evidence means normalized LK agreement ≥0.1, as implemented in `lock_nodes.py` and explained in `PR1_lock_nodes.md`, the method linked by `GATE_estimator.md`. The API's raw selected evidence is **not** thresholded at 0.1. A scoped observational wrapper reads the existing coherence calculation to recover agreement; numerical outputs are unchanged. Full lock also requires pooled LK residual norm ≤0.3. The API does not expose agreement: `.scratch/kwm_patch.diff` proposes an optional return field, left unapplied; the local wrapper is the workaround.\n"
              "- Gate candidates are current locked nodes with accumulated selected evidence >1e−6; the pinned static node is excluded. Scores use updated recurrent velocities. Update is exactly s=1, W+=e, gain=e/max(W,1e−6), step=clip(gain·residual,±0.3), rho=clip(rho+step,±0.75). Each clip resets independently. Gaussian aggregation uses radius=ceil(3σ), via the primitive's radius=None option.\n"
              "- Null v̂ is the immediate `measure_nodes` result under that same recurrent tracker. Alarm uses agreement ≥0.1 and |v̂|>0.15. Primary denominator is all 17 node-steps; the stricter 16 moving-node denominator is also shown. Evidence-weighted mean includes selected evidence even on unlocked nodes; the pinned node contributes zero.\n"
              "- The gate uses the same rounded-cell cone and rel_x≥1 visibility definition as the probe. Hungarian assignment is performed once against every jointly visible block; speed bins then filter assigned truths. An unassigned truth is a miss, as is an assigned distance >0.5. Matched statistics include every assigned pair, including >0.5 errors, and exclude unassigned pairs. False alarms count active nodes with no visible truth within 0.5, independently of uniqueness; overall and criterion-band FA denominators include all active nodes; the three narrow speed-bin FA denominators use node speed. A missing denominator is shown as —.\n"
              "- Full 256-channel tokens, gate and pre-gate h_new are cached in fp16; projection and all measurements are fp32. B=32 zero-map writes retain all five velocity copies during attention and assert equality before selecting one. CPU SDPA replaces the materialized attention fast path; model mathematics and weights are unchanged.\n"
              "- Probe: 14 frames per clip (0,10,…,130), visible rounded cells with rel_x≥1, no occlusion filtering, 3×3×256 zero-padded features. Half-open residual convention is p−floor(p+.5). Five outer clip groups and a disjoint inner clip split; StandardScaler and ridge fit only training data; ridge solves use float64 after the initial fp32 fit emitted conditioning warnings. α∈{1,100,10000} is selected separately for sub-cell, integer, colour and shuffled tasks. Colour is one-hot ridge classification. Shuffled training/validation/test labels are independently permuted inside their respective folds. Pooled held-out R² is primary. Negative shuffled R² is retained, not converted into a positive ratio. Distance bins are [1,5), [5,9), [9,13).\n",
              "\n## Released-model sanity\n",
              f"Eight-clip mean MSE **{float(sanity['mse']):.9f}**, 70 context + 70 autoregressive frames, {float(sanity['seconds']):.1f} seconds. The 1000-clip reference is 0.000625; these eight deterministic first clips are a loading/alignment check, not an estimate of the full-split mean. Predictions are clamped to [0,1] for the released metric; raw MSE is also recorded.\n",
              table(read("released_sanity.csv"), [("clip","Clip"),("mse","MSE"),("raw_mse","Raw MSE"),("turn_mse","Turn MSE"),("nonturn_mse","Nonturn MSE"),("turn_frames","Turn frames")]),
              "\n## Map geometry evidence\n",
              "`compute_fov_mask` agrees with the independently called helper: 177 cells, rows 7..25, columns 16..32. Agent at (16,16), forward is +column. Empirical activity supports **row=16+rel_z**, column=16+rel_x. Compare each off-axis block's candidate cell to its mirrored cell over 20 integer clips; cell-centered scores subtract each cell's temporal mean.\n"]
    if (RESULTS/"data_audit.csv").exists():
        audit = read("data_audit.csv")
        audit_rows = []
        for split in sorted({r["split"] for r in audit}):
            rr = [r for r in audit if r["split"] == split]
            audit_rows.append(dict(split=split, clips=len(rr),
                                   **{key:sum(int(r[key]) for r in rr) for key in ("left","right","forward","noop")},
                                   identity_error=f"{max(float(r['relative_velocity_identity_max_error']) for r in rr):.2g}",
                                   forward_error=f"{max(float(r['forward_displacement_max_error']) for r in rr):.2g}"))
        chunks += ["The stored velocity identity was checked directly: `block_vel_agent + outgoing_camera_agent = rotate(block_vel_world, heading_t)`. Counts below are the 139 incoming transitions per clip. No unknown action codes were found.\n",
                   table(audit_rows, [("split","Split"),("clips","Clips"),("left","Left"),("right","Right"),("forward","Forward"),("noop","No-op"),("identity_error","Max velocity identity error"),("forward_error","Max forward-distance error")])]
    geometry = read("geometry.csv")
    geo = []
    for signal in ("tokens", "g"):
        for normalization in ("raw", "cell_centered"):
            rr = [r for r in geometry if r["signal"] == signal and r["normalization"] == normalization]
            n = sum(int(r["blocks"]) for r in rr)
            geo.append(dict(signal=signal, normalization=normalization, n=n,
                            plus=f"{sum(float(r['plus_sum']) for r in rr)/n:.6f}",
                            minus=f"{sum(float(r['minus_sum']) for r in rr)/n:.6f}"))
    chunks += [table(geo, [("signal","Signal"),("normalization","Normalization"),("n","Block observations"),("plus","Plus sign"),("minus","Minus sign")]),
               "\n## Test 1: sub-cell probe\n",
               table(probe, [("variant","Lift"),("step_type","Steps"),("distance","Distance"),("axis","Axis"),("n","N"),("r2","Held-out R²"),("shuffled_r2","Shuffled R²"),("integer_r2","Integer R²"),("color_accuracy","Colour accuracy"),("color_balanced_accuracy","Balanced colour accuracy")]),
               "\n## Test 2: background null\n",
               f"Criterion evaluated at σ={best_sigma:g}, selected from the primary K=8 gate by passing status, then matched median, then miss rate. All widths are shown. K3D-int is a moving-block contrast and is not scored as a null. Null split source comments identify CPU rendering, while K3D-v/int validation uses NVIDIA-rendered source videos; all computation here is CPU.\n",
               table([{**r, "passed": "not a null" if r["split"] == "k3d_int_validation" else r["passed"]} for r in null], [("split","Split"),("variant","Lift"),("sigma","σ"),("clips","Clips"),("node_steps","Node-steps"),("turn_steps","Skipped turns"),("alarm_rate","Alarm /17"),("moving_node_alarm_rate","Alarm /16"),("evidence_weighted_speed","Evidence-weighted speed"),("passed","Null threshold")]),
               "\n### Static-block raw velocity distribution\n",
               "Raw CSVs and quantiles cover the 16 moving nodes; the omitted pinned node is identically zero. Unfiltered estimates include unobserved nodes at their current lattice/offset values. `lock_evidence` selects normalized agreement≥0.1. The full empirical per-node samples are in `null_static_raw_velocities.csv`; quantiles for all three splits are in the distribution CSVs.\n"]
    distribution = read("null_k3d_null_static_distribution.csv")
    dist = defaultdict(dict)
    for r in distribution:
        key = (r["variant"],r["sigma"],r["field"],r["subset"])
        dist[key][r["quantile"]] = number(r["value"])
        dist[key]["n"] = r["n"]
    dd = [dict(variant=k[0],sigma=k[1],field=k[2],subset=k[3],**v) for k,v in dist.items()]
    chunks.append(table(dd, [("variant","Lift"),("sigma","σ"),("field","Variable"),("subset","Subset"),("n","N"),("0","Min"),("0.05","p5"),("0.5","Median"),("0.95","p95"),("1","Max")]))
    chunks += ["\n## Test 3: estimator gate\n",
               "Primary criterion: matched median≤0.1 and misses≤30% for speed 0.25–1.0, nonturn steps, best σ. K=8 is the default; K=32 and h_new are reported as variants. Nearest-integer is an independent per-truth assignment to the five FloWM velocities, without uniqueness or evidence restrictions.\n"]
    criterion_rows = [r for r in gate if r["variant"] != "nearest_integer" and r["truth_mode"] == "literal"
                      and r["delay"] == "0" and r["step_type"] == "nonturn" and r["speed_bin"] == "0.25-1.0"]
    chunks.append(table(criterion_rows, [("variant","Lift"),("k","K"),("sigma","σ"),
                                        ("median","Median"),("p90","p90"),("miss_rate","Miss"),
                                        ("false_alarm_rate","False alarm"),("passed","Criterion met")]))
    for mode in ("literal", "intrinsic_incoming"):
        for delay in ("0", "1"):
            for category in ("nonturn", "turn"):
                rr = [r for r in gate if r["truth_mode"] == mode and r["delay"] == delay and r["step_type"] == category]
                chunks += [f"\n### {mode}; delay {delay}; {category}\n",
                           table(rr, [("variant","Lift/reference"),("k","K"),("sigma","σ"),("speed_bin","Speed bin"),("steps","Steps"),("truths","Truths"),("assigned","Assigned"),("median","Median"),("p90","p90"),("miss_rate","Miss"),("nodes","Nodes"),("false_alarm_rate","False alarm")])]
    chunks += ["\n## Commands, runtime, artifacts and validation\n",
               "Run from the repository root. `run_cpu.sh` fixes the supplied Python interpreter, CPU visibility, eight-thread limits, offline W&B, and scratch/cache paths before importing Python packages. Exact executed commands and their wall times are in `results/commands.csv`.\n"]
    runtime = []
    for split in ("k3d_int_validation", "k3d_v_validation", "k3d_null_noblocks", "k3d_null_static"):
        rr = read(f"lift_{split}.csv")
        runtime.append(dict(stage=f"Original lift: {split}", clips=len(rr),
                            seconds=f"{sum(float(r['seconds']) for r in rr):.2f}"))
    for variant in ("tokens", "h_new"):
        rr = read(f"probe_{variant}_summary.csv")
        runtime.append(dict(stage=f"Nested probe: {variant}", clips=len({r['clip'] for r in read(f'probe_{variant}_clips.csv')}),
                            seconds=f"{max(float(r['seconds']) for r in rr):.2f}"))
    write_csv("runtime.csv",runtime)
    chunks += ["Original lift times below are sums of measured per-clip wall times, including decode and preprocessing. They remain recorded when a resumed command reuses caches; the short cached command times are not the original lift cost. Probe times include feature extraction and cross-validation. Campaign command times below include imports and CSV scoring.\n",
               table(runtime,[("stage","Stage"),("clips","Clips"),("seconds","Measured seconds")])]
    if (RESULTS/"commands.csv").exists():
        chunks.append(table(read("commands.csv"), [("command","Command"),("seconds","Wall seconds"),("status","Exit status")]))
    if (RESULTS/"supplemental_commands.csv").exists():
        chunks += ["Initial runs, calibration and final checks are listed below with explicit timing bases. They are additional to the campaign commands; a sum of these values is not elapsed time because initial calibration briefly overlapped caching. Commands use the CPU/scratch environment shown in README.\n",
                   table(read("supplemental_commands.csv"), [("command","Command"),("seconds","Seconds"),("timing_basis","Timing basis"),("status","Exit status")])]
    if (RESULTS/"kwm_revision_parity.csv").exists():
        parity = read("kwm_revision_parity.csv")
        chunks += ["\n### Dependency provenance audit\n",
                   "KWM changed externally from `0190d250f6d2c6a8bc0ebd77e92bb90ad3301083` to `2e73903d7ef5e6268cf6bca3a9703a3b9971ccf9` during this task. Its final estimator files were last written at 16:39:30 EST, before the first full null process launched at 16:41:41; the commit was recorded at 16:46:45. All final null and gate trajectories therefore used the latter file contents. `_fixed_projection` did not change. The added residual-mask mode was not used: `static_residual=True` remains subtraction. An independent comparison of the original and final measurement code on 12 real nonturn pairs (four each from moving, empty and static clips), random nonzero rho, both lift variants, both K and all σ produced bitwise-identical velocities, evidence and lock bits in all 144 pair/configuration cases. This does not establish equivalence for unused modes. Initial and final hashes are retained separately in `provenance.csv`. No KWM source was edited by this work.\n",
                   table(parity, [("variant","Lift"),("k","K"),("sigma","σ"),("pairs","Pairs"),("bitwise_equal","Bitwise equal"),("max_difference","Max difference")])]
    chunks += ["\nAll per-clip measurements are CSV files under `k3d/results/`; binary lifts, trajectories, logs, probe predictions, and the proposed commit message are under `k3d/.scratch/`. README documents restart/resume and the meaning of every metric.\n",
               "\nAdditional execution notes are recorded in `k3d/results/execution_notes.csv`. Source/checkpoint hashes are in `k3d/results/provenance.csv`. The `tests/k3d/test_stage1.py` checks cover geometry and turn signs, empirical sign evidence, dataset alignment, five-channel lift identity, box round-trip and projections, Hungarian uniqueness/empty cases/cutoffs, static pinning, turn-state round trip and bitwise agreement-observer parity.\n"]
    if (RESULTS/"execution_notes.csv").exists():
        chunks.append(table(read("execution_notes.csv"), [("item","Item"),("detail","Detail")]))
    dest = ROOT / "experiments/flowm_bench/docs/K3D_stage1_results.md"
    dest.write_text("\n\n".join(chunks) + "\n")
    print(dest, flush=True)


if __name__ == "__main__":
    main()
