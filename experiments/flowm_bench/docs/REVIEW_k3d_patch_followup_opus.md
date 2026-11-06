# Adversarial review: b2e2f08 + 413f471 (reported-only patch-motion follow-up)

Scope: `experiments/flowm_bench/k3d/trained_lift.py`, `scripts/kwm3d_pilot_patch_report.py`,
`experiments/flowm_bench/k3d/pilot/followup_patch.sh`, `tests/k3d/test_trained_lift.py`,
`tests/kwm3d/test_pilot_patch_isolation.py`, against design doc Addendum 2 and its Amendment (2026-11-06).
Everything ran on CPU (`CUDA_VISIBLE_DEVICES=""`, 8 threads). I wrote nothing outside
`/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad/review/patch_followup/`.
I did not modify any systemd unit. The one exception is the pytest run you allowed: its conftest writes temporary
files under `algorithms/mem_wm/backbones/kwm3d/.scratch/tmp`.

State at review time: `k3d-pilot-chain` is a transient unit (`CollectMode=inactive-or-failed`) that runs
`$S/k3d/pilot/chain_pilot.sh`. That file is byte-identical to the repo copy. The unit is `active`, and its log
holds only the "armed" line. `outputs/k3d_pilot/` does not exist yet. The scratch copy of `followup_patch.sh` is
identical to the repo copy. Suite: `38 passed`.

## Verdict per question

| Q | Problem found? | Summary |
|---|---|---|
| 1 Comparability | **Yes, one Major (interpretation, not code)** | The pipeline matches stage 1 exactly: same clips, frames, pairs, search, projections, error and summary (verified). The A-oracle write *definition* differs from the C/stage-1 write even at identical weights (12 % relative difference). The report does not say so. Arm-C assertions cannot plausibly fail. The failure path works end to end. |
| 2 Isolation | **No path found.** Test limitations are Minor | The stats run before the follow-up can start, and the stats script reads nothing beyond its arguments. The tests are real: 6 of 7 mutations were caught. Only an obfuscated read of the real production path, or a read with no effect, escapes. |
| 3 Shell script | **Yes: one Major, several Minor** | Bash reads the script incrementally, so editing it while it waits corrupts execution (demonstrated). A wrong or missing unit name, or a re-armed chain with an old marker in its log, fires the follow-up early. No path trips the chain's overwrite guard. No path touches a GPU. PASS detection and `--missing` quoting are correct. |
| 4 trained_lift / report | **No correctness bug.** Minor metadata and robustness items | Cache keyed by sha256 and arm, strict pair-set check before any CSV is written, stage handling, batch-size defaults and the pooled gate path are all correct. |

No Blocker found.

---

## Major

### MJ1. `followup_patch.sh` is read incrementally by bash: editing the file while the unit waits corrupts execution
*Evidence.* Bash parses the `while` loop (lines 14–19) as a unit, but reads the lines after the loop from the file
offset only when the loop exits. I ran a small script waiting in a loop, rewrote the file with a longer header (as a
fix commit or a `git checkout` would), then released the loop. Bash executed a fragment of the new file:
`s.sh: line 3: made: command not found`, then continued (scratch `bashread/`). The follow-up is designed to sleep for
days. Fixes from this review will edit the repo copy, and the scratch `$G` copy is identical to it. Whichever copy is
launched must never be edited afterwards. The same applies to `chain_pilot.sh`, which nobody should touch.
The python code (`trained_lift.py`, report script) is also loaded from the working tree *at fire time*, but only
`git rev-parse --short HEAD` is logged (line 27), with no dirty flag.

*Fix.*
- Wrap the whole body in `main() { ... }; main "$@"; exit $?`, or in `{ ...; exit; }`, so bash parses it before running anything.
- Launch from a dedicated frozen copy, e.g. `cp followup_patch.sh $G/followup_patch.frozen.sh; chmod a-w`, never from the repo path.
- Log `git status --porcelain | wc -l` and `sha256sum` of `trained_lift.py`, `patch_motion.py`, `lift.py` and the report script at fire time.

### MJ2. The A-oracle lift is not definition-comparable with stage-1 "tokens", and the report does not say so
*What matches* (verified by code reading and experiments):
- Shape `[140,177,256]`.
- Cone order: boolean `[H,W]` mask indexing in `pooled_tokens` (`trained_lift.py:38-41`) and `nonzero(view(-1))` in `extract_view_tokens` are both row-major. `tokens_to_box` uses the same order. `tests/k3d/test_trained_lift.py::test_trained_pooled_write_matches_estimator_imprint` pins this against the estimator imprint.
- Post-gate on a zero map in both cases: C tokens = `g*h_new` exactly (`(1-g)*0`). A-oracle = `gate*proposed`.
- Max-pooling over V on a zero map gives the same zero input.
- Inputs: `ReadOnlyClips` raw frames, the same convention `FloWMVideo` feeds the backbone (`algorithms/flowm_video.py:204-209`, no normalisation).

*What differs.* The per-node write (C, stage 1) runs joint attention over 5×177 identical map tokens plus 64 image
tokens. The pooled write (`kwm3d_models.py:126-142`) runs it over 177 + 64. The duplicated keys change the attention
mass. **With the same released weights**, loaded into arm `c` (via `observation_lift`) and arm `c_pw` (via
`observation_write`), the tokens differ by `max|Δ| = 4.19`, relative L2 = **0.119**, mean cosine 0.9925
(`exp/defdiff.py`, 16 frames of clip 0/0000).

A trained-A-oracle row minus the released row therefore mixes "what training did" with "pooled vs per-node write".
The C row has no such confound. The report prints only the one-line `Lift:` definition and does not warn about this.
Neither lift includes the kinetic `map_conv`, which is correct per the spec, since the estimator reads the write.

*Fix (reported-only, cheap).*
- Add a reference row "frozen released weights, pooled write": run `trained_lift --checkpoint <v2_dynamic.ckpt> --arm c_pw --stage patch`, about 7 CPU-min for 200 clips at batch 8, and pass it with `--run`.
- Add one sentence to the report: "A-oracle uses a V-pooled single-copy write; compare it with the released pooled-write row, not only with the per-node stage 1 row."
- Also state that A-oracle's write was trained only through reconstruction (oracle velocities), never through an estimator.

---

## Minor

1. **Early or false firing of the follow-up** (`followup_patch.sh:12-19`).
   - `systemctl --user is-active <name>` prints `inactive` for a nonexistent unit. A typo in `PATCH_CHAIN_UNIT`, or arming the follow-up when the chain unit is not running, fires it after 60 s. Verified: `PATCH_CHAIN_UNIT=nonexistent-review-unit` → "no terminal marker (chain unit inactive)" after 60.16 s, then "PATCH SKIPPED" with a "both checkpoints missing" report.
   - The chain log is appended (`>> $LOG`). If the chain is ever re-armed after a `PILOT STOPPED`, the old marker fires a new follow-up immediately.
   - Neither case can make `chain_pilot.sh` refuse to start. Its guard (`chain_pilot.sh:18`) checks only `$O/{c,a_oracle}/checkpoints`. The follow-up creates only `$O`, `$O/pilot_patch_motion.md`, `$O/patch_*.log` and `$O/patch_motion/<arm>`, plus `$O/gate_a_oracle.log` after a PASS. Hydra's `run.dir=$O/$arm` copes with an existing `$O`.
   - The damage is a stale, misleading `pilot_patch_motion.md` and a follow-up that never re-runs. If it fired while evaluation was still running, the gate check would see no `pilot_stats_140.md` and skip the gate. `H0` would hash missing files, and the stats appearing later would trigger "PATCH FAILED: statistics files changed".
   - *Fix:* at arm time, require `is-active == active` and refuse to arm otherwise. Record `wc -l` of the chain log at arm time and grep only the lines after it.
2. **Exit status.** The script always exits 0. Line 64 ends with a successful `say` even on "PATCH FAILED", and the missing branch exits 0 even when the report fails (line 40). *Fix:* `exit $BAD`, and set `BAD` from the report's rc in the missing branch.
3. **Report wording and failure reason.** A failed run is rendered as "trained C: test not run — patch test failed (exit 1), see …log" (verified with a forced arm-C `AssertionError`). The test *was* run. The assertion text sits in `patch_motion/c/patch_run.json` (`status: failed`, `error: …`) but is not surfaced. *Fix:* use the wording "test failed", and pass the `error` field through.
4. **The report does not enforce comparability of the run size.** `render` checks that summary pairs equal `info["pairs"]` but not that they equal the released `all/all` pairs (44,990), nor that `clips == 200`. A stray `PATCH_CLIPS` would print 2-clip trained rows next to 200-clip released rows (verified: my 2-clip simulation rendered fine). *Fix:* require `info["pairs"] == released[(k,"all","all")]["pairs"]`, or print a loud warning.
5. **Gate trigger when stats 280 failed.** If the chain ends `PILOT FAILED: statistics 280` after writing a `pilot_stats_140.md` that says PASS, the gate runs. The decision is defined on horizon 140, so this is arguably right. Document it, or also require the `PILOT DONE` marker. The PASS regex `^Pilot 140: \*\*PASS\*\*` matches the heading at `kwm3d_pilot_stats.py:115` exactly, and cannot match the smoke or 280 headings.
6. **Misleading `patch_run.json` metadata** (`trained_lift.py:194-203`). Patch-only runs record `k: 8, sigma: 1, residual_mode: mask, truth: intrinsic_incoming`, which are gate settings, next to `Ks: [8,32,256]`. *Fix:* write the gate keys only when stage ∈ {all, gate}.
7. **Inconsistent failure records.** An `AssertionError` in the lift writes a failure `{stage}_run.json`. A pair-set `ValueError` (`check_pair_set`) writes nothing. Both exit non-zero, and the follow-up handles both as "patch test failed". *Fix:* write the same failure JSON for the `ValueError`.
8. **Isolation-test blind spots** (`tests/kwm3d/test_pilot_patch_isolation.py`). See Q2 below. *Fix:*
   - Record opens with `sys.addaudithook` (the `open`, `os.listdir` and `os.scandir` events) instead of monkeypatching Python-level `open` and `torch.load`.
   - Additionally create the decoy files at the real production locations, or assert that no path outside the explicit arguments is touched.
9. **One parity test is tautological.** `test_c_lift_equals_stage1_observation_tokens` compares `lift_tokens(…,'c')` with `observation_lift` on a random model, and `lift_tokens` *is* `observation_lift`. The meaningful parity evidence lives only in untracked `experiments/flowm_bench/k3d/.scratch/patch_parity_released/`. *Fix:* add a CPU test that runs `patch_pairs` on 1–3 stage-1 cached clips (`lift.read_cache`) and compares against `followup_patch_tokens_k{8,32,256}_pairs.csv`. My `exp/parity.py` does this in seconds and matches exactly.
10. Cosmetic: "pilot chain ended: …" prints two dates. The script is mode 664, so it must be launched as `bash followup_patch.sh`.

---

## Q1 details: comparability and arm-C assertions

**The pipeline is the same as stage 1 (verified):**
- `patch_pairs` (`trained_lift.py:51-80`) is a line-for-line copy of the stage-1 loop (`patch_motion.py:136-161`):
  - the same 16-frame windows and turn exclusion;
  - the `vis[t] & vis[t-1]` pair set;
  - rounded previous cell minus `BOX`;
  - `single_patch_search` with ego from `camera_action`;
  - `intrinsic_incoming` targets;
  - `inf` for unusable pairs;
  - the same row columns.
- `summarize` is imported from `patch_motion` directly. K=8/32 use `projected_box`. K=256 uses `tokens_to_box(tokens.float())`, the same as stage 1.
- The C lift batch is 32 and fp16-cached, as in `lift.py:93-94`. Clips are the first 200 of `k3d_v_validation`, frames 0–139.
- **Experiment `exp/parity.py`:** `trained_lift.patch_pairs` on stage-1 cached released tokens, clips 0/0000, #57 and #199, at K=8, 32 and 256: **639 rows per K, 0 mismatches** against the stage-1 pair CSVs, all 15 columns.
- **Author's full run.** The author ran the released checkpoint through the arm-C trained path (`.scratch/patch_parity_released/`). I checked that its `patch_k{8,32,256}_summary.csv` files are **byte-identical** to `results/followup_patch_tokens_k{8,32,256}_summary.csv`. I did not regenerate that run (about 21 CPU-min). The files predate the commit by about 10 min, so I verified the artefact, not that the committed code produced it.
- **Experiment `exp/run_small.py`:** both CPU test checkpoints (step-2, C and A-oracle) run `--stage patch` through `tl.run` with my own cache dir. 2 clips, 482 pairs, `pair_set_matches_stage1: true`.

**Arm-C assertions on a trained checkpoint: failure is not plausible.**
- The per-node write sees V identical zero-map tokens. There is no V embedding: kwm3d forces `v_channel_identity_emb=False` (`kwm3d_models.py:74-80`), so `assert v_channel_id_embed is None` holds.
- Attention is permutation-equivariant over identical tokens, so the outputs are identical across nodes.
- `tokens == g*h_new` holds exactly, because `(1-g)*0 == 0`.
- **Experiment `exp/node_eq.py`.** On the step-2 C checkpoint and on a heavily perturbed copy (every map-processor parameter plus Gaussian noise of its own std; qkv weights ×3), `max|node_i - node_0|` is **0.0** for tokens, `g` and `h_new`, and `observation_lift` passes.
- The only realistic failure is NaN weights from a diverged training. `allclose` would then fail, which is the right outcome.
- **End to end, if it fails** (verified with a forced `AssertionError`, `sim/failc/`):
  - `trained_lift` writes `patch_motion/c/patch_run.json` with `status: failed` and the error, and exits 1. No CSVs are written, no cache file is left behind, and there is no fallback.
  - The follow-up logs `patch c exit 1` and passes `--missing "trained C=patch test failed …"`.
  - The report is still written, with the A-oracle rows and the C line. The stats-hash check passes, the gate decision runs as usual, and the log ends "PATCH FAILED" with unit exit 0 (Minor 2).

## Q2 details: isolation
- **Temporal.** The stats run inside the chain *before* `PILOT DONE` (`chain_pilot.sh:69-75`). The follow-up starts only after a terminal marker or after the unit has gone. The stats script receives only explicit dump paths and `--output`.
- **Write side.**
  - The report refuses any `--output` named `pilot_stats*`, after `resolve()` too (`kwm3d_pilot_patch_report.py:103-104`).
  - The follow-up writes no stats files and hashes them before and after the patch stage.
  - `trained_lift` writes only to `--output-dir` and its sha256-keyed cache.
  - Grep shows no `PILOT` string in the stats or bench output that could fake a marker.
- **Tests are real.** I copied the stats script, report, tests and conftest into scratch, injected reads before `dest.write_text`, and ran the two isolation tests:

| Mutation | Result |
|---|---|
| M0 unmodified | 4 passed |
| M1 literal `Path('pilot_patch_motion.md').read_text()` | **caught** (4 failed: AST and open recorder) |
| M2 obfuscated name, builtin `open` read, no effect | **caught** (open recorder) |
| M4 obfuscated `os.path.exists` flips the output | **caught** (output equality across clean and decoy cases) |
| M5 obfuscated `os.open` read whose length is appended | **caught** (output equality) |
| M6 obfuscated `numpy.genfromtxt` of `patch_motion/c/patch_k8_summary.csv` | **caught** |
| M3 obfuscated `os.open` read with no effect | not caught (harmless, since the verdict is unaffected) |
| M7 obfuscated *absolute production path* `outputs/k3d_pilot/pilot_patch_motion.md` that flips PASS→FAIL | **not caught**: the fixture never creates that path, and `exists` is not recorded |

Conclusion: the tests would catch any plausible regression, including any literal mention of "patch" in the stats
script. They do not prove isolation against deliberately obfuscated reads of production paths (Minor 8).

## Q3 details: follow-up script
Problems found: MJ1, Minor 1, Minor 2 and Minor 5. Checked and correct:
- **Marker regex** `PILOT (DONE|STOPPED|FAILED)` covers every terminal `say` in `chain_pilot.sh` (lines 18, 36, 40, 51, 65, 71, 74, 75).
- **Unconfirmed exits.** Exits without a marker (`cd` failure at line 12, a kill) are covered by the unit-gone check, which re-confirms after 60 s. Transient-unit GC means a finished chain reads as `inactive`. If `systemctl --user` itself fails, the state is empty and the script keeps waiting (fail-safe, but it would wait forever if there is also no marker).
- **Missing checkpoint** (`sim/missing1`): the test is not run for either arm. The report names the missing file and says the present one was not tested because both were not produced. Exit 0 with "PATCH SKIPPED". This matches the decision.
- **One arm fails:** the other arm is still run and reported (`sim/failc`).
- **PASS path** (`sim/pass`, stubbed `run_cpu.sh` → `trained_lift` with scratch cache, 2 clips): both patches run, the report is written, and the gate runs on `a_oracle` with `--stage gate` into `patch_motion/a_oracle`. The gate reuses the patch-stage cache and `gate_summary.csv` is appended to the log. With a FAIL heading the gate is skipped (`sim/failc`).
- **Quoting.**
  - `ARGS+=(--missing "$name=…")` keeps names and reasons containing spaces, `|`, `=` and parentheses as single argv items. Verified with a chain line containing `| x=y`.
  - The report splits on the first `=` and escapes `|`.
  - Unquoted `$CL`, `$LOG`, `$O` and `$ck` are safe only because those paths contain no spaces. Quote them anyway.
- **GPU.** None. The script exports `CUDA_VISIBLE_DEVICES=""`, `run_cpu.sh` re-exports it, and the report imports only numpy. `trained_lift` loads with `map_location="cpu"`. No `nvidia-smi` and no `.cuda()`.
- **Overwrite guard** (Minor 1): unaffected.

## Q4 details: `trained_lift.py` and the report
- **Cache.** The path is `SCRATCH/trained_lifts/<sha256 of file>/<arm>/<clip>.pt`. Each entry is validated for sha, shape, fp16 dtype and `lift_definition`. Legacy entries without `lift_definition` are accepted, but those can only be pooled-arm caches with the same definition.
- **Arm checks.** A wrong `--arm` is rejected by `load_checkpoint_model` through `.hydra/config.yaml`, which the pilot runs have. That load happens before any cache read.
- **No cross-contamination.** No cache is shared across checkpoints or arms. Atomic `.tmp` → `replace`.
- **Pair set.** `check_pair_set` compares the (clip, frame, block) set against the stage-1 K=8 list (the K=8, 32 and 256 lists are identical, 44,990 rows each). It also detects duplicates and unknown clips, runs for every K, and runs before any CSV is written.
- **Stage and batch handling.**
  - C is limited to lift and patch in both `run` and `main`, including the stage-1 entry-point dispatch.
  - Default batch is 32 for C (as in stage 1) and 8 for pooled arms. Batching does not change per-frame values.
- **Gate path.** The pooled-arm gate path (`trained_lift.py:153-179`) is functionally unchanged, apart from moving `projected_box(…,8)` after the patch block and adding timing.
- **Report.**
  - Reads the K=8, 32 and 256 summaries, requires every speed and distance bin, and checks the pair-set flag, `Ks`, and the pairs against `patch_run.json`.
  - Formats inf and nan correctly.
  - Weaknesses are Minor 3 and 4, plus MJ2: it does not state the definition caveat.

## Artefacts (all under the scratch review dir)
- `exp/parity.py`: patch-path parity on stage-1 caches.
- `exp/node_eq.py`: node-equality under trained and perturbed weights.
- `exp/defdiff.py`: per-node vs pooled write at the same weights.
- `exp/run_small.py`: small runs on the CPU test checkpoints.
- `exp/stub_run.py` and `exp/followup_stub.sh`: the follow-up with `run_cpu.sh` replaced by a scratch-cache wrapper; that is the only difference.
- `sim/{missing1,nomarker,pass,failc}/`: follow-up simulations. `missing1` and `nomarker` used the unmodified script.
- `iso/`: isolation-test mutation copies.
- `bashread/`: incremental-read demo.
