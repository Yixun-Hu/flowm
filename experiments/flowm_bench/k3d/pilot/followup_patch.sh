#!/bin/bash
# Reported-only patch-motion follow-up to the stage 2 pilot (design doc Addendum 2, amendment 2026-11-06).
# Separate unit: the running chain_pilot.sh is never edited. Waits for the pilot chain to end, then, CPU only:
# patch-motion test on the final A-oracle and C checkpoints (same clips and settings as stage 1), report file
# outputs/k3d_pilot/pilot_patch_motion.md. Never touches the verdict or the statistics files. If a final
# checkpoint is missing, the test is not run and the report says so. After PILOT DONE with a PASS at 70 + 70 it
# also runs the estimator gate on the trained A-oracle lift (as already planned in Addendum 2).
# Launch from a frozen read-only copy (bash reads scripts incrementally); the whole body is inside main() so
# that it is parsed before anything runs. Exit status: 0 done or skipped, 1 failed, 2 refused to arm.
main() {
  local S=/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad
  local G="$S/k3d/pilot"
  local CL="${PATCH_CHAIN_LOG:-$G/chain_pilot.log}" LOG="${PATCH_LOG:-$G/followup_patch.log}"
  local UNIT="${PATCH_CHAIN_UNIT:-k3d-pilot-chain}" CLIPS="${PATCH_CLIPS:-200}" SETTLE="${PATCH_SETTLE:-30}"
  local MARK='PILOT (DONE|STOPPED|FAILED)' st rc
  say() { echo "=== $* [$(date '+%F %T %Z')]" >> "$LOG"; }
  chain_state() { systemctl --user is-active "$UNIT" 2>/dev/null; }
  gone() { [ "$1" = inactive ] || [ "$1" = failed ]; }

  # ---- arm: only against a live chain that has not ended yet (PATCH_FORCE=1 overrides, for a deliberate run
  # after the chain has ended and for rehearsals)
  if [ "${PATCH_FORCE:-0}" != 1 ]; then
    if grep -qE "$MARK" "$CL" 2>/dev/null; then say "PATCH NOT ARMED: $CL already holds a terminal marker (set PATCH_FORCE=1 to run now)"; return 2; fi
    if [ "$(chain_state)" != active ]; then say "PATCH NOT ARMED: unit $UNIT is not active (set PATCH_FORCE=1 to run now)"; return 2; fi
  fi
  say "patch follow-up armed (script sha256 $(sha256sum "$0" | cut -c1-16)); waiting for the pilot chain to end"
  while :; do
    grep -qE "$MARK" "$CL" 2>/dev/null && break
    st=$(chain_state)
    if gone "$st"; then sleep 60; grep -qE "$MARK" "$CL" 2>/dev/null && break; st=$(chain_state); gone "$st" && break; fi
    sleep 600
  done
  sleep "$SETTLE"

  cd /home/yixunhu/codespace/flowm || { say "PATCH FAILED: cannot enter the repository"; return 1; }
  local P=/home/yixunhu/miniconda3/envs/flowm/bin/python
  export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
  local O="${PATCH_O:-outputs/k3d_pilot}"; local PM="$O/patch_motion"
  local TERMINAL; TERMINAL=$(grep -E "$MARK" "$CL" 2>/dev/null | tail -1 | sed -E 's/^=== //')
  [ -z "$TERMINAL" ] && TERMINAL="no terminal marker (chain unit inactive)"
  say "pilot chain ended: $TERMINAL"
  say "code: flowm $(git rev-parse --short HEAD) dirty=$(git status --porcelain | grep -vc '^??'); sha256 $(sha256sum experiments/flowm_bench/k3d/trained_lift.py experiments/flowm_bench/k3d/patch_motion.py experiments/flowm_bench/k3d/lift.py scripts/kwm3d_pilot_patch_report.py scripts/kwm3d_pilot_stats.py | awk '{printf "%s=%s ", $2, substr($1,1,12)}')"
  mkdir -p "$O"
  stats_hash() { sha256sum "$O/pilot_stats_140.md" "$O/pilot_stats_280.md" 2>/dev/null | sha256sum | cut -c1-16; }
  local H0; H0=$(stats_hash)
  local CKA="$O/a_oracle/checkpoints/step-006000.ckpt" CKC="$O/c/checkpoints/step-006000.ckpt"
  local REF=experiments/flowm_bench/k3d/results/patch_released_pooled_write
  local ARGS=(--released-results experiments/flowm_bench/k3d/results)
  [ "$CLIPS" = 200 ] || ARGS+=(--allow-partial)
  report() { "$P" scripts/kwm3d_pilot_patch_report.py "${ARGS[@]}" --output "$O/pilot_patch_motion.md" >> "$LOG" 2>&1; }

  # ---- a final checkpoint is missing: do not run the test, report which one
  if [ ! -f "$CKA" ] || [ ! -f "$CKC" ]; then
    local pair name ck
    for pair in "trained C (per-node write):$CKC" "trained A-oracle (pooled write):$CKA"; do
      name=${pair%%:*}; ck=${pair#*:}
      if [ -f "$ck" ]; then ARGS+=(--missing "$name=the pilot did not produce both final checkpoints (chain: $TERMINAL)")
      else ARGS+=(--missing "$name=final checkpoint $ck is missing (chain: $TERMINAL)"); fi
    done
    report; rc=$?; say "report exit $rc"
    say "PATCH SKIPPED: final checkpoint missing (A-oracle $([ -f "$CKA" ] && echo present || echo missing), C $([ -f "$CKC" ] && echo present || echo missing))"
    return $(( rc != 0 ))
  fi

  # ---- patch-motion test on both final checkpoints (order of the report rows: per-node pair, then pooled pair)
  local BAD=0 arm
  declare -A ROW
  for arm in a_oracle c; do
    [ $arm = c ] && ck="$CKC" || ck="$CKA"
    [ $arm = c ] && name="trained C (per-node write)" || name="trained A-oracle (pooled write)"
    say "patch $arm start"
    bash experiments/flowm_bench/k3d/run_cpu.sh trained_lift --checkpoint "$ck" --arm $arm --stage patch --clips "$CLIPS" --output-dir "$PM/$arm" > "$O/patch_$arm.log" 2>&1
    rc=$?; say "patch $arm exit $rc"
    if [ $rc = 0 ] && [ -f "$PM/$arm/patch_k256_summary.csv" ]; then ROW[$arm]="--run:$name=$PM/$arm"
    else ROW[$arm]="--failed:$name=$PM/$arm"; BAD=1; say "patch $arm failed, see $O/patch_$arm.log"; fi
  done
  ARGS+=("${ROW[c]%%:*}" "${ROW[c]#*:}")
  if [ "$CLIPS" = 200 ] && [ -f "$REF/patch_run.json" ]; then ARGS+=(--run "frozen released weights (pooled write)=$REF")
  else say "released pooled-write reference row omitted (rehearsal clip count or $REF absent)"; fi
  ARGS+=("${ROW[a_oracle]%%:*}" "${ROW[a_oracle]#*:}")
  report; rc=$?; say "report exit $rc"
  [ $rc = 0 ] || BAD=1
  [ "$(stats_hash)" = "$H0" ] || { say "PATCH FAILED: statistics files changed during the follow-up (must never happen)"; return 1; }

  # ---- estimator gate on the trained A-oracle lift: only after PILOT DONE with a PASS at 70 + 70
  if [[ "$TERMINAL" == "PILOT DONE"* ]] && grep -q '^Pilot 140: \*\*PASS\*\*' "$O/pilot_stats_140.md" 2>/dev/null; then
    say "pilot verdict PASS: estimator gate on the trained A-oracle lift start"
    bash experiments/flowm_bench/k3d/run_cpu.sh trained_lift --checkpoint "$CKA" --arm a_oracle --stage gate --clips "$CLIPS" --output-dir "$PM/a_oracle" > "$O/gate_a_oracle.log" 2>&1
    rc=$?; say "gate a_oracle exit $rc"
    if [ $rc = 0 ]; then cat "$PM/a_oracle/gate_summary.csv" >> "$LOG"; else BAD=1; fi
  else
    say "no PILOT DONE with PASS at 70 + 70: estimator gate not run"
  fi
  if [ $BAD = 0 ]; then say "PATCH DONE"; else say "PATCH FAILED: see the lines above"; fi
  return $BAD
}
main "$@"; exit $?
