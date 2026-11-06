#!/bin/bash
# Reported-only patch-motion follow-up to the stage 2 pilot (design doc Addendum 2, amendment 2026-11-06).
# Separate unit: the running chain_pilot.sh is never edited. Waits for the pilot chain to end, then, CPU only:
# patch-motion test on the final A-oracle and C checkpoints (same clips and settings as stage 1), report file
# outputs/k3d_pilot/pilot_patch_motion.md. Never touches the verdict or the statistics files. If a final
# checkpoint is missing, the test is not run and the report says so. After a PASS it also runs the estimator
# gate on the trained A-oracle lift (as already planned in Addendum 2).
S=/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad
G=$S/k3d/pilot; CL=${PATCH_CHAIN_LOG:-$G/chain_pilot.log}; LOG=${PATCH_LOG:-$G/followup_patch.log}
UNIT=${PATCH_CHAIN_UNIT:-k3d-pilot-chain}; CLIPS=${PATCH_CLIPS:-200}; SETTLE=${PATCH_SETTLE:-30}
say() { echo "=== $* $(date)" >> $LOG; }
chain_state() { systemctl --user is-active $UNIT 2>/dev/null; }
say "patch follow-up armed; waiting for the pilot chain to end"
while :; do
  grep -qE "PILOT (DONE|STOPPED|FAILED)" $CL 2>/dev/null && break
  st=$(chain_state)
  if [ "$st" = inactive ] || [ "$st" = failed ]; then sleep 60; st=$(chain_state); { [ "$st" = inactive ] || [ "$st" = failed ]; } && break; fi
  sleep 600
done
sleep $SETTLE
cd /home/yixunhu/codespace/flowm || exit 1
P=/home/yixunhu/miniconda3/envs/flowm/bin/python
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
O=${PATCH_O:-outputs/k3d_pilot}; PM=$O/patch_motion
TERMINAL=$(grep -E "PILOT (DONE|STOPPED|FAILED)" $CL | tail -1 | sed -E 's/^=== //')
[ -z "$TERMINAL" ] && TERMINAL="no terminal marker (chain unit inactive)"
say "pilot chain ended: $TERMINAL; flowm $(git rev-parse --short HEAD)"
mkdir -p $O
stats_hash() { sha256sum $O/pilot_stats_140.md $O/pilot_stats_280.md 2>/dev/null | sha256sum | cut -c1-16; }
H0=$(stats_hash)
CKA=$O/a_oracle/checkpoints/step-006000.ckpt; CKC=$O/c/checkpoints/step-006000.ckpt
ARGS=(--released-results experiments/flowm_bench/k3d/results)
report() { $P scripts/kwm3d_pilot_patch_report.py "${ARGS[@]}" --output $O/pilot_patch_motion.md >> $LOG 2>&1; }
if [ ! -f $CKA ] || [ ! -f $CKC ]; then
  for pair in "trained A-oracle:$CKA" "trained C:$CKC"; do
    name=${pair%%:*}; ck=${pair#*:}
    if [ -f $ck ]; then ARGS+=(--missing "$name=the pilot did not produce both final checkpoints (chain: $TERMINAL)")
    else ARGS+=(--missing "$name=final checkpoint $ck is missing (chain: $TERMINAL)"); fi
  done
  report; say "report exit $?"
  say "PATCH SKIPPED: final checkpoint missing (A-oracle $([ -f $CKA ] && echo present || echo missing), C $([ -f $CKC ] && echo present || echo missing))"
  exit 0
fi
BAD=0
for pair in "a_oracle:trained A-oracle:$CKA" "c:trained C:$CKC"; do
  arm=${pair%%:*}; rest=${pair#*:}; name=${rest%%:*}; ck=${rest#*:}
  say "patch $arm start"
  bash experiments/flowm_bench/k3d/run_cpu.sh trained_lift --checkpoint $ck --arm $arm --stage patch --clips $CLIPS --output-dir $PM/$arm > $O/patch_$arm.log 2>&1
  rc=$?; say "patch $arm exit $rc"
  if [ $rc = 0 ] && [ -f $PM/$arm/patch_k256_summary.csv ]; then ARGS+=(--run "$name=$PM/$arm")
  else ARGS+=(--missing "$name=patch test failed (exit $rc), see $O/patch_$arm.log"); BAD=1; fi
done
report; rc=$?; say "report exit $rc"
[ $rc = 0 ] || BAD=1
[ "$(stats_hash)" = "$H0" ] || { say "PATCH FAILED: statistics files changed during the follow-up (must never happen)"; exit 1; }
if grep -q '^Pilot 140: \*\*PASS\*\*' $O/pilot_stats_140.md 2>/dev/null; then
  say "pilot verdict PASS: estimator gate on the trained A-oracle lift start"
  bash experiments/flowm_bench/k3d/run_cpu.sh trained_lift --checkpoint $CKA --arm a_oracle --stage gate --clips $CLIPS --output-dir $PM/a_oracle > $O/gate_a_oracle.log 2>&1
  rc=$?; say "gate a_oracle exit $rc"
  [ $rc = 0 ] && cat $PM/a_oracle/gate_summary.csv >> $LOG || BAD=1
else
  say "pilot verdict is not PASS: estimator gate not run"
fi
[ $BAD = 0 ] && say "PATCH DONE" || say "PATCH FAILED: see the lines above"
