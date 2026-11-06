#!/bin/bash
# Kinetic 3D stage 2 pilot (design doc Addendum 2, pre-registered d61c113). User decision 2026-11-05: start only
# after the C4 chain has finished, gated on free GPU-1 memory. Order: micro-benchmark -> budget/memory checks ->
# train C -> train A-oracle -> four evaluations -> statistics. Stops (does not improvise) on any failure.
S=/tmp/claude-1016/-home-yixunhu-codespace-flowm/831f1992-2b3c-47ed-ba4f-3bd336e0abb9/scratchpad
C=$S/campaign; G=$S/k3d/pilot; LOG=$G/chain_pilot.log
source $S/gpu_gate.sh
say() { echo "=== $* $(date)" >> $LOG; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' '; }
say "pilot chain armed; waiting for C4 CHAIN DONE"
until grep -q "C4 CHAIN DONE" $C/driver.log; do sleep 1800; done
cd /home/yixunhu/codespace/flowm || exit 1
P=/home/yixunhu/miniconda3/envs/flowm/bin/python
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/algorithms/mem_wm/backbones/kwm3d/.scratch/tmp" XDG_CACHE_HOME="$PWD/algorithms/mem_wm/backbones/kwm3d/.scratch/cache" MPLCONFIGDIR="$PWD/algorithms/mem_wm/backbones/kwm3d/.scratch/matplotlib"
O=outputs/k3d_pilot
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" $O
[ -e $O/c/checkpoints ] || [ -e $O/a_oracle/checkpoints ] && { say "PILOT STOPPED: $O already holds checkpoints; refusing to overwrite"; exit 1; }
COMMON=(algorithm=flowm_video dataset=blockworld ckpt_map=default wandb.mode=offline wandb.entity=local wandb.project=flowm_bench experiment.devices=1 experiment.accelerator=gpu)
say "pilot chain start; flowm $(git rev-parse --short HEAD) dirty=$(git status --porcelain | grep -vc '^??') kwm $(git -C /home/yixunhu/codespace/kwm-wt-k3d rev-parse --short HEAD) generator 7e740ec"

# ---- 1. micro-benchmark (10 optimizer steps per arm: 2 warm-up + 8 timed), batch 8 first, fallback batch 4 x accumulate 2
bench() { # batch accumulate tag
  for arm in c a_oracle; do
    wait_for_gpu_free 30000
    $P scripts/kwm3d_bench.py --device cuda --arm $arm --context 50 --prediction 40 --batch $1 --accumulate $2 --warmup 2 --steps 8 --checkpoint on > $O/bench_${arm}_$3.log 2>&1
    say "bench $arm $3 exit $? free $(free_mib) MiB"
  done
  $P $G/bench_summary.py $O/bench_c_$3.log $O/bench_a_oracle_$3.log $(free_mib)
}
MODE=bs8; SC_EXTRA=""
R=$(bench 8 1 bs8); say "bench bs8: $R"
if [ "${R%% *}" != OK ]; then
  MODE=bs4x2
  R=$(bench 4 2 bs4x2); say "bench bs4x2: $R"
  [ "${R%% *}" != OK ] && { say "PILOT STOPPED: neither batch 8 nor batch 4 x 2 fits in the free memory of GPU 1"; exit 1; }
fi
read -r _ MC MA PC PA DAYS <<< "$R"
say "mode $MODE: s/step C $MC A-oracle $MA; peak reserved MiB C $PC A-oracle $PA; projected $DAYS GPU-days for both trainings"
if [ "$($P -c "print(int($DAYS > 4.0))")" = 1 ]; then say "PILOT STOPPED: projected $DAYS GPU-days exceeds the pre-registered 4-day cap"; exit 1; fi

# ---- 2. trainings, sequential
train() { # arm peak_mib
  local arm=$1 need=$($P -c "print(int($2 * 1.10 + 1024))") sc="exp/blockworld/kwm3d/train/pilot_$1"
  [ $MODE = bs4x2 ] && sc="[exp/blockworld/kwm3d/train/pilot_$1,exp/blockworld/kwm3d/train/pilot_bs4_accum2]"
  wait_for_gpu_free $need
  say "train $arm start ($MODE, needs $need MiB, free $(free_mib) MiB)"
  $P -m main "shortcode=$sc" +name=k3d_pilot_$arm "${COMMON[@]}" hydra.run.dir=$O/$arm > $O/train_$arm.log 2>&1
  local rc=$?
  say "train $arm exit $rc"
  [ $rc = 0 ] && [ -f $O/$arm/checkpoints/step-006000.ckpt ] || { say "PILOT FAILED: train $arm (exit $rc, final checkpoint $(ls $O/$arm/checkpoints 2>/dev/null | tail -1))"; exit 1; }
}
train c $PC
train a_oracle $PA

# ---- 3. evaluations of the final checkpoints (first 1,000 validation clips, 70+70 and 70+210), then statistics
for arm in c a_oracle; do
  for horizon in 140 280; do
    wait_for_gpu_free 12000
    say "eval $arm $horizon start"
    $P -m main "shortcode=exp/blockworld/kwm3d/infer/metrics_${horizon}/k3d_v_70ctx" "+name=k3d_pilot_${arm}_${horizon}" "${COMMON[@]}" \
      "algorithm.backbone.arm=$arm" "load=$PWD/$O/$arm/checkpoints/step-006000.ckpt" "hydra.run.dir=$O/$arm/eval_$horizon" \
      "algorithm.logging.per_clip_dump=$O/$arm/eval_$horizon/dump" > $O/eval_${arm}_$horizon.log 2>&1
    rc=$?; say "eval $arm $horizon exit $rc"
    [ $rc = 0 ] && [ -f $O/$arm/eval_$horizon/dump/per_clip.pt ] || { say "PILOT FAILED: eval $arm $horizon (exit $rc)"; exit 1; }
  done
done
export CUDA_VISIBLE_DEVICES=""
$P scripts/kwm3d_pilot_stats.py --c $O/c/eval_140/dump --a_oracle $O/a_oracle/eval_140/dump --baseline experiments/flowm_bench/k3d/results/pilot_baseline_repeat_140 \
  --reference context_mean=experiments/flowm_bench/k3d/results/pilot_reference_context_mean_140 --reference last_flat=experiments/flowm_bench/k3d/results/pilot_reference_last_flat_140 --reference gray=experiments/flowm_bench/k3d/results/pilot_reference_gray_140 \
  --other-c $O/c/eval_280/dump --other-a-oracle $O/a_oracle/eval_280/dump --output $O/pilot_stats_140.md >> $LOG 2>&1 || { say "PILOT FAILED: statistics 140"; exit 1; }
$P scripts/kwm3d_pilot_stats.py --horizon 280 --c $O/c/eval_280/dump --a_oracle $O/a_oracle/eval_280/dump --baseline experiments/flowm_bench/k3d/results/pilot_baseline_repeat_280 \
  --reference context_mean=experiments/flowm_bench/k3d/results/pilot_reference_context_mean_280 --reference last_flat=experiments/flowm_bench/k3d/results/pilot_reference_last_flat_280 --reference gray=experiments/flowm_bench/k3d/results/pilot_reference_gray_280 \
  --output $O/pilot_stats_280.md >> $LOG 2>&1 || { say "PILOT FAILED: statistics 280"; exit 1; }
say "PILOT DONE"
