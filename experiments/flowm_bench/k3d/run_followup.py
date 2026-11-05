"""Sequential eight-thread follow-up; checkpoint completed stages on disk."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRATCH = HERE/".scratch"
RESULTS = HERE/"results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--after-mask", action="store_true", help="Wait for the already running item 1")
    args = parser.parse_args()
    if args.after_mask:
        deadline = time.monotonic()+3600
        while not (RESULTS/"followup_mask_time.csv").exists():
            if time.monotonic() > deadline:
                raise TimeoutError("Mask stage did not finish; inspect followup_mask.log")
            time.sleep(5)
    commands = []
    dest = RESULTS/"followup_commands.csv"
    if dest.exists():
        commands = list(csv.DictReader(dest.open()))
    stages = [] if args.after_mask else [("mask", [sys.executable,"-m","k3d.followup","mask"])]
    stages += [("tests", [sys.executable,"-m","pytest","tests/k3d/test_stage1.py","-q",
                          "-o","cache_dir=experiments/flowm_bench/k3d/.scratch/pytest_cache"]),
               ("breakdown",[sys.executable,"-m","k3d.alarm_breakdown"]),
               ("patch",[sys.executable,"-m","k3d.patch_motion"]),
               ("memory",[sys.executable,"-m","k3d.memory_lift"])]
    for stage, command in stages:
        if any(r["stage"] == stage and int(r["status"]) == 0 for r in commands):
            continue
        print(time.strftime("%H:%M:%S"),"START",stage,flush=True)
        start = time.perf_counter()
        with (SCRATCH/f"followup_{stage}.log").open("w") as out:
            result = subprocess.run(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
        elapsed = time.perf_counter()-start
        commands.append(dict(stage=stage,command=" ".join(command),seconds=elapsed,status=result.returncode))
        with dest.open("w",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=list(commands[0]))
            writer.writeheader()
            writer.writerows(commands)
        print(time.strftime("%H:%M:%S"),"DONE",stage,result.returncode,f"{elapsed:.1f}s",flush=True)
        if result.returncode:
            raise RuntimeError(f"{stage} failed; see .scratch/followup_{stage}.log")


if __name__ == "__main__":
    main()
