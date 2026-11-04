"""Sequential resumable CPU campaign; run after (or wait for) initial V lifts."""
import argparse
import csv
import hashlib
import shlex
import subprocess
import time
from pathlib import Path
from .common import HERE, ROOT, SCRATCH, RESULTS, CHECKPOINT, clips, write_csv, log
from .lift import cache_path


def provenance():
    paths = [CHECKPOINT, ROOT/"algorithms/mem_wm/backbones/flowm/flowm_models_3d.py",
             ROOT/"algorithms/mem_wm/backbones/flowm/flowm_utils.py", ROOT/"datasets/video/base_video.py",
             Path("/home/yixunhu/codespace/kwm-wt-k3d/kwm/lock_nodes.py"),
             Path("/home/yixunhu/codespace/kwm-wt-k3d/kwm/cells.py"),
             Path("/home/yixunhu/codespace/Miniworld-k3d/scripts/generate_videos.py")]
    rows = []
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""):
                h.update(chunk)
        rows.append(dict(source=str(path), value=h.hexdigest()))
    for path in (ROOT, Path("/home/yixunhu/codespace/kwm-wt-k3d"), Path("/home/yixunhu/codespace/Miniworld-k3d")):
        rows.append(dict(source=f"{path}:git_HEAD", value=subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"], text=True).strip()))
    write_csv("provenance.csv", rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--wait-for-v-lifts", action="store_true")
    args = p.parse_args()
    provenance()
    if args.wait_for_v_lifts:
        paths = clips("k3d_v_validation", args.clips)
        while not all(cache_path(path).exists() for path in paths):
            time.sleep(5)
    commands = []
    if (RESULTS/"commands.csv").exists():
        commands = list(csv.DictReader((RESULTS/"commands.csv").open()))

    def run(module, *arguments):
        cmd = ["bash", str(HERE/"run_cpu.sh"), module, *map(str,arguments)]
        name = "_".join([module,*map(str,arguments)]).replace("--", "")
        log(f"START {shlex.join(cmd)}")
        start = time.perf_counter()
        with (SCRATCH/f"campaign_{name}.log").open("w") as out:
            process = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
        elapsed = time.perf_counter()-start
        commands.append(dict(command=shlex.join(cmd),seconds=f"{elapsed:.3f}",status=process.returncode))
        write_csv("commands.csv",commands)
        log(f"DONE {module}: status={process.returncode}, {elapsed:.1f}s")
        if process.returncode:
            raise RuntimeError(f"{module} failed; inspect {SCRATCH/f'campaign_{name}.log'}")

    # Re-run sanity to retain turn-specific rollout errors; lifts are cached.
    run("released_sanity")
    run("lift", "--split", "k3d_int_validation", "--clips", 20)
    run("geometry", "--clips", 20)
    run("lift", "--split", "k3d_v_validation", "--clips", args.clips)
    run("probe_subcell", "--clips", args.clips)
    for split in ("k3d_null_noblocks", "k3d_null_static"):
        run("lift", "--split", split, "--clips", 50)
    for split, n in (("k3d_null_noblocks",50), ("k3d_null_static",50), ("k3d_int_validation",20)):
        run("null_test", "--split", split, "--clips", n)
    run("estimator_gate", "--clips", args.clips)
    run("report")


if __name__ == "__main__":
    main()
