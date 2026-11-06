"""Summarise kwm3d_bench JSON logs for the pilot chain.
usage: bench_summary.py <bench_c.log> <bench_a_oracle.log> <free_mib>
prints: OK|NOFIT <mean_c> <mean_a> <peak_reserved_c> <peak_reserved_a> <projected_gpu_days>
NOFIT when a log has no JSON record (crash / out of memory) or a peak exceeds the free memory given."""
import json, sys
def last(path):
    rec = None
    try:
        for line in open(path):
            line = line.strip()
            if line.startswith("{") and '"seconds_per_step_mean"' in line:
                rec = json.loads(line)
    except FileNotFoundError:
        pass
    return rec
c, a = last(sys.argv[1]), last(sys.argv[2]); free = float(sys.argv[3])
if c is None or a is None:
    print("NOFIT nan nan nan nan nan"); sys.exit(0)
mc, ma = c["seconds_per_step_mean"], a["seconds_per_step_mean"]
pc, pa = c.get("max_memory_reserved_mib", float("nan")), a.get("max_memory_reserved_mib", float("nan"))
# data loading is outside the benchmark: measured floor 0.353 s/batch at batch 8 from the NAS (review, 2026-11-06)
days = 6000 * (max(mc, 0.35) + max(ma, 0.35)) / 86400
ok = pc == pc and pa == pa and max(pc, pa) * 1.10 + 1024 <= free
print(("OK" if ok else "NOFIT"), f"{mc:.4f} {ma:.4f} {pc:.0f} {pa:.0f} {days:.3f}")
