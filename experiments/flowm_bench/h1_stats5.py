import json, glob, numpy as np, os
from scipy import stats
roots = [p for p in glob.glob("/home/yixunhu/codespace/*/results")]
CK = ["ckpt_step16000","ckpt_step17000","ckpt_step18000","ckpt_step19000","ckpt_step20000"]
def load(run, tag):
    """mean over ckpts 16k-20k of mse20/mse150 on the held-out seed 20261025; returns (m20, m150, per-ckpt list)"""
    out = []
    for ck in CK:
        fs = []
        for r in roots:
            fs += glob.glob(f"{r}/{run}/eval_seed20261025_{ck}_{tag}.json")
        if not fs: return None
        d = json.load(open(sorted(fs)[0])); out.append((d["mse20"], d["mse150"]))
    a = np.array(out); return a[:,0].mean(), a[:,1].mean(), a
arms = {"lk flowm": ("h1_lk_s{s}", "flowm_predon"), "lk zero": ("h1_lk_s{s}", "zero_predon_deczero"),
        "lock flowm": ("h1_lock_s{s}", "flowm_predon"), "lock zero": ("h1_lock_s{s}", "zero_predon_deczero"),
        "lock ztf30": ("h1_lock_s{s}", "ztf30_predon_deczero_then_frozen30"),
        "flowm native": ("h1_flowm_s{s}", "flowm"), "flowm zero": ("h1_flowm_s{s}", "zero_deczero")}
res = {}
for name, (pat, tag) in arms.items():
    res[name] = {}
    for s in range(5):
        run = pat.format(s=s)
        if s == 0 and run.startswith("h1_l"): run = run.replace("h1_", "pilot_").replace("_s0", "")
        r = load(run, tag)
        if r is not None: res[name][s] = r
    row = "  ".join(f"s{s}: {res[name][s][0]:.4f}/{res[name][s][1]:.4f}" for s in sorted(res[name]))
    m20 = np.mean([v[0] for v in res[name].values()]); m150 = np.mean([v[1] for v in res[name].values()])
    print(f"{name:14s} n={len(res[name])}  {row}  | mean {m20:.4f}/{m150:.4f}")
print("seed 4 per-ckpt mse150: lk zero", np.round(res['lk zero'][4][2][:,1],4).tolist() if 4 in res['lk zero'] else None, "lock zero", np.round(res['lock zero'][4][2][:,1],4).tolist() if 4 in res['lock zero'] else None)
def paired(a, b):
    seeds = sorted(set(res[a]) & set(res[b]))
    for j, h in ((0, "@20"), (1, "@150")):
        x = np.array([res[a][s][j] for s in seeds]); y = np.array([res[b][s][j] for s in seeds])
        rel = (y - x) / x * 100; t = stats.ttest_rel(y, x)
        print(f"  {a} -> {b} {h}: mean {rel.mean():+.1f}% per-seed {np.round(rel,1).tolist()} p={t.pvalue:.3f} (n={len(seeds)})")
for a, b in [("lk flowm","lock flowm"),("lk zero","lock zero"),("lk flowm","lock zero"),("lock flowm","lock zero"),("lock zero","lock ztf30")]:
    paired(a, b)
