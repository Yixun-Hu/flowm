"""Audit the external KWM update without modifying the editable checkout."""
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import time
import torch
from .common import SCRATCH, clips, write_csv, log
from .lift import projected_box
from .geometry import camera_action
from .followup import get_lift, load_trajectory, support_for, append_result, table

KWM = Path("/home/yixunhu/codespace/kwm-wt-k3d")


def snapshot(revision):
    source = subprocess.check_output(["git","-C",str(KWM),"show",f"{revision}:kwm/lock_nodes.py"])
    path = SCRATCH/f"followup_lock_nodes_{revision}.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location(f"kwm._followup_{revision}",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, hashlib.sha256(source).hexdigest()


@torch.inference_mode()
def main():
    start = time.perf_counter()
    original, old_hash = snapshot("2e73903")
    updated, new_hash = snapshot("c7067d3")
    from kwm.collision import named_nodes
    previous,current,prior,actions = [],[],[],[]
    for source in ("obs","mem"):
        for split in ("k3d_null_noblocks","k3d_null_static","k3d_v_validation"):
            path = clips(split,1)[0]
            cache = get_lift(path,source)
            d = load_trajectory(path,source=source)
            times = (~d["turns"]).nonzero().flatten()[[0,int((~d["turns"]).sum())//2,-1]]+1
            boxes = projected_box(cache["tokens"],8)
            previous.append(boxes[times-1]); current.append(boxes[times])
            prior.append(d["prior"][times-1]); actions.append(cache["actions"][times])
    previous,current,prior,actions = map(torch.cat,(previous,current,prior,actions))
    args = (previous,current,prior,named_nodes("v17"),camera_action(actions))
    rows = []
    for mode in ("subtract","mask"):
        for region in ("all","near","far"):
            for sigma in (.5,1.):
                kwargs = dict(world=33,box=(7,16),support_mask=support_for(region),sigma=sigma,
                              radius=None,static_residual=mode,return_diagnostics=True)
                a = original.measure_nodes(*args,**kwargs)
                b = updated.measure_nodes(*args,**kwargs)
                equal = all(torch.equal(x,y) for x,y in zip(a[:3],b[:3]))
                equal &= all(torch.equal(a[3][name],b[3][name]) for name in a[3])
                difference = max((x-y).abs().max().item() for x,y in zip(a[:3],b[:3]))
                assert equal, f"dependency changed outputs: {mode}/{region}/{sigma}"
                rows.append(dict(source="observation-only + memory post-gate max-V tokens",mode=mode,
                                 region=region,sigma=sigma,pairs=len(previous),bitwise_equal=equal,
                                 max_difference=difference))
    seconds = time.perf_counter()-start
    write_csv("followup_kwm_parity.csv",rows)
    write_csv("followup_kwm_audit_time.csv",[dict(seconds=seconds)])
    write_csv("followup_kwm_revisions.csv",[
        dict(revision="2e73903",sha256=old_hash,stages="mask; fast tests; breakdown"),
        dict(revision="c7067d3",sha256=new_hash,stages="attribution; memory; patch imports (interpolation independent)"),
    ])
    append_result("dependency-audit","### Dependency update audit\n\n" +
        "The external KWM checkout changed from `2e73903` to `c7067d3` during execution. `lock_nodes.py` was updated at 18:43:18 EST (commit 18:44:03), after the mask, tests and breakdown processes had imported the original code, and before patch/attribution/memory processes started. The sole diff adds optional `support_mask_key=None` forwarding; this campaign never supplies that argument. No KWM source was edited by this work. Both revisions were loaded from git into local scratch snapshots and compared on 18 real observation/memory pairs, including adaptive priors, all three support masks, both residual modes and both sigma values. All three outputs and every diagnostic tensor were bitwise identical in all 216 pair/configuration cases. This establishes parity for the used path, not for the new optional feature.\n\n" +
        table(rows,["source","mode","region","sigma","pairs","bitwise_equal","max_difference"]) +
        f"\n\nAudit runtime: {seconds:.2f} s. Source hashes and stage mapping are in `followup_kwm_revisions.csv`; initial provenance remains intact.")
    log(f"KWM audit: 216 pair/configuration cases bitwise identical, {seconds:.2f}s")


if __name__ == "__main__":
    main()
