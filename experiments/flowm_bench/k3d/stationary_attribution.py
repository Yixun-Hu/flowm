"""Locate cone-stationary content at the cells actually supporting null alarms."""
from contextlib import contextmanager
import time
import torch
import kwm.lock_nodes as lock_nodes
from kwm.collision import named_nodes
from .common import clips, clip_id, write_csv, log
from .lift import read_cache, projected_box
from .estimator_gate import trajectory_path
from .followup import OBS, RESULTS, support_for, append_result, table


@contextmanager
def observe_selected_weights():
    original = lock_nodes.two_modes
    captured = {}
    def observer(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["weight"] = result[2].detach().clone()
        return result
    lock_nodes.two_modes = observer
    try:
        yield captured
    finally:
        lock_nodes.two_modes = original


@torch.inference_mode()
def main():
    if (RESULTS/"followup_stationary_attribution_summary.csv").exists():
        return
    start = time.perf_counter()
    lattice = named_nodes("v17")
    support = support_for("all")
    overlap = support.clone()
    overlap[:, :-1] &= support[:, 1:]
    overlap[:, -1] = False
    rows = []
    max_difference = 0.
    for path in clips("k3d_null_noblocks",50):
        cache = read_cache(path)
        boxes = projected_box(cache["tokens"],8)
        data = torch.load(trajectory_path(path,"tokens",8,1.),weights_only=True)
        times = (cache["actions"][1:] == 2).nonzero().flatten()+1
        for lo in range(0,len(times),8):
            t = times[lo:lo+8]
            prior = data["velocity"][(t-2).clamp_min(0)].clone()
            prior[t == 1] = lattice
            with observe_selected_weights() as observed:
                velocity,evidence,locked = lock_nodes.measure_nodes(
                    boxes[t-1],boxes[t],prior,lattice,torch.tensor([[1.,0.]]).expand(len(t),-1),
                    world=33,box=(7,16),support_mask=support,sigma=1.,radius=None,static_residual="subtract")
            # Batch shape can change fp32 rounding. Audit the reconstruction rather
            # than silently treating newly computed weights as original weights.
            delta = (velocity-data["measured"][t-1]).abs().max().item()
            max_difference = max(max_difference,delta)
            assert delta < 2e-3, f"prior reconstruction changed estimates: {delta}"
            weight = observed["weight"]
            alarm = (data["agreement"][t-1] >= .1) & (data["measured"][t-1].norm(dim=-1) > .15)
            unit = alarm & ((data["measured"][t-1]-torch.tensor([1.,0.])).norm(dim=-1) <= .25)
            same = (boxes[t]-boxes[t-1]).square().sum(1)
            ego = (boxes[t]-boxes[t-1].roll(-1,-1)).square().sum(1)
            for name, nodes in (("all_alarms",alarm),("near_ego_forward",unit)):
                w = (weight*nodes[:,:,None,None]).sum(1)*overlap
                for region in ("all","near","far"):
                    wr = w*support_for(region)
                    mass = wr.sum((-2,-1))
                    for j,tt in enumerate(t):
                        rows.append(dict(source=OBS,clip=clip_id(path),frame=int(tt),subset=name,region=region,
                            mass=float(mass[j]),same_ssd_mass=float((wr[j]*same[j]).sum()),
                            ego_ssd_mass=float((wr[j]*ego[j]).sum()),
                            same_better_mass=float((wr[j]*(same[j] < ego[j])).sum())))
    write_csv("followup_stationary_attribution_steps.csv",rows)
    pooled = []
    for subset in ("all_alarms","near_ego_forward"):
        for region in ("all","near","far"):
            rr = [r for r in rows if r["subset"] == subset and r["region"] == region]
            mass = sum(r["mass"] for r in rr)
            a = sum(r["same_ssd_mass"] for r in rr)
            b = sum(r["ego_ssd_mass"] for r in rr)
            pooled.append(dict(source=OBS,subset=subset,region=region,pairs=len(rr),
                evidenced_pairs=sum(r["mass"] > 0 for r in rr),mass=mass,
                same_ssd=a/max(mass,1e-12),ego_ssd=b/max(mass,1e-12),
                same_to_ego=a/max(b,1e-12),same_better_mass=sum(r["same_better_mass"] for r in rr)/max(mass,1e-12)))
    seconds = time.perf_counter()-start
    write_csv("followup_stationary_attribution_summary.csv",pooled)
    write_csv("followup_stationary_attribution_time.csv",[dict(seconds=seconds,max_estimate_difference=max_difference)])
    append_result("stationary-attribution","### 2. Evidence-localized cone-stationarity — observation-only tokens\n\n" +
        table(pooled,["source","subset","region","pairs","evidenced_pairs","mass","same_ssd","ego_ssd","same_to_ego","same_better_mass"]) +
        "\n\nAdditional attribution check because whole-region SSD did not establish global far-field stationarity. All 3,939 empty-room forward pairs are re-evaluated at their cached subtraction priors, sigma=1/K=8. A local observer returns the production `two_modes` result unchanged and copies its selected pixel weights. We compare same-cell and ego-transported **plain** imprint SSD at those selected alarm-evidence locations, on common valid support. `near_ego_forward` restricts alarm velocities to radius .25 around (+1,0). This distinguishes selected nuisance cells from the rest of the far field; it does not distinguish walls, floor texture, or learned positional features. " +
        f"The maximum reconstructed/cached velocity component difference was {max_difference:.3g}; runtime {seconds:.2f} s.")
    log(f"stationary attribution {seconds:.2f}s max difference={max_difference:.3g}")


if __name__ == "__main__":
    main()
