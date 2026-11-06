"""Oracle-location 3x3 SSD motion visibility, independent of node estimation."""
import time
from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F
from .common import clips, clip_id, truth, write_csv, log, SCRATCH
from .geometry import rounded_cells, visible, camera_action, BOX
from .lift import read_cache, tokens_to_box, projected_box
from .estimator_gate import intrinsic_incoming
from .followup import append_result, table, support_for


@lru_cache(maxsize=4)
def interpolation_matrix(n):
    """Exact real Fourier interpolation on an odd torus, at half-cell spacing."""
    assert n % 2 == 1
    delta = torch.arange(2*n, dtype=torch.float64)[:, None]/2 - torch.arange(n)[None]
    frequency = torch.arange(1, (n-1)//2+1, dtype=torch.float64)
    return ((1+2*torch.cos(2*torch.pi*delta[..., None]*frequency/n).sum(-1))/n).float()


def refine_costs(costs, radius=1.5):
    """Same half-cell grid and axiswise parabola as KWM LocalSearch."""
    side = int(4*radius+1)
    index = costs.argmin(-1)
    center = costs.gather(1, index[:, None])[:, 0]
    iy, ix = index//side, index%side
    refined = []
    for coordinate, stride in ((ix, 1), (iy, side)):
        left = costs.gather(1, (index-stride).clamp_min(0)[:, None])[:, 0]
        right = costs.gather(1, (index+stride).clamp_max(side*side-1)[:, None])[:, 0]
        curvature = left+right-2*center
        offset = (.5*(left-right)/curvature.clamp_min(1e-8)).clamp(-.5,.5)
        interior = ((coordinate > 0) & (coordinate < side-1) & (curvature > 1e-8)
                    & torch.isfinite(left) & torch.isfinite(right))
        refined.append(-radius+.5*(coordinate+torch.where(interior,offset,0)))
    # Edge minima are valid bounded-search results, but cannot be refined.
    boundary = (ix == 0) | (ix == side-1) | (iy == 0) | (iy == side-1)
    return torch.stack(refined,-1), torch.isfinite(center), boundary, center


@torch.inference_mode()
def single_patch_search(previous, current, centers, pair_index=None, ego=None, support=None):
    """Match one 3x3 patch per supplied center; inputs [B,C,19,17].

    Centers are box-local (row,column) at t-1. Search intrinsic displacement in
    [-1.5,1.5]^2 around center-ego, so ego is removed exactly once. The current
    image is Fourier interpolated on the zero-filled 33x33 torus, like LocalSearch.
    Uniform 3x3 SSD is normalized by shared bilinear validity, with >=4.5 valid
    cells required. All visible pairs remain in the scoring denominator.
    """
    if pair_index is None:
        pair_index = torch.arange(len(centers))
    if ego is None:
        ego = previous.new_zeros(len(centers), 2)
    if support is None:
        support = support_for("all")
    assert previous.device.type == "cpu"
    n, channels, h, w = current.shape
    world = current.new_zeros(n,channels,33,33)
    world[:,:,BOX[0]:BOX[0]+h,BOX[1]:BOX[1]+w] = current * support
    matrix = interpolation_matrix(33)
    up = matrix @ world @ matrix.T
    py, px = torch.meshgrid(torch.arange(-1,2),torch.arange(-1,2),indexing="ij")
    patch = torch.stack([py.flatten(),px.flatten()],-1)
    coordinates = centers[:,None]+patch
    in_box = ((coordinates >= 0) & (coordinates < torch.tensor([h,w]))).all(-1)
    iy = coordinates[...,0].clamp(0,h-1)
    ix = coordinates[...,1].clamp(0,w-1)
    prior_valid = in_box & support[iy,ix]
    template = previous[pair_index[:,None],:,iy,ix]  # N,9,C
    axis = torch.arange(-1.5,1.501,.5)
    sy,sx = torch.meshgrid(axis,axis,indexing="ij")
    shifts = torch.stack([sy.flatten(),sx.flatten()],-1)
    coords = coordinates[:,None].float()+shifts[None,:,None]-ego[:,None,None,[1,0]]
    # Candidate validity uses bilinear support, exactly as the production search.
    grid = 2*coords.flip(-1)/torch.tensor([w-1,h-1])-1
    valid = F.grid_sample(support[None,None].float().expand(len(centers),-1,-1,-1),
                          grid,align_corners=True)[:,0] * prior_valid[:,None]
    mass = valid.sum(-1)
    world_coords = ((coords+coords.new_tensor(BOX))*2).round().long().remainder(66)
    costs = []
    for lo in range(0,49,7):
        q = world_coords[:,lo:lo+7]
        samples = up[pair_index[:,None,None],:,q[...,0],q[...,1]]
        ssd = (samples-template[:,None]).square().sum(-1)
        cost = (ssd*valid[:,lo:lo+7]).sum(-1)/mass[:,lo:lo+7].clamp_min(1e-8)
        costs.append(cost.masked_fill(mass[:,lo:lo+7] < 4.5,float("inf")))
    costs = torch.cat(costs,1)
    velocity, usable, boundary, cost = refine_costs(costs)
    return dict(velocity=velocity, usable=usable, boundary=boundary, cost=cost)


def summarize(rows, source, k):
    output = []
    for group, bins in (("all", [("all", lambda r: True)]),
                        ("speed", [(".25-.5", lambda r: .25 <= r["speed"] < .5),
                                   (".5-.75", lambda r: .5 <= r["speed"] < .75),
                                   (".75-1", lambda r: .75 <= r["speed"] <= 1.000001)]),
                        ("distance", [("[1,5)", lambda r: 1 <= r["distance"] < 5),
                                      ("[5,9)", lambda r: 5 <= r["distance"] < 9),
                                      ("[9,13)", lambda r: 9 <= r["distance"] < 13),
                                      ("[13,17)", lambda r: 13 <= r["distance"] < 17)]),
                        ("support_distance", [("rounded_x<=6", lambda r: r["cell_x"] <= 6),
                                              ("rounded_x>=7", lambda r: r["cell_x"] >= 7)])):
        for name, predicate in bins:
            rr = [r for r in rows if predicate(r)]
            errors = np.array([r["error"] for r in rr])
            output.append(dict(source=source,k=k,group=group,bin=name,pairs=len(rr),
                               usable=sum(r["usable"] for r in rr),
                               median=float(np.median(errors)) if len(rr) else float("nan"),
                               p90=(float("inf") if len(rr) and np.isinf(np.sort(errors)[int(np.ceil(.9*(len(rr)-1)))])
                                    else float(np.quantile(errors,.9)) if len(rr) else float("nan")),
                               within_025=float((errors <= .25).mean()) if len(rr) else float("nan"),
                               boundary_fraction=float(np.mean([r["boundary"] for r in rr])) if rr else float("nan")))
    return output


@torch.inference_mode()
def main():
    from .trained_lift import dispatch_checkpoint
    if dispatch_checkpoint("patch"):
        return
    start = time.perf_counter()
    paths = clips("k3d_v_validation",200)
    for variant,k in (("tokens",8),("tokens",32),("h_new",8),("h_new",32),("tokens",256)):
        rows = []
        source = "observation-only "+variant
        for ci,path in enumerate(paths):
            dest = SCRATCH/"patch_motion"/f"{clip_id(path).replace('/','_')}_{variant}_k{k}_v1.pt"
            if dest.exists():
                rows.extend(torch.load(dest,weights_only=True))
                continue
            cache = read_cache(path)
            boxes = (tokens_to_box(cache[variant].float())[0] if k == 256 else projected_box(cache[variant],k))
            labels = truth(path)
            pos = labels["block_pos_agent"][:140]
            vis = visible(pos)
            actions = cache["actions"]
            local = []
            for lo in range(1,140,16):
                ts = torch.arange(lo,min(lo+16,140))
                ts = ts[(actions[ts] != 0) & (actions[ts] != 1)]
                pair,block = (vis[ts] & vis[ts-1]).nonzero(as_tuple=True)
                if not len(pair):
                    continue
                t = ts[pair]
                centers = rounded_cells(pos[t-1,block])-torch.tensor(BOX)
                result = single_patch_search(boxes[ts-1],boxes[ts],centers,pair,
                                             camera_action(actions[t]))
                targets = torch.stack([intrinsic_incoming(labels,int(tt))[bb] for tt,bb in zip(t,block)])
                error = (result["velocity"]-targets).norm(dim=-1)
                error[~result["usable"]] = float("inf")
                for i in range(len(t)):
                    local.append(dict(source=source,k=k,clip=clip_id(path),frame=int(t[i]),block=int(block[i]),
                        speed=float(labels["block_speed"][t[i]-1,block[i]]),
                        distance=float(pos[t[i]-1,block[i],0]),cell_x=int(centers[i,1]),
                        vx=float(result["velocity"][i,0]),vy=float(result["velocity"][i,1]),
                        true_vx=float(targets[i,0]),true_vy=float(targets[i,1]),error=float(error[i]),
                        usable=bool(result["usable"][i]),boundary=bool(result["boundary"][i])))
            dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save(local,dest.with_suffix(".tmp"))
            dest.with_suffix(".tmp").replace(dest)
            rows.extend(local)
            if (ci+1)%10 == 0:
                log(f"patch {variant} K={k} {ci+1}/200")
        summary = summarize(rows,source,k)
        prefix = f"followup_patch_{variant}_k{k}"
        write_csv(prefix+"_pairs.csv",rows)
        write_csv(prefix+"_summary.csv",summary)
        per_clip = []
        for path in paths:
            cc = [r for r in rows if r["clip"] == clip_id(path)]
            per_clip.append(dict(clip=clip_id(path),**summarize(cc,source,k)[0]))
        write_csv(prefix+"_clips.csv",per_clip)
        append_result(f"patch-{variant}-{k}",f"### 3. Direct block-motion visibility — {source}, K={k}\n\n" +
            table(summary,["source","k","group","bin","pairs","usable","median","p90","within_025","boundary_fraction"]) +
            "\n\nAll jointly cone-visible block/nonturn pairs from 200 clips, with the original no-occlusion-filter convention. Previous 3×3 patch is centred at the rounded previous block cell. Search uses a half-cell grid in ±1.5 intrinsic cells around the ego-transported centre, uniform patch SSD, exact Fourier interpolation on the zero-filled 33×33 torus, and axiswise parabolic refinement as in LocalSearch. Shared bilinear cone validity normalizes SSD; at least 4.5 of nine cells are required. Unusable pairs count as infinite error and are included in the within-.25 denominator. Boundary minima are retained and reported. Distance is previous rel_x; K=256 is raw tokens, without projection. This is an oracle-location visibility diagnostic, with no preregistered pass threshold; it is not a mathematical upper bound on every possible estimator.")
        log(str(summary[0]))
    seconds = time.perf_counter()-start
    write_csv("followup_patch_time.csv",[dict(seconds=seconds)])
    log(f"DONE patch {seconds:.2f}s")


if __name__ == "__main__":
    main()
