"""CPU checks; no cache generation and no network access (<1 minute)."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / "experiments/flowm_bench/k3d/.scratch"
os.environ.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                  WANDB_MODE="offline", MPLCONFIGDIR=str(SCRATCH/"mpl"),
                  XDG_CACHE_HOME=str(SCRATCH/"cache"), TMPDIR=str(SCRATCH))
sys.path.insert(0, str(ROOT / "experiments/flowm_bench"))

import csv
import numpy as np
import pytest
import torch
from k3d.geometry import ROW_SIGN, cone_mask, rounded_cells, visible, camera_action
from k3d.common import load_model, ReadOnlyClips, clips, companion
from k3d.lift import observation_lift, tokens_to_box, box_to_tokens, projection
from k3d.matching import match_velocities, nearest_integer
from k3d.measurement import NodeTracker, agreement_diagnostic


def test_geometry_sign_and_actions():
    assert ROW_SIGN == 1
    assert rounded_cells(torch.tensor([[4., 2.], [4., -2.]])).tolist() == [[18,20],[14,20]]
    model = load_model()
    mask = model.map_processor.compute_fov_mask(torch.device("cpu"))
    assert torch.equal(mask, cone_mask()) and int(mask.sum()) == 177
    assert mask.nonzero().amin(0).tolist() == [7,16]
    assert mask.nonzero().amax(0).tolist() == [25,32]
    # A point ahead moves to agent-right on a left turn. In map coordinates
    # this must move down: the released roll_map is clockwise for action 0.
    h = torch.zeros(1,5,33,33,1)
    h[0,0,16,20,0] = 1
    p = model.map_processor
    assert p.roll_map(h, torch.tensor([0]))[0,0,20,16,0] == 1
    assert p.roll_map(h, torch.tensor([1]))[0,0,12,16,0] == 1
    assert p.roll_map(h, torch.tensor([2]))[0,0,16,19,0] == 1
    assert torch.equal(p.roll_map(h, torch.tensor([4])), h)
    assert camera_action(torch.tensor([0,1,2,4])).tolist() == [[0.,0.],[0.,0.],[1.,0.],[0.,0.]]


def test_empirical_geometry_evidence():
    path = ROOT / "experiments/flowm_bench/k3d/results/geometry.csv"
    if not path.exists():
        pytest.skip("Run geometry CLI to generate the empirical evidence")
    rows = list(csv.DictReader(path.open()))
    for signal in ("tokens", "g"):
        subset = [r for r in rows if r["signal"] == signal and r["normalization"] == "cell_centered"]
        assert sum(float(r["plus_sum"])-float(r["minus_sum"]) for r in subset) > 0


def test_lift_shape_identity_and_dataset_alignment():
    path = clips("k3d_int_validation",1)[0]
    item = ReadOnlyClips([path])[0]
    raw = torch.load(companion(path,"actions.pt"), weights_only=False)["actions"]
    assert torch.equal(item["conds"].long(), torch.cat([torch.zeros(1,dtype=torch.long),raw[:139]]))
    model = load_model()
    torch.backends.mha.set_fastpath_enabled(False)
    lifted = observation_lift(model, item["videos"][:2])
    assert set(lifted) == {"tokens","g","h_new"}
    assert all(v.shape == (2,177,256) for v in lifted.values())
    assert torch.allclose(lifted["tokens"], lifted["g"]*lifted["h_new"])
    # The function asserts identity across all five channels before dropping V.
    assert lifted["tokens"].abs().sum() > 0


def test_box_roundtrip_projection_and_boundary():
    x = torch.randn(2,177,256)
    box, mask = tokens_to_box(x)
    assert box.shape == (2,256,19,17) and mask.shape == (19,17)
    assert torch.equal(box_to_tokens(box), x)
    assert torch.count_nonzero(box[:,:,~mask]) == 0
    for k in (8,32):
        p = projection(k)
        assert p.shape == (k,256)
        assert torch.allclose(p@p.T, torch.eye(k), atol=1e-6)
    assert not visible(torch.tensor([0.,0.]))
    assert rounded_cells(torch.tensor([1.5,-.5])).tolist() == [16,18]


def test_hungarian_unique_misses_false_alarms_and_empty_cases():
    r = match_velocities([[.25,0],[.75,0],[0,2]], [[.26,0],[.76,0]])
    np.testing.assert_allclose(r["error"], [.01,.01])
    assert not r["miss"].any() and r["false"].tolist() == [False,False,True]
    r = match_velocities([[0,0]], [[.1,0],[.2,0]])
    assert sum(r["miss"]) == 1 and sum(r["assignment"] >= 0) == 1
    assert not match_velocities([[0,0]], [[.5,0]])["miss"][0]
    assert match_velocities([], [[1,0]])["miss"][0]
    assert match_velocities([[1,0]], [])["false"][0]
    np.testing.assert_allclose(nearest_integer(np.array([[.3,.4],[1,0]])), [.5,0])


def test_static_pin_and_turn_state():
    tracker = NodeTracker(1, 1.)
    x = torch.randn(1,8,19,17)
    result = tracker.step(x,x,torch.tensor([4]))
    assert torch.count_nonzero(result["evidence"]) == 0
    assert not result["active"].any()
    assert torch.equal(tracker.rho[:,0], torch.zeros(1,2))
    tracker.rho[0,1] = torch.tensor([.2,-.1])
    original = tracker.nu.clone()
    tracker.rotate(0,0)
    tracker.rotate(0,1)
    assert torch.equal(tracker.nu, original)


def test_agreement_observer_preserves_production_measurement():
    from kwm.lock_nodes import measure_nodes
    tracker = NodeTracker(1,1.)
    g = torch.Generator().manual_seed(42)
    previous = torch.randn(1,8,19,17,generator=g)
    current = previous.roll(1,-1) + .01*torch.randn(previous.shape,generator=g)
    args = (previous,current,tracker.nu,tracker.lattice,torch.zeros(1,2))
    kwargs = dict(world=33,box=(7,16),support_mask=cone_mask()[7:26,16:33],
                  sigma=1.,radius=None,static_residual=True)
    with torch.inference_mode():
        original = measure_nodes(*args,**kwargs)
        with agreement_diagnostic() as diagnostic:
            observed = measure_nodes(*args,**kwargs)
    assert all(torch.equal(a,b) for a,b in zip(original,observed))
    assert diagnostic["agreement"].shape == (1,17)
    assert (diagnostic["agreement"] >= 0).all()


def test_single_patch_subpixel_motion_and_ego():
    from k3d.patch_motion import single_patch_search, interpolation_matrix
    # Smooth translated content, with the intrinsic/ego decomposition explicit.
    yy, xx = torch.meshgrid(torch.arange(19)+7, torch.arange(17)+16, indexing="ij")
    def scene(dx, dy):
        channels = []
        for ox, oy, width in ((0.,0.,1.2),(.7,-.4,1.5),(-.5,.8,1.)):
            channels.append(torch.exp(-((xx-24-ox-dx)**2+(yy-16-oy-dy)**2)/(2*width**2)))
        return torch.stack(channels)
    velocity = torch.tensor([[.3,-.6],[.3,-.6],[1.,0.]])
    ego = torch.tensor([[0.,0.],[1.,0.],[1.,0.]])
    previous = scene(0,0)[None].expand(3,-1,-1,-1)
    current = torch.stack([scene(*(v-a)) for v,a in zip(velocity,ego)])
    result = single_patch_search(previous,current,torch.tensor([[9,8]]).expand(3,-1),ego=ego)
    assert result["usable"].all() and not result["boundary"].any()
    torch.testing.assert_close(result["velocity"],velocity,atol=.08,rtol=0)
    # Exact Fourier interpolation preserves integer samples.
    torch.testing.assert_close(interpolation_matrix(33)[::2],torch.eye(33),atol=1e-6,rtol=0)
    # Empty support is an explicit unusable result, never a valid zero match.
    empty = single_patch_search(previous[:1],current[:1],torch.tensor([[9,8]]),
                                support=torch.zeros(19,17,dtype=torch.bool))
    assert not empty["usable"].any()


def test_alarm_breakdown_bookkeeping():
    from k3d.alarm_breakdown import action_categories, breakdown_counts
    # Placeholder action 0 is NOT a turn at t=0. Turns at incoming t=3 and t=7.
    actions = torch.tensor([0,2,4,0,2,4,2,1,4,2])
    assert action_categories(actions).tolist() == [
        "forward","turn-adjacent","turn","turn-adjacent","no-op",
        "turn-adjacent","turn","turn-adjacent","forward"]
    v = torch.zeros(9,17,2)
    v[:,1] = torch.tensor([1.,0.])
    v[:,2] = torch.tensor([0.,1.])
    agreement = torch.zeros(9,17)
    agreement[:,1] = .1
    agreement[:,2] = .099
    turns = (actions[1:] == 0) | (actions[1:] == 1)
    active = agreement >= .1
    d = dict(actions=actions,turns=turns,measured=v,velocity=v,agreement=agreement,active=active,
             evidence=active.float())
    for kind in ("alarm","locked"):
        rows, _, selected, _ = breakdown_counts(d,kind)
        overall = rows[0]
        assert overall["detections"] == 7 and overall["node_steps"] == 7*17
        assert overall["near_forward"] == 7 and overall["near_zero"] == 0
        assert overall["evidenced_detections"] == overall["locked_detections"] == 7
        assert sum(r["detections"] for r in rows[1:]) == overall["detections"]
        assert sum(r["node_steps"] for r in rows[1:]) == overall["node_steps"]
        assert rows[-1]["steps"] == 2 and rows[-1]["detections"] == 0
        assert not selected[turns].any()


def test_all_node_evidence_handles_multiple_truths_and_zero_denominator():
    from k3d.followup import evidence_step
    nodes = torch.tensor([[0.,0.],[.3,0.],[-.8,0.],[1.,1.]])
    target = torch.tensor([[.3,0.],[-.8,0.]])
    r = evidence_step(torch.tensor([0.,2.,3.,1.]),nodes,target)
    assert r["wins"] == 1 and r["any_win"] and not r["all_win"]
    assert r["best_wrong_ratio"] == pytest.approx(1/3)
    r = evidence_step(torch.tensor([0.,0.,0.,1.]),nodes,target)
    assert np.isinf(r["best_wrong_ratio"]) and not r["any_win"]
    assert np.isnan(evidence_step(torch.zeros(4),nodes,target)["best_wrong_ratio"])


def test_memory_context_adapter_matches_released_context_writes():
    from k3d.memory_lift import context_writes
    model = load_model()
    torch.backends.mha.set_fastpath_enabled(False)
    item = ReadOnlyClips(clips("k3d_int_validation",1))[0]
    x = item["videos"][:3][None]
    a = torch.tensor([[0,2,0]])  # ignore first action, then forward, then turn
    captured = []
    def hook(module, args, output):
        captured.append(module.extract_view_tokens(output,args[2])[0].max(1).values.half())
    with torch.inference_mode():
        handle = model.map_processor.register_forward_hook(hook)
        try:
            model(x,a,torch.tensor([[4]]))
        finally:
            handle.remove()
        reference = torch.stack(captured[:3],1)
        observed = context_writes(model,x,a)
    assert torch.equal(reference,observed)
