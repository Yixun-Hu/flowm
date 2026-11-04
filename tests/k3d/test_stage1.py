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
