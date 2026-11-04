import ast
import math
from pathlib import Path
import numpy as np
import pytest
import torch

from algorithms.mem_wm.backbones.kwm3d.geometry import RIGHT_ROW_SIGN
from algorithms.mem_wm.backbones.kwm3d.metrics import (
    block_region_mse, block_region_splits, project_block_corners, project_block_masks, BlockRegionMetric)

ROOT=Path(__file__).resolve().parents[2]


def test_stage1_row_sign():
    path=ROOT/'experiments/flowm_bench/k3d/geometry.py'
    if not path.exists(): pytest.skip('stage 1 geometry not yet present')
    constants={}
    for statement in ast.parse(path.read_text()).body:
        if isinstance(statement,ast.Assign):
            for target in statement.targets:
                if isinstance(target,ast.Name) and target.id in ('ROW_SIGN','RIGHT_ROW_SIGN'):
                    constants[target.id]=ast.literal_eval(statement.value)
    assert constants, 'stage 1 row sign constant not found'
    assert set(constants.values())=={RIGHT_ROW_SIGN}


def test_analytic_projection():
    d=5.
    points=project_block_corners((d,0))
    focal=64/math.tan(math.pi/6)
    assert points[:,0].min()==pytest.approx(63.5-focal*.35/(d-.35))
    assert points[:,0].max()==pytest.approx(63.5+focal*.35/(d-.35))
    assert points[:,1].min()==pytest.approx(63.5)
    assert points[:,1].max()==pytest.approx(63.5+focal*1.5/(d-.35))
    pos=torch.tensor([[[d,0.],[d,2.]]])
    valid=torch.ones(1,2,dtype=torch.bool)
    mask=project_block_masks(pos,valid,per_block=True)
    rows,cols=mask[0,0].nonzero(as_tuple=True)
    assert abs(int(cols.min())-points[:,0].min())<1.5
    assert abs(int(cols.max())-points[:,0].max())<1.5
    assert abs(int(rows.max())-points[:,1].max())<1.5
    assert mask[0,1].nonzero()[:,1].float().mean()>cols.float().mean()
    assert not project_block_masks(torch.tensor([[[-3.,0.]]]),torch.ones(1,1,dtype=torch.bool)).any()
    assert project_block_masks(torch.tensor([[[.2,0.]]]),torch.ones(1,1,dtype=torch.bool)).any()


def test_region_mse_splits_and_accumulation():
    pred=torch.ones(1,2,3,128,128)
    target=torch.zeros_like(pred)
    meta=dict(block_pos_agent=torch.tensor([[[[5.,0.]],[[5.,0.]]]]),
              block_vel_agent=torch.zeros(1,2,1,2),block_speed=torch.tensor([[[.4],[.6]]]),
              block_valid=torch.ones(1,2,1,dtype=torch.bool))
    actions=torch.tensor([[0,4]])
    result=block_region_splits(pred,target,meta,actions)
    assert result['block_mse']==1
    assert result['block_mse_turn']==1 and result['block_mse_nonturn']==1
    assert result['block_mse_speed_0.25_0.5']==1 and result['block_mse_speed_0.5_0.75']==1
    assert torch.isnan(result['block_mse_speed_0.75_1.0'])
    assert torch.isnan(block_region_mse(pred,target,torch.zeros(1,2,128,128,dtype=torch.bool)))
    metric=BlockRegionMetric()
    metric.update(pred,target,meta,actions)
    metric.update(pred*2,target,meta,actions)
    assert metric.compute()['block_mse']==2.5
    metric.reset()
    assert metric.pixels.count_nonzero()==0


def test_real_mask_iou():
    if not (ROOT/'data/blockworld/k3d_v_validation').exists(): pytest.skip('k3d_v_validation absent')
    from scripts.kwm3d_check_masks import check_masks
    report=check_masks()
    assert report['frames']==20
    assert report['median_iou']>=.6, report
