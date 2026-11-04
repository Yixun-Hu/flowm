from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from algorithms.flowm_video import FloWMVideo
from algorithms.mem_wm.backbones import create_main_model
from algorithms.mem_wm.backbones.flowm.flowm_models_3d import ViTBlockworldModel
from algorithms.mem_wm.backbones.kwm3d import KineticBlockworldModel

ROOT=Path(__file__).resolve().parents[2]


def test_all_configs_and_screening():
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configurations')):
        for arm in ('c','c_pw','a_oracle','a','b'):
            cfg=compose(config_name='config',overrides=[f'shortcode=exp/blockworld/kwm3d/train/{arm}'])
            assert cfg.algorithm.backbone.arm==arm
            assert cfg.algorithm.backbone.model_type=='kwm3d'
            assert cfg.algorithm.backbone.checkpoint_steps and cfg.dataset.load_block_info
            assert cfg.experiment.training.batch_size==8
            assert list(cfg.experiment.training.splits_included)==['k3d_v_training']
        cfg=compose(config_name='config',overrides=['shortcode=[exp/blockworld/kwm3d/train/a,exp/blockworld/kwm3d/screening_finetune]'])
        assert cfg.dataset.max_frames==90 and cfg.algorithm.tasks.prediction.context_frames==50
        assert cfg.algorithm.load_model_state_mode=='released'
        assert cfg.experiment.training.max_steps==5000 and cfg.load is None
        cfg=compose(config_name='config',overrides=['shortcode=exp/blockworld/kwm3d/infer/metrics_140/k3d_v_70ctx'])
        assert cfg.dataset.n_frames==140 and cfg.algorithm.tasks.prediction.context_frames==70
        assert 'block_mse' in cfg.algorithm.logging.metrics


def test_backbone_factory():
    cfg=OmegaConf.create(dict(model_type='kwm3d',arm='b',hidden_channels=32,map_proc_encoder_depth=1,
                             decoder_depth=1,num_heads=4,input_shape='(3,32,32)',patch_size=8))
    assert isinstance(create_main_model(cfg),KineticBlockworldModel)


def test_metadata_is_signature_gated_and_released_mode_loads():
    # Real algorithm/model configuration; no trainer, GPU, or training launch.
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configurations')):
        cfg=compose(config_name='config',overrides=['shortcode=[exp/blockworld/kwm3d/train/b,exp/blockworld/kwm3d/screening_finetune]'])
    if not (ROOT/cfg.algorithm.load_model_state).exists(): pytest.skip('released checkpoint absent')
    algo=FloWMVideo(cfg.algorithm)
    algo.configure_model()
    assert algo._backbone_accepts_metadata
    assert hasattr(algo,'block_region_metric')
    assert set(algo._metrics('prediction').keys())=={'mse','psnr'}
    assert algo._metadata_kwargs({'test':1})=={'metadata':{'test':1}}
    # Verify the same introspection path on an unmodified FloWM backbone.
    with patch('algorithms.mem_wm.backbones.create_main_model',return_value=ViTBlockworldModel()):
        algo.cfg.load_model_state=None
        algo.configure_model()
    assert not algo._backbone_accepts_metadata
    assert algo._metadata_kwargs({'test':1})=={}


def test_validation_block_metric_logging(monkeypatch):
    from algorithms.common import BasePytorchAlgo
    from algorithms.mem_wm.backbones.kwm3d.metrics import BlockRegionMetric
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configurations')):
        cfg=compose(config_name='config',overrides=['shortcode=exp/blockworld/kwm3d/train/b'])
    cfg.algorithm.tasks.prediction.context_frames=2
    cfg.algorithm.logging.n_metrics_frames=3  # only first prediction contributes
    algo=FloWMVideo(cfg.algorithm)
    algo._trainer=SimpleNamespace(sanity_checking=False)
    algo.block_region_metric=BlockRegionMetric()
    gt=torch.zeros(1,4,3,128,128)
    pred=torch.ones_like(gt)*.5
    pred[:,3]=1  # excluded by n_metrics_frames
    metadata=dict(block_pos_agent=torch.tensor([[[[5.,0.]]]*4]),
                  block_vel_agent=torch.zeros(1,4,1,2),block_valid=torch.ones(1,4,1,dtype=torch.bool))
    batch=(gt[:,:2],torch.full((1,2),4),torch.tensor([[0,4]]),gt[:,2:],gt,metadata)
    existing=[];logged=[]
    monkeypatch.setattr(algo,'_sample_all_videos',lambda *args: ({'gt':gt,'prediction':pred},metadata))
    monkeypatch.setattr(algo,'_update_metrics',lambda videos,**kwargs:existing.append(videos))
    monkeypatch.setattr(algo,'_log_videos',lambda *args,**kwargs:None)
    monkeypatch.setattr(algo,'log_dict',lambda values,**kwargs:logged.append(values))
    monkeypatch.setattr(BasePytorchAlgo,'on_validation_epoch_end',lambda *args,**kwargs:None)
    algo.validation_step(batch,0)
    algo.on_validation_epoch_end()
    assert len(existing)==1  # original metrics still run
    assert logged[0]['validation/prediction/block_mse']==.25
    assert logged[0]['validation/prediction/block_mse_turn']==.25
    assert torch.isnan(logged[0]['validation/prediction/block_mse_nonturn'])
    assert not algo.block_region_metric._update_called
