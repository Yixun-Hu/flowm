"""CPU pilot contract tests: configs, metrics, seeding, optimizer steps and CLI."""
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch

from algorithms.flowm_video import FloWMVideo
from algorithms.mem_wm.backbones.kwm3d.metrics import BlockRegionMetric
from algorithms.mem_wm.backbones.kwm3d.per_clip import PerClipDump, clip_stats, pooled
from algorithms.common.metrics.video import VideoMetric, SharedVideoMetricModelRegistry

ROOT = Path(__file__).resolve().parents[2]


def config(shortcode, extra=()):
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configurations")):
        return compose(config_name="config", overrides=[f"shortcode={shortcode}", *extra])


@pytest.mark.parametrize("arm", ["c", "a_oracle"])
def test_pilot_training_config(arm):
    path = f"exp/blockworld/kwm3d/train/pilot_{arm}"
    cfg = config(path)
    assert cfg.seed == 0 and cfg.load is None and cfg.algorithm.load_model_state is None
    assert cfg.algorithm.backbone.arm == arm and cfg.algorithm.backbone.model_type == "kwm3d"
    assert cfg.algorithm.backbone.oracle_velocity == "incoming_object"
    assert cfg.algorithm.backbone.checkpoint_steps and cfg.algorithm.backbone.map_conv == "c4"
    assert cfg.dataset.n_frames == cfg.dataset.max_frames == 90
    assert cfg.algorithm.tasks.prediction.context_frames == 50
    assert cfg.algorithm.teacher_forcing == .125
    assert cfg.dataset.load_block_info and cfg.dataset.metadata_read_only
    assert cfg.dataset.validation_first_clips
    assert list(cfg.algorithm.logging.metrics) == ["mse", "psnr", "block_mse"]
    exp = cfg.experiment
    assert list(exp.tasks) == ["training"] and exp.devices == 1 and not exp.ema.enable
    assert exp.training.strategy == exp.validation.strategy == "auto"
    assert exp.training.precision == "32-true" and exp.training.max_epochs == -1
    assert exp.training.max_steps == 6000 and exp.training.batch_size == 8
    assert exp.training.optim.accumulate_grad_batches == 1
    assert exp.training.lr == 1e-4 and exp.training.weight_decay == 1e-3
    assert list(exp.training.optim.optimizer_beta) == [.9, .99]
    assert exp.training.optim.gradient_clip_val == 1 and exp.training.optim.opt_name == "adamw"
    assert exp.training.lr_scheduler.name == "constant"
    assert exp.training.checkpointing.every_n_train_steps == 1000
    assert exp.training.checkpointing.save_top_k == -1
    assert exp.validation.val_every_n_step == 1000 and exp.validation.limit_batch == 4
    assert list(exp.training.splits_included) == ["k3d_v_training"]
    assert list(exp.validation.splits_included) == ["k3d_v_validation"]
    assert cfg.wandb.mode == "offline"
    fallback = config(f"[{path},exp/blockworld/kwm3d/train/pilot_bs4_accum2]")
    assert fallback.experiment.training.max_steps == 6000
    assert fallback.experiment.training.batch_size == 4
    assert fallback.experiment.training.optim.accumulate_grad_batches == 2
    assert fallback.experiment.validation.val_every_n_step == 2000
    from utils.hydra_utils import validate_shortcode_params
    validate_shortcode_params([f"shortcode=[{path},exp/blockworld/kwm3d/train/pilot_bs4_accum2]"], str(ROOT / "configurations"))


@pytest.mark.parametrize("horizon", [140, 280])
def test_evaluation_config_and_selection(horizon, monkeypatch):
    from datasets.video.blockworld import BlockWorldVideoDataset
    cfg = config(f"exp/blockworld/kwm3d/infer/metrics_{horizon}/k3d_v_70ctx")
    assert cfg.dataset.n_frames == cfg.dataset.max_frames == horizon
    assert cfg.algorithm.tasks.prediction.context_frames == 70
    assert cfg.dataset.num_default_clips == 1000
    assert list(cfg.algorithm.logging.metrics) == ["mse", "psnr", "ssim", "block_mse"]
    assert list(cfg.experiment.validation.splits_included) == ["k3d_v_validation"]
    assert not cfg.experiment.validation.dataloader.shuffle
    monkeypatch.setattr(BlockWorldVideoDataset, "build_metadata", lambda *a: pytest.fail("NAS write attempted"))
    cfg.dataset.num_default_clips = 3
    dataset = BlockWorldVideoDataset(cfg.dataset, split="k3d_v_validation", purpose="validation")
    assert len(dataset) == 3
    assert [dataset.get_clip_location(i) for i in range(3)] == [(0, 0), (1, 0), (2, 0)]
    assert [str(m["video_paths"]) for m in dataset.metadata] == sorted(str(m["video_paths"]) for m in dataset.metadata)


def samples():
    torch.manual_seed(17)
    pred, gt = torch.rand(3, 5, 3, 24, 24)*2-.5, torch.rand(3, 5, 3, 24, 24)
    pos = torch.tensor([3., 0.]).expand(3, 5, 3, 2).clone()
    meta = dict(block_pos_agent=pos, block_valid=torch.ones(3, 5, 3, dtype=torch.bool),
                block_speed=torch.tensor([.3, .6, .9]).expand(3, 5, 3),
                path=["first.mp4", "second.mp4", "third.mp4"], dataset_index=torch.arange(3))
    meta["block_valid"][2] = False
    actions = torch.tensor([4, 4, 0, 4, 1]).expand(3, -1)
    return pred, gt, meta, actions


@pytest.mark.parametrize('reduction_drift', [0., 5e-6, 2e-5])
def test_dump_matches_logged_metrics_across_batches(tmp_path, monkeypatch, caplog, reduction_drift):
    cfg = config("exp/blockworld/kwm3d/train/pilot_c")
    cfg.algorithm.tasks.prediction.context_frames = 2
    cfg.algorithm.logging.n_metrics_frames = 4
    cfg.algorithm.logging.per_clip_dump = str(tmp_path)
    algo = FloWMVideo(cfg.algorithm)
    algo.metrics_prediction = VideoMetric(SharedVideoMetricModelRegistry(), ["mse", "psnr"])
    algo.block_region_metric = BlockRegionMetric()
    algo._trainer = SimpleNamespace(sanity_checking=False, world_size=1, val_dataloaders=[None],
                                    state=SimpleNamespace(fn="VALIDATING"))
    algo.on_validation_epoch_start()
    logged = {}
    monkeypatch.setattr(algo, "log_dict", lambda d, **kw: logged.update(d))
    monkeypatch.setattr(algo, "_log_videos", lambda *a, **kw: None)
    pred, gt, meta, actions = samples()
    for i, ids in enumerate((slice(0, 2), slice(2, 3))):
        mm = {k: v[ids] for k, v in meta.items()}
        pp, gg = pred[ids], gt[ids]
        monkeypatch.setattr(algo, "_sample_all_videos", lambda *a, p=pp, g=gg, m=mm: ({"prediction": p, "gt": g}, m))
        batch = (gg[:, :2], actions[ids, :2], actions[ids, 2:], gg[:, 2:], gg, mm)
        algo.validation_step(batch, i)
    metric = algo.metrics_prediction.metrics['mse']
    compute = metric.compute
    monkeypatch.setattr(metric, 'compute', lambda: compute() * (1 + reduction_drift))
    algo.on_validation_epoch_end()
    dump = torch.load(tmp_path / "per_clip.pt", weights_only=True)
    assert dump["paths"] == meta["path"] and dump["n_frames"] == 4
    assert dump["element_count"][:, 0].tolist() == [2*3*24*24]*3
    assert dump["block_pixel_count"][2].sum() == 0
    assert (tmp_path / "per_clip.csv").exists()
    for name, value in zip(dump["metric_names"], pooled(dump)):
        key = "prediction/mse" if name == "mse" else "validation/prediction/"+name
        if name == 'mse':
            value *= 1 + reduction_drift
        torch.testing.assert_close(value, logged[key].double(), rtol=1e-6, atol=0)
    marker = tmp_path / 'pooled_check_failed.json'
    assert marker.exists() == (reduction_drift > 1e-5)
    if marker.exists():
        report = json.loads(marker.read_text())
        assert report['rtol'] == 1e-5
        assert report['mismatches']['mse']['pooled'] == float(dump['pooled'][0])
        assert report['mismatches']['mse']['logged'] == float(logged['prediction/mse'])
        assert 'POOLED METRIC CHECK FAILED' in caplog.text and 'pooled=' in caplog.text and 'logged=' in caplog.text
    else:
        assert 'POOLED METRIC CHECK FAILED' not in caplog.text
    assert not algo.block_region_metric._update_called
    assert not algo.metrics_prediction.metrics["mse"]._update_called
    cfg.algorithm.logging.per_clip_dump = None
    algo.on_validation_epoch_start()
    assert algo._clip_dump is None


def test_dump_empty_regions_and_duplicate_identity(tmp_path):
    pred, gt, meta, actions = samples()
    stats = clip_stats(pred, gt, meta, actions)
    dump = PerClipDump(0, 5)
    dump.append(stats, meta)
    dump.append(stats, meta)
    with pytest.raises(ValueError, match="Duplicate"):
        dump.write(tmp_path)


def test_seed_option_and_workers(monkeypatch):
    from utils.seed_utils import seed_run
    from experiments.data_modules.base_data_module import BaseDataModule, seed_worker
    cfg = config("exp/blockworld/kwm3d/train/pilot_c")
    def draw():
        return random.random(), np.random.random(), torch.rand(1).item()
    seed_run(0)
    first = draw()
    seed_run(0)
    assert draw() == first
    state = torch.random.get_rng_state()
    seed_run(None)
    assert torch.equal(state, torch.random.get_rng_state())
    assert BaseDataModule(cfg, {}).generator.initial_seed() == 0
    monkeypatch.setattr(torch, "initial_seed", lambda: 123)
    seed_worker(0)
    value = (random.random(), np.random.random())
    seed_worker(0)
    assert (random.random(), np.random.random()) == value
    monkeypatch.setattr(torch, "initial_seed", lambda: 124)
    seed_worker(1)
    assert (random.random(), np.random.random()) != value


def test_lightning_counts_optimizer_steps_with_accumulation(tmp_path):
    import lightning.pytorch as pl
    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.batches = 0
            self.validations = []
        def training_step(self, batch, index):
            self.batches += 1
            return self.weight.square().sum()
        def validation_step(self, batch, index):
            self.validations.append(self.global_step)
        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters())
    model = Tiny()
    trainer = pl.Trainer(accelerator="cpu", devices=1, max_steps=2, max_epochs=-1,
        accumulate_grad_batches=2, val_check_interval=4, check_val_every_n_epoch=None,
        limit_val_batches=1, num_sanity_val_steps=0, logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False, default_root_dir=str(tmp_path))
    data = torch.utils.data.DataLoader(torch.ones(8), batch_size=1)
    trainer.fit(model, data, data)
    assert trainer.global_step == 2 and model.batches == 4 and model.validations == [2]


def test_benchmark_cpu_cli():
    cmd = [sys.executable, str(ROOT / "scripts/kwm3d_bench.py"), "--device", "cpu", "--arm", "a_oracle",
           "--context", "10", "--prediction", "5", "--batch", "1", "--steps", "1", "--warmup", "0",
           "--checkpoint", "on", "--accumulate", "2"]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    record = next(json.loads(line) for line in result.stdout.splitlines() if line.startswith('{'))
    assert record["device"] == "cpu" and record["accumulate"] == 2
    assert record["context"] == 10 and record["prediction"] == 5
    assert record["seconds_per_step_mean"] > 0
    assert record["seconds_per_step_mean"] == record["seconds_per_step_median"]
    assert "max_memory_allocated_mib" not in record
    assert record["backbone"]["hidden_channels"] == 256
    assert record["float32_matmul_precision"] == "high"


@pytest.mark.parametrize("arm", ["c", "a_oracle"])
def test_kinetic_checkpoint_requires_exact_backbone(arm):
    cfg = config(f"exp/blockworld/kwm3d/train/pilot_{arm}", ["algorithm.backbone.hidden_channels=32",
                 "algorithm.backbone.map_proc_encoder_depth=1", "algorithm.backbone.decoder_depth=1"])
    algo = FloWMVideo(cfg.algorithm)
    algo.configure_model()
    state = {"main_model."+k: v for k, v in algo.main_model.state_dict().items()}
    algo.on_load_checkpoint({"state_dict": dict(state)})
    with pytest.raises(RuntimeError):
        algo.on_load_checkpoint({"state_dict": {**state, "main_model.bad_weight": torch.ones(1)}})
    state.pop(next(iter(state)))
    with pytest.raises(RuntimeError):
        algo.on_load_checkpoint({"state_dict": state})
