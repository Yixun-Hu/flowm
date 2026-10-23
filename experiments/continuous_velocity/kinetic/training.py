"""Validation-selected training for the kinetic study (fork of paper/train.py).

The model specification (align, render, refine, order, kappa, kind) and the
motion setting are immutable per run id, enter the manifest and the resume
guard, and all kinetic sources are hashed. No position, velocity, acceleration
or template enters the objective or the model inputs; true motion is logged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

PARENT = Path(__file__).resolve().parents[1]
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
from paper.train import INPUT_KEYS, atomic_json, atomic_torch, network_input, objective  # noqa: E402
from kinetic.memory import build_kinetic_model  # noqa: E402
from kinetic.scenes import KineticTrainingData, ROOT  # noqa: E402

SPEC_KEYS = ('align', 'render', 'refine', 'order', 'kappa', 'kind')
IMMUTABLE = SPEC_KEYS+('seed', 'lr', 'render_mode', 'motion', 'train_size',
                       'irregular_context', 'batch_size', 'validation_size')
SOURCES = [PARENT/'model.py', PARENT/'data.py', PARENT/'paper'/'mechanism.py', PARENT/'paper'/'data.py',
           PARENT/'paper'/'train.py', PARENT/'kinetic'/'transport.py', PARENT/'kinetic'/'memory.py',
           PARENT/'kinetic'/'scenes.py', Path(__file__)]
DEFAULTS = dict(align='bilinear', render='bilinear', refine='none', order=1, kappa=9., kind='physical',
                steps=5000, batch_size=8, validation_size=64, device='cuda:1', lr=.001, seed=2026,
                render_mode='bilinear', motion='horizontal', train_size=55000,
                irregular_context=False, tag='kinetic_v1')


def source_hashes():
    return {str(src.relative_to(PARENT)).replace('/', '__'): hashlib.sha256(src.read_bytes()).hexdigest()
            for src in SOURCES}


def spec_of(cfg):
    return {key: cfg[key] for key in SPEC_KEYS}


def paths(cfg):
    return (ROOT/f'outputs/continuous_velocity/{cfg["tag"]}/training/{cfg["run_id"]}',
            ROOT/f'downloaded_checkpoints/continuous_velocity/{cfg["tag"]}/{cfg["run_id"]}')


@torch.no_grad()
def validate(model, batches):
    """Weighted loss, MSE, velocity/acceleration errors, mean gate, and the
    x-axis gate-open rate on zero-acceleration sequences (null calibration)."""
    model.eval()
    total, count, gate_open, null_count = np.zeros(5), 0, 0., 0.
    for batch in batches:
        pred, aux = model(**network_input(batch))
        b = len(pred)
        gate = aux['acceleration_gate']
        null = batch['true_acceleration'].abs().sum(-1) == 0
        gate_open += (gate[:, 0] > 0)[null].float().sum().item()
        null_count += null.float().sum().item()
        total += b*np.array([objective(pred, batch['target']).item(),
                            (pred-batch['target']).square().mean().item(),
                            (aux['velocity']-batch['true_velocity']).norm(dim=-1).mean().item(),
                            (aux['acceleration']-batch['true_acceleration']).norm(dim=-1).mean().item(),
                            gate.nan_to_num(0.).mean().item()])
        count += b
    model.train()
    result = list(total/count)
    result.append(gate_open/null_count if null_count else float('nan'))
    return result


def run_training(config):
    cfg = dict(DEFAULTS)
    cfg.update(config)
    cfg['order'] = int(cfg['order']); cfg['kappa'] = float(cfg['kappa'])
    out, ckpt = paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    hashes = source_hashes()
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in IMMUTABLE:
            if old['config'][key] != cfg[key]:
                raise ValueError(f'Cannot reuse run_id {cfg["run_id"]} with changed {key}')
        if old['source_sha256'] != hashes:
            raise ValueError(f'Cannot reuse run_id {cfg["run_id"]} after training source changes')
        if old.get('complete') and old['config']['steps'] >= cfg['steps']:
            return old
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_kinetic_model(spec_of(cfg)).to(cfg['device'])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'])
    data_seed = 10000+cfg['seed']
    data = KineticTrainingData(cfg['device'], seed=data_seed, render_mode=cfg['render_mode'],
                               motion=cfg['motion'], train_size=cfg['train_size'],
                               irregular_context=cfg['irregular_context'])
    state = data.rng.bit_generator.state
    acceleration_state = data.acceleration_rng.bit_generator.state
    data.rng = np.random.default_rng(999)
    data.acceleration_rng = np.random.default_rng(777999)
    validation = [data.batch(min(cfg['batch_size'], cfg['validation_size']-i), validation=True)
                  for i in range(0, cfg['validation_size'], cfg['batch_size'])]
    data.rng = np.random.default_rng()
    data.rng.bit_generator.state = state
    data.acceleration_rng = np.random.default_rng()
    data.acceleration_rng.bit_generator.state = acceleration_state
    start, best = 0, float('inf')
    if (ckpt/'last.pt').exists():
        saved = torch.load(ckpt/'last.pt', map_location=cfg['device'], weights_only=False)
        previous = saved['config']
        for key in IMMUTABLE:
            if previous[key] != cfg[key]:
                raise ValueError(f'Cannot resume changed {key}: {previous[key]} -> {cfg[key]}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start, best = saved['step'], saved['best']
        data.rng.bit_generator.state = saved['data_rng']
        data.acceleration_rng.bit_generator.state = saved['acceleration_rng']
        if start > cfg['steps']:
            raise ValueError('Interrupted run is beyond requested budget; use the saved budget artifact')
    source_dir = out/'source'
    source_dir.mkdir(exist_ok=True)
    for src in SOURCES:
        name = str(src.relative_to(PARENT)).replace('/', '__')
        if (source_dir/name).exists() and hashlib.sha256((source_dir/name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'Training source changed during resume: {name}')
        shutil.copyfile(src, source_dir/name)
    manifest = dict(config=cfg, spec=spec_of(cfg), complete=False,
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    data_seed=data_seed, validation_seed=999, source_sha256=hashes,
                    velocity_labels_used_for_training=False, acceleration_labels_used_for_training=False,
                    position_labels_used_for_training=False, template_given_to_model=False,
                    checkpoint_directory=str(ckpt.resolve()), torch=torch.__version__,
                    started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    atomic_json(manifest_path, manifest)
    began = time.monotonic()
    if 'cuda' in cfg['device']:
        torch.cuda.reset_peak_memory_stats(torch.device(cfg['device']))
    logpath = out/'training.csv'
    append = logpath.exists() and start > 0
    memory = 0
    with logpath.open('a' if append else 'w') as handle:
        writer = csv.writer(handle)
        if not append:
            writer.writerow(['step', 'seconds', 'train_loss', 'velocity_l2', 'acceleration_l2',
                             'val_loss', 'val_mse', 'val_velocity_l2', 'val_acceleration_l2', 'val_gate',
                             'val_gate_open_null', 'temperature', 'weight_scale', 'gpu_peak_mb'])
        for step in range(start, cfg['steps']+1):
            if step == start:
                val = validate(model, validation)
                loss_value = velocity_error = acceleration_error = weight_scale = float('nan')
            else:
                batch = data.batch(cfg['batch_size'])
                optimizer.zero_grad(set_to_none=True)
                pred, aux = model(**network_input(batch))
                loss = objective(pred, batch['target'])
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss in {cfg["run_id"]} at {step}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                loss_value = loss.item()
                velocity_error = (aux['velocity'].detach()-batch['true_velocity']).norm(dim=-1).mean().item()
                acceleration_error = (aux['acceleration'].detach()-batch['true_acceleration']).norm(dim=-1).mean().item()
                weight_scale = aux['weights'].detach().mean().item()
                val = validate(model, validation) if step % 250 == 0 or step == cfg['steps'] else [float('nan')]*6
            if step == start or step % 50 == 0 or step == cfg['steps']:
                memory = torch.cuda.max_memory_allocated(torch.device(cfg['device']))/2**20 if 'cuda' in cfg['device'] else 0
                temperature = model.correspondence.log_temperature.exp().clamp(5, 150).item()
                writer.writerow([step, time.monotonic()-began, loss_value, velocity_error, acceleration_error,
                                 *val, temperature, weight_scale, memory])
                handle.flush()
            if np.isfinite(val[0]):
                improved = val[0] < best
                if improved:
                    best = float(val[0])
                saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                             best=best, data_rng=data.rng.bit_generator.state,
                             acceleration_rng=data.acceleration_rng.bit_generator.state, config=cfg,
                             spec=spec_of(cfg))
                atomic_torch(ckpt/'last.pt', saved)
                if improved:
                    atomic_torch(ckpt/'best.pt', saved)
                if step in (5000, 10000, 20000):
                    shutil.copyfile(ckpt/'best.pt', ckpt/f'budget_{step}.pt')
                if step % 1000 == 0 or step == start or step == cfg['steps']:
                    print(json.dumps(dict(run=cfg['run_id'], step=step, val_loss=float(val[0]),
                                          val_mse=float(val[1]), val_velocity_l2=float(val[2]),
                                          val_gate=float(val[4]), val_gate_open_null=float(val[5]),
                                          seconds=round(time.monotonic()-began, 1))), flush=True)
    manifest.update(complete=True, completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                    best_validation_loss=best, seconds_this_session=time.monotonic()-began,
                    gpu_peak_mb=memory)
    atomic_json(manifest_path, manifest)
    del model, optimizer, data, validation
    if 'cuda' in cfg['device']:
        torch.cuda.empty_cache()
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    run_training(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
