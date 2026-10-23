"""Reproducible, validation-selected training for the paper evidence campaign.

The pilot sources/results are immutable inputs. No position, velocity or digit
template is included in the training objective or formal model inputs.
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
sys.path.insert(0, str(PARENT))
from paper.mechanism import build_model
from paper.data import TrainingData

ROOT = PARENT.parents[1]
INPUT_KEYS = ('context', 'cameras', 'context_times', 'target_cameras', 'target_times')
IMMUTABLE = ('kind', 'seed', 'lr', 'render_mode', 'motion', 'train_size',
             'irregular_context', 'batch_size', 'validation_size')


def atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_torch(path, value):
    temporary = path.with_name(path.name+'.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def network_input(batch):
    return {k: batch[k] for k in INPUT_KEYS}


def objective(prediction, target):
    weights = 1 + 5*(target > .05).to(target.dtype)
    return ((prediction-target).square()*weights).sum()/weights.sum()


@torch.no_grad()
def validate(model, batches):
    model.eval()
    total = np.zeros(4)
    for batch in batches:
        pred, aux = model(**network_input(batch))
        b = len(pred)
        total += b*np.array([objective(pred, batch['target']).item(),
                            (pred-batch['target']).square().mean().item(),
                            (aux['velocity']-batch['true_velocity']).norm(dim=-1).mean().item(), 1])
    model.train()
    return total[:3]/total[3]


def paths(config):
    run_id = config['run_id']
    tag = config.get('tag', 'paper_v1')
    return (ROOT/f'outputs/continuous_velocity/{tag}/training/{run_id}',
            ROOT/f'downloaded_checkpoints/continuous_velocity/{tag}/{run_id}')


def run_training(config):
    cfg = dict(steps=5000, batch_size=8, validation_size=64, device='cuda:1',
               lr=.0003, seed=2026, render_mode='bilinear', motion='horizontal',
               train_size=55000, irregular_context=False, tag='paper_v1')
    cfg.update(config)
    out, ckpt = paths(cfg)
    out.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'manifest.json'
    sources = [PARENT/'model.py', PARENT/'data.py', Path(__file__),
               Path(__file__).parent/'mechanism.py', Path(__file__).parent/'data.py']
    hashes = {str(src.relative_to(PARENT)).replace('/', '__'): hashlib.sha256(src.read_bytes()).hexdigest()
              for src in sources}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in IMMUTABLE:
            if old['config'][key] != cfg[key]:
                raise ValueError(f'Cannot reuse run_id with changed {key}')
        if old['source_sha256'] != hashes:
            raise ValueError('Cannot reuse run_id after training source changes')
        if old.get('complete') and old['config']['steps'] >= cfg['steps']:
            return old
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_model(cfg['kind']).to(cfg['device'])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'])
    data_seed = 10000+cfg['seed']
    data = TrainingData(cfg['device'], seed=data_seed, render_mode=cfg['render_mode'],
                        motion=cfg['motion'], train_size=cfg['train_size'],
                        irregular_context=cfg['irregular_context'])
    state = data.rng.bit_generator.state
    data.rng = np.random.default_rng(999)
    validation = [data.batch(min(cfg['batch_size'], cfg['validation_size']-i), validation=True)
                  for i in range(0, cfg['validation_size'], cfg['batch_size'])]
    data.rng = np.random.default_rng()
    data.rng.bit_generator.state = state
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
        if start > cfg['steps']:
            raise ValueError('Interrupted run is beyond requested budget; use the saved budget artifact')
    source_dir = out/'source'
    source_dir.mkdir(exist_ok=True)
    for src in sources:
        name = str(src.relative_to(PARENT)).replace('/', '__')
        if (source_dir/name).exists() and hashlib.sha256((source_dir/name).read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'Training source changed during resume: {name}')
        shutil.copyfile(src, source_dir/name)
    manifest = dict(config=cfg, complete=False, parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    data_seed=data_seed, validation_seed=999, source_sha256=hashes,
                    velocity_labels_used_for_training=False, position_labels_used_for_training=False,
                    template_given_to_model=False, checkpoint_directory=str(ckpt.resolve()),
                    torch=torch.__version__, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    atomic_json(manifest_path, manifest)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats(torch.device(cfg['device'])) if 'cuda' in cfg['device'] else None
    logpath = out/'training.csv'
    append = logpath.exists() and start > 0
    with logpath.open('a' if append else 'w') as handle:
        writer = csv.writer(handle)
        if not append:
            writer.writerow(['step', 'seconds', 'train_loss', 'velocity_l2', 'val_loss', 'val_mse', 'val_velocity_l2', 'gpu_peak_mb'])
        for step in range(start, cfg['steps']+1):
            if step == start:
                val = validate(model, validation)
                loss_value = velocity_error = float('nan')
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
                val = validate(model, validation) if step % 250 == 0 or step == cfg['steps'] else [float('nan')]*3
            if step == start or step % 50 == 0 or step == cfg['steps']:
                memory = torch.cuda.max_memory_allocated(torch.device(cfg['device']))/2**20 if 'cuda' in cfg['device'] else 0
                writer.writerow([step, time.monotonic()-began, loss_value, velocity_error, *val, memory])
                handle.flush()
            if np.isfinite(val[0]):
                improved = val[0] < best
                if improved:
                    best = float(val[0])
                saved = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                             best=best, data_rng=data.rng.bit_generator.state, config=cfg)
                atomic_torch(ckpt/'last.pt', saved)
                if improved:
                    atomic_torch(ckpt/'best.pt', saved)
                if step in (5000, 10000, 20000):
                    shutil.copyfile(ckpt/'best.pt', ckpt/f'budget_{step}.pt')
                if step % 1000 == 0 or step == start or step == cfg['steps']:
                    print(json.dumps(dict(run=cfg['run_id'], step=step, val_loss=float(val[0]),
                                          val_mse=float(val[1]), seconds=round(time.monotonic()-began, 1))), flush=True)
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
    config = json.loads(args.config.read_text())
    run_training(config)


if __name__ == '__main__':
    main()
