"""Frozen FloWM sweep. Run from repository root with the flowm Python."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/flowm-mpl')
import cv2
import numpy as np
import torch

from measurement import (
    Detector, fit_velocity, periodic_fit, render, unwrap_track, WORLD)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data/fractional_velocity'
CHECKPOINT = ROOT/'downloaded_checkpoints/hf_models/mnistworld-models/mnistworld/flowm-fernn/dynamic_po_flowm.pt'
SPEEDS = [0, .1, .25, .5, .75, .9, 1., 1.1, 1.25, 1.5, 1.75, 1.9, 2.]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_model(device):
    # Load the released module directly, avoiding imports of unrelated 3D models.
    path = ROOT/'algorithms/mem_wm/backbones/flowm/flowm_models_2d.py'
    spec = importlib.util.spec_from_file_location('released_flowm_2d', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.FlowEquivariantRNN(1, 64, 50, 32, v_range=2)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model


def released_case(seed):
    path = DATA/f'released/dynamic_validation_200/0/{seed:04d}'
    meta = torch.load(path.with_suffix('.pt'), map_location='cpu', weights_only=False)
    return path, meta


def load_video(path, length=90):
    cap = cv2.VideoCapture(str(path.with_suffix('.mp4')))
    frames = []
    for _ in range(length):
        ok, f = cap.read()
        if not ok: raise ValueError(f'Short video: {path}')
        frames.append(f.mean(-1)/255.)
    cap.release()
    return np.asarray(frames, np.float32)


@torch.inference_mode()
def predict(model, videos, actions, device):
    x = torch.as_tensor(np.asarray(videos), dtype=torch.float32, device=device)[:, :, None]
    a = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
    return model(x[:, :50], a[:, :50], a[:, 50:]).cpu().numpy()[:, :, 0]


def reproduce(model, out, n, device):
    rows = []
    for start in range(0, n, 4):
        cases = [released_case(i) for i in range(start, min(start+4, n))]
        videos = [load_video(p) for p, m in cases]
        actions = [m['actions'][:90].numpy() for p, m in cases]
        preds = predict(model, videos, actions, device)
        for k, pred in enumerate(preds):
            gt = videos[k][50:]
            rows.append(dict(episode=start+k, mse20=float(((pred[:20]-gt[:20])**2).mean()),
                             mse40=float(((pred-gt)**2).mean()),
                             black_mse20=float((gt[:20]**2).mean())))
            if start+k == 0:
                np.savez_compressed(out/'released_example.npz', frames=videos[k], predictions=pred)
        print(f'released validation {min(start+4,n)}/{n}', flush=True)
    write_csv(out/'released_reproduction.csv', rows)
    return rows


def write_csv(path, rows):
    if not rows: return
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def digit_data():
    with gzip.open(DATA/'t10k-images-idx3-ubyte.gz', 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28).copy()/255.
    with gzip.open(DATA/'t10k-labels-idx1-ubyte.gz', 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8).copy()
    return images, labels


def make_case(seed, speed, mode, images, labels):
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(len(images)))
    template = np.zeros((50, 50), np.float32); template[:28, :28] = images[idx]
    start = rng.integers(0, 50, size=2).astype(float)
    _, meta = released_case(seed)
    actions = meta['actions'][:90].numpy()
    # Metadata stores camera centers; the view begins 16 pixels before the center.
    cameras = np.cumsum(actions, axis=0)-16
    offset = np.array([.5, .5]) if mode == 'half_offset' else np.zeros(2)
    positions = start + np.arange(90)[:, None]*np.array([speed, 0.]) + offset
    frames, visibility = render(template, positions, cameras, interpolate=mode != 'nearest')
    return dict(seed=seed, speed=speed, mode=mode, digit_index=idx, label=int(labels[idx]),
                template=template, positions=positions, actions=actions, cameras=cameras,
                frames=frames, visibility=visibility)


def measure(case, pred):
    detector = Detector(case['template'])
    ctx = [detector(f) for f in case['frames'][:50]]
    det = [detector(f) for f in pred]
    ctx_wrapped = (np.array([d['pos'] for d in ctx])+case['cameras'][:50])%50
    ctx_track, v_base, intercept = periodic_fit(ctx_wrapped)
    baseline = intercept + np.arange(50, 90)[:, None]*v_base
    anchor = intercept + 49*v_base
    wrapped = (np.array([d['pos'] for d in det])+case['cameras'][50:])%50
    wrapped[np.array([d['ghost'] for d in det])] = np.nan
    track, v_pred, pred_intercept = periodic_fit(wrapped)
    if np.isfinite(anchor).all() and np.isfinite(pred_intercept).all():
        track += 50*np.round((anchor+v_pred-pred_intercept)/50)
    else:
        track[:] = np.nan
    # Only choose the coordinate origin using GT. Winding through time uses detections.
    valid_context = np.flatnonzero(np.isfinite(ctx_track).all(1))
    shift = np.zeros(2)
    if len(valid_context):
        t0 = valid_context[0]
        shift = np.round((case['positions'][t0]-ctx_track[t0])/50)*50
    track += shift; baseline += shift
    gt = case['positions'][50:]
    errors = np.linalg.norm(track-gt, axis=1)
    base_errors = np.linalg.norm(baseline-gt, axis=1)
    visible = case['visibility'][50:] >= .35
    present = np.array([d['present'] for d in det])
    ghost = np.array([d['ghost'] for d in det])
    usable = visible & present & ~ghost & np.isfinite(errors)
    rows = []
    for h in [20, 40]:
        u = usable[:h]; vis = visible[:h]
        rows.append(dict(seed=case['seed'], mode=case['mode'], speed=case['speed'],
            grid_distance=min(abs(case['speed']-k) for k in [0,1,2]), horizon=h,
            digit_index=case['digit_index'], label=case['label'],
            predicted_vx=v_pred[0], predicted_vy=v_pred[1], tracker_vx=v_base[0],
            ade=float(np.mean(errors[:h][u])) if u.any() else np.nan,
            endpoint_error=errors[h-1] if usable[h-1] else np.nan,
            tracker_ade=float(np.mean(base_errors[:h][vis])) if vis.any() else np.nan,
            tracker_endpoint=base_errors[h-1] if visible[h-1] else np.nan,
            visible_frames=int(vis.sum()), scored_frames=int(u.sum()),
            missing_frames=int((vis & ~present[:h]).sum()),
            ghost_frames=int((vis & ghost[:h]).sum()),
            ambiguous_frames=int((vis & present[:h] & ~np.isfinite(errors[:h])).sum()),
            context_detected=sum(d['present'] for d in ctx),
            context_visible=int((case['visibility'][:50] >= .35).sum()),
            mse=float(((pred[:h]-case['frames'][50:50+h])**2).mean())))
    arrays = dict(predictions=pred, frames=case['frames'], positions=case['positions'],
                  actions=case['actions'], cameras=case['cameras'], visibility=case['visibility'],
                  track=track, baseline=baseline, errors=errors, tracker_errors=base_errors,
                  present=present, ghost=ghost, usable=usable,
                  amplitude=np.array([d['amplitude'] for d in det]),
                  quality=np.array([d['quality'] for d in det]), template=case['template'])
    return rows, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=ROOT/'outputs/fractional_velocity/main')
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--modes', nargs='+', default=['nearest', 'bilinear', 'half_offset'])
    parser.add_argument('--speeds', nargs='+', type=float, default=SPEEDS)
    parser.add_argument('--reproduce', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--spacing', type=float, choices=[1.,.5,.25], default=1.)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'sequences').mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = load_model(args.device)
    equivalence = None
    if args.spacing != 1:
        from densify import apply_densification
        reference = load_model(args.device)
        integer_variant = apply_densification(load_model(args.device),1.)
        path, meta = released_case(0)
        sample = load_video(path)
        original = predict(reference,[sample],[meta['actions'][:90].numpy()],args.device)
        modified = predict(integer_variant,[sample],[meta['actions'][:90].numpy()],args.device)
        equivalence = float(np.max(np.abs(original-modified)))
        assert equivalence == 0, f'Integer transport differs: {equivalence}'
        del reference, integer_variant
        apply_densification(model,args.spacing)
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        checkpoint_sha256=sha(CHECKPOINT), checkpoint_hf_revision='72affe592a5b184d51b37b87ad5358362bbcaa1e',
        source_sha256=sha(ROOT/'algorithms/mem_wm/backbones/flowm/flowm_models_2d.py'),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name() if 'cuda' in args.device else 'cpu',
        context=50, prediction=40, world=50, window=32, channels=model.num_v,
        integer_transport_max_abs_difference=equivalence,
        transport='released' if args.spacing==1 else 'accumulated offset, nearest integer, no hidden-state interpolation',
        tracking='oracle digit identity; robust circular constant-velocity fit for winding; no true position or velocity in tracker',
        timestamp=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.reproduce: reproduce(model, args.out, args.reproduce, args.device)
    images, labels = digit_data(); all_rows=[]
    for seed in range(args.seeds):
        cases = [make_case(seed,speed,mode,images,labels) for mode in args.modes
                 for speed in (args.speeds if mode != 'half_offset' else [0.,1.,2.])]
        for start in range(0,len(cases),args.batch_size):
            batch=cases[start:start+args.batch_size]
            paths=[args.out/'sequences'/f"{seed:03d}_{c['mode']}_{c['speed']:g}.npz" for c in batch]
            if all(p.exists() for p in paths):
                preds=[np.load(p)['predictions'] for p in paths]
            else:
                preds=predict(model,[c['frames'] for c in batch],[c['actions'] for c in batch],args.device)
            for c,pred,path in zip(batch,preds,paths):
                rows,arrays=measure(c,pred); all_rows.extend(rows)
                np.savez_compressed(path,**arrays)
        write_csv(args.out/'metrics.csv',all_rows)
        print(f"seed {seed+1}/{args.seeds}, {len(all_rows)//2} sequences",flush=True)
    assert all(torch.equal(initial[k],v.cpu()) for k,v in model.state_dict().items())
    manifest['weights_unchanged_after_run']=True
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__ == '__main__': main()
