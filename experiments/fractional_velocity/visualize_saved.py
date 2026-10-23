"""Lightweight visualizations of saved runs; never loads a model or uses CUDA."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/flowm-mpl')
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]/'outputs/fractional_velocity'
OUT = ROOT/'visualizations'


def load(speed, folder='main'):
    return np.load(ROOT/folder/'sequences'/f'000_nearest_{speed:g}.npz')


def main():
    OUT.mkdir(exist_ok=True)
    original, dense = load(.5), load(.5, 'dense05')
    # Fixed first seed. Choose snapshots solely for GT visibility, not model output.
    horizons = [2, 6, 14, 19, 28, 32]
    assert all(original['visibility'][49+h] > .7 for h in horizons)
    fig, axes = plt.subplots(3, 6, figsize=(12, 6.2))
    rows = [('Ground truth', original['frames'][50:]),
            ('Original FloWM\n25 channels', original['predictions']),
            ('Same frozen weights\n81 channels', dense['predictions'])]
    for r, (label, frames) in enumerate(rows):
        for j, h in enumerate(horizons):
            ax = axes[r, j]
            ax.imshow(frames[h-1], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(f'+{h} frames', fontsize=11)
            if j == 0: ax.set_ylabel(label, fontsize=11)
    fig.suptitle('Fractional-velocity failure and recovery: vx = 0.5 px/frame', fontsize=15)
    fig.text(.5, .025, 'Seed 0 · 50 context frames · nearest-pixel rendering · identical camera actions\n'
             'Snapshots selected for >70% ground-truth visibility. All panels use the same 0–1 intensity scale.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    fig.savefig(OUT/'failure_and_recovery.png', dpi=170)
    fig.savefig(OUT/'failure_and_recovery.pdf'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for folder, label, color in [('main','Original: 25 channels','#ba3545'),
                                  ('dense05','Same weights: 81 channels','#187a70')]:
        with open(ROOT/folder/'metrics.csv') as f:
            rows = [r for r in csv.DictReader(f) if r['mode']=='nearest' and r['horizon']=='40']
        speeds = sorted({float(r['speed']) for r in rows})
        values, low, high = [], [], []
        for v in speeds:
            rs = [r for r in rows if float(r['speed']) == v]
            assert len(rs) == 32
            misses = np.array([int(r['missing_frames']) for r in rs])
            visible = np.array([int(r['visible_frames']) for r in rs])
            value = misses.sum()/visible.sum()*100
            rng = np.random.default_rng(42); ids = rng.integers(32, size=(2000,32))
            boot = misses[ids].sum(1)/visible[ids].sum(1)*100
            lo, hi = np.quantile(boot, [.025,.975])
            values.append(value); low.append(value-lo); high.append(hi-value)
        ax.errorbar(speeds, values, yerr=[low,high], marker='o', capsize=3, color=color, label=label)
    ax.set(xlabel='True horizontal velocity (px/frame)', ylabel='Missing detections (% of visible GT frames)',
           title='The original model loses fractional-speed digits', ylim=(-3,105))
    ax.set_xticks([0,.25,.5,.75,1,1.25,1.5,1.75,2]); ax.grid(alpha=.2)
    ax.legend(loc='lower center', fontsize=10)
    fig.text(.5,.01,'32 paired seeds · 40-frame prediction · 95% seed-bootstrap intervals · nearest-pixel rendering',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(OUT/'missing_detection_curve.png',dpi=170)
    fig.savefig(OUT/'missing_detection_curve.pdf');plt.close(fig)

    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try: font=ImageFont.truetype(font_path,18); small=ImageFont.truetype(font_path,14)
    except OSError: font=small=ImageFont.load_default()
    cases=[(v,load(v),load(v,'dense05')) for v in [.5,1.,1.5]]
    frames=[]
    for t in range(40):
        canvas=Image.new('RGB',(800,760),'white'); draw=ImageDraw.Draw(canvas)
        draw.text((16,12),f'Frozen FloWM: prediction +{t+1:02d} / 40 (50 context frames)',font=font,fill='black')
        for j,name in enumerate(['Ground truth','Original: 25 channels','Same weights: 81']):
            draw.text((124+j*220,48),name,font=small,fill='black')
        for r,(v,a,b) in enumerate(cases):
            y=85+r*215
            draw.text((12,y+75),f'vx = {v:g}',font=font,fill='black')
            for j,im in enumerate([a['frames'][50+t],a['predictions'][t],b['predictions'][t]]):
                tile=Image.fromarray((np.clip(im,0,1)*255).astype(np.uint8)).resize((192,192),Image.Resampling.NEAREST).convert('RGB')
                canvas.paste(tile,(125+j*220,y))
        draw.text((16,735),'Seed 0; all 40 frames, original moving camera; shared intensity scale [0,1].',font=small,fill='black')
        frames.append(canvas)
    frames[0].save(OUT/'paired_rollout.gif',save_all=True,append_images=frames[1:],duration=250,loop=0)
    print(OUT)


if __name__=='__main__': main()
