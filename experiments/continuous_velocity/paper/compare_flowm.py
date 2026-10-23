"""CPU-only comparison of saved paper-model and frozen FloWM predictions.

Keeps this post-hoc visualization separate from the completed paper campaign.
Re-scores FloWM using the audited v2 evaluator, verifies exact scene pairing,
and never selects examples or frames using either model's prediction quality.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/flowm-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper.data import ROOT, make_case
from paper.metrics import evaluate_case, METRICS_VERSION

OUT = ROOT/'outputs/continuous_velocity/flowm_comparison'
PAPER = ROOT/'outputs/continuous_velocity/paper_v1'
SPEEDS = dict(bilinear=(0., .37, .5, .63, 1., 1.23, 1.5, 2.),
              nearest=(0., .5, 1., 1.5, 2.))
MODELS = ('flowm25', 'flowm81', 'ours')
LABELS = ('FloWM · 25 channels\nreleased, frozen',
          'FloWM · 81 channels\nsame frozen weights',
          'Ours · continuous velocity\nfractional-velocity training')
COLORS = ('#d45e52', '#d09b34', '#087f8c')
TRAIN_SEEDS = (2027, 2028, 2029, 2030)
METRICS = {'mse': ('squared_error_sum', 'pixel_count'),
           'success': ('frame_success_frames', 'visible_frames')}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_baseline(model, seed, mode, speed):
    if mode == 'bilinear':
        directory = ROOT/f'outputs/continuous_velocity/pilot/evaluation/{model}/sequences'
    else:
        variant = 'main' if model == 'flowm25' else 'dense05'
        directory = ROOT/f'outputs/fractional_velocity/{variant}/sequences'
    return directory/f'{seed:03d}_{mode}_{speed:g}.npz'


def saved_ours(mode, speed):
    return (PAPER/'evaluation/physical_s2027/core/examples'
            /f'seed000__{mode}__vx{speed:+.6g}_vy+0__mnist_regular_released.npz')


def read_npz(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def assert_pair(saved, case):
    for key in ('frames', 'positions', 'cameras', 'actions', 'template', 'visibility'):
        if not np.array_equal(saved[key], case[key]):
            raise ValueError(f'Unpaired scene: {key}')


def read_rows(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            try:
                row[key] = float(value)
            except ValueError:
                pass
    return rows


def collect():
    rows, sources = [], []
    for train_seed in TRAIN_SEEDS:
        directory = PAPER/f'evaluation/physical_s{train_seed}/core'
        manifest = json.loads((directory/'manifest.json').read_text())
        assert manifest['complete'] and manifest['metric_schema_version'] == METRICS_VERSION
        for filename, digest in manifest['source_sha256'].items():
            assert sha(ROOT/filename) == digest, filename
        sources.append(dict(path=str(directory/'metrics.csv'), sha256=sha(directory/'metrics.csv'),
                            manifest=manifest))
        for row in read_rows(directory/'metrics.csv'):
            if row['horizon'] == 40 and row['speed'] in SPEEDS[row['mode']]:
                row.update(comparison_model='ours', train_seed=train_seed)
                rows.append(row)
    for seed in range(32):
        for mode, speeds in SPEEDS.items():
            for speed in speeds:
                case = make_case(seed, speed, mode, horizon=40)
                for model in MODELS[:2]:
                    path = saved_baseline(model, seed, mode, speed)
                    saved = read_npz(path)
                    assert_pair(saved, case)
                    measured, _ = evaluate_case(case, saved['predictions'], horizons=(40,))
                    measured[0].update(comparison_model=model, train_seed=-1)
                    rows.extend(measured)
                    sources.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path)))
        if seed % 8 == 7:
            print(f'Re-scored and verified matched scenes: {seed+1}/32', flush=True)
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with (OUT/'paired_metrics_v2.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)
    return rows, sources


def aggregate(rows, model, mode, speeds, metric):
    selected = [r for r in rows if r['comparison_model'] == model
                and r['mode'] == mode and r['speed'] in speeds]
    train_seeds = sorted({r['train_seed'] for r in selected})
    num, den = METRICS[metric]
    numerators, denominators = [], []
    for ts in train_seeds:
        part = [r for r in selected if r['train_seed'] == ts]
        assert len(part) == 32*len(speeds)
        assert len({(r['seed'], r['speed']) for r in part}) == len(part)
        numerators.append([sum(r[num] for r in part if r['seed'] == s) for s in range(32)])
        denominators.append([sum(r[den] for r in part if r['seed'] == s) for s in range(32)])
    numerators, denominators = np.array(numerators), np.array(denominators)
    values = numerators.sum(1)/denominators.sum(1)
    # Identical sampled scene clusters across methods and fitted models.
    indices = np.random.default_rng(20261013).integers(32, size=(4000, 32))
    bootstrap = (numerators[:, indices].sum(2)/denominators[:, indices].sum(2)).mean(0)
    return dict(mean=float(values.mean()), training_values=values.tolist(),
                training_sd=float(values.std(ddof=1)) if len(values)>1 else None,
                ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                scene_sequences=32*len(speeds), fitted_models=len(values)), bootstrap


def save_figure(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT/f'{name}.{ext}', dpi=190, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def quant_plot(rows):
    groups = [('bilinear', (.37, .63, 1.23), 'Bilinear · held-out v = 0.37, 0.63, 1.23'),
              ('nearest', (.5, 1.5), 'Nearest · v = 0.5, 1.5 (on FloWM 81 grid)')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.2))
    summary = {}
    for rid, (mode, speeds, title) in enumerate(groups):
        summary[mode] = dict(speeds=speeds, models={}, comparisons={})
        for cid, metric in enumerate(METRICS):
            ax = axes[rid, cid]; boot = {}; means = {}
            factor = 100 if metric == 'success' else 1
            for j, model in enumerate(MODELS):
                result, boot[model] = aggregate(rows, model, mode, speeds, metric)
                summary[mode]['models'].setdefault(model, {})[metric] = result
                value = result['mean']*factor; means[model] = result['mean']
                lo, hi = np.array(result['ci95'])*factor
                ax.bar(j, value, color=COLORS[j], width=.62, zorder=2)
                ax.errorbar(j, value, yerr=[[max(0,value-lo)], [max(0,hi-value)]],
                            fmt='none', color='#303b43', capsize=4, lw=1.2, zorder=3)
                if model == 'ours':
                    vals = np.array(result['training_values'])*factor
                    ax.scatter(j+np.linspace(-.15,.15,len(vals)), vals, s=18,
                               facecolor='white', edgecolor='#173c41', zorder=4)
                ax.text(j, hi+(2.2 if metric=='success' else .001),
                        f'{value:.1f}%' if metric=='success' else f'{value:.5f}',
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
            for base in MODELS[:2]:
                difference = boot['ours']-boot[base]
                summary[mode]['comparisons'].setdefault(base,{})[metric] = dict(
                    ours_minus_baseline=means['ours']-means[base],
                    paired_scene_ci95=np.quantile(difference,[.025,.975]).tolist())
            ax.set_xticks(range(3), ['FloWM 25', 'FloWM 81', 'Ours'])
            ax.set_title(title+('\nImage error ↓' if metric=='mse' else '\nLocalization success ↑'),fontsize=12)
            ax.set_ylabel('Mean squared pixel error' if metric=='mse' else 'Visible target frames (%)')
            ax.set_ylim(0, max(ax.get_ylim()[1]*1.16, .012) if metric=='mse' else 114)
            ax.set_axisbelow(True); ax.grid(axis='y',alpha=.2)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Continuous-velocity prediction vs. frozen FloWM', fontsize=18, fontweight='bold')
    fig.text(.5,.015,'50 context + 40 predicted frames · 32 paired scenes per speed · ours: 4 fresh training seeds\n'
             'Bars: pooled metrics, then mean across fitted models. Whiskers: conditional 95% scene-cluster CIs; dots: training seeds.\n'
             'FloWM: released frozen weights; ours: trained on fractional velocities. Training conditions are not matched.\n'
             'Nearest subset excludes v = 0.37, 0.63, 1.23: ours falls to 46.27% success on those nearest-rendered holdouts.\n'
             'Success: visible target detected without detected duplication, periodic position error ≤2 px; not an unwrapped trajectory score.',
             ha='center',fontsize=9,linespacing=1.5)
    fig.tight_layout(rect=(0,.15,1,.95),h_pad=2.5)
    save_figure(fig,'aggregate_comparison')
    fig, axes=plt.subplots(1,2,figsize=(12,4.7))
    for j,model in enumerate(MODELS):
        for ax,metric in zip(axes,METRICS):
            values=[aggregate(rows,model,'bilinear',(v,),metric)[0]['mean'] for v in SPEEDS['bilinear']]
            ax.plot(SPEEDS['bilinear'],np.array(values)*(100 if metric=='success' else 1),
                    'o-',color=COLORS[j],label=['Frozen FloWM 25','Frozen FloWM 81','Ours (4 training seeds)'][j])
    for ax in axes:
        ax.set_xlabel('True horizontal velocity (px/frame)');ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Image MSE ↓');axes[1].set_ylabel('Localization success (%) ↑')
    axes[1].set_ylim(-3,104);axes[0].legend(fontsize=9)
    fig.suptitle('Velocity sweep includes integer-speed controls · bilinear rendering',fontsize=15)
    fig.text(.5,.01,'32 matched scenes per speed · same 50-frame context and 40-frame forecast · frozen FloWM vs. fractional-trained ours',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95));save_figure(fig,'velocity_sweep')
    return summary


def examples(mode, speeds):
    result={}
    for speed in speeds:
        case=make_case(0,speed,mode,horizon=40)
        result[speed]={'truth':case['frames'][50:], 'visibility':case['visibility'][50:]}
        for model in MODELS:
            path=saved_ours(mode,speed) if model=='ours' else saved_baseline(model,0,mode,speed)
            saved=read_npz(path);assert_pair(saved,case)
            result[speed][model]=saved['predictions']
    return result


def qualitative(mode, speeds):
    cases=examples(mode,speeds)
    columns=[]
    for speed in speeds:
        # Three temporal bins; pick maximum GT visibility, with earliest ties.
        for start,stop in ((0,13),(13,26),(26,40)):
            h=start+int(np.argmax(cases[speed]['visibility'][start:stop]))
            columns.append((speed,h))
    fig,axes=plt.subplots(4,len(columns),figsize=(14,6.5))
    fig.subplots_adjust(left=.15,right=.99,top=.86,bottom=.13,hspace=.16,wspace=.12)
    for row,key in enumerate(('truth',*MODELS)):
        for col,(speed,h) in enumerate(columns):
            ax=axes[row,col]
            ax.imshow(cases[speed][key][h],cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(f'v = {speed:g}\n+{h+1} frames',fontsize=10)
            if col==0:
                label='Ground truth' if row==0 else LABELS[row-1]
                ax.set_ylabel(label,rotation=0,ha='right',va='center',labelpad=12,
                              fontsize=10,color='black' if row==0 else COLORS[row-1])
    fig.suptitle(f'Same digit, camera and context · {mode} rendering',fontsize=18,fontweight='bold')
    fig.text(.5,.035,'Fixed test scene 0; ours uses first fresh training seed 2027. No model-quality-based example selection.\n'
             'Snapshots maximize ground-truth visibility within three fixed time bins; all panels share intensity scale [0, 1].\n'
             'FloWM uses frozen released weights; ours was trained on fractional velocities. See aggregate results for all 32 scenes.',
             ha='center',fontsize=9,linespacing=1.4)
    save_figure(fig,f'qualitative_{mode}')
    return [dict(mode=mode,speed=s,horizon=h+1,gt_visibility=float(cases[s]['visibility'][h])) for s,h in columns]


def animation():
    speeds=(.37,.63,1.23);cases=examples('bilinear',speeds)
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,19);small=ImageFont.truetype(font_path,14)
    frames=[]
    for h in range(40):
        canvas=Image.new('RGB',(1010,785),'white');draw=ImageDraw.Draw(canvas)
        draw.text((20,12),f'Paired fractional-velocity forecast   +{h+1:02d} / 40 frames',font=font,fill='#172e38')
        for col,label in enumerate(['Ground truth','Frozen FloWM 25','Frozen FloWM 81','Ours (trained)']):
            draw.text((155+col*211,49),label,font=small,fill='black')
        for row,speed in enumerate(speeds):
            y=83+row*215
            draw.text((15,y+75),f'v = {speed:g}',font=font,fill='black')
            for col,key in enumerate(('truth',*MODELS)):
                im=Image.fromarray(np.uint8(np.clip(cases[speed][key][h],0,1)*255))
                canvas.paste(im.resize((192,192),Image.Resampling.NEAREST).convert('RGB'),(150+col*211,y))
        draw.text((20,739),'Bilinear · test scene 0 · training seed 2027 · all 40 frames, shared intensity scale [0,1]',font=small,fill='black')
        draw.text((20,762),'Frozen released FloWM vs. fractional-velocity-trained ours; training conditions differ.',font=small,fill='black')
        frames.append(canvas)
    frames[0].save(OUT/'paired_rollout.gif',save_all=True,append_images=frames[1:],duration=240,loop=0)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--figures-only',action='store_true',help='Reuse this comparison\'s audited v2 CSV; do not rescore.')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    source_digest=sha(Path(__file__))
    if args.figures_only:
        previous=json.loads((OUT/'summary.json').read_text())
        assert previous['metrics_sha256']==sha(Path(__file__).with_name('metrics.py'))
        rows=read_rows(OUT/'paired_metrics_v2.csv');sources=previous['sources']
        for source in sources:
            assert sha(ROOT/source['path'])==source['sha256'],source['path']
        scoring_digest=previous.get('scoring_source_sha256',previous['source_sha256'])
    else:
        rows,sources=collect();scoring_digest=source_digest
    summary=quant_plot(rows)
    selected=qualitative('bilinear',(.37,.63,1.23))+qualitative('nearest',(.5,1.,1.5))
    animation()
    metadata=dict(posthoc_visualization=True,model_inference_performed=False,device='cpu',
                  metrics_version=METRICS_VERSION,source_sha256=source_digest,
                  scoring_source_sha256=scoring_digest,
                  paired_metrics_sha256=sha(OUT/'paired_metrics_v2.csv'),
                  metrics_sha256=sha(Path(__file__).with_name('metrics.py')),
                  test_scene_seeds=list(range(32)),ours_training_seeds=TRAIN_SEEDS,
                  training_comparison='Frozen released FloWM vs fractional-trained ours; unmatched training.',
                  interval_definition='Paired scene-cluster bootstrap, conditional on fitted models; not future-training uncertainty.',
                  paired_baseline_sequences=832,summary=summary,selected_snapshots=selected,sources=sources)
    baseline_manifests=[ROOT/f'outputs/fractional_velocity/{variant}/manifest.json' for variant in ('main','dense05')]
    baseline_manifests += [ROOT/f'outputs/continuous_velocity/pilot/evaluation/{model}/manifest.json' for model in MODELS[:2]]
    metadata['baseline_manifests']=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),contents=json.loads(p.read_text())) for p in baseline_manifests]
    assert sha(Path(__file__))==source_digest, 'Comparison source changed during execution'
    (OUT/'summary.json').write_text(json.dumps(metadata,indent=2))
    pictures=['qualitative_bilinear','aggregate_comparison','velocity_sweep','qualitative_nearest']
    body=''.join(f'<figure><a href="{p}.pdf"><img src="{p}.png" alt="{p}"></a></figure>' for p in pictures)
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Ours vs. frozen FloWM</title>
    <style>body{font:17px/1.55 system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#19323d}img{width:100%;height:auto}figure{margin:28px 0}a{color:#087f8c}.note{background:#edf5f5;padding:18px;border-radius:8px}</style>
    <h1>Continuous-velocity prediction vs. frozen FloWM</h1>
    <p class="note">All predictions are real saved outputs; no new model inference. The same digits, initial positions, camera actions and target frames are paired exactly. Ours is trained on fractional velocities; FloWM uses its released frozen checkpoint. This is a stress-test comparison, not a matched-training architecture comparison.</p>
    <p>Ours: four fresh training seeds, 5,000-update budget, validation-selected checkpoints. Qualitative examples: fixed test scene 0 and first fresh training seed 2027. Main bilinear aggregate: 32 scenes at each held-out speed 0.37, 0.63 and 1.23. Nearest control: 32 scenes at speeds 0.5 and 1.5, which lie on the 81-channel velocity grid. This subset excludes the difficult nearest-rendered holdouts 0.37, 0.63 and 1.23, on which ours obtains only <strong>46.27% success</strong>; matched saved FloWM predictions at those nearest speeds are unavailable. See <a href="../paper_v1/research_findings.md">the full transfer results</a>.</p>
    <p>FloWM predictions were re-scored with the audited v2 evaluator. Localization success requires a visible target detection without detected duplication and periodic position error ≤2 px. It does not measure whole-world winding. Confidence intervals resample paired test scenes, conditional on fitted models. Original, potentially biased CV-fit metrics are not used here.</p>'''+body+'''
    <h2>Every forecast frame</h2><img src="paired_rollout.gif" alt="40-frame paired forecast animation">
    <p><a href="summary.json">Numerical results and provenance</a> · <a href="paired_metrics_v2.csv">Per-sequence metrics</a></p></html>'''
    (OUT/'report.html').write_text(html)
    print(json.dumps(summary,indent=2));print(f'Created comparison: {OUT}',flush=True)


if __name__=='__main__':
    main()
