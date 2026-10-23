"""Aggregate completed runs, paired seed intervals, and research figures."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/flowm-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS={'physical':'Physical attention','relaxed':'Physical + free corrections','offset':'Unconstrained offsets',
        'physical_untrained':'Untrained physical model','oracle_velocity':'True velocity diagnostic',
        'flowm25':'Frozen FloWM 25','flowm81':'Frozen FloWM 81'}
COLORS={'physical':'#087f8c','relaxed':'#6b4c9a','offset':'#757575','physical_untrained':'#99a945',
        'oracle_velocity':'#555555','flowm25':'#c44536','flowm81':'#dc9a23'}


def ci(values):
    v=np.array(values,float);v=v[np.isfinite(v)]
    if not len(v):return [None,None,None]
    rng=np.random.default_rng(2026);boot=v[rng.integers(len(v),size=(2000,len(v)))].mean(1)
    return [float(v.mean()),*map(float,np.quantile(boot,[.025,.975]))]


def save(fig,path):
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=170);fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def summarize(g):
    visible=g.visible_frames.sum()
    return dict(n_sequences=len(g),missing_rate=float(g.missing_frames.sum()/visible),
                ghost_rate=float(g.ghost_frames.sum()/visible),coverage=float(g.scored_frames.sum()/visible),
                ade=float(g.ade.mean()),mse=float(g.mse.mean()),
                measured_velocity_mae=float(abs(g.predicted_vx-g.speed).mean()),
                latent_velocity_mae=float(abs(g.latent_vx-g.speed).mean()) if 'latent_vx' in g and g.latent_vx.notna().any() else None,
                velocity_fit_count=int(g.predicted_vx.notna().sum()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/continuous_velocity/pilot'))
    args=p.parse_args();root=args.root;figs=root/'figures';figs.mkdir(exist_ok=True)
    datasets={}
    for name in LABELS:
        path=root/'evaluation'/name
        if not (path/'manifest.json').exists():continue
        manifest=json.loads((path/'manifest.json').read_text())
        if not manifest.get('complete'):continue
        df=pd.read_csv(path/'metrics.csv');df['model']=name
        df['missing_rate']=df.missing_frames/df.visible_frames
        df['ghost_rate']=df.ghost_frames/df.visible_frames
        df['coverage']=df.scored_frames/df.visible_frames
        datasets[name]=df
    all_data=pd.concat(datasets.values(),ignore_index=True);all_data.to_csv(root/'all_metrics.csv',index=False)
    test=all_data[(all_data.horizon==40)&(all_data['mode']=='bilinear')]
    summary={name:summarize(g) for name,g in test.groupby('model')}
    held=test[test.speed.isin([.37,.63,1.23])]
    holdout={name:summarize(g) for name,g in held.groupby('model')}
    nearest=all_data[(all_data.horizon==40)&(all_data['mode']=='nearest')]
    controls={name:summarize(g) for name,g in nearest.groupby('model')}
    nearest_heldout={name:summarize(g) for name,g in nearest[nearest.speed.isin([.37,.63,1.23])].groupby('model')}
    comparison=[]
    physical=held[held.model=='physical']
    for name in datasets:
        if name=='physical':continue
        joined=physical.merge(held[held.model==name],on=['seed','speed'],suffixes=('_physical','_other'))
        for metric in ['mse','missing_rate','ghost_rate','coverage','ade']:
            delta=joined.assign(delta=joined[metric+'_physical']-joined[metric+'_other']).groupby('seed').delta.mean()
            comparison.append(dict(other=name,metric=metric,physical_minus_other=ci(delta),n_seeds=int(delta.notna().sum())))
    (root/'summary.json').write_text(json.dumps(dict(all_speeds=summary,heldout_speeds=holdout,nearest_control=controls,nearest_heldout=nearest_heldout,paired=comparison),indent=2))
    table=test.groupby(['model','speed'])[['latent_vx','predicted_vx','ade','mse','missing_rate','ghost_rate','coverage']].mean()
    table.to_csv(root/'by_speed.csv')

    shown=[n for n in ['physical','relaxed','offset','flowm25','flowm81'] if n in datasets]
    fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
    for name in shown:
        g=test[test.model==name]
        column='predicted_vx'
        means=g.groupby('speed')[column].mean()
        counts=g.groupby('speed')[column].count()
        means=means.where(counts>=32)
        axes[0].plot(means.index,means.values,marker='o',label=LABELS[name],color=COLORS[name])
        coverage=g.groupby('speed').coverage.mean()
        axes[1].plot(coverage.index,coverage.values,marker='.',color=COLORS[name])
    axes[0].plot([0,2],[0,2],'k:',label='Correct velocity')
    axes[0].set(ylabel='Velocity (px/frame)',title='64 paired sequences per speed; velocity means require at least 32 valid fits')
    axes[0].legend(fontsize=8,ncol=2)
    axes[1].set(xlabel='True horizontal velocity',ylabel='Scored / visible frames',ylim=(-.03,1.03))
    for ax in axes:
        for v in [.37,.63,1.23]:ax.axvline(v,color='gray',alpha=.2)
    save(fig,figs/'velocity_and_coverage')

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name in shown:
        g=test[test.model==name].groupby('speed')
        for ax,metric in zip(axes,['missing_rate','ghost_rate','mse']):
            values=g[metric].mean();ax.plot(values.index,values.values,marker='.',label=LABELS[name],color=COLORS[name])
    for ax,title in zip(axes,['Missing / visible frames','Ghost / visible frames','Image MSE']):
        ax.set(xlabel='True horizontal velocity',title=title);ax.grid(alpha=.2)
    axes[2].set_yscale('log');axes[0].legend(fontsize=7)
    save(fig,figs/'failure_rates_and_image_error')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in ['physical','offset','relaxed']:
        log=pd.read_csv(root/name/'training.csv').dropna(subset=['val_mse'])
        axes[0].plot(log.step,log.val_mse,label=LABELS[name],color=COLORS[name])
        axes[1].plot(log.step,log.val_velocity_mae,label=LABELS[name],color=COLORS[name])
    axes[0].set(xlabel='Training updates',ylabel='Validation MSE',yscale='log');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Training updates',ylabel='Correspondence-fit velocity MAE (both axes)')
    save(fig,figs/'training_curves')

    # Fixed seed and visibility-based selection, independent of model predictions.
    original=np.load(root/'evaluation/physical/sequences/000_bilinear_0.37.npz')
    visible=np.flatnonzero(original['visibility'][50:]>.7)
    selection=visible[np.linspace(0,len(visible)-1,6).astype(int)]
    names=[n for n in ['flowm25','flowm81','physical','relaxed','offset'] if n in datasets]
    fig,axes=plt.subplots(len(names)+1,6,figsize=(12,2*(len(names)+1)))
    frames=[original['frames'][50:]]+[np.load(root/f'evaluation/{name}/sequences/000_bilinear_0.37.npz')['predictions'] for name in names]
    for row,(name,images) in enumerate(zip(['Ground truth']+[LABELS[n] for n in names],frames)):
        for col,t in enumerate(selection):
            ax=axes[row,col];ax.imshow(images[t],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if col==0:ax.set_ylabel(name,fontsize=9)
            if row==0:ax.set_title(f'+{t+1} frames')
    fig.suptitle('Held-out speed 0.37 px/frame · seed 0 · bilinear rendering · common intensity scale [0,1]')
    save(fig,figs/'heldout_rollout')

    table_rows=[]
    for name,row in holdout.items():
        table_rows.append(f"<tr><td>{LABELS[name]}</td><td>{row['mse']:.5f}</td><td>{row['missing_rate']:.1%}</td><td>{row['ghost_rate']:.1%}</td><td>{row['coverage']:.1%}</td><td>{row['ade']:.3f}</td></tr>")
    phys=holdout['physical'];untrained=holdout['physical_untrained'];near=nearest_heldout['physical'];oracle=holdout['oracle_velocity']
    interpretation=f'''<h2>Findings and limits</h2>
<p>The trained physical model estimates held-out horizontal velocities with MAE {phys['latent_velocity_mae']:.4f} px/frame, with {phys['missing_rate']:.3%} missing detections and no detected ghosts. Training reduces image MSE by {1-phys['mse']/untrained['mse']:.1%} relative to the same untrained physical architecture. The free-correction control performs almost identically; the paired interval does not establish an advantage for strict enforcement. The unconstrained-offset model optimizes much less successfully within the same budget.</p>
<p><strong>Rendering transfer remains a limitation.</strong> On nearest-pixel versions of the same held-out speeds, the physical model has ADE {near['ade']:.2f} px, image MSE {near['mse']:.5f}, missing rate {near['missing_rate']:.2%}, ghost rate {near['ghost_rate']:.2%}, and coverage {near['coverage']:.1%}. It was trained on bilinear sequences only. Thus the pilot does not establish rendering-independent velocity estimation.</p>
<p>Supplying true velocity in the separately labeled diagnostic reduces bilinear held-out MSE to {oracle['mse']:.6f}; this identifies velocity inference as a major remaining bottleneck. A plausible follow-up is longer-baseline correspondence refinement and training/evaluation across both observation renderers. These follow-ups have not been run. This pilot also does not isolate learned correspondence improvements from learned memory/decoder improvements.</p>'''
    report='''<!doctype html><meta charset="utf-8"><title>Continuous velocity pilot</title>
<style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#18252b}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}code{background:#eee}</style>
<h1>Continuous-velocity physics attention: pilot results</h1>
<p>One training seed; 5,000 updates per learned model, batch size 8. Official MNIST train identities for training and validation; official test identities for evaluation. 50 context frames, 40 predictions, 64 paired test sequences per speed. Camera trajectories and digit/start selections are paired across models. The three held-out speed intervals are [0.34,0.40], [0.60,0.66], and [1.20,1.26].</p>
<p>The learned model receives images, camera actions/coordinates, and timestamps. True velocity, position, and digit templates are not inputs. The true-velocity row is a separately labeled diagnostic. The image detector uses the known digit template equally for every evaluated model.</p>
<h2>Held-out speeds: 0.37, 0.63, 1.23</h2>
<table><tr><th>Model</th><th>Image MSE</th><th>Missing</th><th>Ghosts</th><th>Coverage</th><th>Conditional ADE (px)</th></tr>'''+''.join(table_rows)+'''</table>
<p>Missing and ghost rates use sufficiently visible ground-truth frames. ADE excludes missing, ghost, and ambiguous tracks; compare it together with coverage. MSE includes all pixels and prediction frames. No detector-based exclusion applies to MSE.</p>'''+interpretation+'''
<h2>Architecture and controls</h2>
<p>Translation-constrained correspondence attention scores camera-compensated feature matches over short time gaps. A differentiable weighted constant-velocity fit determines one continuous velocity. Temporal attention fuses aligned observations into a 50×50 canonical memory at the final context time. Future frames sample this fixed memory using cumulative displacement. The unconstrained model instead predicts offsets with an MLP. The stronger relaxed control starts with identical physical motion, then learns free per-time residual offsets. All models share the appearance encoder, memory attention, and decoder designs.</p>
<p>Finite spatial sampling remains; there is no enumerated velocity-channel bank. Correspondence search is bounded. This single rigid object assumption is stronger than the original FloWM task. Frozen FloWM comparisons establish task improvement but do not isolate training-data or architecture effects. The matched trained controls address the latter only within this pilot.</p>
<h2>Velocity and trajectory coverage</h2><img src="figures/velocity_and_coverage.png">
<p>All curves use the same image-derived velocity estimator. Means are shown only when at least 32 of 64 sequences have a valid fit; gaps denote insufficient estimates. Missing or ambiguous tracks cannot supply a velocity estimate; coverage is shown alongside the curve. The CSV separately records the learned models' latent displacement slopes.</p>
<h2>Failures and image fidelity</h2><img src="figures/failure_rates_and_image_error.png">
<h2>Held-out rollout</h2><img src="figures/heldout_rollout.png">
<p>Seed 0 was fixed in advance. Snapshots were selected only for ground-truth visibility greater than 70%; every panel uses the same intensity scale.</p>
<h2>Learning curves</h2><img src="figures/training_curves.png">
<p>Velocity labels are used for diagnostic logging only, never for the training loss. Offset models' correspondence-fit diagnostics are not their final predicted motion.</p>
<h2>Scope and limitations</h2><p>One optimization seed is a pilot, not a robustness study. Nearest-rendering controls are a transfer test after bilinear training. Detector localization has a pixel-scale floor and may undercount fragmented ghosts. Runtime was measured on a shared GPU and is not an isolated speed benchmark. Multi-object association, acceleration, deformation, and an implicit appearance field remain untested.</p>
<p>Machine-readable results: <a href="summary.json">summary and paired seed intervals</a>, <a href="all_metrics.csv">all metrics</a>, <a href="by_speed.csv">per-speed summary</a>.</p>'''
    (root/'report.html').write_text(report)
    print(json.dumps(dict(heldout=holdout,all_speeds=summary),indent=2))


if __name__=='__main__':main()
