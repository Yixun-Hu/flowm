"""Create diagnostic plots and seed-clustered statistical summaries."""
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


def interval(values):
    values=np.asarray(values,float); values=values[np.isfinite(values)]
    if not len(values):return np.nan,np.nan,np.nan
    rng=np.random.default_rng(2026)
    means=values[rng.integers(len(values),size=(2000,len(values)))].mean(1)
    return float(values.mean()),*np.quantile(means,[.025,.975])


def errorbar(ax,groups,column,label,**kwargs):
    xs=[]; ys=[]; los=[]; his=[]
    for x,g in groups:
        mean,lo,hi=interval(g[column]); xs.append(x); ys.append(mean);los.append(mean-lo);his.append(hi-mean)
    ax.errorbar(xs,ys,yerr=[los,his],label=label,capsize=2,**kwargs)


def save(fig,path):
    fig.tight_layout();fig.savefig(path.with_suffix('.png'),dpi=180)
    fig.savefig(path.with_suffix('.pdf'));plt.close(fig)


def prepare(directory):
    x=pd.read_csv(directory/'metrics.csv')
    x['missing_rate']=x.missing_frames/x.visible_frames
    x['ghost_rate']=x.ghost_frames/x.visible_frames
    x['coverage']=x.scored_frames/x.visible_frames
    x['velocity_abs_error']=abs(x.predicted_vx-x.speed)
    x['tracker_velocity_abs_error']=abs(x.tracker_vx-x.speed)
    x['visible_fraction']=x.visible_frames/x.horizon
    x['context_visible_fraction']=x.context_visible/50
    extras=[]
    for row in x.itertuples():
        z=np.load(directory/'sequences'/f'{row.seed:03d}_{row.mode}_{row.speed:g}.npz')
        vis=z['visibility'][50:50+row.horizon]>=.35
        good=z['usable'][:row.horizon]
        extras.append(dict(success_rate=float(((z['errors'][:row.horizon]<=2)&good).sum()/max(vis.sum(),1)),
            amplitude=float(z['amplitude'][:row.horizon][vis].mean()) if vis.any() else np.nan,
            fit_quality=float(z['quality'][:row.horizon][vis].mean()) if vis.any() else np.nan))
    return pd.concat([x,pd.DataFrame(extras)],axis=1)


def adjusted_effect(df,column):
    """Within-seed OLS; seed bootstrap, speed + speed^2 + visibility controls."""
    data=df.dropna(subset=[column]).copy()
    names=['grid_distance','speed','speed2','visible_fraction','context_visible_fraction']
    data['speed2']=data.speed**2
    def fit(d):
        cols=names+[column]
        centered=d[cols]-d.groupby('seed')[cols].transform('mean')
        return np.linalg.lstsq(centered[names].values,centered[column].values,rcond=None)[0][0]
    if len(data)<20:return dict(coefficient=None,n=len(data))
    coefficient=fit(data);seeds=data.seed.unique();rng=np.random.default_rng(42);coefs=[]
    for _ in range(1000):
        sample=pd.concat([data[data.seed==s].assign(seed=i) for i,s in enumerate(rng.choice(seeds,len(seeds)))])
        coefs.append(fit(sample))
    return dict(coefficient=float(coefficient),ci95=np.quantile(coefs,[.025,.975]).tolist(),n=len(data),
                controls=names[1:],seed_fixed_effects=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/fractional_velocity'))
    args=p.parse_args();root=args.root;out=root/'figures';out.mkdir(exist_ok=True)
    frames={}
    for folder,label in [('main','Original (25)'),('dense05','Δ=0.5 (81)'),('dense025','Δ=0.25 (289)')]:
        if (root/folder/'metrics.csv').exists():frames[label]=prepare(root/folder)
    original=frames['Original (25)'];nearest=original[(original['mode']=='nearest') & (original.horizon==40)]
    fig,axs=plt.subplots(2,1,figsize=(8,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for label,x in frames.items():
        x=x[(x['mode']=='nearest')&(x.horizon==40)]
        errorbar(axs[0],x.groupby('speed'),'predicted_vx',label,marker='o',linestyle='none')
        errorbar(axs[1],x.groupby('speed'),'coverage',label,marker='o')
    errorbar(axs[0],nearest.groupby('speed'),'tracker_vx','Continuous tracker',marker='x',linestyle='--',color='black')
    axs[0].plot([0,2],[0,2],':',color='gray');axs[0].set(ylabel='Estimated vx (px/frame)',title='Velocity curve: absent detections remain unmeasurable')
    axs[1].set(xlabel='True vx (px/frame)',ylabel='Scored / visible frames',ylim=(-.04,1.04))
    axs[0].legend(fontsize=8);save(fig,out/'velocity_quantization')

    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for i,h in enumerate([20,40]):
        x=original[(original['mode']=='nearest')&(original.horizon==h)]
        # Aggregate within seed before CIs to preserve paired observations.
        grouped=x.groupby(['seed','grid_distance'])[['ade','endpoint_error','tracker_ade','missing_rate']].mean().reset_index()
        errorbar(axs[0,i],grouped.groupby('grid_distance'),'ade','FloWM conditional ADE',marker='o')
        errorbar(axs[0,i],grouped.groupby('grid_distance'),'tracker_ade','Tracker ADE',marker='x')
        errorbar(axs[1,i],grouped.groupby('grid_distance'),'missing_rate','Missing detections',marker='o',color='tab:red')
        axs[0,i].set(title=f'Horizon {h}',ylabel='Mean trajectory error (px)');axs[0,i].legend(fontsize=8)
        axs[1,i].set(xlabel='Distance to nearest velocity channel',ylabel='Missing / visible frames',ylim=(-.04,1.04))
    save(fig,out/'error_vs_grid_distance')

    fig,axs=plt.subplots(3,2,figsize=(11,9),sharex=True)
    for i,speed in enumerate([.5,1.,1.5]):
        for label,folder in [('Original (25)','main'),('Δ=0.5 (81)','dense05'),('Δ=0.25 (289)','dense025')]:
            if label not in frames:continue
            zs=[np.load(f) for f in sorted((root/folder/'sequences').glob(f'*_nearest_{speed:g}.npz'))]
            errors=np.array([np.where(z['usable'],z['errors'],np.nan) for z in zs])
            vis=np.array([z['visibility'][50:]>=.35 for z in zs]);usable=np.array([z['usable'] for z in zs])
            count=np.isfinite(errors).sum(0);mean=np.nansum(errors,axis=0)/np.maximum(count,1);mean[count==0]=np.nan
            axs[i,0].plot(np.arange(1,41),mean,label=label)
            axs[i,1].plot(np.arange(1,41),usable.sum(0)/np.maximum(vis.sum(0),1),label=label)
            if label=='Original (25)':
                base=np.array([np.where(z['visibility'][50:]>=.35,z['tracker_errors'],np.nan) for z in zs])
                axs[i,0].plot(np.arange(1,41),np.nanmean(base,axis=0),'--',color='black',label='Tracker')
        axs[i,0].plot(np.arange(1,41),np.arange(1,41)*min(abs(speed-k) for k in [0,1,2]),':',color='gray',label='T d(v,V), reference only')
        axs[i,0].set(ylabel=f'v={speed}: error (px)');axs[i,1].set(ylabel='Scored / visible',ylim=(-.04,1.04))
    axs[0,0].legend(fontsize=7);axs[-1,0].set_xlabel('Prediction horizon');axs[-1,1].set_xlabel('Prediction horizon')
    save(fig,out/'error_vs_horizon')

    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for mode,g in original[original.horizon==40].groupby('mode'):
        errorbar(axs[0],g.groupby('speed'),'missing_rate',mode,marker='o')
        errorbar(axs[1],g.groupby('speed'),'amplitude',mode,marker='o')
        errorbar(axs[2],g.groupby('speed'),'ghost_rate',mode,marker='o')
    for ax,title in zip(axs,['Missing detections','Matched-template amplitude','Ghost detections']):
        ax.set(xlabel='True vx',title=title);ax.legend(fontsize=8)
    save(fig,out/'appearance_controls')

    columns=['predicted_vx','tracker_vx','ade','endpoint_error','tracker_ade','missing_rate','ghost_rate','coverage','success_rate','amplitude']
    all_data=pd.concat([x.assign(model=label) for label,x in frames.items()])
    all_data.to_csv(root/'all_metrics.csv',index=False)
    all_data.groupby(['model','mode','horizon','speed'])[columns].mean().to_csv(root/'aggregate_metrics.csv')
    statistics={}
    for metric in ['ade','missing_rate','success_rate']:
        statistics[metric+'_vs_grid_distance']=adjusted_effect(nearest,metric)
    paired=[]
    for label,x in frames.items():
        if label=='Original (25)':continue
        x=x[(x['mode']=='nearest')&(x.horizon==40)]
        merged=nearest.merge(x,on=['seed','speed'],suffixes=('_orig','_dense'))
        for group,sel in [('on_grid',merged.speed.isin([0,1,2])),('half_grid',merged.speed.isin([.5,1.5])),('quarter_grid',merged.speed.isin([.25,.75,1.25,1.75])),('near_grid',merged.speed.isin([.1,.9,1.1,1.9]))]:
            g=merged[sel]
            for metric in ['missing_rate','success_rate','ade']:
                diff=g.assign(delta=g[metric+'_dense']-g[metric+'_orig']).groupby('seed').delta.mean()
                mean,lo,hi=interval(diff)
                paired.append(dict(model=label,group=group,metric=metric,delta=mean,ci_low=lo,ci_high=hi,n_seeds=int(diff.notna().sum())))
    statistics['paired_densification']=paired
    (root/'statistics.json').write_text(json.dumps(statistics,indent=2))
    print(nearest.groupby('speed')[columns].mean().round(4).to_string())
    print(json.dumps(statistics,indent=2))


if __name__=='__main__':main()
