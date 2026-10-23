"""Explanatory figures for RESULTS.md: transport after training, the nearest-rendering
estimation problem, and the localization-success metric. Reads the completed
kinetic_v1 evaluations and writes PNGs next to this file (kinetic/figures/).
Run: python experiments/continuous_velocity/kinetic/explain_figures.py
"""
import sys, glob, json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; R=ROOT/'outputs/continuous_velocity/kinetic_v1'; OUT=Path(__file__).resolve().parent/'figures'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'experiments/fractional_velocity')); from measurement import Detector
C=dict(base='#2a78d6',spectral='#eb6834',newton='#1baf7a',other='#8a8984',oracle='#0b0b0b'); SURF='#fcfcfb'; T1='#0b0b0b'; T2='#52514e'
plt.rcParams.update({'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.edgecolor':'#c3c2b7','axes.labelcolor':T2,'xtick.color':T2,'ytick.color':T2,'text.color':T1,'axes.spines.top':False,'axes.spines.right':False,'font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','axes.grid':True,'grid.color':'#e6e5e1','grid.linewidth':.6,'axes.axisbelow':True})
VAR=['base','alignS','renderS','spectral','newton','spectral_newton']; SEEDS=range(2027,2032); HELD=[.37,.63,1.23]
def rows(v,s,oracle=False):
    p=R/f'evaluation/A_{v}_s{s}{"_oracle" if oracle else ""}/core/metrics.csv'
    return pd.read_csv(p) if p.exists() else None
def pooled(d): return d.squared_error_sum.sum()/d.pixel_count.sum()
def succ(d): return d.frame_success_frames.sum()/d.visible_frames.sum()
tab={}
for v in VAR:
    for s in SEEDS:
        d=rows(v,s); 
        if d is None: continue
        d=d[d.horizon==40]
        for mode in ['bilinear','nearest']:
            m=d[d['mode']==mode]; h=m[m.speed.isin(HELD)]
            tab[(v,s,mode)]=dict(mse=pooled(h),success=succ(h),verr=h.correspondence_velocity_error.mean(),
                                 per_speed={sp:pooled(m[m.speed==sp]) for sp in sorted(m.speed.unique())},
                                 per_speed_succ={sp:succ(m[m.speed==sp]) for sp in sorted(m.speed.unique())})
    d=rows(v,2027,True)
    if d is not None:
        d=d[d.horizon==40]
        for mode in ['bilinear','nearest']:
            m=d[(d['mode']==mode)&d.speed.isin(HELD)]; tab[(v,'oracle',mode)]=dict(mse=pooled(m),success=succ(m),per_speed_succ={sp:succ(d[(d['mode']==mode)&(d.speed==sp)]) for sp in sorted(d.speed.unique())})
col=lambda v: C.get(v, C['newton'] if 'newton' in v else C['spectral'] if v in('alignS','renderS') else C['other'])
def strip(ax,mode,log=False):
    for i,v in enumerate(VAR):
        ys=[tab[(v,s,mode)]['mse'] for s in SEEDS]
        ax.scatter(np.full(len(ys),i)+np.linspace(-.12,.12,len(ys)),ys,s=22,color=col(v),zorder=3,edgecolor=SURF,linewidth=.8)
        ax.hlines(np.mean(ys),i-.25,i+.25,color=col(v),linewidth=2)
        if (v,'oracle',mode) in tab: ax.scatter([i],[tab[(v,'oracle',mode)]['mse']],marker='D',s=30,facecolor=SURF,edgecolor=C['oracle'],linewidth=1.2,zorder=4)
    ax.set_xticks(range(len(VAR))); ax.set_xticklabels(['base','alignS','renderS','spectral','newton','spectral\n+newton'])
    if log: ax.set_yscale('log')
    ax.set_ylabel('image MSE (held-out speeds, horizon 40)')
# ---------- Figure 1: transport after training
fig,axs=plt.subplots(1,3,figsize=(13,4.2),constrained_layout=True)
ax=axs[0]; strip(ax,'bilinear'); ax.set_ylim(0,None); ax.set_title('Bilinear rendering: MSE per training seed (5 seeds)')
ax.scatter([],[],marker='D',facecolor=SURF,edgecolor=C['oracle'],label='same checkpoint, true velocity given (oracle)'); ax.scatter([],[],color=T2,s=22,label='one training seed'); ax.legend(frameon=False,loc='upper right',fontsize=8)
ax=axs[1]
for v,lab in [('base','base (bilinear transport)'),('spectral','spectral transport'),('newton','Newton refinement')]:
    for k,s in enumerate(SEEDS):
        p=glob.glob(str(R/f'training/A_{v}_bilinear_horizontal_s{s}_lr*/training.csv'))
        if not p: continue
        t=pd.read_csv(p[0]).dropna(subset=['val_mse']); ax.plot(t.step,t.val_mse,color=col(v),linewidth=1.2,alpha=.75,label=lab if k==0 else None)
ax.set_yscale('log'); ax.set_xlabel('training update'); ax.set_ylabel('validation image MSE'); ax.set_title('Validation loss during training (5 seeds each)'); ax.legend(frameon=False,fontsize=8)
ax=axs[2]; sp=sorted(tab[('base',2027,'bilinear')]['per_speed'])
for v,lab in [('base','base'),('spectral','spectral'),('newton','Newton')]:
    M=np.array([[tab[(v,s,'bilinear')]['per_speed'][x] for x in sp] for s in SEEDS])
    ax.plot(sp,M.mean(0),color=col(v),marker='o',markersize=4,linewidth=1.6,label=lab)
    ax.fill_between(sp,M.min(0),M.max(0),color=col(v),alpha=.15,linewidth=0)
for x in HELD: ax.axvline(x,color='#c3c2b7',linewidth=.8,linestyle=':')
ax.set_xlabel('true speed (px/frame); dotted = held-out speeds'); ax.set_ylabel('image MSE, horizon 40'); ax.set_title('Bilinear MSE by test speed (band = seed range)'); ax.legend(frameon=False,fontsize=8)
fig.savefig(OUT/'transport_after_training.png',dpi=160); plt.close(fig)
# ---------- Figure 2: nearest rendering
fig,axs=plt.subplots(2,2,figsize=(13,8.4),constrained_layout=True)
ax=axs[0,0]; strip(ax,'nearest',log=True); ax.set_title('Nearest rendering: MSE per seed (log scale) vs. true-velocity oracle')
ax.scatter([],[],marker='D',facecolor=SURF,edgecolor=C['oracle'],label='same checkpoint, true velocity given'); ax.legend(frameon=False,fontsize=8,loc='center right')
ax=axs[0,1]
for j,(mode,c) in enumerate([('bilinear','#2a78d6'),('nearest','#eb6834')]):
    for i,v in enumerate(VAR):
        ys=[tab[(v,s,mode)]['verr'] for s in SEEDS]; x=i+(-.17 if j==0 else .17)
        ax.scatter(np.full(5,x)+np.linspace(-.06,.06,5),ys,s=18,color=c,zorder=3,edgecolor=SURF,linewidth=.6,label=mode if i==0 else None); ax.hlines(np.mean(ys),x-.14,x+.14,color=c,linewidth=2)
ax.set_xticks(range(len(VAR))); ax.set_xticklabels(['base','alignS','renderS','spectral','newton','spectral\n+newton']); ax.set_yscale('log'); ax.set_ylabel('velocity estimate error (px/frame)'); ax.set_title('Velocity estimation error of the same models, by renderer'); ax.legend(frameon=False,fontsize=8,title='test rendering',title_fontsize=8)
ax=axs[1,0]
for v,lab in [('base','base'),('newton','Newton'),('spectral','spectral')]:
    M=np.array([[tab[(v,s,'nearest')]['per_speed_succ'][x] for x in sp] for s in SEEDS]); ax.plot(sp,100*M.mean(0),color=col(v),marker='o',markersize=4,linewidth=1.6,label=lab)
o=[100*tab[('base','oracle','nearest')]['per_speed_succ'][x] for x in sp]; ax.plot(sp,o,color=C['oracle'],linestyle='--',linewidth=1.2,marker='D',markersize=4,markerfacecolor=SURF,label='base + true velocity')
ax.set_ylim(-3,103); ax.set_xlabel('true speed (px/frame)'); ax.set_ylabel('localization success (%)'); ax.set_title('Nearest rendering: localization success by speed (horizon 40)'); ax.legend(frameon=False,fontsize=8)
ax=axs[1,1]; ax.axis('off')
name='seed000__nearest__v0+0.63_a+0__mnist_regular_released.npz'
ex={k:np.load(R/f'evaluation/A_{k}_s2027{"_oracle" if k=="oracle" else ""}/core/examples/{name}',allow_pickle=True) for k in ['base','newton']}
ex['oracle']=np.load(R/f'evaluation/A_base_s2027_oracle/core/examples/{name}',allow_pickle=True)
vis=np.flatnonzero(ex['base']['visible']); ks=[int(vis[1]),int(vis[len(vis)//2]),int(vis[-2])]; fr=ex['base']['frames']; ti=ex['base']['target_indices']
rowsL=[('ground truth',[fr[ti[k]] for k in ks]),('base prediction',[ex['base']['predictions'][k] for k in ks]),('Newton prediction',[ex['newton']['predictions'][k] for k in ks]),('base + true velocity',[ex['oracle']['predictions'][k] for k in ks])]
sub=axs[1,1].get_subplotspec().subgridspec(5,3,wspace=.04,hspace=.18,height_ratios=[.45,1,1,1,1])
for r,(lab,ims) in enumerate(rowsL):
    for c,(k,im) in enumerate(zip(ks,ims)):
        a=fig.add_subplot(sub[r+1,c]); a.imshow(im,cmap='gray',vmin=0,vmax=1); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values(): s.set_visible(False)
        if c==0: a.set_ylabel(lab,fontsize=8,rotation=0,ha='right',va='center',labelpad=4)
        if r==0: a.set_title(f'+{k+1} frames',fontsize=8,fontweight='normal')
vb=ex['base']['velocity']; vest=json.loads(str(ex['base']['model_diagnostics']))['correspondence_velocity']
axs[1,1].set_title(f'Nearest example: true v {vb[0]:.2f}, base estimate {vest[0]:.2f} px/frame',loc='left',pad=2)
fig.savefig(OUT/'nearest_estimation_problem.png',dpi=160); plt.close(fig)
# ---------- Figure 3: localization success metric
fig=plt.figure(figsize=(13,4.2),constrained_layout=True); gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.6])
def panel(ax,mode,k):
    e=np.load(R/f'evaluation/A_base_s2027/core/examples/seed000__{mode}__v0+0.63_a+0__mnist_regular_released.npz',allow_pickle=True)
    pred=e['predictions'][k]; det=Detector(e['template'])(pred); cam=e['cameras'][e['target_indices'][k]]; gt=e['positions'][e['target_indices'][k]]
    ax.imshow(pred,cmap='gray',vmin=0,vmax=1); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    yy,xx=np.indices(e['template'].shape); com=np.array([(e['template']*xx).sum(),(e['template']*yy).sum()])/e['template'].sum()
    gw=(gt-cam+com)%50  # digit centre in window coordinates
    ax.add_patch(plt.Circle((gw[0],gw[1]),2.,fill=False,color='#1baf7a',linewidth=1.5)); ax.plot(gw[0],gw[1],'+',color='#1baf7a',markersize=10,mew=1.5)
    if det['present']: dp=(det['pos']+com)%50; ax.plot(dp[0],dp[1],'x',color='#eb6834',markersize=9,mew=1.8)
    ax.set_xlim(-.5,31.5); ax.set_ylim(31.5,-.5)
    err=float(e['frame_errors'][k]); ok=bool(e['frame_success'][k])
    ax.set_title(f'{mode} rendering, frame +{k+1}\ndetector (x) vs truth (+): '+(f'{err:.2f} px → {"success" if ok else "fail"}' if np.isfinite(err) else 'digit not detected → missing'),fontsize=9)
    return e
panel(fig.add_subplot(gs[0]),'bilinear',17); panel(fig.add_subplot(gs[1]),'nearest',17)
ax=fig.add_subplot(gs[2])
for mode,c in [('bilinear','#2a78d6'),('nearest','#eb6834')]:
    e=np.load(R/f'evaluation/A_base_s2027/core/examples/seed000__{mode}__v0+0.63_a+0__mnist_regular_released.npz',allow_pickle=True)
    fe=e['frame_errors']; t=np.arange(1,41); ax.plot(t,fe,color=c,marker='o',markersize=3.5,linewidth=1.2,label=f'{mode}: success {100*e["frame_success"].sum()/max(1,e["visible"].sum()):.0f}%')
    miss=~e['present']&e['visible']; ax.scatter(t[miss],np.full(miss.sum(),11.5),marker='|',color=c,s=60,linewidth=1.5)
    if mode=='nearest':
        for k in np.flatnonzero(~e['visible']): ax.axvspan(k+.5,k+1.5,color='#e6e5e1',linewidth=0,zorder=0)
ax.axhline(2,color='#1baf7a',linewidth=1.2,linestyle='--'); ax.text(40.5,2.1,'2 px tolerance',color='#1baf7a',fontsize=8,ha='right',va='bottom')
ax.text(40.5,11.7,'| = detector found nothing (missing);  grey = digit out of view, not scored',fontsize=8,ha='right',va='bottom',color=T2); ax.set_ylim(0,12.5); ax.set_xlabel('frames after last context frame'); ax.set_ylabel('detector position − true position (px)'); ax.set_title('Per-frame localization error, base model, true v = 0.63 px/frame'); ax.legend(frameon=False,fontsize=8,loc='center left')
fig.savefig(OUT/'localization_success_metric.png',dpi=160); plt.close(fig)
# ---------- numbers
print('held-out h40 pooled MSE / success / velocity error, mean over seeds')
for mode in ['bilinear','nearest']:
    for v in VAR:
        m=np.array([[tab[(v,s,mode)][k] for k in ('mse','success','verr')] for s in SEEDS]); o=tab.get((v,'oracle',mode))
        print(f'{mode:8s} {v:16s} mse {m[:,0].mean():.5f}±{m[:,0].std(ddof=1):.5f}  succ {100*m[:,1].mean():5.1f}%  verr {m[:,2].mean():.4f}' + (f'  | oracle mse {o["mse"]:.5f} succ {100*o["success"]:.1f}%' if o else ''))
e=ex['base']; print('example nearest v0.63: base est v', vest, 'newton est', json.loads(str(ex['newton']['model_diagnostics']))['correspondence_velocity'], 'present frames base', int(e['present'].sum()), 'newton', int(ex['newton']['present'].sum()))
print('wrote', sorted(p.name for p in OUT.glob('*.png')))
