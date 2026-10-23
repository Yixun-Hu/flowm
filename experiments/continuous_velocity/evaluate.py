"""Paired evaluation using the existing detector, plus latent velocity diagnostics."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from model import ContinuousMemory
from data import ROOT,SPEEDS
sys.path.insert(0,str(ROOT/'experiments/fractional_velocity'))
from run import digit_data,make_case,measure,write_csv,load_model,predict
from densify import apply_densification


def setup(name,tag,device):
    if name in ['flowm25','flowm81']:
        model=load_model(device)
        if name=='flowm81':apply_densification(model,.5)
        return model,dict(checkpoint='released',train_steps=0)
    torch.manual_seed(2026)
    kind=name if name in ['offset','relaxed'] else 'physical'
    model=ContinuousMemory(kind).to(device)
    metadata=dict(train_steps=0)
    if name!='physical_untrained':
        path=ROOT/f'downloaded_checkpoints/continuous_velocity/{tag}/{kind}/best.pt'
        saved=torch.load(path,map_location=device,weights_only=False)
        model.load_state_dict(saved['model'],strict=True)
        metadata=dict(checkpoint=str(path.resolve()),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),train_steps=saved['step'])
    model.eval().requires_grad_(False)
    return model,metadata


@torch.inference_mode()
def infer(model,name,cases,device):
    if name in ['flowm25','flowm81']:
        result=predict(model,[c['frames'] for c in cases],[c['actions'] for c in cases],device)
        return result,[{} for _ in cases]
    batch=len(cases)
    images=torch.tensor(np.array([c['frames'][:50] for c in cases]),device=device)
    cameras=torch.tensor(np.array([c['cameras'] for c in cases]),device=device,dtype=torch.float32)
    times=torch.arange(90,device=device,dtype=torch.float32)[None].expand(batch,-1)
    kwargs={}
    if name=='oracle_velocity':
        kwargs['velocity_override']=torch.tensor([[c['speed'],0] for c in cases],device=device,dtype=torch.float32)
    result,aux=model(images,cameras[:,:50],times[:,:50],cameras[:,50:],times[:,50:],**kwargs)
    v=(aux['offsets'][:,-1]-aux['offsets'][:,0])/39
    extras=[dict(latent_vx=float(v[i,0]),latent_vy=float(v[i,1]),
                 correspondence_vx=float(aux['velocity'][i,0]),correspondence_vy=float(aux['velocity'][i,1])) for i in range(batch)]
    return result.cpu().numpy(),extras


def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=['physical','offset','physical_untrained','oracle_velocity'])
    p.add_argument('--seeds',type=int,default=64);p.add_argument('--modes',nargs='+',default=['bilinear','nearest'])
    p.add_argument('--speeds',nargs='+',type=float,default=SPEEDS)
    p.add_argument('--device',default='cuda:1');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--tag',default='pilot');args=p.parse_args();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    images,labels=digit_data()
    for name in args.models:
        out=ROOT/f'outputs/continuous_velocity/{args.tag}/evaluation/{name}'
        out.mkdir(parents=True,exist_ok=True);(out/'sequences').mkdir(exist_ok=True)
        model,metadata=setup(name,args.tag,args.device);rows=[];inference_seconds=0;inferred=0
        expected={f'{seed:03d}_{mode}_{speed:g}.npz' for seed in range(args.seeds) for mode in args.modes for speed in args.speeds}
        for seed in range(args.seeds):
            cases=[make_case(seed,speed,mode,images,labels) for mode in args.modes for speed in args.speeds]
            for start in range(0,len(cases),args.batch_size):
                batch=cases[start:start+args.batch_size]
                for c in batch:
                    filename=f"{seed:03d}_{c['mode']}_{c['speed']:g}.npz"
                    c['output']=out/'sequences'/filename;c['cached']=c['output']
                    if name in ['flowm25','flowm81'] and not c['cached'].exists():
                        source='main' if name=='flowm25' else 'dense05'
                        c['cached']=ROOT/'outputs/fractional_velocity'/source/'sequences'/filename
                pending=[c for c in batch if not c['cached'].exists()]
                calculated={}
                if pending:
                    torch.cuda.synchronize();began=time.monotonic()
                    preds,extras=infer(model,name,pending,args.device)
                    torch.cuda.synchronize();inference_seconds+=time.monotonic()-began;inferred+=len(pending)
                    calculated={str(c['output']):(pred,extra) for c,pred,extra in zip(pending,preds,extras)}
                for c in batch:
                    if c['cached'].exists():
                        with np.load(c['cached']) as saved:
                            pred=saved['predictions'];extra=json.loads(str(saved['model_diagnostics'])) if 'model_diagnostics' in saved else {}
                    else:pred,extra=calculated[str(c['output'])]
                    measured,arrays=measure(c,pred)
                    for r in measured:r.update(extra);r['model']=name
                    rows.extend(measured)
                    arrays['model_diagnostics']=np.array(json.dumps(extra))
                    np.savez_compressed(c['output'],**arrays)
            write_csv(out/'metrics.csv',rows)
            print(f'{name}: seed {seed+1}/{args.seeds}, {len(rows)//2} sequences',flush=True)
        metadata.update(arguments=vars(args),complete=True,sequences=len(rows)//2,
                        evaluated_names=sorted(expected),new_inference_sequences=inferred,inference_seconds=inference_seconds,
                        oracle=name=='oracle_velocity',parameters=sum(p.numel() for p in model.parameters()))
        (out/'manifest.json').write_text(json.dumps(metadata,indent=2));del model;torch.cuda.empty_cache()


if __name__=='__main__':main()
