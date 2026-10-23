"""Train a small physics-attention pilot or its matched learned-offset control."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from model import ContinuousMemory
from data import TrainingData,ROOT


def network_input(batch):
    return {k:batch[k] for k in ['context','cameras','context_times','target_cameras','target_times']}


def objective(prediction,target):
    weights=1+5*(target>.05).to(target.dtype)
    return ((prediction-target).square()*weights).sum()/weights.sum()


@torch.no_grad()
def validate(model,batches):
    model.eval();rows=[]
    for batch in batches:
        pred,aux=model(**network_input(batch))
        rows.append([objective(pred,batch['target']).item(),(pred-batch['target']).square().mean().item(),
                     (aux['velocity']-batch['true_velocity']).abs().mean().item()])
    model.train();return np.mean(rows,0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['physical','offset','relaxed'],required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--device',default='cuda:1');p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--lr',type=float,default=.0003);p.add_argument('--tag',default='pilot')
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/f'outputs/continuous_velocity/{args.tag}/{args.kind}';out.mkdir(parents=True,exist_ok=True)
    ckpt=ROOT/f'downloaded_checkpoints/continuous_velocity/{args.tag}/{args.kind}';ckpt.mkdir(parents=True,exist_ok=True)
    model=ContinuousMemory(args.kind).to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    data=TrainingData(args.device,seed=1234)
    state=data.rng.bit_generator.state;data.rng=np.random.default_rng(999)
    validation=[data.batch(args.batch_size,validation=True) for _ in range(4)]
    data.rng=np.random.default_rng();data.rng.bit_generator.state=state
    start=0;best=float('inf')
    if args.resume and (ckpt/'last.pt').exists():
        saved=torch.load(ckpt/'last.pt',weights_only=False,map_location=args.device)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['step'];best=saved['best'];data.rng.bit_generator.state=saved['data_rng']
    manifest=dict(arguments=vars(args),parameters=sum(p.numel() for p in model.parameters()),
        train_digit_ids='official MNIST train 0:55000',validation_digit_ids='official MNIST train 55000:60000',
        evaluation_digit_ids='official MNIST test',train_camera_episodes='64:224',validation_camera_episodes='224:256',
        evaluation_camera_episodes='0:64',withheld_velocity_intervals=[[x-.03,x+.03] for x in [.37,.63,1.23]],
        velocity_labels_used_for_training=False,position_labels_used_for_training=False,template_given_to_model=False,
        source_sha256={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['model.py','data.py','train.py']},
        torch=torch.__version__,gpu=torch.cuda.get_device_name(torch.device(args.device)),start_time=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    logpath=out/'training.csv';writeheader=not args.resume or not logpath.exists()
    f=open(logpath,'a' if args.resume else 'w');writer=csv.writer(f)
    if writeheader:writer.writerow(['step','seconds','train_loss','velocity_mae','val_loss','val_mse','val_velocity_mae','gpu_peak_mb'])
    began=time.monotonic();model.train()
    for step in range(start,args.steps+1):
        if step==start:
            val=validate(model,validation);loss_value=float('nan');velocity_mae=float('nan')
        else:
            batch=data.batch(args.batch_size);optimizer.zero_grad(set_to_none=True)
            pred,aux=model(**network_input(batch));loss=objective(pred,batch['target'])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            loss_value=loss.item();velocity_mae=(aux['velocity'].detach()-batch['true_velocity']).abs().mean().item()
            val=validate(model,validation) if step%250==0 or step==args.steps else [float('nan')]*3
        if step==start or step%50==0 or step==args.steps:
            seconds=time.monotonic()-began;memory=torch.cuda.max_memory_allocated(torch.device(args.device))/2**20
            writer.writerow([step,seconds,loss_value,velocity_mae,*val,memory]);f.flush()
            print(json.dumps(dict(kind=args.kind,step=step,seconds=round(seconds,1),loss=loss_value,
                                 velocity_mae=velocity_mae,val_loss=float(val[0]),gpu_peak_mb=round(memory))),flush=True)
        if np.isfinite(val[0]):
            improved=val[0]<best
            if improved:best=float(val[0])
            saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,
                       data_rng=data.rng.bit_generator.state,arguments=vars(args))
            torch.save(saved,ckpt/'last.pt')
            if improved:torch.save(saved,ckpt/'best.pt')
    f.close();manifest['complete']=True;manifest['seconds']=time.monotonic()-began
    manifest['best_validation_loss']=best;(out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
