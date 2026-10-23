"""Timestamp and motion-composition checks on trained models, without retraining."""
import argparse
import json
import sys
import numpy as np
import torch
from data import ROOT
from evaluate import setup
sys.path.insert(0,str(ROOT/'experiments/fractional_velocity'))
from run import digit_data,make_case


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:1');args=p.parse_args()
    torch.set_num_threads(2);images,labels=digit_data();result={}
    for name in ['physical','relaxed']:
        model,_=setup(name,'pilot',args.device);records=[]
        for seed in range(16):
            cases=[make_case(seed,v,'bilinear',images,labels) for v in [.37,.63,1.,1.23]]
            frames=torch.tensor(np.stack([c['frames'] for c in cases]),device=args.device)
            cameras=torch.tensor(np.stack([c['cameras'] for c in cases]),device=args.device).float()
            times=torch.arange(90,device=args.device).float()[None].expand(4,-1)
            full,aux=model(frames[:,:50],cameras[:,:50],times[:,:50],cameras[:,50:],times[:,50:])
            ix=torch.tensor(list(range(0,49,2))+[49],device=args.device)
            sparse,saux=model(frames[:,ix],cameras[:,ix],times[:,ix],cameras[:,50:],times[:,50:])
            dt=torch.tensor([[7.,13.,20.]],device=args.device).expand(4,-1)
            displacement=model.offsets(dt,aux['velocity'],aux['summary'])
            composition=(displacement[:,0]+displacement[:,1]-displacement[:,2]).norm(dim=-1)
            for k,c in enumerate(cases):
                records.append(dict(seed=seed,speed=c['speed'],full_mse=float((full[k]-frames[k,50:]).square().mean()),
                    sparse_mse=float((sparse[k]-frames[k,50:]).square().mean()),
                    full_vx=float(aux['velocity'][k,0]),sparse_vx=float(saux['velocity'][k,0]),
                    composition_error=float(composition[k])))
        result[name]=dict(records=records,full_mse=np.mean([r['full_mse'] for r in records]),
                         sparse_mse=np.mean([r['sparse_mse'] for r in records]),
                         max_composition_error=max(r['composition_error'] for r in records))
    (ROOT/'outputs/continuous_velocity/pilot/diagnostics.json').write_text(json.dumps(result,indent=2))
    print({k:{a:b for a,b in v.items() if a!='records'} for k,v in result.items()})


if __name__=='__main__':main()
