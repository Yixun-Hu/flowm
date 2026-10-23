"""Paired synthetic data; official MNIST train/test identity separation."""
import gzip
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/fractional_velocity'
HOLDOUT=[.37,.63,1.23]
SPEEDS=[0,.1,.25,.37,.5,.63,.75,.9,1,1.1,1.23,1.25,1.5,1.75,1.9,2]


def read_digits(train):
    prefix='train' if train else 't10k'
    with gzip.open(DATA/f'{prefix}-images-idx3-ubyte.gz','rb') as f:
        images=np.frombuffer(f.read(),np.uint8,offset=16).reshape(-1,28,28).copy()
    with gzip.open(DATA/f'{prefix}-labels-idx1-ubyte.gz','rb') as f:
        labels=np.frombuffer(f.read(),np.uint8,offset=8).copy()
    return images,labels


def camera_bank():
    return np.stack([torch.load(DATA/f'released/dynamic_validation_200/0/{i:04d}.pt',weights_only=False)['actions'][:90].numpy() for i in range(256)])


def render(digits,positions,cameras,mode='bilinear'):
    """Directly sample the original sprite at each absolute position, never recur."""
    b,n,_=positions.shape;device=digits.device
    yy,xx=torch.meshgrid(torch.arange(32,device=device),torch.arange(32,device=device),indexing='ij')
    world=torch.stack([xx,yy],-1)[None,None]+cameras[:,:,None,None]
    if mode=='nearest':positions=(positions+.5).floor()
    relative=(world-positions[:,:,None,None])%50
    lo=relative.floor().long();frac=relative-lo
    canvas=digits.new_zeros(b,50,50);canvas[:,:28,:28]=digits
    flattened=canvas.flatten(1)
    output=digits.new_zeros(b,n,32,32)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        ix=(lo[...,0]+dx)%50;iy=(lo[...,1]+dy)%50
        vals=flattened.gather(1,(iy*50+ix).reshape(b,-1)).reshape(b,n,32,32)
        weight=(frac[...,0] if dx else 1-frac[...,0])*(frac[...,1] if dy else 1-frac[...,1])
        output=output+vals*weight
    return output


class TrainingData:
    def __init__(self,device,seed=1234):
        images,_=read_digits(True)
        self.images=torch.from_numpy(images).to(device).float()/255
        self.actions=torch.from_numpy(camera_bank()).to(device)
        self.device=device;self.rng=np.random.default_rng(seed)

    def batch(self,batch_size=8,validation=False):
        rng=self.rng
        indices=rng.integers(55000,60000,batch_size) if validation else rng.integers(0,55000,batch_size)
        episodes=rng.integers(224,256,batch_size) if validation else rng.integers(64,224,batch_size)
        velocity=rng.uniform(0,2,batch_size)
        for i in range(batch_size):
            while min(abs(velocity[i]-v) for v in HOLDOUT)<.03:velocity[i]=rng.uniform(0,2)
        integer=rng.random(batch_size)<.15
        velocity[integer]=rng.integers(0,3,integer.sum())
        starts=rng.integers(0,50,(batch_size,2))
        targets=np.array([50,54,59,69,79,89])
        times=np.concatenate([np.arange(50),targets])
        times=torch.as_tensor(times,device=self.device).float()[None].expand(batch_size,-1)
        v=torch.as_tensor(np.column_stack([velocity,np.zeros(batch_size)]),device=self.device,dtype=torch.float32)
        p=torch.as_tensor(starts,device=self.device,dtype=torch.float32)[:,None]+times[:,:,None]*v[:,None]
        cameras=self.actions[episodes].cumsum(1)-16
        cameras=cameras[:,times[0].long()]
        frames=render(self.images[indices],p,cameras)
        # Only images/cameras/times go to the NN; v is diagnostic logging only.
        return dict(context=frames[:,:50],cameras=cameras[:,:50].float(),context_times=times[:,:50],
                    target_cameras=cameras[:,50:].float(),target_times=times[:,50:],target=frames[:,50:],true_velocity=v)
