"""Correspondence attention, physical velocity fitting, and co-moving memory.

Formal predictions use no target images, digit template, true position, or true
velocity. The optional velocity override is exclusively for an oracle diagnostic.
Memory is a 50x50 periodic field in the coordinate system co-moving with the
object and coinciding with world coordinates at the last context time.
"""
import torch
from torch import nn
import torch.nn.functional as F


def fit_velocity(displacements, dt, weights):
    numerator = (weights[..., None]*dt[..., None]*displacements).sum(1)
    denominator = (weights*dt.square()).sum(1, keepdim=True).clamp_min(1e-6)
    return numerator/denominator


def world_embed(features, cameras, size=50):
    b,t,c,h,w=features.shape
    yy,xx=torch.meshgrid(torch.arange(h,device=features.device),torch.arange(w,device=features.device),indexing='ij')
    ix=(xx[None,None]+cameras[:,:,0,None,None].long())%size
    iy=(yy[None,None]+cameras[:,:,1,None,None].long())%size
    indices=(iy*size+ix).flatten(2)
    canvas=features.new_zeros(b,t,c,size*size)
    canvas.scatter_(-1,indices[:,:,None].expand(-1,-1,c,-1),features.flatten(3))
    mask=features.new_zeros(b,t,1,size*size)
    mask.scatter_(-1,indices[:,:,None],1)
    return canvas.reshape(b,t,c,size,size),mask.reshape(b,t,1,size,size)


def periodic_sample(fields, coordinates):
    """BCHW periodic fields sampled at Bx...x2 pixel coordinates."""
    n,c,h,w=fields.shape
    output_shape=coordinates.shape[1:-1]
    coords=coordinates.remainder(h).reshape(n,-1,1,2)+1
    grid=2*coords/torch.tensor([w+1,h+1],device=fields.device,dtype=fields.dtype)-1
    padded=F.pad(fields,(1,1,1,1),mode='circular')
    sampled=F.grid_sample(padded,grid,align_corners=True,mode='bilinear')
    return sampled.reshape(n,c,*output_shape)


class CorrespondenceAttention(nn.Module):
    def __init__(self, channels=4):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(1,12,3,padding=1,bias=False),nn.SiLU(),
                                   nn.Conv2d(12,channels,3,padding=1,bias=False))
        self.log_temperature=nn.Parameter(torch.tensor(3.7))
        self.confidence=nn.Sequential(nn.Linear(3,12),nn.SiLU(),nn.Linear(12,1))
        nn.init.zeros_(self.confidence[-1].weight);nn.init.zeros_(self.confidence[-1].bias)
        yy,xx=torch.meshgrid(torch.arange(50),torch.arange(50),indexing='ij')
        shifts=torch.stack([(xx+25)%50-25,(yy+25)%50-25],-1).reshape(-1,2).float()
        self.register_buffer('shifts',shifts)

    def forward(self,images,cameras,times):
        b,t,h,w=images.shape
        learned=self.encoder(images.reshape(b*t,1,h,w)).reshape(b,t,-1,h,w)
        # Raw intensity is an appearance skip, not privileged object information.
        descriptors=torch.cat([images[:,:,None],learned],2)
        fields,mask=world_embed(descriptors,cameras)
        fft=torch.fft.rfft2(fields)
        energy=torch.fft.rfft2(fields.square().sum(2))
        maskfft=torch.fft.rfft2(mask[:,:,0])
        raw_energy=torch.fft.rfft2(fields[:,:,0].square())
        first=[];second=[];groups=[]
        for g,lag in enumerate([1,2,4]):
            starts=torch.linspace(0,t-lag-1,12,device=images.device).long()
            first.append(starts);second.append(starts+lag);groups.extend([g]*len(starts))
        i=torch.cat(first);j=torch.cat(second)
        dt=times[:,j]-times[:,i]
        numerator=torch.fft.irfft2((fft[:,i].conj()*fft[:,j]).sum(2),s=(50,50)).flatten(2)
        denom=torch.fft.irfft2(energy[:,i].conj()*maskfft[:,j]+maskfft[:,i].conj()*energy[:,j],s=(50,50)).flatten(2)
        raw_overlap=torch.fft.irfft2(raw_energy[:,i].conj()*maskfft[:,j]+maskfft[:,i].conj()*raw_energy[:,j],s=(50,50)).flatten(2).clamp_min(0)
        similarity=(2*numerator/denom.clamp_min(1e-5)).clamp(-1,1)
        # A bounded spatial search uses pixel coordinates, not a bank of velocity channels.
        allowed=(self.shifts.abs()[None,None] <= (2.5*dt[:,:,None,None]+1)).all(-1)
        allowed=allowed & (raw_overlap>2)
        logits=similarity*self.log_temperature.exp().clamp(5,150)
        logits=logits.masked_fill(~allowed,-10000)
        attention=logits.softmax(-1)
        displacement=attention@self.shifts
        match=(attention*similarity).sum(-1)
        support=(attention*raw_overlap).sum(-1)
        entropy=-(attention*attention.clamp_min(1e-10).log()).sum(-1)/8
        conf_input=torch.stack([match,entropy,torch.log1p(support)/6],-1)
        confidence=self.confidence(conf_input).squeeze(-1).sigmoid()
        weights=confidence*match.clamp_min(0).pow(4)*(support/(support+10))*allowed.any(-1)
        velocity=fit_velocity(displacement,dt,weights)
        for _ in range(2):
            residual=(displacement/dt[:,:,None]-velocity[:,None]).square().sum(-1)
            robust=weights/(1+residual/.15**2)
            velocity=fit_velocity(displacement,dt,robust)
        # Identical observation summary for the unconstrained-offset control.
        summaries=[]
        for g in range(3):
            sl=slice(g*12,(g+1)*12); wg=weights[:,sl]
            summaries.append((wg[:,:,None]*displacement[:,sl]/dt[:,sl,None]).sum(1)/wg.sum(1,keepdim=True).clamp_min(1e-6))
        summary=torch.cat(summaries+[torch.stack([weights[:,g*12:(g+1)*12].mean(1) for g in range(3)],-1)],-1)
        return fields,mask,velocity,summary,dict(displacements=displacement,dt=dt,weights=weights)


class ContinuousMemory(nn.Module):
    def __init__(self,kind='physical',channels=4):
        super().__init__();assert kind in ['physical','offset','relaxed'];self.kind=kind
        self.correspondence=CorrespondenceAttention(channels)
        self.memory_attention=nn.Sequential(nn.Conv2d(channels+1,8,1),nn.SiLU(),nn.Conv2d(8,1,1))
        nn.init.zeros_(self.memory_attention[-1].weight);nn.init.zeros_(self.memory_attention[-1].bias)
        self.decoder=nn.Sequential(nn.Conv2d(channels+1,16,3,padding=1,padding_mode='circular'),nn.SiLU(),
                                   nn.Conv2d(16,1,3,padding=1,padding_mode='circular'))
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)
        if kind in ['offset','relaxed']:
            self.offset_head=nn.Sequential(nn.Linear(10,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
            if kind=='relaxed':nn.init.zeros_(self.offset_head[-1].weight)
            else:nn.init.normal_(self.offset_head[-1].weight,std=.001)
            nn.init.zeros_(self.offset_head[-1].bias)
        yy,xx=torch.meshgrid(torch.arange(50),torch.arange(50),indexing='ij')
        self.register_buffer('world_grid',torch.stack([xx,yy],-1).float())
        self.register_buffer('view_grid',self.world_grid[:32,:32].clone())

    def offsets(self,dt,velocity,summary):
        if self.kind=='physical':return dt[:,:,None]*velocity[:,None]
        s=summary[:,None].expand(-1,dt.shape[1],-1)
        args=torch.cat([s,dt[:,:,None]/40],-1)
        zero=torch.cat([s,torch.zeros_like(dt[:,:,None])],-1)
        learned=40*(self.offset_head(args)-self.offset_head(zero))
        if self.kind=='relaxed':learned=learned+dt[:,:,None]*velocity[:,None]
        return learned

    def forward(self,context,cameras,context_times,target_cameras,target_times,velocity_override=None):
        b,t,h,w=context.shape
        fields,mask,velocity,summary,details=self.correspondence(context,cameras,context_times)
        if velocity_override is not None:
            if self.kind!='physical':raise ValueError('Oracle override only defined for physical model')
            velocity=velocity_override
        dt=context_times-context_times[:,-1:]
        offsets=self.offsets(dt,velocity,summary)
        coords=self.world_grid[None,None]+offsets[:,:,None,None]
        aligned=periodic_sample(fields.reshape(b*t,-1,50,50),coords.reshape(b*t,50,50,2)).reshape(b,t,-1,50,50)
        visible=periodic_sample(mask.reshape(b*t,1,50,50),coords.reshape(b*t,50,50,2)).reshape(b,t,1,50,50)
        # Undo zero-padding attenuation at camera boundaries before temporal fusion.
        values=aligned/visible.clamp_min(1e-4)
        scores=self.memory_attention(values.reshape(b*t,-1,50,50)).reshape(b,t,1,50,50)
        weights=(scores.clamp(-8,8).exp()*visible)
        memory=(weights*values).sum(1)/weights.sum(1).clamp_min(1e-5)
        canonical=(memory[:,:1]+self.decoder(memory)).clamp(0,1)
        future_dt=target_times-context_times[:,-1:]
        future_offsets=self.offsets(future_dt,velocity,summary)
        q=self.view_grid[None,None]+target_cameras[:,:,None,None]-future_offsets[:,:,None,None]
        n=target_times.shape[1]
        prediction=periodic_sample(canonical[:,None].expand(-1,n,-1,-1,-1).reshape(b*n,1,50,50),q.reshape(b*n,32,32,2)).reshape(b,n,32,32)
        return prediction,dict(velocity=velocity,offsets=future_offsets,memory=canonical,summary=summary,**details)
