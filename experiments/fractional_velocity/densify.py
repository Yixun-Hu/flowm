"""Test-time accumulated-offset transport. No learned parameters are changed.

Each fractional channel moves by R(t*v)-R((t-1)*v), R(x)=floor(x+.5).
Thus hidden tensors undergo integer rolls only, and cumulative translation is
within half a pixel of t*v. This is a rasterized fractional channel, not a fully
continuous transport operator. At integer velocities it equals released FloWM.
"""
import types
import numpy as np
import torch


def accumulated_forward(self, u_t, h_prev, action):
    b = u_t.shape[0]
    u_conv = self.conv_u(u_t)
    _, c, height, width = u_conv.shape
    full = torch.zeros(b,c,self.world_size,self.world_size,device=h_prev.device)
    full[:,:,:height,:width] = u_conv
    full = full.unsqueeze(1).expand(-1,self.num_v,-1,-1,-1)
    t = self.transport_step
    shifts = np.floor(t*self.velocity_array+.5)-np.floor((t-1)*self.velocity_array+.5)
    shifted = torch.stack([torch.roll(h_prev[:,i],(int(dy),int(dx)),(2,3))
                           for i,(dx,dy) in enumerate(shifts)],dim=1)
    # Same ego-motion ordering and convolution as the released cell.
    ego = []
    for i in range(b):
        dx,dy = -action[i]
        ego.append(torch.roll(shifted[i],(int(dy.item()),int(dx.item())),(2,3)))
    shifted = torch.stack(ego,dim=0)
    h_conv = self.conv_h(shifted.reshape(b*self.num_v,c,self.world_size,self.world_size))
    self.transport_step += 1
    return self.activation(full+h_conv.reshape(b,self.num_v,c,self.world_size,self.world_size))


def apply_densification(model, spacing):
    assert spacing in [1.,.5,.25]
    values = np.arange(-2,2+spacing/2,spacing)
    cell = model.cell
    cell.v_list = [(float(vx),float(vy)) for vx in values for vy in values]
    cell.velocity_array = np.asarray(cell.v_list)
    cell.num_v = len(cell.v_list); model.num_v = cell.num_v
    cell.forward = types.MethodType(accumulated_forward,cell)
    model.register_forward_pre_hook(lambda module,args:setattr(module.cell,'transport_step',0))
    return model
