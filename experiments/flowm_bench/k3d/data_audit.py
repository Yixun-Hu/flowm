"""Read-only audit of stored action codes and outgoing velocity semantics."""
import torch
from .common import clips, clip_id, companion, truth, write_csv


def main():
    rows = []
    for split, n in (("k3d_v_validation",200), ("k3d_int_validation",20),
                     ("k3d_null_noblocks",50), ("k3d_null_static",50)):
        for path in clips(split,n):
            labels = truth(path)
            actions = torch.load(companion(path,"actions.pt"), map_location="cpu", weights_only=False)
            a = actions["actions"][:139]
            delta = actions["delta_xz"][:140]
            angle = labels["agent_dir"][:140]
            c,s = angle.cos(),angle.sin()
            camera = torch.stack([delta[:,0]*c-delta[:,1]*s,delta[:,0]*s+delta[:,1]*c],-1)
            world = labels["block_vel_world"][:140,:,[0,2]]
            intrinsic = torch.stack([world[...,0]*c[:,None]-world[...,1]*s[:,None],
                                     world[...,0]*s[:,None]+world[...,1]*c[:,None]],-1)
            error = (labels["block_vel_agent"][:140] + camera[:,None] - intrinsic).abs()
            rows.append(dict(split=split, clip=clip_id(path), left=int((a==0).sum()), right=int((a==1).sum()),
                             forward=int((a==2).sum()), noop=int((a==4).sum()),
                             unknown_action=int(~torch.isin(a,torch.tensor([0,1,2,4])).all()),
                             forward_displacement_max_error=float((delta[:139][a==2].norm(dim=-1)-1).abs().max()) if (a==2).any() else 0.,
                             relative_velocity_identity_max_error=float(error.max()) if error.numel() else 0.))
    write_csv("data_audit.csv", rows)


if __name__ == "__main__":
    main()
