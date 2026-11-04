from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from torch.utils.data._utils.collate import default_collate
from datasets.video.blockworld import BlockWorldVideoDataset

ROOT=Path(__file__).resolve().parents[2]


def lightweight_dataset(path,start,n_frames=7,skip=1,load=True):
    """Use the real __getitem__ without rebuilding shared on-disk indices."""
    from decord import VideoReader,cpu
    length=len(VideoReader(str(path),ctx=cpu(),num_threads=1))
    ds=BlockWorldVideoDataset.__new__(BlockWorldVideoDataset)
    ds.cfg=SimpleNamespace(load_block_info=load,cond_alignment='t-1->t',data_cond_alignment='t->t+1')
    ds.metadata=[dict(video_paths=path,video_pts=torch.arange(length))]
    ds.idx_remap=[start]; ds.cumulative_sizes=[length-n_frames+1]
    ds.n_frames=n_frames;ds.frame_skip=skip;ds.use_preprocessed_latents=False
    ds.use_depth=False;ds.purpose='validation';ds.external_cond_dim=5
    ds.cond_loading_style='action_int';ds.transform=lambda x:x
    ds.is_latent_preprocessing_expt=False
    return ds


@pytest.mark.parametrize('skip',[1,2])
def test_real_block_alignment(skip):
    paths=sorted((ROOT/'data/blockworld/k3d_v_validation').glob('*/*_rgb.mp4'))
    if not paths: pytest.skip('blockworld data absent')
    path=paths[0];start=17
    ds=lightweight_dataset(path,start,skip=skip)
    sample=ds[0]
    info=torch.load(path.with_name(path.stem.removesuffix('_rgb')+'_block_info.pt'),map_location='cpu',weights_only=False)
    n=info['num_blocks']
    expected=info['block_pos_agent'][start:start+ds.n_frames:skip]
    assert torch.equal(sample['metadata']['block_pos_agent'][:,:n],expected)
    assert sample['videos'].shape[0]==sample['metadata']['block_valid'].shape[0]
    assert sample['metadata']['block_valid'][:,:n].all()
    assert not sample['metadata']['block_valid'][:,n:].any()
    batch=default_collate([sample,sample])
    assert batch['metadata']['block_pos_agent'].shape==(2,len(expected),10,2)
    assert batch['metadata']['block_valid'].dtype==torch.bool


def test_opt_out_unchanged():
    paths=sorted((ROOT/'data/blockworld/k3d_v_validation').glob('*/*_rgb.mp4'))
    if not paths: pytest.skip('blockworld data absent')
    old=lightweight_dataset(paths[0],10,load=False)[0]
    new=lightweight_dataset(paths[0],10,load=True)[0]
    for key,value in old.items():
        if key=='metadata':
            for k,v in value.items(): assert new[key][k]==v
        elif isinstance(value,torch.Tensor): assert torch.equal(value,new[key])
        else: assert value==new[key]
    assert set(old['metadata'])=={'path','clip'}


def test_block_padding(tmp_path):
    n=2;pos=torch.arange(5*n*2).reshape(5,n,2).float()
    path=tmp_path/'0000_rgb.mp4'
    torch.save(dict(num_blocks=n,block_pos_agent=pos,block_vel_agent=pos),tmp_path/'0000_block_info.pt')
    ds=BlockWorldVideoDataset.__new__(BlockWorldVideoDataset)
    ds.n_frames=7;ds.frame_skip=2
    result=ds.load_block_metadata(dict(video_paths=path,video_pts=torch.arange(5)),3,5)
    assert torch.equal(result['block_pos_agent'][0,:n],pos[3])
    assert result['block_valid'][0,:n].all()
    assert not result['block_valid'][1:].any()


def test_corrected_oracle_incoming_alignment():
    paths=sorted((ROOT/'data/blockworld/k3d_v_validation').glob('*/*_rgb.mp4'))
    if not paths: pytest.skip('blockworld data absent')
    path=paths[0]; start=17
    ds=lightweight_dataset(path,start)
    metadata=ds[0]['metadata']
    info=torch.load(path.with_name(path.stem.removesuffix('_rgb')+'_block_info.pt'),weights_only=False,map_location='cpu')
    n=info['num_blocks'];heading=info['agent_dir'][start]
    world=info['block_vel_world'][start-1,:, (0,2)]
    expected=torch.stack((heading.cos()*world[:,0]-heading.sin()*world[:,1],
                          heading.sin()*world[:,0]+heading.cos()*world[:,1]),-1)
    torch.testing.assert_close(metadata['block_vel_object_incoming'][0,:n],expected)
    # Raw field remains exactly what the generator wrote.
    torch.testing.assert_close(metadata['block_vel_agent'][0,:n],info['block_vel_agent'][start],rtol=0,atol=0)
