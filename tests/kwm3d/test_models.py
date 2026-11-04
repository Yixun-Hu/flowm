import copy
from pathlib import Path

import pytest
import torch
from kwm.rot import rot_vec
from kwm.transport import stream
from algorithms.mem_wm.backbones.flowm.flowm_models_3d import ViTBlockworldModel
from algorithms.mem_wm.backbones.kwm3d.kwm3d_models import ARMS, KineticBlockworldModel

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "downloaded_checkpoints/hf_models/blockworld-models/blockworld/dynamic/flowm/v2_dynamic.ckpt"
SMALL = dict(embed_dim=32, map_proc_encoder_depth=1, dec_depth=1, num_heads=4,
             input_shape=(3,32,32), img_patch_size=8, world_size=4)


def inputs(batch=1, context=4, prediction=3, size=32):
    torch.manual_seed(312)
    x = torch.rand(batch, context, 3, size, size)
    a = torch.tensor([4, 2, 4, 4]).repeat(batch, 1)[:, :context]
    z = torch.tensor([4, 0, 2]).repeat(batch, 1)[:, :prediction]
    metadata = dict(block_vel_agent=torch.rand(batch, context+prediction, 10, 2) - .5,
                    block_valid=torch.arange(10)[None,None,:].expand(batch, context+prediction, -1) < 6)
    return x, a, z, metadata


@pytest.mark.parametrize("arm", ARMS)
def test_forward_backward(arm):
    m = KineticBlockworldModel(arm=arm, **SMALL)
    x,a,z,meta = inputs()
    y = m(x,a,z,metadata=meta)
    y.square().mean().backward()
    assert y.shape == (1,3,3,32,32)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.map_processor.patch.proj.weight.grad.norm() > 0
    if arm == 'a':
        assert torch.equal(m.last_node_state[:,0], torch.zeros(1,4))


@pytest.mark.parametrize("arm", ["c_pw", "a"])
def test_checkpoint_gradients(arm):
    torch.manual_seed(102)
    plain = KineticBlockworldModel(arm=arm, checkpoint_steps=False, **SMALL)
    checked = copy.deepcopy(plain)
    checked.checkpoint_steps = True
    x,a,z,meta = inputs()
    target = torch.rand(1,3,3,32,32)
    results = []
    for m in (plain, checked):
        torch.manual_seed(17)
        y = m(x,a,z,target_seq=target,teacher_forcing_ratio=.5,metadata=meta)
        y.square().mean().backward()
        results.append(y)
    torch.testing.assert_close(*results, rtol=0, atol=1e-7)
    for (name,p),(_,q) in zip(plain.named_parameters(), checked.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            relative = (p.grad-q.grad).norm() / p.grad.norm().clamp_min(1e-12)
            assert relative <= 1e-5, (name, relative)


@pytest.mark.parametrize("arm,lattice", [("c_pw",None),("b",None),("a","v9"),("a","v17"),("a_oracle",None)])
def test_pooled_token_count(arm,lattice):
    m=KineticBlockworldModel(arm=arm,lattice=lattice, **{**SMALL,"world_size":16,"input_shape":(3,128,128),"img_patch_size":16})
    seen=[]
    hook=m.map_processor.blocks[0].register_forward_pre_hook(lambda _,args: seen.append(args[0].shape[1]))
    h=m.map_processor.init_map(1,torch.device('cpu'))
    m._write(h,torch.rand(1,3,128,128),m._get_fov_mask(h.device),torch.ones(1,h.shape[1],dtype=torch.bool))
    hook.remove()
    assert seen == [177+64]


@pytest.mark.parametrize("arm", ["a","b"])
def test_turn_and_stream(arm):
    m=KineticBlockworldModel(arm=arm,**SMALL)
    torch.manual_seed(42)
    h=torch.randn(2,len(m.lattice),33,33,3)
    state=torch.rand(2,len(m.lattice),4)
    state[:,0]=0
    turned,st=m.turn_state(h,state,1)
    restored,st2=m.turn_state(turned,st,-1)
    torch.testing.assert_close(restored,h,rtol=0,atol=0)
    torch.testing.assert_close(st2,state,rtol=0,atol=0)
    nu=m.lattice[None]+state[...,:2]
    new_nu=m.lattice[None]+st[...,:2]
    torch.testing.assert_close(new_nu,rot_vec(nu[:,m.perms[-1]],1),rtol=0,atol=1e-6)
    ego=torch.zeros(2,2)
    f=h.permute(0,1,4,2,3)
    left=stream(torch.rot90(f,-1,(-2,-1)),rot_vec(nu),ego,c4=True)
    right=torch.rot90(stream(f,nu,ego,c4=True),-1,(-2,-1))
    torch.testing.assert_close(left,right,rtol=0,atol=1e-5)


def test_oracle_masked_slots_inert():
    m=KineticBlockworldModel(arm='a_oracle',**SMALL).eval()
    x,a,z,meta=inputs()
    with torch.no_grad():
        first,h=m(x,a,z,metadata=meta,return_hidden=True)
        meta['block_vel_agent'][~meta['block_valid']]=1234
        second,h2=m(x,a,z,metadata=meta,return_hidden=True)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    torch.testing.assert_close(h,h2,rtol=0,atol=0)
    assert h[:,7:].count_nonzero() == 0
    assert m.last_nodes[:,7:].count_nonzero() == 0


def test_prediction_freezes_estimator_and_turn_invalidates(monkeypatch):
    m=KineticBlockworldModel(arm='a',checkpoint_steps=False,**SMALL)
    calls=[]
    original=m.refiner.forward
    def measurement(*args,**kwargs):
        calls.append(args[0].clone())
        return original(*args,**kwargs)
    monkeypatch.setattr(m.refiner,'forward',measurement)
    x,a,z,meta=inputs()
    a[:]=torch.tensor([4,0,4,4])
    m(x,a,z,metadata=meta)
    assert len(calls)==1  # only pair 2->3, never the prediction frames


@pytest.mark.parametrize('arm',ARMS)
def test_load_released(arm):
    if not CKPT.exists(): pytest.skip('released checkpoint absent')
    source=torch.load(CKPT,map_location='cpu',weights_only=False)['state_dict']
    m=KineticBlockworldModel(arm=arm)
    missing,unexpected=m.load_released(source)
    expected={'map_conv.weight'} if arm in ('b','a','a_oracle') else set()
    if arm=='a': expected|={'P','refiner.n_s'}
    assert set(missing)==expected and not unexpected
    if arm in ('c','c_pw'):
        h=torch.randn(1,5,3,3,256)
        torch.testing.assert_close(m._mix(h),h,rtol=0,atol=0)
    broken=dict(source); broken['bogus']=torch.tensor(0.)
    with pytest.raises(AssertionError): m.load_released(broken)


def test_c_released_parity_two_real_clips():
    paths=sorted((ROOT/'data/blockworld/k3d_v_validation').glob('*/*_rgb.mp4'))[:2]
    if not CKPT.exists() or len(paths)<2: pytest.skip('released checkpoint or two real clips absent')
    from decord import VideoReader,cpu
    kw=dict(v_range=1,v_channel_maxpool_decode=True,v_channel_mixing_mode='2d_v_channel_flattened')
    baseline=ViTBlockworldModel(**kw).eval()
    m=KineticBlockworldModel(arm='c',checkpoint_steps=True).eval()
    source=torch.load(CKPT,map_location='cpu',weights_only=False)['state_dict']
    baseline.load_state_dict({k.removeprefix('main_model.'):v for k,v in source.items()},strict=True)
    m.load_released(source,rebuild_mixing=False)
    for path in paths:
        video=torch.from_numpy(VideoReader(str(path),ctx=cpu(),num_threads=1).get_batch(list(range(5))).asnumpy()).permute(0,3,1,2).float()/255
        actions=torch.load(path.with_name(path.stem.removesuffix('_rgb')+'_actions.pt'),weights_only=False,map_location='cpu')['actions']
        actions=torch.as_tensor(actions[:4]).long()
        actions=torch.cat((torch.tensor([4]),actions))[None]
        with torch.no_grad():
            expected=baseline(video[None,:3],actions[:,:3],actions[:,3:],return_hidden=True)
            actual=m(video[None,:3],actions[:,:3],actions[:,3:],return_hidden=True)
        for p,q in zip(expected,actual): torch.testing.assert_close(p,q,rtol=0,atol=1e-6)


def test_lock_arithmetic_box_and_gradient(monkeypatch):
    import algorithms.mem_wm.backbones.kwm3d.kwm3d_models as module
    m=KineticBlockworldModel(arm='a',**{**SMALL,'world_size':16})
    previous=torch.zeros(1,8,19,17)
    current=torch.ones_like(previous,requires_grad=True)
    state=torch.zeros(1,17,4)
    state[...,2]=2
    captured={}
    def measured(prev,cur,nu,lattice,action,world,*args,**kwargs):
        captured.update(kwargs)
        assert world==33
        delta=cur.mean()*torch.full((1,17,2),.2)
        return delta,torch.ones(1,17),torch.ones(1,17)
    monkeypatch.setattr(module,'lock_measurement',measured)
    support=m._get_fov_mask(torch.device('cpu'))[7:26,16:33]
    updated=m.refiner(previous,current,state,m.lattice,torch.zeros(1,2),(7,16),support)
    assert captured['box']==(7,16) and captured['static_residual']
    assert torch.equal(captured['support_mask'],support)
    assert captured['sigma']==1 and captured['radius'] is None
    torch.testing.assert_close(updated[:,1:,:2],torch.full((1,16,2),.2/3))
    assert (updated[:,1:,2]==3).all() and (updated[:,1:,3]==1).all()
    assert updated[:,0].count_nonzero()==0
    updated[...,:2].sum().backward()
    assert m.refiner.n_s.grad.abs()>0 and current.grad.abs().sum()>0


def test_observation_imprint_has_no_memory_input():
    m=KineticBlockworldModel(arm='a',**{**SMALL,'world_size':16})
    frame=torch.rand(1,3,32,32)
    snapshot,box,support=m.observation_imprint(frame)
    assert snapshot.shape==(1,8,19,17) and box==(7,16) and int(support.sum())==177
    assert snapshot[:,:,~support].count_nonzero()==0
    # Projection of a V-node zero-map write equals the economical one-node lift.
    zero=m.map_processor.init_map(1,torch.device('cpu'))
    written=m._pooled_write(zero,frame,m._get_fov_mask(zero.device),torch.ones(1,17,dtype=torch.bool))
    expected=torch.einsum('kc,bhwc->bkhw',m.P,written[:,0])[:,:,7:26,16:33]
    torch.testing.assert_close(snapshot,expected,rtol=0,atol=1e-6)


def test_oracle_incoming_object_option():
    m=KineticBlockworldModel(arm='a_oracle',oracle_velocity='incoming_object',**SMALL)
    _,_,_,meta=inputs()
    meta['block_vel_object_incoming']=torch.full_like(meta['block_vel_agent'],.25)
    nodes,valid=m._oracle(meta,2,torch.device('cpu'))
    torch.testing.assert_close(nodes[:,1:7],torch.full((1,6,2),.25))
    assert nodes[:,7:].count_nonzero()==0
    meta.pop('block_vel_object_incoming')
    with pytest.raises(ValueError,match='block_vel_object_incoming'):
        m._oracle(meta,2,torch.device('cpu'))


@pytest.mark.parametrize('conv',['c4','plain'])
def test_map_conv_identity(conv):
    m=KineticBlockworldModel(arm='b',map_conv=conv,**SMALL)
    x=torch.randn(1,9,9,9,32)
    torch.testing.assert_close(m._mix(x),x,rtol=0,atol=0)
