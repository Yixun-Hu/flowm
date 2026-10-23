import sys
from pathlib import Path
import unittest
import numpy as np
import torch
from model import fit_velocity,periodic_sample,world_embed,ContinuousMemory
from data import render,read_digits,TrainingData

torch.set_num_threads(1)


class PhysicsTests(unittest.TestCase):
    def test_fit_and_gradients(self):
        dt=torch.tensor([[1.,2.,4.]])
        velocity=torch.tensor([[.37,-.63]])
        displacement=(dt[:,:,None]*velocity[:,None]).requires_grad_()
        weights=torch.tensor([[.3,.4,.9]],requires_grad=True)
        fitted=fit_velocity(displacement,dt,weights)
        torch.testing.assert_close(fitted,velocity)
        fitted.square().sum().backward()
        self.assertTrue(torch.isfinite(displacement.grad).all())
        self.assertGreater(displacement.grad.abs().sum().item(),0)

    def test_camera_embedding(self):
        image=torch.arange(32*32).reshape(1,1,1,32,32).float()
        cameras=torch.tensor([[[49.,-14.]]])
        field,mask=world_embed(image,cameras)
        yy,xx=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
        coords=torch.stack([xx,yy],-1)[None]+cameras[:,0,None,None]
        recovered=periodic_sample(field[:,0],coords)
        torch.testing.assert_close(recovered,image[:,0],atol=.002,rtol=1e-5)

    def test_renderer_matches_existing_generator(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fractional_velocity'))
        from measurement import render as numpy_render
        digit=torch.tensor(read_digits(False)[0][0]).float()/255
        template=np.zeros((50,50),np.float32);template[:28,:28]=digit.numpy()
        positions=np.array([[49.37,3.5],[51.74,-.25],[20.,23.]],np.float32)
        cameras=np.array([[0,0],[47,-16],[4,10]])
        for mode in ['nearest','bilinear']:
            actual=render(digit[None],torch.tensor(positions)[None],torch.tensor(cameras)[None],mode)[0]
            expected,_=numpy_render(template,positions,cameras,mode=='bilinear')
            np.testing.assert_allclose(actual.numpy(),expected,atol=1e-5)

    def test_time_composition(self):
        model=ContinuousMemory();v=torch.tensor([[.37,.63]]);summary=torch.zeros(1,9)
        delta=model.offsets(torch.tensor([[3.,7.,10.]]),v,summary)
        torch.testing.assert_close(delta[:,0]+delta[:,1],delta[:,2])

    def test_relaxed_control_has_identical_initial_motion(self):
        torch.manual_seed(2026);physical=ContinuousMemory('physical')
        torch.manual_seed(2026);relaxed=ContinuousMemory('relaxed')
        for key,param in physical.state_dict().items():torch.testing.assert_close(param,relaxed.state_dict()[key])
        dt=torch.tensor([[-49.,-10.,0.,1.,40.]])
        velocity=torch.tensor([[.37,.2]]);summary=torch.randn(1,9)
        torch.testing.assert_close(physical.offsets(dt,velocity,summary),relaxed.offsets(dt,velocity,summary))

    def test_forward_backward_and_camera_origin(self):
        torch.manual_seed(2026);data=TrainingData('cpu');batch=data.batch(2)
        model=ContinuousMemory()
        keys=['context','cameras','context_times','target_cameras','target_times']
        args={k:batch[k] for k in keys}
        prediction,aux=model(**args)
        self.assertEqual(prediction.shape,batch['target'].shape)
        self.assertTrue(torch.isfinite(aux['velocity']).all())
        (prediction-batch['target']).square().mean().backward()
        self.assertGreater(model.correspondence.log_temperature.grad.abs().item(),0)
        print('Untrained velocity:',aux['velocity'].detach().tolist(),'GT:',batch['true_velocity'].tolist())
        with torch.no_grad():
            args['cameras']=args['cameras']+torch.tensor([7.,-11.])
            args['target_cameras']=args['target_cameras']+torch.tensor([7.,-11.])
            shifted,aux2=model(**args)
        torch.testing.assert_close(shifted,prediction,atol=2e-4,rtol=1e-3)


if __name__=='__main__':unittest.main()
