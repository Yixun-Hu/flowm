"""Scientific checks: wraps, paired conditions, tracker, misses, ghosts."""
import unittest
import numpy as np
from measurement import Detector, periodic_fit, render, translate


class MeasurementsTest(unittest.TestCase):
    def setUp(self):
        self.template = np.zeros((50, 50), np.float32)
        self.template[2:11, 3:6] = 1
        self.template[8:11, 3:11] = .8

    def test_integer_renderers_are_identical(self):
        for pos in [(0,0), (49,49), (100,-51)]:
            np.testing.assert_array_equal(translate(self.template,pos),
                                          translate(self.template,pos,True))

    def test_fractional_shift_does_not_accumulate_blur(self):
        x = translate(self.template,(50.5,0),True)
        np.testing.assert_array_equal(x,translate(self.template,(.5,0),True))
        self.assertAlmostEqual(float(x.sum()),float(self.template.sum()),places=5)

    def test_unwrapped_motion_crosses_multiple_worlds(self):
        t=np.arange(90); gt=np.column_stack([47+1.5*t, 3+0*t])
        wrapped=gt%50; wrapped[::3]=np.nan
        track,v,origin=periodic_fit(wrapped)
        np.testing.assert_allclose(v,[1.5,0],atol=1e-6)
        valid=np.isfinite(track).all(1)
        displacement=track[valid]-gt[valid]
        np.testing.assert_allclose(displacement-displacement[0],0,atol=1e-6)
        self.assertGreater(np.ptp(track[valid,0]),100)

    def test_blank_and_duplicate_detections(self):
        d=Detector(self.template)
        self.assertFalse(d(np.zeros((32,32)))['present'])
        one=translate(self.template,(1,2))[:32,:32]
        two=one+translate(self.template,(17,17))[:32,:32]
        self.assertTrue(d(one)['present']); self.assertFalse(d(one)['ghost'])
        self.assertTrue(d(two)['present']); self.assertTrue(d(two)['ghost'])

    def test_camera_compensation_recovers_world_velocity(self):
        rng=np.random.default_rng(7)
        t=np.arange(90); pos=np.column_stack([12+.5*t, 5+0*t])
        cameras=rng.integers(-60,60,(90,2))
        frames,visibility=render(self.template,pos,cameras)
        d=Detector(self.template)
        wrapped=(np.array([d(f)['pos'] for f in frames])+cameras)%50
        _,v,_=periodic_fit(wrapped)
        np.testing.assert_allclose(v,[.5,0],atol=.02)


if __name__=='__main__': unittest.main()
