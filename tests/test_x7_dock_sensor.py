"""Offline contracts for simulated distance uncertainty."""
from importlib.util import find_spec
import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from audit_x7_dock_sensor import POS_ERROR_BOUND, correct_linkwise, correct_sample, movement_bound


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class SensorContracts(unittest.TestCase):
    def test_quaternion_sign_is_irrelevant(self):
        a=np.array([[0.,0.,0.,0.,0.,0.,1.]])
        b=a.copy();b[0,3:]*=-1
        self.assertEqual(movement_bound(a,b,np.array([1.])),0.)

    def test_delayed_distance_uses_worst_link_motion_and_positive_error(self):
        a=np.array([[0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,1.]])
        b=a.copy();b[0,0]=.004;b[1,1]=.010
        raw=np.array([.05,.012,.09]); radii=np.array([1.,2.])
        d,m=correct_sample(raw,a,b,radii,False)
        self.assertEqual(m,.010);np.testing.assert_allclose(d,raw-.010-POS_ERROR_BOUND)
        base,_=correct_sample(raw,a,b,radii,True);np.testing.assert_array_equal(base,raw)

    def test_linkwise_uses_motion_of_the_relevant_link_without_dropping_bound(self):
        a=np.array([[0.,0.,0.,0.,0.,0.,1.],[0.,0.,0.,0.,0.,0.,1.]])
        b=a.copy(); b[0,0]=.001; b[1,0]=.020
        raw=np.array([[.015,.090],[.060,.080],[.070,.070]])
        got,_=correct_linkwise(raw,a,b,np.ones(2))
        np.testing.assert_allclose(got, [.015-.001-POS_ERROR_BOUND,.060-.001-POS_ERROR_BOUND,.070-.020-POS_ERROR_BOUND])

    def test_rotation_of_a_bounded_link_is_accounted_as_chord(self):
        a=np.array([[0.,0.,0.,0.,0.,0.,1.]])
        theta=.04;b=a.copy();b[0,5]=np.sin(theta/2);b[0,6]=np.cos(theta/2)
        self.assertAlmostEqual(movement_bound(a,b,np.array([2.])),4*np.sin(theta/2),places=10)


if __name__=='__main__':
    unittest.main()
