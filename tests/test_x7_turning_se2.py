"""Deterministic tests of common SE(2) retargeting; no simulator launches."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
HAS_RUNTIME=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS_RUNTIME:
    import numpy as np
    from generate_x7_turning_se2 import CANDIDATES, blend_correction, compose, invert, resample_motion, transfer
    from run_x7_turning_carry import LAYOUTS, WXY, drive_segment, waypoints


@unittest.skipUnless(HAS_RUNTIME,'X7 runtime dependencies unavailable')
class TurningSE2Test(unittest.TestCase):
    def test_non_commutative_rigid_transforms_compose_and_invert(self):
        a,b=np.array([1.,2.,.3]),np.array([.2,.4,-.6])
        np.testing.assert_allclose(compose(invert(a),compose(a,b)),b)
        self.assertFalse(np.allclose(compose(a,b),compose(b,a)))

    def test_orientation_transfer_does_not_displace_a_fixed_pivot(self):
        source=np.array([1.5,.6,-.45]); target=np.array([1.5,.6,-.55]); d=compose(target,invert(source))
        for s in (0.,.3,1.):
            warped=compose(blend_correction(np.zeros(3),d,s),source)
            np.testing.assert_allclose(warped[:2],source[:2],atol=1e-12)
            np.testing.assert_allclose(warped[2],source[2]+s*(target[2]-source[2]),atol=1e-12)

    def test_resample_keeps_endpoint_and_limits(self):
        path=np.column_stack([np.linspace(0.,2.0,30),np.zeros(30),np.linspace(0.,.6,30)])
        ref=resample_motion(path)
        self.assertGreater(len(ref),30)
        np.testing.assert_allclose(ref[-1,:3],path[-1])
        np.testing.assert_array_equal(ref[-1,3:],np.zeros(3))
        self.assertTrue(np.all(abs(ref[:,3:])<=np.array(WXY)*.86+1e-10))

    def test_both_arms_copied_as_pair_and_heldout_pose_not_emitted(self):
        w=waypoints(LAYOUTS['source-ccw']); seg=[drive_segment(a,b) for (_,a),(_,b) in zip(w,w[1:])]
        edges=np.r_[3,3+np.cumsum([len(s) for s in seg])]; records=[]
        for (_,a),(_,b),i,j in zip(w,w[1:],edges[:-1],edges[1:]): records.append(dict(name='route',start_inclusive=int(i),end_exclusive=int(j)))
        armjaw=np.arange(16)/100
        end=np.tile(np.r_[w[-1][1],np.zeros(3)],(3,1))
        base=np.vstack([np.zeros((3,6)),*seg,end]); u=np.column_stack([base,np.tile(armjaw,(len(base),1))])
        p=np.r_[np.arange(3), np.full(edges[-1]-3,4),[5,6,9]]
        source=dict(action=u,phase=p,row={'waypoints':w,'route_segments':records})
        for candidate in CANDIDATES.values():
            action,phase,_,_=transfer(source,candidate)
            np.testing.assert_array_equal(action[:,6:],np.tile(armjaw,(len(action),1)))
            np.testing.assert_array_equal(action[:3],u[:3])
            self.assertNotEqual(candidate['goal'],LAYOUTS['heldout-angle']['goal'])
            self.assertEqual(np.count_nonzero(phase==6),1)


if __name__=='__main__':
    unittest.main()
