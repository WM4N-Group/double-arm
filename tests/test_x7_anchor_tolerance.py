"""Pose-bias and full-support unit contracts; no physics is launched."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
AVAILABLE=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if AVAILABLE:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from audit_x7_anchor_tolerance import CASES, LAYOUTS, check_footprint, perceived_pose
    from run_x7_turning_carry import PARK_Y, SUPPORT_Z, dock_pose, r2


@unittest.skipUnless(AVAILABLE,'X7 runtime dependencies unavailable')
class AnchorToleranceContracts(unittest.TestCase):
    def test_target_normal_rotates_with_the_true_fixture(self):
        l=LAYOUTS['north']; obs=perceived_pose(l,*CASES['dock-in-10mm']); q=dock_pose(l)
        np.testing.assert_allclose(r2(q[2]).T@(dock_pose(obs)[:2]-q[:2]),[0.,-.01],atol=1e-12)
        self.assertEqual(l['tote'],obs['tote']); self.assertEqual(l['goal'][2],obs['goal'][2])

    def test_pickup_bias_never_mutates_true_geometry(self):
        l=LAYOUTS['south']; old=l.copy(); obs=perceived_pose(l,*CASES['combined'])
        self.assertEqual(l,old); self.assertNotEqual(obs,l)
        np.testing.assert_allclose(np.array(obs['tote'])-l['tote'],[.005,0.,1.],atol=1e-12)

    def synthetic(self,l,local_y_shift=0.):
        dock=dock_pose(l); n=32; dx,dy,angle=l['tote']; xy=dock[:2]+r2(dock[2])@[dx,PARK_Y+dy+local_y_shift]
        state=np.zeros((n,13));state[:,:3]=[*xy,SUPPORT_Z]
        state[:,3:7]=Rotation.from_euler('z',l['goal'][2]+angle,degrees=True).as_quat()
        jaws=np.zeros((n,4,2));jaws[:,:,0]=.034
        return {'phase':np.full(n,9),'object_post':state,'on_table':np.tile([0,1],(n,1)),'gripper_post':jaws}

    def test_support_needs_the_bottom_not_only_center_or_a_contact_flag(self):
        l=LAYOUTS['south']; inside=check_footprint(self.synthetic(l),l); outside=check_footprint(self.synthetic(l,.035),l)
        self.assertTrue(inside['full_bottom_supported']); self.assertFalse(outside['full_bottom_supported'])
        self.assertGreater(inside['bottom_support_margin_m'],.001); self.assertLess(outside['bottom_support_margin_m'],0.)


if __name__=='__main__':
    unittest.main()
