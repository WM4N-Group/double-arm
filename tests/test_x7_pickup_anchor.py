"""Test shared SE(2) pickup anchors without launching a simulator."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS_RUNTIME=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS_RUNTIME:
    import numpy as np
    from scipy.spatial.transform import Rotation
    from generate_x7_pickup_anchor import PILOT, object_anchor
    from run_x7_turning_carry import LAYOUTS, r2, PARK_Y, HANDLE_HALF_SPAN


@unittest.skipUnless(HAS_RUNTIME,'X7 runtime dependencies unavailable')
class PickupAnchorContract(unittest.TestCase):
    def test_translation_and_rotation_applied_about_tote_center(self):
        xyz,q=object_anchor((.008,-.012,1.8),[0.,PARK_Y,1.4],Rotation.identity())
        np.testing.assert_allclose(xyz,[.008,PARK_Y-.012,1.4],atol=1e-12)
        np.testing.assert_allclose(q.as_euler('zyx',degrees=True),[1.8,0,0],atol=1e-12)

    def test_one_common_rigid_delta_preserves_two_arm_relative_geometry(self):
        a=np.array([-HANDLE_HALF_SPAN,PARK_Y,1.4]); b=np.array([HANDLE_HALF_SPAN,PARK_Y,1.4]); r=Rotation.identity()
        for l in PILOT.values():
            a1,r1=object_anchor(l['tote'],a,r); b1,r2_=object_anchor(l['tote'],b,r)
            np.testing.assert_allclose(r1.inv().apply(b1-a1),b-a,atol=1e-12)
            np.testing.assert_allclose((r1.inv()*r2_).as_rotvec(),[0,0,0],atol=1e-12)
            self.assertGreater(np.linalg.norm(a1[:2]-a[:2]),.005)
            self.assertNotEqual(l['goal'],LAYOUTS['heldout-angle']['goal'])

    def test_wrist_orientation_rotates_in_chassis_not_in_wrist_local_frame(self):
        q=Rotation.from_euler('xyz',[1.2,.3,-.2]); _,v=object_anchor((0.,0.,10.),[0.,0.,0.],q)
        np.testing.assert_allclose(v.as_matrix()[:2],r2(np.deg2rad(10.))@q.as_matrix()[:2],atol=1e-12)
        np.testing.assert_allclose(v.as_matrix()[2],q.as_matrix()[2],atol=1e-12)


if __name__=='__main__':
    unittest.main()
