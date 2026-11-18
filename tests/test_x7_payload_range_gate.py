"""Pure geometry and unknown-packet contracts; no physics is started."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS_RUNTIME=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS_RUNTIME:
    import numpy as np
    from audit_x7_payload_range_gate import MODEL_ERROR_M, MOUNT_PAD_M, PITCH_M, merged_rear, payload_mounts
    from run_x7_turning_carry import PARK_Y,r2


@unittest.skipUnless(HAS_RUNTIME,'X7 runtime dependencies unavailable')
class PayloadRangeGateTests(unittest.TestCase):
    def test_fusion_never_treats_an_unknown_region_as_clear(self):
        self.assertEqual(merged_rear(.32,None,0.),-1.)
        self.assertEqual(merged_rear(-1.,.32,0.),-1.)

    def test_payload_error_margin_does_not_relax_fixed_body_measurement(self):
        self.assertAlmostEqual(merged_rear(.03,.01,.001),.011-MODEL_ERROR_M)
        self.assertEqual(merged_rear(.025,.12,.001),.025)

    def test_mounts_move_with_calibration_and_cover_both_projected_axes(self):
        raw=payload_mounts((0,0,0));dx,dy,yaw=.005,-.01,np.pi/6
        moved=payload_mounts((dx,dy,30));local=(moved[:,:2]-[dx,PARK_Y+dy])@r2(yaw)
        np.testing.assert_allclose(local[:,0],raw[:,0],atol=1.e-12)
        np.testing.assert_allclose(local[:,1],.0955+MOUNT_PAD_M,atol=1.e-12)
        np.testing.assert_array_equal(moved[:,2],raw[:,2]);self.assertEqual(len(raw),1638)
        self.assertLessEqual(max(np.diff(np.unique(raw[:,0])).max(),np.diff(np.unique(raw[:,2])).max()),PITCH_M+1.e-12)


if __name__=='__main__':unittest.main()
