"""Offline tests for calibrated X7 range-grid geometry and blind spots."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from audit_x7_chassis_range_proxy import MOUNT_PAD_M, RANGE_M, REAR_X, DENSE_REAR_X, gaps_from_fractions, odom_correction, transform_mounts


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class RangeProxyContracts(unittest.TestCase):
    def test_same_range_fraction_does_not_require_object_id(self):
        np.testing.assert_allclose(gaps_from_fractions([0.,.5,1.]),MOUNT_PAD_M+RANGE_M*np.array([0.,.5,1.]))
        with self.assertRaises(ValueError):gaps_from_fractions([np.nan])

    def test_rear_probe_rotation_uses_only_base_pose(self):
        mount=np.array([[.035,.25,.27]])
        origin,end=transform_mounts(mount,np.array([2.,3.,np.pi/2]),1)
        np.testing.assert_allclose(origin,[[1.75,3.035,.27]],atol=1e-12)
        np.testing.assert_allclose(end-origin,[[-RANGE_M,0.,0.]],atol=1e-12)
        self.assertGreaterEqual(odom_correction([0.,0.,0.],np.array([0.,.01,.01])),.01)

    def test_narrow_counterexample_not_sampled_by_rear_grid(self):
        self.assertFalse(np.any(abs(REAR_X-.035)<.006)); self.assertTrue(np.any(abs(REAR_X)<.12)); self.assertTrue(np.any(abs(DENSE_REAR_X-.035)<.006))


if __name__=='__main__':
    unittest.main()
