"""Offline checks of the X7 anchored hybrid projections."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
AVAILABLE=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if AVAILABLE:
    import numpy as np
    from train_x7_anchored_turning_bc import constrain_arms, constrain_dock
    from train_x7_turning_trajectory_bc import BLOCK_PHASES, DEV
    from run_x7_turning_carry import dock_pose


@unittest.skipUnless(AVAILABLE,'X7 runtime dependencies unavailable')
class AnchoredContracts(unittest.TestCase):
    def test_terminal_station_projection_updates_se2_only_and_only_after_dock_approach(self):
        l=DEV['dev-cw']; n=[20]*14; targets=np.zeros((sum(n),22)); expect=dock_pose(l)
        targets[:,:3]=expect+[.020,-.007,.02]; targets[:,6:]=.03
        got,delta=constrain_dock(targets,n,l)
        np.testing.assert_allclose(got[-1,:3],expect,rtol=0,atol=1e-12)
        np.testing.assert_allclose(delta,[-.02,.007,-.02],rtol=0,atol=1e-12)
        np.testing.assert_array_equal(targets[:sum(n[:8])],got[:sum(n[:8])])
        np.testing.assert_array_equal(targets[:,6:],got[:,6:])

    def test_refuses_large_terminal_error(self):
        with self.assertRaisesRegex(ValueError,'beyond'):
            constrain_dock(np.zeros((280,22)),[20]*14,DEV['dev-cw'])

    def test_paired_prior_overrides_contact_not_final_settle_or_base_or_jaws(self):
        ph=np.repeat(BLOCK_PHASES,20); a=np.zeros((len(ph),22)); a[:,-2:]=.03
        b,speed=constrain_arms(a,ph,np.full((10,14),.08),np.arange(10))
        self.assertLess(speed,1.25)
        np.testing.assert_array_equal(b[:,np.r_[:6,20:22]],a[:,np.r_[:6,20:22]])
        for stage in range(1,8): np.testing.assert_array_equal(b[ph==stage,6:20],.08)
        np.testing.assert_array_equal(b[ph==9],a[ph==9])
        np.testing.assert_array_equal(b[0],a[0])


if __name__=='__main__':
    unittest.main()
