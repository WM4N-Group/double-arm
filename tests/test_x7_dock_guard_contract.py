"""State-machine contracts for the float-valued simulation distance input."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
AVAILABLE=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if AVAILABLE:
    import numpy as np
    from test_x7_dock_guard import DockGuard, GUARD_SCAN_M, STABLE_SUBSTEPS, RETREAT_SUBSTEPS, clearance_trigger
    from run_x7_turning_carry import WXY


@unittest.skipUnless(AVAILABLE,'X7 runtime dependencies unavailable')
class DockGuardContracts(unittest.TestCase):
    def test_trigger_moves_outward_with_approach_speed(self):
        self.assertGreater(clearance_trigger(np.array([0.,-.25,0.]),0.),clearance_trigger(np.zeros(3),0.))
        self.assertEqual(clearance_trigger(np.array([0.,.25,0.]),0.),clearance_trigger(np.zeros(3),0.))

    def test_scan_range_covers_worst_trigger_with_drive_speed_limits(self):
        vmax=np.linalg.norm(WXY[:2]);q=np.zeros((3,2));q[1,1]=-vmax
        self.assertLess(clearance_trigger(q[:,1],0.),GUARD_SCAN_M)
        a=np.zeros(22); a[0]=1.
        g=DockGuard(0.); cmd,_=g.update(0,q,[GUARD_SCAN_M,GUARD_SCAN_M],[0,0],a)
        self.assertEqual(g.state,'pass');np.testing.assert_array_equal(cmd,a)

    def test_one_dropout_latches_without_automatic_resume_or_retreat(self):
        g=DockGuard(0.); q=np.zeros((3,2)); u=np.zeros(22);u[-2:]=.006
        g.update(0,q,None,[0,0],u)
        opening=u.copy();opening[-2:]=.034
        for k in range(1,STABLE_SUBSTEPS+RETREAT_SUBSTEPS+1):
            a,_=g.update(k,q,[.08,.03],[0,0],opening)
            np.testing.assert_array_equal(a[-2:],u[-2:]);np.testing.assert_array_equal(a[:6],0.)
        self.assertEqual(g.state,'hold'); self.assertIsNone(g.retreat_origin)

    def test_B_hazard_brakes_then_retreats_along_estimated_station_normal(self):
        g=DockGuard(np.pi/2);q=np.zeros((3,2));u=np.zeros(22);u[-2:]=.006
        for k in range(STABLE_SUBSTEPS+2):
            a,_=g.update(k,q,[.08,.007],[0,0],u)
        self.assertEqual(g.state,'retreat');self.assertLess(a[0],0.);self.assertAlmostEqual(a[1],0.)
        np.testing.assert_array_equal(a[-2:],u[-2:])

    def test_retreat_cancels_and_holds_if_observation_unavailable(self):
        g=DockGuard(0.);q=np.zeros((3,2));u=np.zeros(22);u[-2:]=.006
        for k in range(STABLE_SUBSTEPS+2):g.update(k,q,[.08,.007],[0,0],u)
        self.assertEqual(g.state,'retreat');a,_=g.update(STABLE_SUBSTEPS+3,q,None,[0,0],u)
        self.assertEqual(g.state,'hold');np.testing.assert_array_equal(a[:6],0.)


if __name__=='__main__':
    unittest.main()
