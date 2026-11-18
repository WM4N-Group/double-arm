"""Offline checks for the restricted retirement/retreat envelope."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from audit_x7_certified_retreat import DURATION_STEPS, EnvelopeGuard, REAR_OBSTACLE_SPEED_BOUND, RETREAT_M, correct_moving_rear


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class RetreatEnvelopeContract(unittest.TestCase):
    def make(self):
        g=EnvelopeGuard(0.);g.state='retreat';g.retreat_origin=np.zeros(3);g.hold=np.zeros(3);g.held_cmd=np.r_[np.zeros(20),.006,.006]
        u=np.r_[np.zeros(20),.034,.034]
        return g,u

    def test_rear_sample_age_penalty_does_not_change_front(self):
        l=np.array([.08,.012,.017]);c=correct_moving_rear(l,24)
        np.testing.assert_array_equal(l[:2],c[:2]);self.assertAlmostEqual(c[2],l[2]-.1*REAR_OBSTACLE_SPEED_BOUND)

    def test_noisy_but_still_qualified_front_bound_allows_full_retreat(self):
        g,u=self.make()
        for i in range(DURATION_STEPS):
            a,_=g.update(i,np.zeros((3,2)),[.08,.011 if i%2 else .020],[0,0],u)
            np.testing.assert_array_equal(a[-2:],g.held_cmd[-2:])
        self.assertEqual(g.state,'hold');self.assertAlmostEqual(a[1],RETREAT_M)

    def test_missing_or_insufficient_front_bound_holds_at_measured_state(self):
        for obs in (None,[.08,.0052]):
            g,u=self.make();q=np.zeros((3,2));q[1,0]=.005
            a,_=g.update(14,q,obs,[0,0],u)
            self.assertEqual(g.state,'hold');np.testing.assert_allclose(a[:3],q[:,0]);np.testing.assert_allclose(a[3:6],0.)
            np.testing.assert_allclose(a[-2:],g.held_cmd[-2:]);self.assertEqual(g.s,0)


if __name__=='__main__':
    unittest.main()
