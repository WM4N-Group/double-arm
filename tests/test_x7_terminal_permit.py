"""Pure contracts for the X7 modeled brake proof."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    import numpy as np
    from audit_x7_terminal_permit import PAUSE, PROOF, insert_proof, permission, plan


@unittest.skipUnless(HAS,'Remote X7 dependencies unavailable')
class TerminalPermitTest(unittest.TestCase):
    def test_proof_requirements_fail_closed(self):
        self.assertTrue(permission(.004,.001,.28,8*PROOF))
        for a in ((.014,.001,.28,8*PROOF),(.004,.008,.28,8*PROOF),(.004,.001,.12,8*PROOF),
                  (.004,.001,.28,8*PROOF-1),(.004,.001,float('nan'),8*PROOF)):
            self.assertFalse(permission(*a))

    def test_proof_pause_does_not_change_endpoint_or_release_schedule_relative_to_route(self):
        u=np.arange(10*22).reshape(10,22).astype(float);ph=np.r_[0,1,2,3,4,4,4,4,5,6]
        v,p=insert_proof(u,ph,6)
        np.testing.assert_array_equal(v[:6],u[:6]);np.testing.assert_array_equal(v[6+PAUSE:],u[6:]);self.assertTrue(np.all(v[6:6+PAUSE,3:6]==0.))
        np.testing.assert_array_equal(p[6+PAUSE:],ph[6:]);self.assertEqual(len(set(plan())),20)
