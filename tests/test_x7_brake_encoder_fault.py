"""Watchdog contracts for the modeled separate brake sensor."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    import numpy as np
    from audit_x7_brake_encoder_fault import LAG,MAX_AGE,plan,take_packet


@unittest.skipUnless(HAS,'Remote X7 dependencies unavailable')
class BrakeEncoderFaultTest(unittest.TestCase):
    def test_delayed_packet_contract_not_current_truth_fallback(self):
        q=np.array([[1.,.3],[2.,.01]])
        np.testing.assert_array_equal(take_packet((80,q),80+LAG),q)
        self.assertIsNone(take_packet((80,q),80+LAG-1))
        self.assertIsNone(take_packet((80,q),80+MAX_AGE+1))
        self.assertIsNone(take_packet((80,None),80+LAG))
        self.assertIsNone(take_packet((80,np.ones((3,2))),80+LAG))
        self.assertIsNone(take_packet((80,np.full((2,2),np.nan)),80+LAG))

    def test_full_pair_manifest(self):
        p=plan();self.assertEqual(len(set(p)),30)
        self.assertEqual(sum(c=='push-frozen-backup-drop' for _,c in p),6)
