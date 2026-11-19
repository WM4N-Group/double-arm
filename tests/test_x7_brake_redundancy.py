"""Contracts for independently timed changes in modeled brake authority."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    from audit_x7_brake_redundancy import BACKUP_MAX_N,BRAKE_MAX_N,DERATED_N,limits,plan
    from audit_x7_brake_monitor import DELAY,feedback_valid,probe


@unittest.skipUnless(HAS,'Remote X7 dependencies unavailable')
class BrakeRedundancyTest(unittest.TestCase):
    def test_small_probe_cannot_bound_remaining_high_load_authority(self):
        for t in (320,328):
            actual=min(probe(t),DERATED_N)
            self.assertTrue(feedback_valid((t,actual),t+DELAY))
        self.assertLess(DERATED_N,BRAKE_MAX_N)

    def test_bounded_parallel_force_and_common_loss_fixture(self):
        self.assertEqual(len(set(plan())),36)
        self.assertEqual(limits('late-derated-single',91,90),(DERATED_N,0.))
        self.assertEqual(limits('late-derated-redundant',89,90),(BRAKE_MAX_N,BACKUP_MAX_N))
        self.assertEqual(limits('late-derated-redundant',90,90),(DERATED_N,BACKUP_MAX_N))
        self.assertEqual(limits('late-open-redundant',90,90),(0.,BACKUP_MAX_N))
        self.assertEqual(limits('late-common-loss',90,90),(0.,0.))
