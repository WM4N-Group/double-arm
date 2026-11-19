"""Offline contract checks for sensorless engagement scheduling."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    from audit_x7_passive_brake import CASES,STRONG,WEAK,passive_capacity,plan
    from sweep_x7_passive_delay import DELAYS


@unittest.skipUnless(HAS,'Remote X7 dependencies unavailable')
class PassiveBrakeTest(unittest.TestCase):
    def test_no_engagement_before_latch_or_delay(self):
        self.assertEqual(passive_capacity(900,None,CASES['clear-armed']),(0.,0.,0.))
        for delay,name in ((0,'lost-immediate'),(24,'lost-delay100ms'),(60,'lost-delay250ms')):
            self.assertEqual(passive_capacity(900+delay-1,900,CASES[name]),(0.,0.,0.))
            self.assertEqual(passive_capacity(900+delay,900,CASES[name]),STRONG)
        self.assertEqual(passive_capacity(900,900,CASES['lost-weak']),WEAK)

    def test_positive_negative_and_delayed_cases_are_separate(self):
        self.assertEqual(len(set(plan())),36)
        self.assertEqual(passive_capacity(1000,100,CASES['lost-no-passive']),(0.,0.,0.))
        self.assertEqual(list(DELAYS.values()),[(6,STRONG),(12,STRONG),(18,STRONG)])
