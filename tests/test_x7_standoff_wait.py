"""Pure standoff contracts: reserve not cleared by an ordinary valid packet."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    from audit_x7_standoff_wait import RESERVE_M, decide_standoff, plan, spec


@unittest.skipUnless(HAS,'X7 dependencies unavailable')
class StandoffTest(unittest.TestCase):
    def test_reserve_uses_distance_not_fixture(self):
        self.assertEqual(decide_standoff(None,16,0.,0.)[0],'unknown')
        self.assertEqual(decide_standoff(.10,16,0.,0.)[0],'standoff_clearance')
        self.assertIsNone(decide_standoff(.319,16,0.,0.)[0])
        self.assertAlmostEqual(decide_standoff(.25,16,0.,0.)[2],RESERVE_M+.006)

    def test_fixture_clocks_cannot_follow_stop_time(self):
        self.assertEqual(len(set(plan())),36)
        starts=[]
        for case in ('push-early','push-staging','push-late'):
            params,k,d=spec(case,900,1020,1200);starts.append(k-44);self.assertEqual(params['force_n'],620.);self.assertFalse(d)
        self.assertEqual(starts,[996,1056,1156]);self.assertTrue(spec('push-with-missing-range',900,1020,1200)[2])
