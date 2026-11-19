"""Pure contract tests for the delayed front-range stop."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    from audit_x7_transport_front_stop import MAX_AGE, packet_bound, panel, trip


@unittest.skipUnless(HAS,'X7 physics/torch dependencies unavailable')
class TransportFrontStopTest(unittest.TestCase):
    def test_missing_invalid_and_stale_frames_veto(self):
        for raw,age,d in ((None,17,0.),(float('nan'),16,0.),(.03,MAX_AGE+1,0.),(.03,16,-.01),(.03,-1,0.)):
            self.assertEqual(trip(raw,age,d,.1)[0],'unknown')

    def test_age_corrected_motion_and_braking_suppress_near_entry(self):
        self.assertIsNone(trip(.022,16,0.,0.)[0])
        self.assertEqual(trip(.022,16,.011,.11)[0],'near')
        self.assertLess(packet_bound(.022,17,.011),packet_bound(.022,16,0.))

    def test_unique_cases_and_fault_panels(self):
        rows=panel();self.assertEqual(len(rows),20);self.assertEqual(len(set(rows)),20)
        self.assertEqual(sum(s=='range-dropout' for _,s in rows),4)
