from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_pi0_relative_state_stem import exposure_counts
from pi0_relative_full_profile import full_schedule
from pi0_single_demo_fit import transition_indices


class ExposureAuditTests(unittest.TestCase):
    def test_actual_schedule_not_20000_full_demo_repetitions(self):
        r=exposure_counts(full_schedule(1714,636),1714,transition_indices(1714,636))
        self.assertEqual(r['optimizer_updates'],20000)
        self.assertEqual(r['unique_frames'],1714)
        self.assertEqual(r['ordinary_frame_visit_min_max'],[5,6])
        self.assertEqual(sum(r['visits_by_frame'].values()),20000)
        self.assertGreater(r['transition_frame_visit_min_max'][0],1100)

    def test_partial_coverage_rejected(self):
        with self.assertRaises(ValueError):exposure_counts([0,0,1],3,[1])
        with self.assertRaises(ValueError):exposure_counts([0,1,2],3,[4])


if __name__=='__main__':unittest.main()
