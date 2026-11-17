from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_pi0_grasp_retention import first_persistent_contact_loss


class RetentionTimelineTests(unittest.TestCase):
    def test_initial_noncontact_is_not_loss_and_brief_dropout_is_not_persistent(self):
        values=[False]*20+[True]*20+[False]*2+[True]*2+[False]*15
        self.assertEqual(first_persistent_contact_loss(values,20),44)
        self.assertIsNone(first_persistent_contact_loss([True]*20,0))
    def test_interval_must_remain_long_enough_after_grasp_start(self):
        self.assertIsNone(first_persistent_contact_loss([False]*20,15,10))
        self.assertEqual(first_persistent_contact_loss([False]*30,15,10),15)


if __name__=='__main__':unittest.main()
