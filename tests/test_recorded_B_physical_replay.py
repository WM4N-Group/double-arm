from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_recorded_B_physical import completed_recorded_B


class RecordedTapeTests(unittest.TestCase):
    def test_completion_requires_actual_recorded3000_commands_not_exit0_or_expert1714(self):
        result={'stage':'recorded_B_replay_completed_NOT_live_policy','aborted':False,'recorded_B_actions_executed':3000,'no_assistance_all_checked_samples_confirmed':True}
        self.assertTrue(completed_recorded_B(0,result))
        for k,v in [('recorded_B_actions_executed',1714),('aborted',True),('no_assistance_all_checked_samples_confirmed',False),('stage','recorded_B_replay_aborted_NOT_live_policy')]:
            x=dict(result);x[k]=v;self.assertFalse(completed_recorded_B(0,x))
        self.assertFalse(completed_recorded_B(7,result))


if __name__=='__main__':unittest.main()
