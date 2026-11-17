from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from finalize_official_raw_pool import COMPLETE,outcome


class FinalizeTests(unittest.TestCase):
    def test_five_episode_pool_needs_final_gate_not_live_log_successes(self):
        s={'stage':'official_shard_generation','target_unique_demos':5,'verified_unique_demos':0}
        self.assertEqual(outcome(s),'pending')
        s.update(stage=COMPLETE,active_pid=0,verified_unique_demos=5,source_grasping_mode='assisted',policy_training='not_run')
        self.assertEqual(outcome(s),'ready')
        s['active_pid']=999;self.assertEqual(outcome(s),'failed')

    def test_failed_dependency_is_not_bypassed(self):
        self.assertEqual(outcome({'stage':'generation_failed_or_interrupted'}),'failed')
        self.assertEqual(outcome({'stage':COMPLETE,'error':'final gate'}),'failed')


if __name__=='__main__':unittest.main()
