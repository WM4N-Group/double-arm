from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_pi0_local_grasp import STEPS,replay_gate,review_label
from pi0_review_contract import physical_task_passed,training_split_physical_task_passed


class LocalGraspScopeTests(unittest.TestCase):
    def valid(self):return {'mode':'source_action_replay','steps':STEPS,'aborted':False,'local_cup_grasp_report_seen':True,'object_pose_write_count':0,'initial_state_load_count':1}
    def test_replay_gate_is_real_replay_completion_not_prediction_or_task_success(self):
        self.assertTrue(replay_gate(self.valid()))
        for key,value in [('mode','learned_pi0_local_grasp'),('steps',STEPS-1),('aborted',True),('local_cup_grasp_report_seen',False),('object_pose_write_count',1),('initial_state_load_count',2)]:
            x=self.valid();x[key]=value;self.assertFalse(replay_gate(x))
    def test_partial_assisted_result_cannot_be_full_physical_or_paper_success(self):
        x={'mode':'learned_pi0_local_grasp','local_cup_grasp_report_seen':True,'task_success':False,'privileged_initialization':True}
        self.assertFalse(physical_task_passed(x));self.assertFalse(training_split_physical_task_passed(x))
        self.assertIn('NOT FULL TASK OR PHYSICAL SUCCESS',review_label(x))
        x['local_cup_grasp_report_seen']=False;self.assertIn('FAILED LOCAL GRASP',review_label(x))
        self.assertIn('NOT LEARNED',review_label(self.valid()))


if __name__=='__main__':unittest.main()
