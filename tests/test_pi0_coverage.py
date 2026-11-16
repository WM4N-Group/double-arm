from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_pi0_coverage_pipeline import source_replay_passed


class CoverageGateTests(unittest.TestCase):
    def test_teacher_replay_is_only_an_executability_gate_not_policy_success(self):
        good={'mode':'recorded_action_replay_audit','learned_policy':False,'initial_state_load_count':1,
              'num_recorded_actions':1714,'completed_actions':1714,'teacher_action_reads':1714,
              'model_inference_calls':0,'goal_success':True,'grasped_ever':[True,False],'policy_task_success':False}
        self.assertTrue(source_replay_passed(good));self.assertFalse(good['policy_task_success'])
        for key,value in [('learned_policy',True),('completed_actions',1713),('goal_success',False),
                          ('grasped_ever',[False,False]),('initial_state_load_count',2),('error','failure')]:
            with self.subTest(key=key):self.assertFalse(source_replay_passed({**good,key:value}))
