from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_source0_physical import dependency_released,replay_completed,valid_freeze_sentinel_repair


class PhysicalSourceDependencyTests(unittest.TestCase):
    def good(self):return {'stage':'train-physical-diagnostic_FAILED','active_pid':0,'training':'completed'}
    def test_empty_gpu_between_training_and_eval_is_not_release(self):
        self.assertTrue(dependency_released(self.good(),False))
        self.assertFalse(dependency_released(self.good(),True))
        for field,value in [('stage','train-physical-diagnostic'),('active_pid',42),('training','in_progress')]:
            s=self.good();s[field]=value;self.assertFalse(dependency_released(s,False))
    def test_exit0_cannot_hide_aborted_or_empty_actual_replay(self):
        result={'stage':'source_replay_aborted_NOT_learned_policy','aborted':True,'teacher_source_actions_executed':0}
        self.assertFalse(replay_completed(0,result))
        result={'stage':'source_replay_completed_NOT_learned_policy','aborted':False,'teacher_source_actions_executed':1714,'no_assistance_all_checked_samples_confirmed':True}
        self.assertTrue(replay_completed(0,result));self.assertFalse(replay_completed(7,result))
        result['error']='hidden';self.assertFalse(replay_completed(0,result))
    def test_repair_only_known_zero_command_freeze_audit_abort(self):
        result={'stage':'source_replay_aborted_NOT_learned_policy','aborted':True,'teacher_source_actions_executed':0,'reset_physics_steps':0,'no_assistance_checked_samples':1,'error':"ValueError('Physical/AG Python/USD absence not confirmed at initial 0')"}
        rejected={'grasping_mode':'physical','errors':["_ag_freeze_gripper."+a+": TypeError: Expected recorded Boolean, not an unknown truthy object" for a in ('left','right')], 'gripper_freeze':{'left':-1,'right':-1},**{f:{'left':0,'right':0} for f in ('constraint_active','object_in_hand','usd_ag_constraint_present')}}
        self.assertTrue(valid_freeze_sentinel_repair(result,rejected))
        rejected['usd_ag_constraint_present']['left']=1;self.assertFalse(valid_freeze_sentinel_repair(result,rejected))

    def test_assisted_pass_alone_never_releases_B_chain(self):
        self.assertFalse(dependency_released({'stage':'train-author-diagnostic','active_pid':0,'training':'completed'},False))


if __name__=='__main__':unittest.main()
