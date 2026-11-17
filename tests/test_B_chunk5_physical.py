from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_B_chunk5_physical import actual_complete


class Chunk5CompletionTests(unittest.TestCase):
    def good(self):return {'episodes':[{'steps':3000,'aborted':False}],'execute_steps':5,'grasping_mode':'physical','evaluation_split':'train','teacher_action_reads':0,'initial_state_load_count':1,'no_assistance_all_checked_samples_confirmed':True,'no_assistance_checked_samples':6021}
    def test_actual_unaborted_train_conditions_required_not_placeholder_or_goalflag(self):
        self.assertTrue(actual_complete(self.good()))
        for k,v in [('episodes',[]),('execute_steps',10),('grasping_mode','assisted'),('evaluation_split','validation'),('teacher_action_reads',1),('no_assistance_checked_samples',6020),('no_assistance_all_checked_samples_confirmed',False)]:
            r=self.good();r[k]=v;self.assertFalse(actual_complete(r))
        r=self.good();r['episodes'][0]['aborted']=True;self.assertFalse(actual_complete(r))
    def test_short_episode_has_actual_matching_assistance_samples_and_is_not_assumed_task_success(self):
        r=self.good();r['episodes'][0]['steps']=1500;r['no_assistance_checked_samples']=3021
        self.assertTrue(actual_complete(r));self.assertNotIn('task_success',r)


if __name__=='__main__':unittest.main()
