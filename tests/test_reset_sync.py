import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from reset_sync_contract import select_reset_sync_candidate


class ResetSyncTests(unittest.TestCase):
    def fixture(self):
        def shot(step,mae):return {'physics_steps':step,'rgb_mae_0_255':dict.fromkeys(('head_image','left_wrist_image','right_wrist_image'),mae),
            'root_translation_m':.001,'root_yaw_abs_difference_rad':.001,'joint_max_abs_difference_rad':.002,'xy_rotation_determinant':.999}
        return {'mode':'reset_sync_image_audit_only','grasping_mode':'physical','initial_state_load_count':1,'teacher_action_reads':0,
            'simulator_control_calls':0,'inference_calls':0,'assisted_constraint_seen':False,'object_pose_write_count':0,'series_completed':True,
            'snapshots':[shot(0,90.),shot(1,2.),shot(20,3.)]}
    def test_first_aligned_snapshot_not_task_success(self):
        self.assertEqual(select_reset_sync_candidate(self.fixture()),1)
    def test_untrusted_or_incomplete_audit_fails_closed(self):
        for key,value in [('grasping_mode','assisted'),('series_completed',False),('object_pose_write_count',1),('assisted_constraint_seen',True),('teacher_action_reads',1),('inference_calls',1)]:
            r=self.fixture();r[key]=value;self.assertIsNone(select_reset_sync_candidate(r))
    def test_all_cameras_and_robot_alignment_required(self):
        for field,value in [('root_translation_m',.1),('root_yaw_abs_difference_rad',.2),('joint_max_abs_difference_rad',.2),('xy_rotation_determinant',.97)]:
            r=self.fixture();r['snapshots']=r['snapshots'][1:2];r['snapshots'][0][field]=value;self.assertIsNone(select_reset_sync_candidate(r))
        for rgb in ({'head_image':1.},dict.fromkeys(('head_image','left_wrist_image','right_wrist_image'),float('nan'))):
            r=self.fixture();r['snapshots']=r['snapshots'][1:2];r['snapshots'][0]['rgb_mae_0_255']=rgb;self.assertIsNone(select_reset_sync_candidate(r))
