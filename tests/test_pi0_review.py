"""Pure-stdlib acceptance/fingerprint fixtures; no local simulation or assets."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi0_review_contract import fingerprint_checkpoint,physical_task_passed


class Pi0ReviewTests(unittest.TestCase):
    def test_complete_inference_tree_is_hashed_not_just_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'29999';checkpoint.mkdir()
            (checkpoint/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs':1}))
            for name in ('params','assets'):
                (checkpoint/name).mkdir();(checkpoint/name/'payload').write_bytes(b'fixture')
            (checkpoint/'train_state').mkdir();(checkpoint/'train_state/optimizer').write_bytes(b'not needed for inference')
            first=fingerprint_checkpoint(checkpoint)
            self.assertEqual(len(first['files']),3)
            self.assertEqual(first['step_index'],29999)
            (checkpoint/'params/payload').write_bytes(b'changed actual parameters')
            self.assertNotEqual(first['sha256'],fingerprint_checkpoint(checkpoint)['sha256'])
            (checkpoint/'assets/linked').symlink_to(checkpoint/'params/payload')
            with self.assertRaises(ValueError):fingerprint_checkpoint(checkpoint)

    def test_unfinished_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'29999';checkpoint.mkdir()
            (checkpoint/'_CHECKPOINT_METADATA').write_text(json.dumps({'commit_timestamp_nsecs':None}))
            with self.assertRaises(ValueError):fingerprint_checkpoint(checkpoint)

    def test_success_requires_physical_contact_movement_and_zero_pose_writes(self):
        result={'mode':'closed_loop_neural_policy','algorithm':'official_openpi_pi0_lora32','task_success':True,
            'privileged_state':False,'planner_enabled':False,'teacher_action_reads':0,'initial_state_load_count':1,
            'grasping_mode':'physical','assisted_constraint_seen':False,'object_pose_write_count':0,
            'episodes':[{'initial_state_load_count':1,'split':'validation','sustained_success':True,'stable_success_frames':30,
                'cup_specific_grasp_seen':True,'max_cup_lift_m':.04,'base_path_length_m':.6,'object_pose_write_count':0}]}
        self.assertTrue(physical_task_passed(result))
        for field,value in [('task_success',False),('assisted_constraint_seen',True),('object_pose_write_count',1),('teacher_action_reads',1)]:
            changed=copy.deepcopy(result);changed[field]=value;self.assertFalse(physical_task_passed(changed))
        for field,value in [('cup_specific_grasp_seen',False),('max_cup_lift_m',.02),('base_path_length_m',.4),('initial_state_load_count',2),('stable_success_frames',29)]:
            changed=copy.deepcopy(result);changed['episodes'][0][field]=value;self.assertFalse(physical_task_passed(changed))
