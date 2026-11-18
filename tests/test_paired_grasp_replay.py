from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Numerical fixtures run on the server')
class PairedGraspReplayTest(unittest.TestCase):
    def fixtures(self):
        from compare_paired_grasp_replay import BOUNDARY_FIELDS
        a = np.zeros((4,21), np.float32)
        pose = np.array([0.,0.,0.,0.,0.,0.,1.])
        r = dict(completed_actions=4, aborted=False, initial_state_load_count=1, reset_physics_steps=20,
                 explicit_object_pose_writes=0, policy_calls=0, recorded_boundary_fields=list(BOUNDARY_FIELDS))
        t = dict(actions=a.copy(), initial_proprio=np.zeros(21), initial_root_pose=pose.copy(),
                 initial_cup_pose=pose.copy(), initial_left_eef_pose=pose.copy(),
                 post_step_cup_poses=np.tile(pose,(4,1)), post_step_left_eef_poses=np.tile(pose,(4,1)),
                 post_step_left_cup_contacts=np.array([0,2,2,1]), grasp_boundaries=np.zeros((29,8), np.int8))
        assisted = {k:v.copy() for k,v in t.items()}
        assisted['grasp_boundaries'][24:, [0,2,4,6]] = 1
        assisted['post_step_cup_poses'][1:, 2] = .02
        return a, dict(r,grasping_mode='assisted'), assisted, dict(r,grasping_mode='physical'), t

    def test_post_boundary_index_and_equivalent_quaternion(self):
        from compare_paired_grasp_replay import compare
        a, ar, at, pr, pt = self.fixtures()
        pt['initial_cup_pose'][3:] *= -1
        d = compare(a, ar, at, pr, pt)
        self.assertEqual(d['first_assisted_left_cup_joint_observed_post_step'],1)
        self.assertEqual(d['first_assisted_left_cup_joint_observed_boundary'], {'step': 1, 'phase': 'post'})
        self.assertEqual(d['pre_attachment_max_cup_separation_m'],0.)
        self.assertAlmostEqual(d['max_cup_separation_m'],.02)

    def test_reject_assistance_unknown_and_misaligned_initial_state(self):
        from compare_paired_grasp_replay import compare
        for key, index, value in [('grasp_boundaries',(23,6),1),('grasp_boundaries',(22,0),-1),
                                  ('initial_cup_pose',0,.002),('actions',(0,0),1),('actions',(0,0),-0.0),('post_step_left_cup_contacts',2,-1)]:
            with self.subTest(key=key, index=index):
                a, ar, at, pr, pt = self.fixtures(); pt[key][index]=value
                with self.assertRaises(ValueError):
                    compare(a, ar, at, pr, pt)

    def test_report_no_attachment_as_no_event(self):
        from compare_paired_grasp_replay import compare
        a, ar, at, pr, pt = self.fixtures()
        at['grasp_boundaries'].fill(0)
        d = compare(a,ar,at,pr,pt)
        self.assertIsNone(d['first_assisted_left_cup_joint_observed_post_step'])
        self.assertIsNone(d['onset_window'])
