from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Numerical fixtures run on server')
class ExpertReleaseScoreTests(unittest.TestCase):
    def fixture(self, path, *, opens=True, inside=True, aid=False):
        from compare_paired_grasp_replay import BOUNDARY_FIELDS
        d=path/'physical'; d.mkdir(); n=34
        actions=np.zeros((n,21),np.float32); actions[:,13]=-1; actions[3:,13]=1 if opens else -1
        pose=np.array([0.,0.,0.,0.,0.,0.,1.]); cups=np.tile(pose,(n,1)); cups[1:3,2]=.1
        contact=np.zeros(n,np.int8); contact[1:3]=2; grip=contact>0
        widths=np.zeros(n,np.float32); widths[3:]=.07 if opens else .0
        flags=np.zeros(n,bool); flags[3:]=inside; boundaries=np.zeros((21+n*2,8),np.int8)
        if aid:boundaries[22,[0,2,6]]=1
        np.savez(d/'trace.npz',actions=actions,initial_proprio=np.zeros(21),initial_root_pose=pose,initial_cup_pose=pose,
                 initial_left_eef_pose=pose,post_step_cup_poses=cups,post_step_left_eef_poses=np.tile(pose,(n,1)),
                 post_step_left_cup_contacts=contact,grasp_boundaries=boundaries,post_step_cup_grasp_flags=grip,
                 inside_goal_flags=flags,post_step_left_gripper_width=widths)
        r=dict(kind='expert_tape_physical_not_live_policy',stage='replay_complete_NOT_live_policy',grasping_mode='physical',
               completed_actions=n,aborted=False,initial_state_load_count=1,reset_physics_steps=20,explicit_object_pose_writes=0,
               policy_calls=0,recorded_boundary_fields=list(BOUNDARY_FIELDS),
               runtime_motor_contract={'gripper_left':{'mode':'binary','inverted':False,'command_input_limits':[0,1],'command_output_limits':[-1,1]}})
        (d/'result.json').write_text(json.dumps(r)); return actions

    def test_release_requires_open_width_goal_and_no_constraint(self):
        from run_expert_physical_pilot import score
        for kw,expected in [({},'completed_release_stable_inside'),({'opens':False},'held_lift_no_verified_release_inside'),
                            ({'inside':False},'held_lift_no_verified_release_inside'),({'aid':True},None)]:
            with self.subTest(kw=kw), tempfile.TemporaryDirectory() as temp:
                p=Path(temp); a=self.fixture(p,**kw)
                if expected is None:
                    with self.assertRaises(ValueError):score(p,a)
                else:self.assertEqual(score(p,a)['outcome'],expected)
