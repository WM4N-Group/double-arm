"""Evaluation-only trace fixtures; no local robotics simulation."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


class MotionAuditIsolation(unittest.TestCase):
    def test_object_poses_are_only_for_audits_and_not_new_acceptance_criteria(self):
        source=(ROOT/'scripts/evaluate_pi0_r1.py').read_text()
        self.assertIn('episode.update(cup_motion_metrics(',source)
        self.assertIn("inputs.update(state=as_numpy(obs['prop_state']).copy(),prompt=PROMPT)",source)
        gate=(ROOT/'scripts/pi0_review_contract.py').read_text()
        self.assertNotIn('grasp_supported_base_path_m',gate)


@unittest.skipUnless(importlib.util.find_spec('numpy'),'Numerical checks run in remote environment')
class MotionAuditNumerics(unittest.TestCase):
    def test_transport_requires_consecutive_contact_and_lift(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from rollout_motion_metrics import cup_motion_metrics
        result=cup_motion_metrics([[0,0,.04],[.2,0,.04],[.4,0,.04],[.6,0,.04]],
            [[0,0,0],[.2,0,0],[.4,0,0],[.6,0,0]],[True,True,False,True],[0,0,0])
        self.assertAlmostEqual(result['grasp_supported_base_path_m'],.2)
        self.assertEqual(result['cup_grasp_report_and_lift_frames'],3)
        no_grasp=cup_motion_metrics([[0,0,.04],[1,0,.04]],[[0,0,0],[1,0,0]],[False,False],[0,0,0])
        self.assertEqual(no_grasp['grasp_supported_base_path_m'],0)
        no_lift=cup_motion_metrics([[0,0,.02],[1,0,.02]],[[0,0,0],[1,0,0]],[True,True],[0,0,0])
        self.assertEqual(no_lift['grasp_supported_base_path_m'],0)

    def test_first_step_cup_motion_and_shape_failures_are_visible(self):
        from rollout_motion_metrics import cup_motion_metrics
        result=cup_motion_metrics([[.4,0,0]],[[0,0,0]],[False],[0,0,0])
        self.assertAlmostEqual(result['max_cup_step_displacement_m'],.4)
        self.assertEqual(result['grasp_supported_base_path_m'],0)
        with self.assertRaises(ValueError):cup_motion_metrics([[0,0,0]],[],[False],[0,0,0])
