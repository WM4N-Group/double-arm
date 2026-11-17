from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physical_grasp_observation import snapshot_no_assistance,no_assistance_confirmed


class NoAssistanceEvidenceTests(unittest.TestCase):
    def robot(self):
        return SimpleNamespace(grasping_mode='physical',_ag_obj_constraints={'left':None,'right':None},
            _ag_obj_in_hand={'left':None,'right':None},_ag_freeze_gripper={'left':False,'right':False},
            eef_links={a:SimpleNamespace(prim_path='/robot/'+a) for a in ('left','right')})
    def test_explicit_physical_and_actual_empty_usd_is_only_no_assistance_not_grasp_success(self):
        r=self.robot();paths=[]
        report=snapshot_no_assistance(r,lambda p:paths.append(p) or False)
        self.assertTrue(no_assistance_confirmed(report))
        self.assertEqual(paths,['/robot/left/ag_constraint','/robot/right/ag_constraint'])
        self.assertNotIn('grasp_success',report);self.assertNotIn('task_success',report)
        self.assertIsNone(r._ag_obj_constraints['left'])
    def test_physical_label_cannot_hide_a_live_usd_constraint(self):
        report=snapshot_no_assistance(self.robot(),lambda p:p.endswith('left/ag_constraint'))
        self.assertFalse(no_assistance_confirmed(report));self.assertEqual(report['usd_ag_constraint_present']['left'],1)
    def test_missing_or_failed_engine_query_stays_unknown_and_rejects(self):
        report=snapshot_no_assistance(self.robot())
        self.assertFalse(no_assistance_confirmed(report));self.assertEqual(report['usd_ag_constraint_present']['left'],-1)
        report=snapshot_no_assistance(self.robot(),lambda p:None)
        self.assertFalse(no_assistance_confirmed(report));self.assertEqual(report['usd_ag_constraint_present']['right'],-1)
    def test_assisted_object_freeze_or_mode_reject_without_engine_object_truth_test(self):
        class EngineObject:
            def __bool__(self):raise AssertionError('Never truth-test a simulator object')
        for attribute,value in (('_ag_obj_constraints',EngineObject()),('_ag_obj_in_hand',EngineObject()),('_ag_freeze_gripper',True)):
            robot=self.robot();getattr(robot,attribute)['left']=value
            report=snapshot_no_assistance(robot,lambda p:False);self.assertFalse(no_assistance_confirmed(report))
        robot=self.robot();robot.grasping_mode='assisted'
        self.assertFalse(no_assistance_confirmed(snapshot_no_assistance(robot,lambda p:False)))
    def test_missing_error_status_is_not_a_complete_observation(self):
        report=snapshot_no_assistance(self.robot(),lambda p:False)
        del report['errors'];self.assertFalse(no_assistance_confirmed(report))
        report['errors']=None;self.assertFalse(no_assistance_confirmed(report))

    def test_none_stays_unknown_without_exact_pinned_source_but_known_sentinel_preserves_raw(self):
        import tempfile,hashlib
        from unittest import mock
        robot=self.robot();robot._ag_freeze_gripper={'left':None,'right':None}
        report=snapshot_no_assistance(robot,lambda p:False)
        self.assertFalse(no_assistance_confirmed(report));self.assertEqual(report['gripper_freeze']['left'],-1)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manipulation_robot.py';path.write_text('audited fixture source')
            self.assertFalse(no_assistance_confirmed(snapshot_no_assistance(robot,lambda p:False,path)))
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            with mock.patch('physical_grasp_observation.PINNED_MANIPULATION_SHA256',digest):
                report=snapshot_no_assistance(robot,lambda p:False,path)
                self.assertTrue(no_assistance_confirmed(report))
                self.assertIsNone(report['gripper_freeze_raw']['left'])
                self.assertIsNone(robot._ag_freeze_gripper['left'])
                self.assertEqual(report['freeze_none_inactive_source_sha256'],digest)
                path.write_text('changed')
                self.assertFalse(no_assistance_confirmed(snapshot_no_assistance(robot,lambda p:False,path)))

    def test_missing_arm_is_not_assumed_empty(self):
        robot=self.robot();del robot._ag_obj_constraints['right']
        report=snapshot_no_assistance(robot,lambda p:False)
        self.assertEqual(report['constraint_active']['right'],-1);self.assertFalse(no_assistance_confirmed(report))


if __name__=='__main__':unittest.main()
