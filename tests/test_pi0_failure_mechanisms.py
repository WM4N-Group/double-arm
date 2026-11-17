import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_pi0_failure_mechanisms import motion_metrics, intervene, digest_inputs
try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'NumPy numerical tests run on server')
class FailureMechanismTests(unittest.TestCase):
    def setUp(self):
        self.inputs={'head_image':np.zeros((4,4,3),np.uint8),'left_wrist_image':np.ones((4,4,3),np.uint8),
            'right_wrist_image':np.full((4,4,3),2,np.uint8),'state':np.arange(21,dtype=np.float32),'prompt':'pick'}
        self.other={k: v.copy() if hasattr(v,'copy') else v for k,v in self.inputs.items()}
        self.other['head_image'][:]=50; self.other['state'][:]=-3

    def test_intervention_changes_only_named_modalities_and_never_mutates(self):
        before=digest_inputs(self.inputs)
        for name,changed in [('other_frame_rgb',{'head_image'}),('other_frame_state',{'state'}),
                             ('different_prompt',{'prompt'}),('gray_head_only',{'head_image'}),
                             ('gray_wrists_only',{'left_wrist_image','right_wrist_image'})]:
            result=intervene(self.inputs,self.other,name)
            actual={key for key in result if not np.array_equal(result[key],self.inputs[key])}
            self.assertEqual(actual,changed)
        self.assertEqual(before,digest_inputs(self.inputs))
        self.assertEqual(before,digest_inputs(intervene(self.inputs,self.other,'repeat_original')))
        with self.assertRaises(ValueError):intervene(self.inputs,self.other,'bad')

    def test_hold_baseline_uses_right_arm_state_offset_not_gripper(self):
        state=np.arange(21,dtype=float); ref=np.zeros((50,21))
        ref[:,3:13]=state[3:13];ref[:,14:20]=state[13:19]
        metrics=motion_metrics(ref,ref,state,np.ones(21))
        for name in ('trunk','left_arm','right_arm'):
            self.assertEqual(metrics[name]['hold_current_q_rmse_rad'],0)
            self.assertIsNone(metrics[name]['error_to_hold_rmse_ratio'])
        self.assertNotIn('hold_current_q_rmse_rad',metrics['left_gripper'])

    def test_error_ratio_distinguishes_precise_motion_from_low_absolute_error(self):
        state=np.zeros(21);ref=np.zeros((50,21));ref[:,7:13]=.001
        prediction=ref.copy();prediction[:,7:13]+=.01
        metrics=motion_metrics(prediction,ref,state,np.ones(21))
        self.assertAlmostEqual(metrics['left_arm']['error_to_hold_rmse_ratio'],10)
        self.assertAlmostEqual(metrics['left_arm']['hold_current_q_rmse_rad'],.001)
        ref[:,13]=1;prediction[:,13]=.3
        metrics=motion_metrics(prediction,ref,state,np.ones(21))
        self.assertEqual(metrics['left_gripper']['effective_command_agreement_fraction'],0)

    def test_bad_shapes_nonfinite_and_scale_rejected(self):
        ref=np.zeros((50,21));state=np.zeros(21);std=np.ones(21)
        with self.assertRaises(ValueError):motion_metrics(ref[:10],ref,state,std)
        with self.assertRaises(ValueError):motion_metrics(ref,ref,state,std,0)
        ref[0,0]=np.nan
        with self.assertRaises(ValueError):motion_metrics(ref,ref,state,std)

@unittest.skipIf(np is None,'NumPy numerical tests run on server')
class GraspOnsetSelectionTests(unittest.TestCase):
    def test_effective_threshold_and_missing_transition(self):
        from probe_pi0_grasp_onset import onset_frames
        self.assertEqual(onset_frames([1]*20+[.49]*30),(20,35))
        for values in ([1,1],[-1,-1],[np.nan,1],[]):
            with self.assertRaises(ValueError):onset_frames(values)

if __name__=='__main__':unittest.main()
