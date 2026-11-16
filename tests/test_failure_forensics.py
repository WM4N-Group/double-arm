from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
except ImportError:np=None


@unittest.skipIf(np is None,'Numerical fixtures run remotely')
class FailureForensicsTests(unittest.TestCase):
    def test_close_steps_are_not_separate_attempts_and_raw_base_clipping_is_counted(self):
        from analyze_pi0_failed_rollout import action_metrics
        a=np.zeros((12,21));a[:,13]=1;a[3:,13]=-1;s=np.zeros_like(a);s[:4,19]=.1
        raw=a.copy();raw[5,0]=.2
        report=action_metrics(a,s,raw,a,s,5)
        self.assertEqual(report['close_command_runs'],[[3,11]])
        self.assertEqual(report['negative_jaw_command_steps'],9)
        self.assertEqual(report['first_close_step'],3)
        self.assertEqual(report['raw_native_base_output_clipped_steps'],1)
        self.assertEqual(report['one_step_tracking_rmse_rad'],0)
