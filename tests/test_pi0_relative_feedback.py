from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_pi0_relative_feedback import feedback_gain

try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'numpy unavailable locally; run real arithmetic on server')
class FeedbackProbeTests(unittest.TestCase):
    def test_native_target_response_anchor_follow_vs_correct(self):
        original=np.zeros((50,21),np.float32)
        delta=np.full(6,.02,np.float32)
        changed=original.copy();changed[:,7:13]=delta
        self.assertAlmostEqual(feedback_gain(original,changed,delta)['native_target_response_gain'],1.,places=6)
        self.assertEqual(feedback_gain(original,original,delta)['native_target_response_gain'],0.)
        with self.assertRaises(ValueError):feedback_gain(original,changed,np.zeros(6))
        with self.assertRaises(ValueError):feedback_gain(original,changed,np.full(6,np.nan))


if __name__=='__main__':unittest.main()
