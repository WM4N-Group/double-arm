"""Fail-closed packet and exercise-plan contracts."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    import numpy as np
    from audit_x7_brake_monitor import DELAY, MAX_AGE, PAUSE, READY_FRAMES, exercise_plan, feedback_valid, plan, probe


@unittest.skipUnless(HAS,'Remote X7 dependencies unavailable')
class BrakeMonitorTest(unittest.TestCase):
    def test_age_out_of_range_missing_and_wrong_wave_fail(self):
        good=probe(800)
        self.assertTrue(feedback_valid((800,good+.6),800+DELAY))
        self.assertFalse(feedback_valid((800,good),800+DELAY-1))
        self.assertFalse(feedback_valid((800,good),800+MAX_AGE+1))
        self.assertFalse(feedback_valid((800,None),800+DELAY))
        self.assertFalse(feedback_valid((800,0.),800+DELAY))
        self.assertNotEqual(probe(800),probe(808))
        self.assertFalse(feedback_valid((808,good),808+DELAY))

    def test_confirmation_holds_before_original_motion_resumes(self):
        u=np.arange(10*22).reshape(10,22).astype(float);ph=np.r_[0,1,2,3,4,4,4,4,5,6]
        v,p=exercise_plan(u,ph,6)
        end=6+PAUSE+READY_FRAMES
        np.testing.assert_array_equal(v[:6],u[:6]);np.testing.assert_array_equal(v[end:],u[6:]);self.assertTrue(np.all(v[6:end,3:6]==0.))
        self.assertTrue(np.all(p[6:end]==4));self.assertEqual(len(set(plan())),36)
