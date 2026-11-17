from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from probe_pi0_consistent_feedback import nominal_gate,perturb_q

try:
    import numpy as np
except ImportError:np=None


class ConsistentSensorGateTests(unittest.TestCase):
    def good(self):return {'source_rgb_mae_0_255':{'head':5.,'left':6.,'right':5.},'source_body_q_max_abs_rad':1e-4,'root_translation_during_refresh_m':1e-5,'object_pose_write_count':0,'assisted_constraint_seen':False}
    def test_unknown_unstable_or_assisted_initialization_not_scored(self):
        self.assertTrue(nominal_gate(self.good()))
        for key,value in [('source_body_q_max_abs_rad',.02),('root_translation_during_refresh_m',.1),('object_pose_write_count',1),('assisted_constraint_seen',True)]:
            r=self.good();r[key]=value;self.assertFalse(nominal_gate(r))
        r=self.good();r['source_rgb_mae_0_255']['left']=float('nan');self.assertFalse(nominal_gate(r))
        r=self.good();r['source_rgb_mae_0_255'].pop('right');self.assertFalse(nominal_gate(r))

    @unittest.skipIf(np is None,'numpy server fixture required')
    def test_initial_perturbation_respects_unchanged_bounds_and_does_not_mutate_q(self):
        q=np.array([0.,1.,2.,3.,4.,5.],np.float32);copy=q.copy()
        lo=q-.01;hi=q+.01
        np.testing.assert_allclose(perturb_q(q,lo,hi,1),hi)
        np.testing.assert_allclose(perturb_q(q,lo,hi,-1),lo)
        np.testing.assert_array_equal(perturb_q(q,lo,hi,0),q)
        np.testing.assert_array_equal(q,copy)
        with self.assertRaises(ValueError):perturb_q(q,lo,hi,2)
        with self.assertRaises(ValueError):perturb_q(q,hi,lo,1)


if __name__=='__main__':unittest.main()
