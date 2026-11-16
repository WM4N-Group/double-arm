from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi0_accumulation_preflight import accumulation_plan
try:
    import numpy as np
except ImportError:np=None


class AccumulationPlanTests(unittest.TestCase):
    def test_paper_update_count_is_not_microstep_count(self):
        p=accumulation_plan(1,64,50000)
        self.assertEqual(p['required_micro_steps'],3200000);self.assertEqual(p['sample_exposures'],3200000)
        self.assertEqual(p['accumulation_steps'],64);self.assertFalse(p['pi0_gpu_memory_fit_verified'])
        with self.assertRaises(ValueError):accumulation_plan(3,64,50000)


@unittest.skipIf(np is None,'Numerical fixture runs remotely')
class FrozenEndpointTests(unittest.TestCase):
    def test_endpoint_rule_is_inclusive_not_adjacent_differences(self):
        from frozen_segment_audit import frozen_segment_mask
        q=np.array([0.,1.,2.,2.,1.,0.])[:,None];mask,quiet=frozen_segment_mask(q)
        self.assertTrue(quiet[0]);self.assertTrue(mask.all())
    def test_motion_short_sequence_overlap_and_strict_threshold(self):
        from frozen_segment_audit import frozen_segment_mask
        mask,_=frozen_segment_mask(np.arange(10,dtype=float)[:,None]*.01);self.assertFalse(mask.any())
        mask,_=frozen_segment_mask(np.zeros((4,26)));self.assertFalse(mask.any())
        mask,_=frozen_segment_mask(np.zeros((8,26)));self.assertTrue(mask.all())
        q=np.zeros((6,1));q[-1]=1e-3;mask,_=frozen_segment_mask(q);self.assertFalse(mask.any())
        with self.assertRaises(ValueError):frozen_segment_mask(np.array([[float('nan')]]))
