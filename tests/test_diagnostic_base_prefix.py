from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnostic_base_prefix import replace_initial_base_only
try:
    import numpy as np
except ImportError:np=None


@unittest.skipIf(np is None,'server numpy test')
class BasePrefixTests(unittest.TestCase):
    def test_only_base_prefix_changes_no_expert_arms_grippers_or_input_mutation(self):
        b=np.ones((3000,21),np.float32);s=np.full((1714,21),2.,np.float32)
        out=replace_initial_base_only(b,s);self.assertEqual(out.shape,(1200,21))
        np.testing.assert_array_equal(out[:682,:3],2.);np.testing.assert_array_equal(out[:,3:],1.)
        np.testing.assert_array_equal(out[682:,:3],1.);np.testing.assert_array_equal(b,1.);np.testing.assert_array_equal(s,2.)
    def test_wrong_tape_or_unknown_values_rejected(self):
        with self.assertRaises(ValueError):replace_initial_base_only(np.ones((1714,21),np.float32),np.ones((1714,21),np.float32))
        b=np.ones((3000,21),np.float32);b[0,0]=np.nan
        with self.assertRaises(ValueError):replace_initial_base_only(b,np.ones((1714,21),np.float32))


if __name__=='__main__':unittest.main()
