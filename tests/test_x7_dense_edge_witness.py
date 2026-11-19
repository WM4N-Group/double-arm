"""CPU-only invariants for the experimental transverse X7 witness."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from audit_x7_dense_edge_witness import RANGE_M, WITNESS_Y_PITCH_M, witness_fusion, witness_plan


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class DenseWitnessTest(unittest.TestCase):
    def test_side_clear_does_not_lift_unrelated_longitudinal_returns(self):
        expected=.04
        self.assertEqual(witness_fusion(.043/RANGE_M,expected,[1.,1.]),(1.,'verified_side_clear'))
        for d in (.004,.034,.090):
            f=d/RANGE_M
            self.assertEqual(witness_fusion(f,expected,[1.,1.]),(f,'preserved'))

    def test_missing_or_occupied_side_stays_inhibited_even_with_no_forward_return(self):
        for f,reason in (([1.,np.nan],'missing'),([1.,1.01],'missing'),([1.,.44],'occupied')):
            value,status=witness_fusion(1.,.040,f)
            self.assertTrue(np.isnan(value))
            self.assertEqual(status,reason)
        self.assertTrue(np.isnan(witness_fusion(np.nan,.040,[1.,1.])[0]))

    def test_mirror_geometry_and_coverage_not_per_fixture(self):
        lo=np.array([[-.168,-.275,1.318]]);hi=np.array([[.168,.063,1.382]])
        mounts=np.array([[.16893,-.316,1.360],[-.16893,-.316,1.360],[.170,-.316,1.360]])
        rows=witness_plan(mounts,lo,hi,[0,1,2])
        self.assertEqual([(r[0],r[1]) for r in rows],[(0,1),(1,-1)])
        for i,sign,_,a,b in rows:
            self.assertTrue((sign*a[:,0]>sign*b[:,0]).all())
            np.testing.assert_array_equal(a[:,1:],b[:,1:])
            self.assertLess(a[0,1],mounts[i,1])
            self.assertGreater(a[-1,1],mounts[i,1]+RANGE_M)
            self.assertLessEqual(float(np.diff(a[:,1]).max()),WITNESS_Y_PITCH_M+1e-12)
