"""Offline checks for terminal time-scaling and independent force clocks."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    import numpy as np
    from audit_x7_preslow_transport import STRETCH, plan, retime, spec
    from train_x7_turning_trajectory_bc import BLOCK_PHASES


@unittest.skipUnless(HAS,'Remote X7 dependencies unavailable')
class PreslowTest(unittest.TestCase):
    def test_only_terminal_transport_block_changes(self):
        counts=[20]*14;ph=np.repeat(BLOCK_PHASES,counts)
        u=np.tile(np.arange(len(ph),dtype=float)[:,None],(1,22))
        a,p,start,old_end,new_end=retime(u,ph,counts,STRETCH)
        self.assertEqual((start,old_end,new_end),(160,180,210))
        np.testing.assert_array_equal(a[:start],u[:start]);np.testing.assert_array_equal(a[new_end:],u[old_end:])
        np.testing.assert_array_equal(p[new_end:],ph[old_end:]);self.assertTrue(np.all(p[start:new_end]==4))
        np.testing.assert_allclose(a[new_end-1,3:6],u[old_end-1,3:6]/STRETCH)
        np.testing.assert_allclose(a[new_end-1,:3],u[old_end-1,:3])

    def test_fixed_and_late_exposures_are_distinct_and_prespecified(self):
        push,k=spec('slow-push-fixed',1100,1400);late,j=spec('slow-push-late',1100,1400)
        self.assertEqual((k,j),(1100,1400));self.assertEqual(push,late)
        self.assertEqual(len(set(plan('bench'))),8);self.assertEqual(len(set(plan('panel'))),16)
