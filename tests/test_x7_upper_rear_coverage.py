"""Offline geometry checks for extending the X7 fixed-body range profile."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from audit_x7_upper_rear_coverage import cad_boxes, envelope_grid, MOUNT_PAD_M
    from audit_x7_rear_coverage import rear_curtain


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class UpperRearCoverageContract(unittest.TestCase):
    cad={'parts_in_cad_frame':[
        {'center':[-1.41,-.24,.01], 'size':[.71,.95,.64]},
        {'center':[-.98,-.03,0.], 'size':[.24,.2,.30]},
        {'center':[-.86,0.,0.], 'size':[.12,.22,.30]},
    ]}

    def test_grid_uses_inclusive_fixed_body_slabs_not_one_bounding_box(self):
        lo,hi=cad_boxes(self.cad); g=envelope_grid(self.cad); low=rear_curtain(self.cad)
        self.assertLess(low[:,2].max(),g[:,2].max())
        np.testing.assert_allclose(g[:,2].min(),lo[:,2].min())
        np.testing.assert_allclose(g[:,2].max(),hi[:,2].max())
        for low,high in zip(lo,hi):
            mask=((g[:,0]>=low[0]-1e-12)&(g[:,0]<=high[0]+1e-12)&
                  (g[:,2]>=low[2]-1e-12)&(g[:,2]<=high[2]+1e-12))
            self.assertTrue(mask.any())
            self.assertTrue(np.all(g[mask,1]>=high[1]+MOUNT_PAD_M-1e-12))

    def test_overlap_rays_start_beyond_outer_surface(self):
        g=envelope_grid(self.cad); lo,hi=cad_boxes(self.cad)
        hits=(abs(g[:,0])<.1)&(g[:,2]>.7)&(g[:,2]<.714)
        self.assertTrue(hits.any())
        np.testing.assert_allclose(g[hits,1],hi[0,1]+MOUNT_PAD_M)
        after=(abs(g[:,0])<.1)&(g[:,2]>.77)&(g[:,2]<.84)
        self.assertTrue(after.any())
        np.testing.assert_allclose(g[after,1],hi[1,1]+MOUNT_PAD_M)

    def test_rejects_invalid_cad_extent(self):
        bad={'parts_in_cad_frame':[{'center':[0,.2,0], 'size':[.1,0,.2]}]}
        with self.assertRaises(ValueError):envelope_grid(bad)


if __name__=='__main__': unittest.main()
