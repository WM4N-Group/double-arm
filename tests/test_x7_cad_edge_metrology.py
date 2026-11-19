"""Pure geometry contracts for the isolated CAD edge audit."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
HAS = all(find_spec(n) is not None for n in ('numpy', 'scipy', 'PIL', 'pybullet'))
if HAS:
    import numpy as np
    from audit_x7_cad_edge_metrology import first_nominal_entry, frame, raw_ray_fractions, unframe


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class CADEdgeTest(unittest.TestCase):
    def test_parallel_outside_is_not_a_geometric_hit(self):
        lo, hi = np.array([[-.168, -.275, 1.318]]), np.array([[.168, .063, 1.382]])
        self.assertIsNone(first_nominal_entry([.168932,-.316,1.36175],[.168932,.004,1.36175],lo,hi))
        self.assertAlmostEqual(first_nominal_entry([.168,-.316,1.36175],[.168,.004,1.36175],lo,hi),.041/.320)

    def test_cross_view_and_reverse_use_segment_not_infinite_line(self):
        lo, hi = np.array([[0.,0.,0.]]), np.array([[1.,1.,1.]])
        self.assertAlmostEqual(first_nominal_entry([2.,.5,.5],[-1.,.5,.5],lo,hi),1./3)
        self.assertIsNone(first_nominal_entry([2.,.5,.5],[1.001,.5,.5],lo,hi))

    def test_geometry_comparison_does_not_depend_on_identity(self):
        q = np.array([3.325,.111,-.473]); pt = np.array([[.168932,-.316,1.36],[.18,-.31,1.36]])
        np.testing.assert_allclose(unframe(frame(pt,q),q),pt,atol=1e-12)
        np.testing.assert_array_equal(raw_ray_fractions([{'kind':'cad_link','fraction':.25}]),
                                      raw_ray_fractions([{'kind':'probe_actor','fraction':.25}]))
