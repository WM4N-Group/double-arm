"""Geometry and unknown propagation tests for complementary rear observations."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
HAS = all(find_spec(n) is not None for n in ('numpy', 'scipy', 'PIL', 'pybullet'))
if HAS:
    import numpy as np
    from audit_x7_split_rear_rays import RANGE_M, combine_split, partition_mounts


@unittest.skipUnless(HAS, 'X7 runtime dependencies unavailable')
class SplitRearRaysTest(unittest.TestCase):
    def test_origin_and_projection_are_per_cell(self):
        lo, hi = np.array([[-.10, -.03, .50]]), np.array([[.10, .30, .60]])
        # First is shadowed despite the CAD exit being beyond one range;
        # second misses in X, third starts inside, fourth sees face too far.
        mounts = np.array([[.09, -.07, .55], [.11, -.07, .55], [0., .01, .55], [0., -.40, .55]])
        (direct, split, unknown), faces = partition_mounts(mounts, lo, hi)
        np.testing.assert_array_equal(direct, [1, 3])
        np.testing.assert_array_equal(split, [0])
        np.testing.assert_array_equal(unknown, [2])
        np.testing.assert_allclose(faces, [[-.03, .30]])

    def test_unobserved_inter_cad_cavity_is_not_skipped(self):
        lo = np.array([[-1., -.02, 0.], [-1., .03, 0.]])
        hi = np.array([[1., .02, 1.], [1., .06, 1.]])
        (direct, split, unknown), _ = partition_mounts(np.array([[0., -.07, .5]]), lo, hi)
        self.assertEqual((len(direct), len(split), unknown.tolist()), (0, 0, [0]))

    def test_short_prefix_measurement_keeps_metric_range(self):
        length = .030
        out = combine_split([.40, 1., 1., np.nan, 1., -1.], [1., .20, 1., 1., np.nan, .5], length)
        np.testing.assert_allclose(out[:3], [.40 * length / RANGE_M, .20, 1.])
        self.assertTrue(np.isnan(out[3:]).all())
