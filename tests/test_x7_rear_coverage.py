"""Rear depth curtain contracts (no local simulation)."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
HAS_RUNTIME = all(find_spec(n) is not None for n in ('numpy', 'pybullet', 'scipy', 'PIL'))
if HAS_RUNTIME:
    import numpy as np
    from audit_x7_rear_coverage import CoverageWorld, PITCH_M, rear_curtain, summarize_cells
    from audit_x7_chassis_range_proxy import MOUNT_PAD_M, RANGE_M
    from test_x7_dock_guard import DockGuard


@unittest.skipUnless(HAS_RUNTIME, 'X7 runtime dependencies unavailable')
class CurtainTests(unittest.TestCase):
    def test_masked_and_corrupt_cells_are_unknown_not_no_hit(self):
        self.assertEqual(summarize_cells([1., 1., .05], [True, False, True]), (None, 1, 1))
        self.assertEqual(summarize_cells([1., np.nan, -1.], [True] * 3), (None, 2, 0))

    def test_valid_occupied_and_valid_no_return_remain_distinct(self):
        self.assertEqual(summarize_cells([1., 1.], [True] * 2), (RANGE_M + MOUNT_PAD_M, 0, 0))
        d, unknown, hits = summarize_cells([1., .01], [True] * 2)
        self.assertAlmostEqual(d, .01 * RANGE_M + MOUNT_PAD_M)
        self.assertEqual((unknown, hits), (0, 1))

    def test_braking_latches_a_delivered_unknown_even_if_later_packet_recovers(self):
        w=CoverageWorld.__new__(CoverageWorld); w.guard=DockGuard(0); w.rear_events=[]
        w.guard.state='brake'; q=np.zeros((3,2))
        w.observe_active_packet(40,24,np.array([.32,.014,-1.]),None,q)
        w.observe_active_packet(48,32,np.array([.32,.014,.31]),None,q)
        self.assertTrue(w.guard.cannot_retreat)
        self.assertEqual(w.guard.state,'brake')
        self.assertEqual([x['sample_k'] for x in w.rear_events],[24])

    def test_active_retreat_stops_at_measured_position_on_partial_loss(self):
        w=CoverageWorld.__new__(CoverageWorld);w.guard=DockGuard(0);w.rear_events=[]
        w.guard.state='retreat';q=np.zeros((3,2));q[1,0]=.004
        w.observe_active_packet(72,56,np.array([.32,.027,-1.]),None,q)
        self.assertEqual(w.guard.state,'hold');self.assertTrue(w.guard.cannot_retreat)
        np.testing.assert_array_equal(w.guard.hold,q[:,0])

    def test_grid_covers_both_projected_dimensions_to_slab_edges(self):
        cad={'parts_in_cad_frame':[{'center':[-1.41,-.24,.01], 'size':[.71,.95,.64]},
                                    {'center':[-.5,-.1,0.], 'size':[.15,.3,.3]}]}
        cells=rear_curtain(cad);x,z=np.unique(cells[:,0]),np.unique(cells[:,2])
        np.testing.assert_allclose([x[0],x[-1],z[0],z[-1]],[-.33,.31,.005,.715])
        self.assertLessEqual(max(np.diff(x).max(),np.diff(z).max()),PITCH_M+1.e-12)
        self.assertEqual(len(cells),len(x)*len(z))


if __name__=='__main__':
    unittest.main()
