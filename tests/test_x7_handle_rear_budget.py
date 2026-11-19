"""Offline handle geometry and post-stop obstacle allowance checks."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from audit_x7_handle_rear_budget import APPROACH_BUDGET_M,OBSTACLE_TRAVEL,BUFFER_M,entry_margin,handle_mounts
    from audit_x7_payload_range_gate import payload_mounts
    from test_x7_dock_guard import RETREAT_M,STATION_CLEARANCE_M


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class HandleBudgetTest(unittest.TestCase):
    def test_entry_allowance_has_a_separate_post_stop_distance_term(self):
        self.assertGreater(APPROACH_BUDGET_M,OBSTACLE_TRAVEL)
        self.assertAlmostEqual(entry_margin()-RETREAT_M-APPROACH_BUDGET_M,STATION_CLEARANCE_M+BUFFER_M)

    def test_mounts_extend_coverage_past_both_main_box_edges(self):
        main=payload_mounts((0.,0.,0.)); handles=handle_mounts((0.,0.,0.))
        self.assertTrue(np.all(abs(handles[:,0])>max(abs(main[:,0]))))
        np.testing.assert_allclose(handles[:,0].min(),-handles[:,0].max())
        self.assertGreater(handles[:,2].max(),main[:,2].max())


if __name__=='__main__': unittest.main()
