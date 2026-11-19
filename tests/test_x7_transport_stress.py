"""Contracts for paired transport perturbations, without physics execution."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    import numpy as np
    from stress_x7_rgb_transport import SCENARIOS, POSES, disturbance, plan, sha_layout
    from collect_x7_transport_bc import layouts
    from run_x7_turning_carry import r2


@unittest.skipUnless(HAS,'X7 physics/torch deps unavailable')
class StressTransportTest(unittest.TestCase):
    def test_fresh_layouts_and_matched_panel(self):
        known={sha_layout(r) for r in layouts()};fresh={sha_layout(r) for r in POSES.values()}
        self.assertFalse(known&fresh);self.assertEqual(len(fresh),4)
        rows=plan();self.assertEqual(len({r['case'] for r in rows}),32)
        pairs={}
        for r in rows:pairs.setdefault((r['layout'],r['scenario']),set()).add(r['mode'])
        self.assertTrue(all(v=={'candidate','nominal'} for v in pairs.values()))

    def test_bias_is_positive_local_y_and_is_limited_to_transport(self):
        for yaw in (-.42,.42):
            b,f,e=disturbance(SCENARIOS['odom-inward-16mm'],900,920,yaw)
            self.assertAlmostEqual(float((b[:2]@r2(yaw))[1]),.016)
            self.assertEqual(e,1.);self.assertFalse(f.any())
            active=[k for k in range(850,960) if disturbance(SCENARIOS['odom-inward-16mm'],k,920,yaw)[0].any()]
            self.assertEqual(active,list(range(849,920))[1:])

    def test_push_and_effort_window_match(self):
        vals=[disturbance(SCENARIOS['push-limited-drive'],k,940,.4) for k in range(890,945)]
        for i,(b,f,e) in enumerate(vals):
            k=i+890;self.assertFalse(b.any())
            if 896<=k<920:
                self.assertEqual(e,.30);self.assertAlmostEqual(float(f[:2]@r2(.4)[:,1]),-620.)
            else:self.assertEqual(e,1.);self.assertFalse(f.any())
