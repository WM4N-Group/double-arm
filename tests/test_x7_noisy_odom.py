"""Offline invariants for the v2 disturbed-data plan."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from collect_x7_noisy_odom import LAG, make_bias, task_plan
    from collect_x7_transport_bc import layouts, sha_layout
    from run_x7_turning_carry import r2


@unittest.skipUnless(HAS,'X7 physics deps unavailable')
class NoisyOdomTest(unittest.TestCase):
    def test_training_layouts_only_and_opposite_sign_pairs(self):
        train={r['case']:sha_layout(r) for r in layouts() if r['split']=='train'}
        plan=task_plan();self.assertEqual(len(plan),48)
        for a,b in zip(plan[::2],plan[1::2]):
            self.assertEqual(a['parent_train_case'],b['parent_train_case'])
            self.assertEqual(sha_layout(a['layout']),train[a['parent_train_case']])
            self.assertLess(a['bias_m']*b['bias_m'],0)
            self.assertNotEqual(a['case'],b['case'])

    def test_bias_and_delayed_oracle_label_are_local_and_do_not_cross_dock(self):
        for c in task_plan():
            b=make_bias(1200,240,990,c);self.assertTrue(b.any());self.assertFalse(b[:240].any());self.assertFalse(b[990-LAG:].any())
            corrected=np.zeros_like(b);corrected[LAG:]=b[:-LAG]
            self.assertFalse(corrected[990:].any())
            local=b[:,:2]@r2(np.radians(c['layout']['goal'][2]));np.testing.assert_allclose(local[:,0],0.,atol=1e-12)
            self.assertAlmostEqual(float(np.max(abs(local[:,1]))),abs(c['bias_m']))
