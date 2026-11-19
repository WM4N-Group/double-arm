"""Pure contracts for the independent payload validation panel."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS_RUNTIME=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS_RUNTIME:
    import numpy as np
    from audit_x7_payload_generalization import MIX, SCENES, pose_for_control
    from audit_x7_anchor_tolerance import LAYOUTS
    from run_x7_turning_carry import dock_pose,r2


@unittest.skipUnless(HAS_RUNTIME,'X7 runtime dependencies unavailable')
class PayloadValidationTests(unittest.TestCase):
    def test_controller_bias_does_not_mutate_the_world(self):
        for scene,bias,overshoot,_ in MIX.values():
            true=SCENES[scene]; snapshot=true.copy(); obs=pose_for_control(true,bias,overshoot)
            a,b=dock_pose(true),dock_pose(obs)
            self.assertEqual(true,snapshot)
            np.testing.assert_allclose(r2(a[2]).T@(b[:2]-a[:2]),[0.,-overshoot],atol=1e-12)
            np.testing.assert_allclose(np.subtract(obs['tote'],true['tote']),bias,atol=1e-12)

    def test_panel_never_reuses_the_calibration_or_previous_holdout_layout(self):
        for l in SCENES.values():
            for v in LAYOUTS.values(): self.assertNotEqual(l['goal'],v['goal'])
            self.assertNotEqual(l['goal'][2],34.)
        clear={l for l,_,_,fixture in MIX.values() if fixture is None}
        self.assertEqual(clear,set(SCENES))


if __name__=='__main__':unittest.main()
