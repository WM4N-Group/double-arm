"""X7 CPU trajectory BC offline data and checkpoint contracts (no sim launch)."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
AVAILABLE=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if AVAILABLE:
    import numpy as np
    from train_x7_turning_trajectory_bc import BLOCK_PHASES, DEV, HELDOUT, K, align, fit, predict


@unittest.skipUnless(AVAILABLE,'X7 runtime dependencies unavailable')
class TrajectoryBCContract(unittest.TestCase):
    def test_alignment_keeps_previous_endpoint_and_never_pulls_across_block(self):
        timeline=np.repeat(BLOCK_PHASES,2)
        n=len(timeline); a=np.tile(np.arange(n)[:,None],(1,22)).astype(float)
        route=[{'start_inclusive':2*i,'end_exclusive':2*i+2} for i in range(4,9)]
        base,arm,jaw,length=align(a,timeline,route)
        self.assertEqual(base.shape,(14,K,3)); self.assertEqual(arm.shape,(14,K,14)); self.assertEqual(jaw.shape,(14,K,2))
        np.testing.assert_array_equal(length,np.full(14,2))
        np.testing.assert_array_equal(arm[1:,0,0],arm[:-1,-1,0])
        np.testing.assert_array_equal(jaw[:,-1,0],np.arange(1,28,2))

    def test_phase_labels_and_kinematic_cadence_are_fitted_without_evaluation(self):
        rows=[]
        for sign in (-1,1):
            for k in range(4):
                l={'goal':(3.2+k*.02,.1,sign*(24+k)),'bay_y':.66,'tote':(k*.001,0.,0.)}
                base=np.zeros((14,K,3)); arm=np.zeros((14,K,14)); jaw=np.full((14,K,2),.034)
                base[:,:,2]=sign*.1; arm[:,:,0]=k*.01
                rows.append(dict(layout=l,base=base,arm=arm,jaw=jaw,durations=np.full(14,15.)))
        checkpoint=fit(rows)
        for l in DEV.values():
            u,ph,durations=predict(checkpoint,l)
            self.assertEqual(u.shape,(210,22)); self.assertTrue(np.isfinite(u).all())
            np.testing.assert_array_equal(ph,np.repeat(BLOCK_PHASES,15)); self.assertEqual(durations,[15]*14)
            self.assertNotEqual(l,HELDOUT)


if __name__=='__main__':
    unittest.main()
