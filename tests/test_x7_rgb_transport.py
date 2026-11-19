"""CPU contracts for the X7 data split and learned transport interface."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet','torch'))
if HAS:
    import numpy as np
    import torch
    from collect_x7_transport_bc import layouts, sha_layout, validate_trace
    from run_x7_turning_carry import dock_pose
    from train_x7_rgb_transport import RGBTransport, feature, transport_prior


@unittest.skipUnless(HAS,'X7 CPU/torch dependencies unavailable')
class RGBTransportTest(unittest.TestCase):
    def test_layout_plan_disjoint_and_unseen_yaw_bands(self):
        rows=layouts();self.assertEqual(rows,layouts())
        self.assertEqual(len({sha_layout(r) for r in rows}),96)
        by={name:[abs(r['goal'][2]) for r in rows if r['split']==name] for name in ('train','dev','heldout')}
        self.assertEqual([len(a) for a in by.values()],[72,12,12])
        self.assertLess(max(by['train']),min(by['dev']))
        self.assertLess(max(by['dev']),min(by['heldout']))

    def test_geometric_prior_does_not_fill_nontransport_actions(self):
        layout=layouts()[0];ref,start,stop=transport_prior(layout,[20]*14)
        self.assertEqual(ref.shape,(280,6));self.assertEqual((start,stop),(80,180))
        self.assertFalse(np.any(ref[:start]));self.assertFalse(np.any(ref[stop:]))
        np.testing.assert_allclose(ref[stop-1,:3],dock_pose(layout),atol=2e-7)
        np.testing.assert_array_equal(ref[stop-1,3:],0.)

    def test_strict_clearance_is_not_rounded(self):
        case={'action':np.zeros((7,22)),'phase':np.zeros(7),'rgb_external':np.zeros((2,128,192,3),dtype=np.uint8),
              'rgb_action_index':np.array([0,6]),'station_clearance':np.array([[[.0049999,.1,0.,0.]]]),
              'station_initialization_clearance':np.array([[.1,.1,0.,0.]])}
        score={'success':True,'object_attachment_constraints_seen':0,'state_writes_during_actions':0}
        foot={'full_bottom_supported':True}
        self.assertFalse(validate_trace(case,score,foot))
        case['station_clearance'][0,0,0]=.005
        self.assertTrue(validate_trace(case,score,foot))
        case['station_clearance'][0,0,2]=1
        self.assertFalse(validate_trace(case,score,foot))
        case['rgb_action_index'][1]=5
        with self.assertRaisesRegex(ValueError,'misaligned'):validate_trace(case,score,foot)

    def test_model_consumes_image_and_state_in_one_chunk(self):
        torch.manual_seed(81);torch.set_num_threads(1)
        layout=layouts()[0]
        dims=len(feature(layout,np.zeros((3,2)),np.zeros((14,2)),np.zeros((4,2)),np.zeros((6,6))))
        m=RGBTransport(dims).eval()
        st=torch.zeros((1,dims));im=torch.zeros((1,128,192,3),dtype=torch.uint8)
        with torch.inference_mode():
            z=m(im,st);self.assertEqual(z.shape,(1,6,6))
            self.assertFalse(torch.equal(z,m(im+180,st)))
            self.assertFalse(torch.equal(z,m(im,st+1)))
