from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi0_relative_joint_diagnostic import joint_encoding,RelativeJointInputs,RelativeJointOutputs,JOINT_ACTION_INDICES,JOINT_STATE_INDICES
try:import numpy as np
except ImportError:np=None


class RelativeJointSchemaTests(unittest.TestCase):
    def test_right_arm_offset_never_uses_gripper_width(self):
        self.assertEqual(JOINT_ACTION_INDICES[-6:],tuple(range(14,20)))
        self.assertEqual(JOINT_STATE_INDICES[-6:],tuple(range(13,19)))
        self.assertNotIn(13,JOINT_ACTION_INDICES);self.assertNotIn(20,JOINT_ACTION_INDICES)


@unittest.skipIf(np is None,'Numerical tests run on remote server')
class RelativeJointNumerics(unittest.TestCase):
    def test_roundtrip_preserves_native_base_grippers_and_current_state(self):
        state=np.arange(21,dtype=np.float32)*.1;actions=np.arange(50*21,dtype=np.float32).reshape(50,21)/1000
        before=state.copy();relative=joint_encoding(actions,state)
        for axis in (0,1,2,13,20):np.testing.assert_array_equal(relative[:,axis],actions[:,axis])
        np.testing.assert_allclose(joint_encoding(relative,state,inverse=True),actions,atol=3e-7)
        np.testing.assert_array_equal(state,before)

    def test_inverse_is_one_chunk_anchor_not_accumulation_or_future_state(self):
        state=np.arange(21,dtype=np.float32);relative=np.zeros((50,21),np.float32);relative[:,7]=.001
        native=RelativeJointOutputs()({'state':np.r_[state,np.zeros(11,np.float32)],'actions':np.pad(relative,((0,0),(0,11)))})['actions']
        np.testing.assert_array_equal(native[:,7],np.full(50,7.001,np.float32))
        self.assertEqual(native[0,14],state[13]);self.assertEqual(native[0,19],state[18])
        np.testing.assert_array_equal(native[:,13],0)

    def test_inputs_keep_real_images_original_proprio_prompt_no_index(self):
        from momagen_pi0_schema import make_pi0_inputs
        data={'state':np.arange(21,dtype=np.float32),'actions':np.zeros((50,21),np.float32),'prompt':'pick'}
        for key in ('head_image','left_wrist_image','right_wrist_image'):data[key]=np.zeros((4,4,3),np.uint8)
        before=make_pi0_inputs(data);after=RelativeJointInputs()(data)
        self.assertEqual(set(before),set(after))
        np.testing.assert_array_equal(after['state'],before['state'])
        for key in before['image']:np.testing.assert_array_equal(after['image'][key],before['image'][key])
        np.testing.assert_allclose(joint_encoding(after['actions'],after['state'],inverse=True),data['actions'])

    def test_reject_unknown_nonfinite_and_wrong_layouts(self):
        for actions,state in ((np.zeros((2,20)),np.zeros(21)),(np.zeros((2,21)),np.zeros(32)),(np.full((2,21),np.nan),np.zeros(21))):
            with self.assertRaises(ValueError):joint_encoding(actions,state)

    def test_real_hdf5_stats_match_all_episode_clamped_horizons_and_keep_nonjoint_axes(self):
        import hashlib
        import tempfile
        import h5py
        from pi0_relative_joint_diagnostic import relative_joint_stats
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'fixture.hdf5';episodes=[];expected=[];rng=np.random.default_rng(12)
            with h5py.File(path,'w') as f:
                for index in range(16):
                    actions=rng.normal(size=(3,21)).astype(np.float32);states=rng.normal(size=(3,21)).astype(np.float32)
                    g=f.create_group(f'data/demo_{index}');g.create_dataset('actions',data=actions);g.create_dataset('obs/prop_state',data=states)
                    episodes.append({'source':str(path),'source_demo':f'demo_{index}','frames':3,
                        'fingerprint':hashlib.sha256(actions.tobytes()+states[0].tobytes()).hexdigest()})
                    for start in range(3):
                        expected.append(joint_encoding(actions[np.minimum(np.arange(start,start+50),2)],states[start]))
            native={'mean':np.arange(21,dtype=float).tolist(),'std':np.ones(21).tolist()}
            result,meta=relative_joint_stats({'split':'train','episodes':episodes},native);expected=np.concatenate(expected)
            axes=list(JOINT_ACTION_INDICES)
            np.testing.assert_allclose(np.asarray(result['mean'])[axes],expected[:,axes].astype(np.float64).mean(axis=0),atol=1e-7)
            np.testing.assert_allclose(np.asarray(result['std'])[axes],expected[:,axes].astype(np.float64).std(axis=0),atol=1e-7)
            for axis in (0,1,2,13,20):
                self.assertEqual(result['mean'][axis],native['mean'][axis]);self.assertEqual(result['std'][axis],native['std'][axis])
            self.assertEqual(meta['label_values_per_dimension'],2400)
            self.assertIsNone(result['q01']);self.assertIsNone(result['q99'])
            with self.assertRaises(ValueError):relative_joint_stats({'split':'validation','episodes':episodes},native)


if __name__=='__main__':unittest.main()
