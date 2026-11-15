"""Lightweight pi0 integration contracts; numeric/image checks run remotely."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


class Pi0SourceContract(unittest.TestCase):
    def test_only_official_pi0_not_pi05_or_fast(self):
        source=(ROOT/'scripts/pi0_r1_config.py').read_text()
        self.assertIn("pi0_config.Pi0Config(",source)
        self.assertNotIn('pi05=True',source)
        self.assertIn('action_horizon=50',source)
        self.assertIn('rank=32',source)
        self.assertIn('freeze_filter=model.get_freeze_filter()',source)
        self.assertIn('ema_decay=None',source)
        self.assertIn('fsdp_devices=1',source)
        self.assertNotIn('DeltaActions(',source)
        self.assertNotIn('AlohaInputs(',source)

    def test_native_visual_converter_never_uploads_or_reads_failed_data(self):
        source=(ROOT/'scripts/convert_momagen_pi0.py').read_text()
        tree=ast.parse(source)
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                             and node.func.attr=='push_to_hub' for node in ast.walk(tree)))
        self.assertIn("task/'tmp'",source)
        self.assertNotIn("task/'tmp_failed'",source)
        self.assertIn("split='validation' if index==0",source)
        self.assertIn("if fingerprint in seen:continue",source)
        self.assertIn('camera_keys(obs)',source)

    def test_training_checks_pretrained_download_and_gpu_allocation(self):
        source=(ROOT/'scripts/run_pi0_pipeline.py').read_text()
        self.assertIn('gpu=resolve_gpu(root)',source)
        self.assertIn("verified.get('status')!='download_verified'",source)
        self.assertIn("'PI0_BATCH_SIZE':'1'",source)
        self.assertIn('wait(a.generation_pid',source)
        self.assertIn("'PI0_TRAIN_STEPS':'20'",source)
        self.assertIn("task_success=False",source)
        self.assertIn("'audit_pi0_parameter_contract.py'",source)
        self.assertIn("stage='pi0_parameter_contract_failed'",source)

    def test_visual_pi0_eval_has_one_reset_and_no_teacher_or_object_pose_inputs(self):
        source=(ROOT/'scripts/evaluate_pi0_r1.py').read_text();tree=ast.parse(source)
        loads=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='load_state']
        self.assertEqual(len(loads),1)
        self.assertNotIn('add_unique_task_poses(',source)
        self.assertNotIn("g['actions']",source)
        self.assertIn("meta['env_kwargs']['init_curobo']=False",source)
        self.assertIn("prompt=PROMPT",source)
        self.assertIn("inputs.update(state=as_numpy(obs['prop_state'])",source)
        self.assertIn('prediction,info=infer_local(inputs,token,args.port)',source)

    def test_evaluation_orchestrator_uses_isolated_runtime(self):
        source=(ROOT/'scripts/run_pi0_evaluation.py').read_text()
        self.assertIn("expected_env=root/'envs/openpi'",source)
        self.assertIn('os.execv(str(interpreter)',source)
        self.assertIn("str(script/'server_env.sh')",source)
        self.assertIn('set -euo pipefail; source',source)
        tree=ast.parse(source)
        top_imports=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any(isinstance(node,ast.Import) and any(alias.name in ('numpy','h5py') for alias in node.names)
                             for node in top_imports))
        self.assertIn("code=run([py,str(script/'run_pi0_evaluation.py')",(ROOT/'scripts/run_pi0_pipeline.py').read_text())

    def test_parameter_audit_is_abstract_cpu_only_and_checks_actual_partition(self):
        source=(ROOT/'scripts/audit_pi0_parameter_contract.py').read_text()
        self.assertIn("os.environ.get('JAX_PLATFORMS') != 'cpu'",source)
        self.assertIn("os.environ.get('CUDA_VISIBLE_DEVICES') != ''",source)
        self.assertIn('jax.eval_shape(',source)
        self.assertIn('train.flat_state().items()',source)
        self.assertIn('count(train) + count(frozen) != count(state)',source)
        self.assertIn("'non_lora_trainable_leaves': non_lora",source)
        tree=ast.parse(source)
        self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                             and node.func.attr in ('value_and_grad','grad') for node in ast.walk(tree)))

    def test_zero_disk_relay_is_not_an_open_proxy(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from relay_public_pi0_assets import allowed_path
        self.assertTrue(allowed_path('/checkpoints/pi0_base/params/_METADATA'))
        for path in ('/other/checkpoint','/checkpoints/pi0_base/../secret',
                     '/checkpoints/pi0_base/%2e%2e/secret','/checkpoints/pi0_base/file?url=https://example.org'):
            self.assertFalse(allowed_path(path))



@unittest.skipUnless(importlib.util.find_spec('numpy') and importlib.util.find_spec('h5py'),
                     'Numerical image/schema tests run on the server, no local installation')
class Pi0Numerics(unittest.TestCase):
    def test_real_three_views_and_native_actions_roundtrip(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import numpy as np
        from momagen_pi0_schema import make_pi0_inputs,unpad_pi0_actions,STATE_NAMES,ACTION_NAMES
        self.assertEqual(len(STATE_NAMES),21);self.assertEqual(len(ACTION_NAMES),21)
        rng=np.random.default_rng(27)
        actions=rng.normal(size=(50,21)).astype('float32')
        data={'state':rng.normal(size=21).astype('float32'),'actions':actions,'prompt':'move and manipulate'}
        for i,key in enumerate(['head_image','left_wrist_image','right_wrist_image']):
            data[key]=np.full((256,256,4),i*50,dtype=np.uint8)
        converted=make_pi0_inputs(data)
        self.assertTrue(all(converted['image_mask'].values()))
        self.assertEqual(set(converted['image']),{'base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'})
        self.assertEqual(converted['image']['right_wrist_0_rgb'][0,0,0],100)
        self.assertEqual(converted['image']['base_0_rgb'].shape,(256,256,3))
        padded=np.pad(converted['actions'],((0,0),(0,11)))
        np.testing.assert_array_equal(unpad_pi0_actions(padded),actions)
        self.assertEqual(converted['prompt'],data['prompt'])
        self.assertNotIn('object::cup',converted)

    def test_cross_environment_rpc_keeps_real_images_and_proprio_exact(self):
        import numpy as np
        sys.path.insert(0,str(ROOT/'scripts'))
        from pi0_rpc import encode_observation,decode_observation
        rng=np.random.default_rng(53)
        inputs={'state':rng.normal(size=21).astype('float32'),'prompt':'pick and transport'}
        for key in ('head_image','left_wrist_image','right_wrist_image'):
            inputs[key]=rng.integers(0,256,size=(256,256,3),dtype=np.uint8)
        restored=decode_observation(encode_observation(inputs))
        for key in ('state','head_image','left_wrist_image','right_wrist_image'):
            np.testing.assert_array_equal(restored[key],inputs[key])
        broken=encode_observation(inputs);broken['left_wrist_image']['shape']=[2,2,3]
        with self.assertRaises(ValueError):decode_observation(broken)

    def test_native_hdf5_checks_real_synchronized_views(self):
        import tempfile
        import h5py
        import numpy as np
        sys.path.insert(0,str(ROOT/'scripts'))
        from convert_momagen_pi0 import inspect_episode
        from momagen_pi0_schema import camera_keys
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory)/'source.hdf5','w') as file:
                group=file.create_group('demo_0');group.create_dataset('actions',data=np.zeros((32,21),np.float32));obs=group.create_group('obs')
                obs.create_dataset('prop_state',data=np.zeros((32,21),np.float32))
                for camera in ['eyes','left_eef_link','right_eef_link']:
                    ds=obs.create_dataset(f'robot_r1::robot_r1:{camera}:Camera:0::rgb',shape=(32,256,256,4),dtype='u1')
                    sample=np.zeros((256,256,4),np.uint8);sample[:128]=200
                    ds[0]=sample;ds[16]=sample;ds[31]=sample
                cameras,actions,_,digest=inspect_episode(group)
                self.assertEqual(len(cameras),3);self.assertEqual(actions.shape,(32,21));self.assertEqual(len(digest),64)
                del obs[cameras['right_wrist_image']]
                with self.assertRaises(ValueError):camera_keys(obs)


if __name__=='__main__':unittest.main()
