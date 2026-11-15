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

    def test_assisted_diagnostics_cannot_be_labelled_as_physical_success(self):
        source=(ROOT/'scripts/run_pi0_evaluation.py').read_text()
        self.assertIn("choices=['physical','configured'],default='physical'",source)
        self.assertIn('physical_task_passed(result)',source)
        self.assertIn('author_config_task_success=',source)
        export=(ROOT/'scripts/export_pi0_review.py').read_text()
        self.assertIn('PI0 ASSISTED DIAGNOSTIC - NOT PHYSICAL SUCCESS',export)
        pipeline=(ROOT/'scripts/run_pi0_pipeline.py').read_text()
        self.assertIn("wait(a.gpu_prerequisite_pid,'waiting_for_allocated_gpu_prerequisite')",pipeline)

    def test_training_redirects_only_its_compilation_cache(self):
        source=(ROOT/'scripts/run_pi0_entrypoint.py').read_text()
        self.assertIn("if option=='jax_compilation_cache_dir':value=str(root/'cache/jax')",source)
        self.assertIn('finally:jax.config.update=original_update',source)
        self.assertIn("runpy.run_path(str(source/'scripts/train.py'),run_name='__main__')",source)

    def test_absolute_base_variant_is_label_only_and_keeps_visual_proprio_inputs(self):
        source=(ROOT/'scripts/pi0_r1_config.py').read_text()
        self.assertIn('use_quantile_norm=absolute',source)
        self.assertIn('pi0_r1_tidy_table_absolute_base_lora32',source)
        tree=ast.parse(source)
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='R1AbsoluteBaseInputs')
        self.assertIn("if 'actions' in data:",ast.get_source_segment(source,cls))
        self.assertIn('result=make_pi0_inputs(data)',ast.get_source_segment(source,cls))
        evaluation=(ROOT/'scripts/evaluate_pi0_r1.py').read_text()
        self.assertIn('action=absolute_base_to_native(target,pose)',evaluation)
        self.assertIn("info.get('action_representation','native')!=args.action_representation",evaluation)
        self.assertIn('np.clip(action[:3],base_lower,base_upper)',evaluation)
        pipeline=(ROOT/'scripts/run_pi0_pipeline.py').read_text()
        self.assertIn('--reuse-prepared-native-dataset',pipeline)
        self.assertIn("'PI0_ACTION_REPRESENTATION':a.action_representation",pipeline)

    def test_sampling_diagnostics_never_control_the_simulator_or_tune_on_holdout(self):
        source=(ROOT/'scripts/benchmark_pi0_sampling.py').read_text()
        self.assertIn("provenance.get('split')!='train'",source)
        self.assertNotIn('--holdout',source)
        self.assertIn('policy.infer(inputs,noise=noise)',source)
        self.assertNotIn('env.step(',source)
        diagnostic=(ROOT/'scripts/diagnose_pi0_actions.py').read_text()
        self.assertIn("norm['norm_stats']['actions']['std']",diagnostic)
        self.assertIn("'teacher_actions_sent_to_model':False",diagnostic)
        self.assertIn("'simulator_control_calls':0",diagnostic)
        service=(ROOT/'scripts/serve_pi0_r1.py').read_text()
        self.assertIn("'--denoising-steps',type=int,default=10",service)
        self.assertIn("choices=['gaussian','zero'],default='gaussian'",service)
        self.assertIn("policy.infer(observation,noise=noise)",service)

    def test_evaluation_exports_separately_labelled_review(self):
        source=(ROOT/'scripts/run_pi0_evaluation.py').read_text()
        self.assertIn('physical_task_passed(result)',source)
        self.assertIn("str(script/'export_pi0_review.py')",source)
        self.assertIn('review_export_exit_status=export_code',source)
        export=(ROOT/'scripts/export_pi0_review.py').read_text()
        self.assertIn('PI0 FAILED ATTEMPT - NOT A SUCCESS DEMO',export)
        self.assertIn('PI0 PHYSICAL SUCCESS - MANUAL REVIEW PENDING',export)
        self.assertIn('fingerprint_checkpoint(checkpoint)',export)
        self.assertNotIn('copytree(',export)

    def test_single_gpu_deployment_can_keep_pi0_on_cpu_without_model_substitution(self):
        source=(ROOT/'scripts/run_pi0_evaluation.py').read_text()
        self.assertIn("choices=['cpu','gpu'],default='cpu'",source)
        self.assertIn("10 if args.inference_platform=='cpu' else 5",source)
        self.assertIn("'--execute-steps',str(execute_steps)",source)
        self.assertIn('free.get(gpu,0)<22000',source)
        service=(ROOT/'scripts/serve_pi0_r1.py').read_text()
        self.assertIn("gpu if args.inference_platform=='gpu' else ''",service)
        self.assertIn("'cuda' if args.inference_platform=='gpu' else 'cpu'",service)
        self.assertIn('create_trained_policy(config,args.checkpoint,sample_kwargs=',service)
        evaluation=(ROOT/'scripts/evaluate_pi0_r1.py').read_text()
        self.assertIn("'real_time_control_claim':False",evaluation)
        self.assertIn("platform=info.get('inference_platform')",evaluation)

    def test_trained_inference_preflight_is_cpu_only_and_not_teacher_replay(self):
        source=(ROOT/'scripts/preflight_pi0_inference.py').read_text()
        self.assertIn("os.environ.get('CUDA_VISIBLE_DEVICES') != ''",source)
        self.assertIn("os.environ.get('JAX_PLATFORMS') != 'cpu'",source)
        self.assertIn('create_trained_policy(register_config(), args.checkpoint)',source)
        self.assertIn('prediction = policy.infer(inputs)',source)
        self.assertIn('actions.shape != (50, 21)',source)
        self.assertNotIn("['actions'][",source)
        self.assertNotIn('env.step(',source)
        self.assertIn("'not_task_evaluation': True",source)

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
    @unittest.skipUnless(importlib.util.find_spec('openpi'),'Official config test runs in isolated openpi environment')
    def test_absolute_quantile_override_survives_official_data_factory(self):
        from unittest.mock import patch
        from openpi import transforms
        from openpi.training import config as official
        from pi0_r1_config import register_config
        with patch.object(official.DataConfigFactory,'_load_norm_stats',return_value=None), \
             patch.object(official,'ModelTransformFactory',return_value=lambda model:transforms.Group()):
            absolute=register_config(action_representation='absolute_base')
            self.assertTrue(absolute.data.create(absolute.assets_dirs,absolute.model).use_quantile_norm)
            native=register_config(action_representation='native')
            self.assertFalse(native.data.create(native.assets_dirs,native.model).use_quantile_norm)

    def test_absolute_base_roundtrip_and_nonintegrating_target_hold(self):
        import numpy as np
        sys.path.insert(0,str(ROOT/'scripts'))
        from pi0_base_targets import native_to_absolute_base,absolute_base_to_native
        yaw=np.array([-.7,1.2,3.13,-3.13]);poses=np.zeros((4,7));poses[:,:2]=[[5,1.5],[2,3],[1,1],[1,1]]
        poses[:,5]=np.sin(yaw/2);poses[:,6]=np.cos(yaw/2)
        native=np.zeros((4,21));native[:,:3]=[[.003,-.002,.001],[.01,.005,-.001],[0,0,.03],[0,0,.02]]
        native[:,3:]=np.random.default_rng(1).normal(size=(4,18))
        targets=native_to_absolute_base(native,poses)
        np.testing.assert_allclose(absolute_base_to_native(targets,poses),native,atol=1e-6)
        pose=poses[0].copy();target=targets[0].copy();pose[:2]=target[:2]
        pose[5]=np.sin(target[2]/2);pose[6]=np.cos(target[2]/2)
        np.testing.assert_allclose(absolute_base_to_native(target,pose)[:3],0,atol=1e-6)

    def test_absolute_overlay_stops_at_episode_boundary_and_is_not_loaded_for_inference(self):
        import tempfile
        import numpy as np
        sys.path.insert(0,str(ROOT/'scripts'))
        from pi0_r1_config import R1AbsoluteBaseInputs,load_absolute_overlay
        inputs={'state':np.zeros(21,np.float32),'prompt':'task'}
        for key in ('head_image','left_wrist_image','right_wrist_image'):inputs[key]=np.zeros((256,256,3),np.uint8)
        inference=R1AbsoluteBaseInputs('/nonexistent-not-read')(inputs)
        self.assertNotIn('index',inference);self.assertNotIn('actions',inference)
        with tempfile.TemporaryDirectory() as directory:
            labels=np.arange(4*21,dtype=np.float32).reshape(4,21)
            np.save(Path(directory)/'train-actions.npy',labels);np.save(Path(directory)/'train-episode-ends.npy',np.array([1,1,3,3]))
            import json,hashlib
            provenance=Path(directory)/'provenance.json';provenance.write_text('{}')
            (Path(directory)/'manifest.json').write_text(json.dumps({'splits':{'train':{
                'frames':4,'source_provenance':str(provenance),
                'source_provenance_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest(),
                'label_sha256':hashlib.sha256(labels.tobytes()).hexdigest()}}}))
            data={**inputs,'actions':np.zeros((50,21),np.float32),'index':1}
            converted=R1AbsoluteBaseInputs(directory)(data)
            np.testing.assert_array_equal(converted['actions'],np.repeat(labels[1:2],50,axis=0))
            self.assertEqual(converted['state'].shape,(21,));self.assertNotIn('index',converted)
            load_absolute_overlay.cache_clear()

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
