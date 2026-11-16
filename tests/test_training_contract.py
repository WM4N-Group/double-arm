"""Lightweight safeguards; numerical BC tests run only where torch is installed."""
import ast
import importlib.util
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


class EvaluationContract(unittest.TestCase):
    def setUp(self):
        self.tree = ast.parse((ROOT / 'scripts/evaluate_state_policy.py').read_text())

    def test_no_state_replay_in_control_loop(self):
        loads = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == 'load_state']
        self.assertEqual(len(loads), 1)
        control = [n for n in ast.walk(self.tree) if isinstance(n, ast.For)
                   and isinstance(n.target, ast.Name) and n.target.id == 'step']
        self.assertEqual(len(control), 1)
        self.assertFalse(any(n is loads[0] for n in ast.walk(control[0])))

    def test_evaluation_does_not_read_teacher_actions(self):
        source = (ROOT / 'scripts/evaluate_state_policy.py').read_text()
        self.assertNotIn("['actions']", source)
        self.assertNotIn('["actions"]', source)
        self.assertIn("meta['env_kwargs']['init_curobo']=False", source)

    def test_code_sync_excludes_binary_robot_assets(self):
        import tempfile
        sys.path.insert(0,str(ROOT / 'scripts'))
        from code_sync_manifest import lightweight
        with tempfile.TemporaryDirectory() as tmp:
            mesh=pathlib.Path(tmp)/'base_link.STL'; mesh.write_bytes(b'solid')
            code=pathlib.Path(tmp)/'train.py'; code.write_text('pass')
            huge=pathlib.Path(tmp)/'huge.txt'
            with huge.open('wb') as f: f.truncate(1024*1024+1)
            self.assertFalse(lightweight(mesh))
            self.assertFalse(lightweight(huge))
            self.assertTrue(lightweight(code))

    def test_observation_adapter_receives_obs_info_pair(self):
        calls=[n for n in ast.walk(self.tree) if isinstance(n,ast.Call)
               and isinstance(n.func,ast.Attribute) and n.func.attr=='get_obs_IL']
        kwargs=[kw.value for n in calls for kw in n.keywords if kw.arg=='di']
        self.assertTrue(kwargs)
        self.assertTrue(all(isinstance(v,ast.Tuple) and len(v.elts)==2 for v in kwargs))

    def test_usd_import_occurs_after_environment_launch(self):
        source=(ROOT/'scripts/evaluate_state_policy.py').read_text()
        self.assertGreater(source.index('from pxr import Gf'),source.index('env=EnvOmniGibson('))

    def test_heldout_split_is_by_episode(self):
        source = (ROOT / 'scripts/train_state_policy.py').read_text()
        self.assertIn('train = [e for e in episodes if e[0] != val_name]', source)
        self.assertIn('val = [e for e in episodes if e[0] == val_name]', source)
        self.assertIn("raise ValueError('Need >=3 successful episodes", source)


class CampaignRestartContract(unittest.TestCase):
    def test_round_allocation_preserves_interrupted_rounds(self):
        import tempfile
        sys.path.insert(0, str(ROOT / 'scripts'))
        from continue_learning_campaign import next_round_directory
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            (root / 'continued-round-1').mkdir()
            sentinel = root / 'continued-round-1' / 'keep.txt'
            sentinel.write_text('preserve')
            (root / 'continued-round-3').mkdir()
            index, path = next_round_directory(root)
            self.assertEqual(index, 4)
            self.assertTrue(path.is_dir())
            self.assertEqual(sentinel.read_text(), 'preserve')
            self.assertEqual(next_round_directory(root)[0], 5)

    def test_generation_seed_avoids_prior_campaign_batches(self):
        import tempfile
        import json
        sys.path.insert(0, str(ROOT / 'scripts'))
        from continue_learning_campaign import next_generation_seed
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            campaign = root/'runs/tidy-learning-campaign-old';campaign.mkdir(parents=True)
            (campaign/'continued-gpu1-seed-400.log').touch()
            batch = root/'runs/tidy_table-data-old';batch.mkdir()
            (batch/'generation-request.json').write_text(json.dumps({'seed': 700}))
            self.assertEqual(next_generation_seed(root, {'gpu1_seed':300}, {}), 800)


class PhysicalGraspContract(unittest.TestCase):
    def test_mode_switch_requires_empty_hands_and_no_constraints(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from grasping_validation import activate_physical_after_reset
        class Robot:
            _grasping_mode='assisted'
            _ag_obj_in_hand={'left':None,'right':None}
            _ag_obj_constraints={'left':None,'right':None}
            @property
            def grasping_mode(self):return self._grasping_mode
        robot=Robot();activate_physical_after_reset(robot)
        self.assertEqual(robot.grasping_mode,'physical')
        for attribute in ('_ag_obj_in_hand','_ag_obj_constraints'):
            blocked=Robot();setattr(blocked,attribute,{'left':object(),'right':None})
            with self.assertRaises(ValueError):activate_physical_after_reset(blocked)
            self.assertEqual(blocked.grasping_mode,'assisted')

    def test_publisher_rejects_assisted_success(self):
        source=(ROOT/'scripts/continue_learning_campaign.py').read_text()
        self.assertIn("result.get('grasping_mode')!='physical'",source)
        self.assertIn("result.get('assisted_constraint_seen') is not False",source)


class SingleGPUAllocationContract(unittest.TestCase):
    def test_only_the_allocated_gpu_can_be_selected(self):
        import tempfile,json
        sys.path.insert(0,str(ROOT/'scripts'))
        from resource_allocation import resolve_gpu,require_dual_gpu_allocation,ALLOWED_GPU_UUID
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'runs').mkdir()
            (root/'runs/RESOURCE_ALLOCATION.json').write_text(json.dumps({'max_gpu_count':1,'allowed_gpu_index':1,'allowed_gpu_uuid':ALLOWED_GPU_UUID}))
            self.assertEqual(resolve_gpu(root),ALLOWED_GPU_UUID)
            self.assertEqual(resolve_gpu(root,'1'),ALLOWED_GPU_UUID)
            self.assertEqual(resolve_gpu(root,ALLOWED_GPU_UUID),ALLOWED_GPU_UUID)
            for requested in ('0','0,1','GPU-reserved'):
                with self.assertRaises(ValueError):resolve_gpu(root,requested)
            with self.assertRaises(ValueError):require_dual_gpu_allocation(root)

    def test_missing_or_malformed_allocation_never_defaults_to_gpu_zero(self):
        import tempfile,json
        sys.path.insert(0,str(ROOT/'scripts'))
        from resource_allocation import resolve_gpu,ALLOWED_GPU_UUID
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'runs').mkdir()
            with self.assertRaises(ValueError):resolve_gpu(root)
            path=root/'runs/RESOURCE_ALLOCATION.json'
            for config in ({},[],{'max_gpu_count':2},{'max_gpu_count':True},
                           {'max_gpu_count':1,'allowed_gpu_index':0,'allowed_gpu_uuid':ALLOWED_GPU_UUID},
                           {'max_gpu_count':1,'allowed_gpu_index':1,'allowed_gpu_uuid':'GPU-foreign'}):
                path.write_text(json.dumps(config))
                with self.subTest(config=config),self.assertRaises(ValueError):resolve_gpu(root)
            path.write_text('{malformed')
            with self.assertRaises(ValueError):resolve_gpu(root)

    def test_old_dual_controller_cannot_revive_from_missing_or_changed_config(self):
        import tempfile,json
        sys.path.insert(0,str(ROOT/'scripts'))
        from resource_allocation import require_dual_gpu_allocation
        with tempfile.TemporaryDirectory() as directory:
            root=pathlib.Path(directory);(root/'runs').mkdir()
            with self.assertRaises(ValueError):require_dual_gpu_allocation(root)
            (root/'runs/RESOURCE_ALLOCATION.json').write_text(json.dumps({'max_gpu_count':2}))
            with self.assertRaises(ValueError):require_dual_gpu_allocation(root)

    def test_object_pose_audit_observes_but_does_not_move_objects(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        from object_pose_audit import ObjectPoseAudit
        class Object:
            position=0
            def set_position_orientation(self,position):self.position=position
        obj=Object();audit=ObjectPoseAudit({'cup':obj})
        self.assertEqual(obj.position,0);self.assertEqual(audit.events,[])
        audit.step=7;obj.set_position_orientation(position=3)
        self.assertEqual(obj.position,3);self.assertEqual(audit.events[0]['step'],7)
        audit.close();obj.set_position_orientation(position=4)
        self.assertEqual(len(audit.events),1)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('h5py'),
                     'Numerical tests are server-only; no local training dependencies')
class StatePolicyNumerics(unittest.TestCase):
    def test_physical_success_writer_keeps_only_one_reset_state(self):
        import tempfile,json
        import numpy as np
        import h5py
        sys.path.insert(0,str(ROOT/'scripts'))
        from physical_demo_writer import append_physical_success
        from train_state_policy import load_episodes
        n=40;obs={'prop_state':np.zeros((n,21)),'prop_eef_basepose':np.zeros((n,38)),
            'base_link_pose':np.zeros((n,7)),'base_qpos':np.zeros((n,3)),
            'grasp_left':np.ones((n,1)),'grasp_right':-np.ones((n,1)),
            'task::teacup.n.02_1':np.zeros((n,7))}
        report={'sustained_success':True,'grasped_ever':[True,False],'base_path_length_m':1.0}
        proof={'grasping_mode':'physical','assisted_constraint_seen':False,'object_pose_write_count':0}
        with tempfile.TemporaryDirectory() as directory:
            output=pathlib.Path(directory)/'physical.hdf5'
            with self.assertRaises(ValueError):
                append_physical_success(output,'demo_0',np.zeros(12),obs,np.zeros((n,21)),{'env_name':'task'},report,{**proof,'grasping_mode':'assisted'})
            self.assertFalse(output.exists())
            append_physical_success(output,'demo_0',np.zeros(12),obs,np.zeros((n,21)),{'env_name':'task'},report,proof)
            with h5py.File(output) as file:
                self.assertEqual(file['data/demo_0/initial_state'].shape,(12,))
                self.assertNotIn('states',file['data/demo_0'])
                self.assertEqual(len(file['data/demo_0/actions']),n)
            episodes,_,_=load_episodes(output);self.assertEqual(len(episodes),1)

    def test_network_and_unique_feature_schema(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import numpy as np
        import torch
        from state_policy_features import feature_keys, vectorize, is_small
        from train_state_policy import ChunkPolicy, JOINT, GRIP, joint_anchor
        obs={'prop_eef_basepose':np.zeros(38),'grasp_left':np.array([-1]),'grasp_right':np.array([1])}
        for i in range(1,4): obs[f'task::plate.n.04_{i}']=np.full(7,i)
        keys=feature_keys(obs); vec=vectorize(obs,keys)
        self.assertEqual(vec.shape,(61,))
        self.assertEqual(len([k for k in keys if k.startswith('task::plate')]),3)
        self.assertFalse(is_small(np.zeros((256,256,3))))
        self.assertEqual(sorted(JOINT+GRIP+list(range(3))),list(range(21)))
        anchor=joint_anchor(np.arange(21,dtype=np.float32))
        np.testing.assert_array_equal(anchor[14:20],np.arange(13,19))
        np.testing.assert_array_equal(anchor[3:13],np.arange(3,13))
        self.assertEqual(anchor[13],0)
        self.assertEqual(anchor[0],0)
        model=ChunkPolicy(len(vec),history=4,horizon=16,hidden=32)
        out=model(torch.zeros(2,4,len(vec)))
        self.assertEqual(tuple(out.shape),(2,16,21))
        out.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_multimodal_policy_loss_and_decoding(self):
        sys.path.insert(0,str(ROOT / 'scripts'))
        import torch
        from train_state_policy import ChunkPolicy
        policy=ChunkPolicy(68,history=4,horizon=16,hidden=32,mixtures=3)
        obs=torch.zeros(2,4,68);target=torch.zeros(2,16,21)
        target[:,:,[13,20]]=1
        loss=policy.mixture_loss(obs,target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters()))
        self.assertEqual(tuple(policy(obs).shape),(2,16,21))

    def test_recurrent_sequence_matches_stepwise_inference_and_reset(self):
        sys.path.insert(0,str(ROOT / 'scripts'))
        import torch
        from train_recurrent_policy import RecurrentPolicy
        torch.manual_seed(3)
        model=RecurrentPolicy(5,hidden=8,layers=2).eval()
        x=torch.randn(1,8,5)
        with torch.no_grad():
            expected=model.forward_sequence(x)
            model.reset_memory();actual=torch.cat([model(x[:,i:i+1]) for i in range(x.shape[1])],dim=1)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
            model.reset_memory();torch.testing.assert_close(model(x[:,:1]),expected[:,:1],atol=1e-6,rtol=1e-5)

    def test_neural_world_goal_policy_forward_and_gradient(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        import torch
        from train_goal_policy import GoalPolicy
        model=GoalPolicy(68,history=4,hidden=32)
        result=model(torch.randn(3,4,68))
        self.assertEqual(tuple(result.shape),(3,1,22))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_memory_policy_has_no_trajectory_order_or_cursor(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        import torch
        from train_memory_policy import NearestStatePolicy
        torch.manual_seed(5)
        centers=torch.randn(50, 12);targets=torch.randn(50, 22)
        query=centers[[1, 30, 9]].reshape(3, 4, 3)
        model=NearestStatePolicy(centers, targets)
        torch.testing.assert_close(model(query)[:,0], targets[[1, 30, 9]])
        order=torch.randperm(50)
        torch.testing.assert_close(model(query), NearestStatePolicy(centers[order],targets[order])(query))
        self.assertFalse(hasattr(model, 'cursor'))

    def test_world_goal_residual_anchor_matches_native_motor_layout(self):
        import numpy as np
        sys.path.insert(0,str(ROOT/'scripts'))
        from policy_action_codecs import world_goal_anchor, encode_world_base_goals
        from train_state_policy import joint_anchor
        rng=np.random.default_rng(13);prop=rng.normal(size=(5,21)).astype('float32')
        motor=joint_anchor(prop);poses=np.zeros((5,7),np.float32);poses[:,:3]=rng.normal(size=(5,3));poses[:,6]=1
        heading=rng.uniform(-np.pi,np.pi,size=5).astype('float32')
        np.testing.assert_allclose(world_goal_anchor(poses,heading,motor),
            encode_world_base_goals(motor,poses,heading),atol=1e-7)
        self.assertEqual(world_goal_anchor(poses[0],heading[0],motor[0]).shape,(22,))

    def test_noisy_anchor_correction_preserves_absolute_supervision(self):
        import numpy as np
        import torch
        sys.path.insert(0,str(ROOT/'scripts'))
        from policy_action_codecs import world_goal_anchor
        from train_feedback_policy import anchor_noise_delta
        from train_state_policy import joint_anchor
        rng=np.random.default_rng(19);raw=rng.normal(size=(8,68)).astype('float32');noise=rng.normal(scale=.01,size=raw.shape).astype('float32')
        def reference(x):
            poses=np.zeros((len(x),7),np.float32);poses[:,:2]=x[:,:2];poses[:,6]=1
            prop=np.zeros((len(x),21),np.float32);prop[:,3:13]=x[:,6:16];prop[:,13:19]=x[:,24:30]
            return world_goal_anchor(poses,x[:,2],joint_anchor(prop))
        expected=reference(raw+noise)-reference(raw)
        actual=anchor_noise_delta(torch.tensor(raw),torch.tensor(noise)).numpy()
        np.testing.assert_allclose(actual,expected,atol=3e-7)
        np.testing.assert_allclose((reference(raw)+2-actual)+reference(raw+noise),
            (reference(raw)+2)+reference(raw),atol=7e-7)

    def test_world_base_goal_codec_roundtrip(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import numpy as np
        from policy_action_codecs import encode_world_base_goals, decode_world_base_goals
        rng = np.random.default_rng(8)
        actions = rng.normal(0, .03, size=(30, 21))
        poses = np.zeros((30, 7));poses[:, :3] = rng.normal(size=(30, 3))
        yaw = rng.uniform(-np.pi, np.pi, size=30)
        poses[:, 5] = np.sin(yaw/2);poses[:, 6] = np.cos(yaw/2)
        headings = rng.uniform(-np.pi, np.pi, size=30)
        goals = encode_world_base_goals(actions, poses, headings)
        self.assertEqual(goals.shape, (30, 22))
        actual = np.concatenate([decode_world_base_goals(goals[i:i+1], poses[i], headings[i])
                                 for i in range(len(actions))])
        np.testing.assert_allclose(actual, actions, atol=3e-7, rtol=1e-5)

    def test_normalized_kernel_forward_shape_and_finiteness(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import torch
        from train_rbf_policy import RBFPolicy
        model = RBFPolicy(torch.randn(16, 20), torch.randn(16, 22), .7, normalized=True)
        output = model(torch.randn(2, 4, 5))
        self.assertEqual(tuple(output.shape), (2, 1, 22))
        self.assertTrue(torch.isfinite(output).all())

    def test_kernel_fit_is_invariant_to_training_row_order(self):
        sys.path.insert(0,str(ROOT / 'scripts'))
        import torch
        from train_rbf_policy import RBFPolicy
        torch.manual_seed(4);x=torch.randn(24,5,dtype=torch.float64);y=torch.randn(24,21,dtype=torch.float64)
        def fit(x,y):
            k=torch.exp(-torch.cdist(x,x).square()/(2*.7**2));k.diagonal().add_(1e-4)
            return RBFPolicy(x,torch.linalg.solve(k,y),.7)
        order=torch.randperm(len(x));query=torch.randn(2,1,5,dtype=torch.float64)
        torch.testing.assert_close(fit(x,y)(query),fit(x[order],y[order])(query),atol=1e-6,rtol=1e-5)


if __name__ == '__main__': unittest.main()
