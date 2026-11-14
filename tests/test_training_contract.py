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

    def test_heldout_split_is_by_episode(self):
        source = (ROOT / 'scripts/train_state_policy.py').read_text()
        self.assertIn('train = [e for e in episodes if e[0] != val_name]', source)
        self.assertIn('val = [e for e in episodes if e[0] == val_name]', source)
        self.assertIn("raise ValueError('Need >=3 successful episodes", source)


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('h5py'),
                     'Numerical tests are server-only; no local training dependencies')
class StatePolicyNumerics(unittest.TestCase):
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


if __name__ == '__main__': unittest.main()
