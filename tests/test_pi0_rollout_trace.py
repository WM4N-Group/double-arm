"""Exercise evaluator finalization without importing or launching a simulator."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
except ImportError:
    np=None


@unittest.skipIf(np is None,'Numerical fixtures run in isolated remote environments')
class RolloutTraceTests(unittest.TestCase):
    def test_abort_retains_only_completed_transitions_and_cannot_pass(self):
        from rollout_motion_metrics import cup_motion_metrics
        source=Path(__file__).resolve().parents[1]/'scripts/evaluate_pi0_r1.py'
        tree=ast.parse(source.read_text())
        function=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='finalize_episode')
        with tempfile.TemporaryDirectory() as directory:
            n=3;result={};saved=[]
            scope={'np':np,'json':json,'args':SimpleNamespace(demo='demo_0',evaluation_split='validation',output=Path(directory)),
                'cup_positions':[np.array([0.,0.,.1])]*n,'post_base_positions':[np.array([float(i),0.,0.]) for i in range(n)],
                'cup_grasp_flags':[True]*n,'eef_poses':[np.zeros((2,7))]*n,'cup_contact_counts':[np.zeros(2,dtype=int)]*n,'initial_cup_position':np.zeros(3),
                'states':[np.full(21,i) for i in range(n+1)],'actions':[np.full(21,i) for i in range(n+1)],
                'targets':[np.full(21,i) for i in range(n+1)],'proposals':[np.full(21,i+.25) for i in range(n+1)],'root_poses':[np.array([0.,0.,0.,0.,0.,0.,1.])]*(n+1),
                'path':[np.array([float(i),0.,0.]) for i in range(n+1)],
                'grasp':[True,False],'stable':30,'predicate':SimpleNamespace(goal_status='fixture'),
                'audit':SimpleNamespace(events=[]),'cup_grasped':True,'max_cup_lift':.1,
                'result':result,'rpc':[{}],'clipped':0,'base_clipped':0,'time':time,'started':time.monotonic(),
                'cup_motion_metrics':cup_motion_metrics,'save':lambda:saved.append(True)}
            exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),scope)
            scope['finalize_episode'](aborted=True)
            self.assertFalse(result['task_success'])
            self.assertTrue(result['episodes'][0]['aborted'])
            self.assertFalse(result['episodes'][0]['sustained_success'])
            self.assertEqual(result['episodes'][0]['steps'],n)
            self.assertEqual(saved,[True])
            with np.load(Path(directory)/'demo_0-trace.npz') as trace:
                for key in ('actions','observations','model_action_targets','model_proposed_targets','pre_step_root_poses','base_positions','post_step_eef_world_poses','post_step_cup_contact_finger_counts'):
                    self.assertEqual(len(trace[key]),n)
                np.testing.assert_array_equal(trace['actions'][-1],np.full(21,n-1))
            scope['cup_positions']=[];result.clear();saved.clear()
            scope['finalize_episode'](aborted=True)
            self.assertEqual(result,{})
            self.assertEqual(saved,[])
