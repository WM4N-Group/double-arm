from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
try:
    import numpy as np
except ImportError:np=None


@unittest.skipIf(np is None,'Numerical fixtures run remotely')
class GraspTelemetryTests(unittest.TestCase):
    def test_read_only_contacts_include_all_cup_links_and_unknown_is_not_zero(self):
        from grasp_telemetry import read_grasp_telemetry
        root=SimpleNamespace(prim_path='/cup/root');handle=SimpleNamespace(prim_path='/cup/handle')
        cup=SimpleNamespace(root_link=root,links={'root':root,'handle':handle})
        class Link:
            def get_position_orientation(self):return np.array([1,2,3]),np.array([0,0,0,1])
        class Robot:
            eef_links={'left':Link(),'right':Link()}
            def _find_gripper_contacts(self,arm):
                if arm=='right':raise RuntimeError('fixture unknown')
                return set(),{'/cup/root':{'finger1'},'/cup/handle':{'finger1','finger2'}}
        poses,counts,errors=read_grasp_telemetry(Robot(),cup)
        self.assertEqual(poses.shape,(2,7));self.assertEqual(counts.tolist(),[2,-1]);self.assertEqual(len(errors),1)
        self.assertTrue(np.isfinite(poses).all())
