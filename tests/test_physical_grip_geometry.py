from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_physical_grip_geometry import cup_in_eef
try:
    import numpy as np
except ImportError:np=None


@unittest.skipIf(np is None,'numpy numerical test runs on server')
class LocalCentroidTests(unittest.TestCase):
    def test_translation_identity_and_inverse_quaternion_rotation(self):
        pose=np.array([[1,2,3,0,0,0,1],[0,0,0,0,0,np.sqrt(.5),np.sqrt(.5)]],float)
        p=np.array([[2,4,6],[1,0,0]],float)
        np.testing.assert_allclose(cup_in_eef(p,pose),[[1,2,3],[0,-1,0]],atol=1e-12)
        pose[1,3:]*=-1;np.testing.assert_allclose(cup_in_eef(p,pose)[1],[0,-1,0],atol=1e-12)
    def test_unknown_or_nonunit_pose_rejected(self):
        pose=np.array([[0,0,0,0,0,0,2.]],float)
        with self.assertRaises(ValueError):cup_in_eef(np.zeros((1,3)),pose)
        pose[0,6]=np.nan
        with self.assertRaises(ValueError):cup_in_eef(np.zeros((1,3)),pose)


if __name__=='__main__':unittest.main()
