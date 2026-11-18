"""Pure geometry checks for reporting X7 held-body sweep bounds."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS:
    import numpy as np
    from audit_x7_held_load_sweep import CORNER,local_aabb,union_bounds
    from run_x7_turning_carry import r2


@unittest.skipUnless(HAS,'X7 runtime dependencies unavailable')
class HeldLoadSweepTests(unittest.TestCase):
    def test_dock_frame_contains_all_transformed_world_corners(self):
        bounds=np.array([[2.,1.,.5],[3.,1.2,1.5]]); frame=np.array([1.,.5,.42]); got=local_aabb(bounds,frame)
        pts=bounds[0]+CORNER*(bounds[1]-bounds[0]); pts[:,:2]=(pts[:,:2]-frame[:2])@r2(frame[2])
        np.testing.assert_allclose(got,[pts.min(axis=0),pts.max(axis=0)])
        self.assertGreater(got[1,0]-got[0,0],.90)

    def test_union_includes_link_and_time_extrema(self):
        v=np.array([[[[-1,0,0],[.3,1,2]],[[0,-1,1],[.8,.1,3]]],
                    [[[-.6,-.34,0],[.8,1.6,2]],[[0,.5,1],[1.,2,2]]]])
        np.testing.assert_array_equal(union_bounds(v),[[-1,-1,0],[1,2,3]])

    def test_malformed_physics_box_does_not_look_empty(self):
        with self.assertRaises(ValueError): local_aabb([[1,0,0],[0,1,1]],np.zeros(3))
        with self.assertRaises(ValueError): local_aabb([[0,np.nan,0],[1,1,1]],np.zeros(3))


if __name__=='__main__':unittest.main()
