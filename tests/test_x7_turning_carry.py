"""Route checks and a remote-only rotated station Bullet fixture."""
from importlib.util import find_spec
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS_RUNTIME=all(find_spec(n) is not None for n in ('numpy','scipy','PIL','pybullet'))
if HAS_RUNTIME:
    import numpy as np
    import pybullet as p
    from run_x7_turning_carry import LAYOUTS, WXY, dock_pose, drive_segment, r2, waypoints
    from x7_station_collision import compound_station, station_clearance, station_parts


@unittest.skipUnless(HAS_RUNTIME,'X7 runtime dependencies unavailable')
class RouteTests(unittest.TestCase):
    def test_simulated_base_reference_limits_and_exact_dock(self):
        for name,l in LAYOUTS.items():
            path=waypoints(l)
            with self.subTest(name=name):
                for (_,a),(_,b) in zip(path,path[1:]):
                    u=drive_segment(a,b)
                    np.testing.assert_allclose(u[-1,:3],b,atol=1e-12)
                    np.testing.assert_array_equal(u[-1,3:],np.zeros(3))
                    self.assertTrue(np.all(abs(u[:,3:])<WXY))
                np.testing.assert_allclose(path[-1][1],dock_pose(l))

    def test_docking_direction_expressed_in_rotated_station_frame(self):
        for l in LAYOUTS.values():
            a,b=[np.array(q) for _,q in waypoints(l)[-2:]]
            np.testing.assert_allclose(r2(b[2]).T@(b[:2]-a[:2]),[0.,-.59],atol=1.e-12)
            self.assertEqual(a[2],b[2])

    def test_holdout_pose_and_yaw_not_source_docks(self):
        held=[dock_pose(l) for l in LAYOUTS.values() if l['split']=='heldout']
        self.assertEqual(len(held),1)
        for l in LAYOUTS.values():
            if l['split']=='source':
                self.assertGreater(np.linalg.norm(held[0][:2]-dock_pose(l)[:2]),.15)
                self.assertGreater(abs(held[0][2]-dock_pose(l)[2]),.10)


@unittest.skipUnless(HAS_RUNTIME and os.environ.get('X7_REMOTE_TESTS')=='1',
                     'Actual Bullet fixtures must run only on the remote server')
class RotatedFixtureTest(unittest.TestCase):
    def test_all_rotated_rendered_station_parts_have_collision(self):
        client=p.connect(p.DIRECT); self.addCleanup(p.disconnect,client)
        dock=dock_pose(LAYOUTS['heldout-angle']); sw=r2(dock[2]);
        parts=station_parts(0.,-.4,1.25,[.5,.6,.7,1])
        station=compound_station(parts,position=[*dock[:2],0.],yaw=float(dock[2]))
        self.assertEqual(len(p.getVisualShapeData(station)),6)
        self.assertEqual(len(p.getCollisionShapeData(station,-1)),6)
        dims=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.002,.002,.002])
        for label,(x,y,z),_,_ in parts:
            with self.subTest(label=label):
                obj=p.createMultiBody(baseMass=1.,baseCollisionShapeIndex=dims,basePosition=[*(dock[:2]+sw@[x,y]),z])
                distances,_,_=station_clearance(obj,[station]); self.assertLess(distances[0],0.)
                p.removeBody(obj)


if __name__=='__main__':
    unittest.main()
