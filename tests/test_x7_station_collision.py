"""Strict station geometry tests. Bullet fixtures are remote opt-in only."""
from importlib.util import find_spec
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
HAS_RUNTIME = all(find_spec(n) is not None for n in ('numpy','pybullet'))
if HAS_RUNTIME:
    import numpy as np
    import pybullet as p
    from x7_station_collision import (STATION_SCAN_M, clipped_bounds, compound_station,
                                      station_clearance, station_parts)


@unittest.skipUnless(HAS_RUNTIME,'X7 collision dependencies unavailable')
class GeometryTests(unittest.TestCase):
    def test_triangle_crossing_slab_without_an_internal_vertex(self):
        t=np.array([[[0,0,0],[2,2,0],[2,0,2]]],dtype=np.float32)
        lo,hi=clipped_bounds(t,.5,1.5)
        np.testing.assert_allclose(lo,[.5,0,0])
        np.testing.assert_allclose(hi,[1.5,1.5,1.5])
        self.assertIsNone(clipped_bounds(t,3.,4.))

    def test_supports_are_beyond_raw_chassis_footprint(self):
        parts=station_parts(0.,-.400,1.25,[.5,.6,.7,1])
        self.assertEqual(len(parts),6)
        for name,xyz,dims,_ in parts:
            if name.endswith('_leg') or name=='rail':
                self.assertLess(xyz[1]+dims[1]/2,-.732)


@unittest.skipUnless(HAS_RUNTIME and os.environ.get('X7_REMOTE_TESTS')=='1',
                     'Actual engine fixtures must only run on the remote server')
class ActualEngineTests(unittest.TestCase):
    def setUp(self):
        self.client=p.connect(p.DIRECT)
        self.addCleanup(p.disconnect,self.client)
        self.parts=station_parts(0.,-.400,1.25,[.5,.6,.7,1])
        self.station=compound_station(self.parts)

    def test_every_visible_structural_part_has_collision_shape(self):
        self.assertEqual(len(p.getVisualShapeData(self.station)),len(self.parts))
        self.assertEqual(len(p.getCollisionShapeData(self.station,-1)),len(self.parts))
        probe=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.003]*3)
        for name,center,_,_ in self.parts:
            with self.subTest(part=name):
                obj=p.createMultiBody(baseMass=1.,baseCollisionShapeIndex=probe,basePosition=center)
                d,hits,_=station_clearance(obj,[self.station])
                self.assertLess(d[0],0.)
                self.assertEqual(hits,[0])  # no simulation step; signed query must catch it independently
                p.removeBody(obj)

    def test_no_hit_is_reported_as_censored_not_infinite(self):
        sh=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.01]*3)
        obj=p.createMultiBody(baseMass=1.,baseCollisionShapeIndex=sh,basePosition=[-5.,0.,1.])
        d,contacts,link=station_clearance(obj,[self.station])
        self.assertEqual((d,contacts,link),([STATION_SCAN_M],[0],[-2]))


if __name__=='__main__':
    unittest.main()
