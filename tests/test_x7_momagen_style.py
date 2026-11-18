"""Synthetic CPU-only checks; no simulator scene is launched."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
HAS_RUNTIME = all(find_spec(name) is not None for name in ('numpy', 'scipy', 'PIL', 'pybullet'))
if HAS_RUNTIME:
    import numpy as np
    from generate_x7_momagen_style import SPLITS, retrieve, segments, target_geometry


@unittest.skipUnless(HAS_RUNTIME, 'X7 runtime dependencies not available')
class X7RetargetContractTest(unittest.TestCase):
    def trace(self):
        t = dict(phase=np.arange(10), object_post=np.zeros((10,13)), object_pre=np.zeros((10,13)),
                 contact_samples=np.zeros((10,8,9)), base_post=np.zeros((10,3,2)),
                 on_table=np.zeros((10,2),dtype=int), gripper_post=np.zeros((10,4,2)))
        t['contact_samples'][2:7,:,:4] = .5
        t['object_post'][3,2] = .1
        t['base_post'][4,0,1] = .15
        t['gripper_post'][7,:,0] = .034
        t['on_table'][7,1] = 1
        return t

    def test_phase_and_measured_events_are_both_preserved(self):
        s = segments(self.trace())
        self.assertEqual([r['start_inclusive'] for r in s['segments']], [0,3,6])
        self.assertEqual([r['end_exclusive'] for r in s['segments']], [3,6,10])
        self.assertEqual(list(s['physical_event_control_indices'].values()), [2,3,4,7])

    def test_missing_real_grasp_rejects_even_with_scripted_phase(self):
        t = self.trace(); t['contact_samples'][2,:,3] = 0
        with self.assertRaisesRegex(ValueError, 'no required phase event'):
            segments(t)

    def test_closed_grippers_do_not_qualify_as_release(self):
        t = self.trace(); t['gripper_post'][7,:,0] = .012
        with self.assertRaises(ValueError):
            segments(t)

    def test_no_contact_with_B_does_not_qualify_as_release(self):
        t = self.trace(); t['on_table'][7,:] = [1,0]
        with self.assertRaises(ValueError):
            segments(t)

    def test_layout_and_seed_bands_do_not_overlap(self):
        geometry = {name:{i:target_geometry(i,name) for i in range(*band)} for name,band in SPLITS.items()}
        self.assertFalse(set(geometry['train']) & set(geometry['validation']))
        self.assertTrue(all(x[3] <= 2.632 or x[3] >= 2.71 for x in geometry['train'].values()))
        self.assertTrue(all(2.651<=x[3]<=2.689 for x in geometry['validation'].values()))

    def test_retrieves_one_source_for_both_arms_with_stable_tie_break(self):
        goal = (.589, -.388, .0, 2.65)
        early = {'seed':7, 'geometry':goal}
        late = {'seed':9, 'geometry':goal}
        self.assertIs(retrieve([late,early],goal),early)


if __name__=='__main__':
    unittest.main()
