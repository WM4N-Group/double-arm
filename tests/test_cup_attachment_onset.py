from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'Numerical fixtures run remotely')
class CupAttachmentOnsetTests(unittest.TestCase):
    def test_world_conversion_removes_robot_motion(self):
        from audit_cup_attachment_onset import root_relative_to_world
        q = 2 ** -.5
        root = [[0, 0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, q, q]]
        np.testing.assert_allclose(root_relative_to_world([[1, 1, 0], [0, -1, 0]], root), [[1, 1, 0]] * 2, atol=1e-12)
        with self.assertRaises(ValueError):
            root_relative_to_world([[1, 0, 0]], [[0] * 7])

    def test_observation_boundary_unknown_not_zero_and_max_is_not_marker(self):
        from audit_cup_attachment_onset import first_true, onset_metrics
        cup = np.array([[0, 0, 0], [.008, 0, 0], [.009, 0, 0], [.009, 0, 0]])
        marker = first_true([False, False, True, True])
        m = onset_metrics(cup, marker, contacts=np.array([-1, 0, 2, 1]))
        self.assertAlmostEqual(m['cup_step_at_marker_mm'], 1.)
        self.assertAlmostEqual(m['max_cup_world_step_mm_within_3_samples'], 8.)
        self.assertIsNone(m['window'][0]['left_contact_links'])
        self.assertIsNone(m['window'][0]['cup_world_step_mm'])
        self.assertEqual(m['window'][1]['left_contact_links'], 0)
        self.assertEqual(onset_metrics(cup, first_true([False] * 4)), {'first_grasp_sample': None})
