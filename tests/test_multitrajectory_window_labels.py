from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_multitrajectory_window_labels import boundary_anchors, episode_indices

try:
    import numpy as np
except ImportError:
    np = None


class WindowLabelsTest(unittest.TestCase):
    def test_no_lag_and_short_episodes_anchor_indices(self):
        self.assertEqual(boundary_anchors(100, 0), [0, 99])
        self.assertEqual(boundary_anchors(100, 49), [0, 50, 51, 99])
        self.assertEqual(boundary_anchors(1, 49), [0])

    @unittest.skipIf(np is None, 'Numerical fixture runs remotely')
    def test_no_cross_episode_and_exact_tail_padding(self):
        for n, lag in ((4, 0), (4, 2), (4, 49), (1, 49)):
            indices, padded = episode_indices(n, lag)
            self.assertTrue((indices < n).all())
            self.assertEqual(int(padded.sum()), min(n, lag))
            self.assertTrue((indices[padded] == n-1).all())
