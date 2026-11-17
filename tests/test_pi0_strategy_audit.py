from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_pi0_training_strategy import probe_indices


class StrategyAuditTests(unittest.TestCase):
    def test_pretrained_control_can_reuse_exact_production_sampling_protocol(self):
        from benchmark_pi0_sampling import sampling_variants
        self.assertEqual(sampling_variants(True),((10,'fixed_gaussian'),))
        self.assertEqual(len(sampling_variants(False)),4)

    def test_sampling_covers_onset_and_episode_clamp_boundaries(self):
        self.assertEqual(probe_indices(100, 40), [0,39,40,50,98,99])
        self.assertEqual(probe_indices(100, None), [0,50,98,99])
        self.assertEqual(probe_indices(30, 0), [0,15,28,29])

    def test_invalid_source_indices_are_not_silently_clamped(self):
        for length, close in ((2,0),(100,100),(100,-1)):
            with self.assertRaises(ValueError):
                probe_indices(length,close)


if __name__ == '__main__':
    unittest.main()
