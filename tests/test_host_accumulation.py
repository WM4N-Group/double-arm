from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pi0_accumulation_preflight import HostGradientAccumulator
try:
    import numpy as np
except ImportError:
    np = None


class HostAccumulatorContractTests(unittest.TestCase):
    def test_invalid_window_is_rejected_before_numerical_imports(self):
        for n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                HostGradientAccumulator(n)


@unittest.skipIf(np is None, 'Numerical fixtures run on server CPU')
class HostAccumulatorNumericsTests(unittest.TestCase):
    def test_mean_then_clip_not_per_micro_clip(self):
        acc = HostGradientAccumulator(64)
        for x in [100., -99.] + [0.] * 62:
            acc.add([np.array([x], np.float32)])
        self.assertEqual(acc.count, 64)
        self.assertEqual(acc.nbytes, 4)
        np.testing.assert_array_equal(acc.mean()[0], np.array([1/64], np.float32))
        np.testing.assert_array_equal(acc.mean()[0], np.array([1/64], np.float32))
        with self.assertRaises(ValueError):
            acc.add([np.zeros(1, np.float32)])

    def test_no_mean_before_full_window_and_rejected_add_is_transactional(self):
        acc = HostGradientAccumulator(2)
        acc.add([np.ones(2, np.float32), np.ones(1, np.float32)])
        with self.assertRaises(ValueError):
            acc.mean()
        for bad in ([np.ones(3, np.float32), np.ones(1, np.float32)],
                    [np.ones(2, np.float32), np.array([float('nan')], np.float32)],
                    [np.ones(2, np.float64), np.ones(1, np.float32)]):
            with self.assertRaises(ValueError):
                acc.add(bad)
            self.assertEqual(acc.count, 1)
        acc.add([np.ones(2, np.float32) * 3, np.ones(1, np.float32) * 5])
        np.testing.assert_array_equal(acc.mean()[0], [2., 2.])
        np.testing.assert_array_equal(acc.mean()[1], [3.])


if __name__ == '__main__':
    unittest.main()
