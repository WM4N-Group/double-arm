"""Offline checks that rear-return attribution cannot silently accept changed traces."""
from importlib.util import find_spec
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
HAS = all(find_spec(n) is not None for n in ('numpy', 'scipy', 'PIL', 'pybullet'))
if HAS:
    import numpy as np
    from verify_x7_rear_return_attribution import compare_trace, explain_event


@unittest.skipUnless(HAS, 'X7 runtime dependencies unavailable')
class RearAttributionTest(unittest.TestCase):
    def test_rejects_even_one_changed_motor_reference(self):
        with tempfile.TemporaryDirectory() as dirname:
            a, b = Path(dirname) / 'a.npz', Path(dirname) / 'b.npz'
            fields = dict(sensor_telemetry=np.zeros((1, 13)), actual_motor_references=np.array([[0., 1.]]))
            np.savez(a, **fields)
            np.savez(b, **fields)
            names, _ = compare_trace(a, b)
            self.assertEqual(names, list(fields))
            np.savez(b, **(fields | {'actual_motor_references': np.array([[0., 1. + 1e-12]])}))
            with self.assertRaisesRegex(ValueError, 'changed archived trace'):
                compare_trace(a, b)

    def test_masked_return_does_not_become_known_through_diagnostics(self):
        packet = {'channels': {'handles': {'calibrated_m': .315}}, 'fused_raw_m': -1.}
        event = {'substep': 32, 'sample_k': 16, 'event': 'rear_coverage_lost'}
        telemetry = np.zeros((1, 13))
        telemetry[0, [0, 3, 4, 8, 11]] = [32, 16, 16, -1., -1.]
        self.assertFalse(explain_event(packet, event, telemetry)['known'])
        telemetry[0, 11] = .20
        with self.assertRaisesRegex(ValueError, 'did not fail closed'):
            explain_event(packet, event, telemetry)
