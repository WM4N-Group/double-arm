"""No simulator/torch dependency: a zero exit code must not count as task success."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('generation_result', ROOT / 'scripts/check_generation_result.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class GenerationQualityGate(unittest.TestCase):
    def test_zero_exit_without_success_is_failure(self):
        self.assertEqual(module.classify(0, 0, 1), 'insufficient_successful_demos')
        self.assertEqual(module.classify(0, 2, 3), 'insufficient_successful_demos')
        self.assertEqual(module.classify(0, 3, 3), 'passed')
        self.assertEqual(module.classify(124, 3, 3), 'interrupted_or_failed')

    def test_failed_stats_produce_failure_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); output = root / 'generated/task'; output.mkdir(parents=True)
            (output / 'important_stats.json').write_text(json.dumps({'num_attempts':8,'num_success':0,'num_failures':8}))
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/check_generation_result.py'),
                                     '--run-dir', str(root), '--exit-status', '0', '--required', '3'],
                                     capture_output=True, text=True)
            self.assertEqual(result.returncode, 6)
            report = json.loads((root / 'generation-result.json').read_text())
            self.assertEqual(report['state'], 'insufficient_successful_demos')
            self.assertEqual(report['training'], 'not_started')
            self.assertEqual(report['num_success'], 0)

    def test_native_config_is_not_patched_by_state_wrapper(self):
        text = (ROOT / 'scripts/run_official_generation.sh').read_text()
        self.assertIn('python -u momagen/scripts/generate_dataset.py', text)
        self.assertNotIn('generate_state_demos.py', text)
        self.assertIn("['generation']['guarantee'] is True", text)


if __name__ == '__main__': unittest.main()
