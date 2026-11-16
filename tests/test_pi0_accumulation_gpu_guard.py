"""No numerical or CUDA imports: gates must precede ALL model initialization."""
from pathlib import Path
import ast
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from preflight_pi0_accumulation_gpu import release_blockers, TERMINAL_STAGES, pinned_upstream_revision, OPENPI_REV
from resource_allocation import ALLOWED_GPU_UUID


class AccumulationReleaseTests(unittest.TestCase):
    def setUp(self):
        self.campaign = Path('/mnt/yuanyue/double-arm/runs/coverage')
        self.status = {'stage': 'coverage_trained_training_condition_still_failed',
                       'controller_pid': 42, 'active_pid': 0}
        self.pointer = {'controller_pid': 42, 'campaign': str(self.campaign),
                        'status_file': str(self.campaign / 'status.json'),
                        'gpu_uuid': ALLOWED_GPU_UUID, 'max_gpu_count': 1}

    def blockers(self, live=(), apps=()):
        return release_blockers(self.status, self.pointer, self.campaign, set(live), apps)

    def test_terminal_dead_workflow_and_empty_card_required(self):
        self.assertEqual(self.blockers(), [])
        for stage in TERMINAL_STAGES:
            self.status['stage'] = stage
            self.assertEqual(self.blockers(), [])

    def test_idle_gap_before_autoevaluation_is_not_release(self):
        for stage in ('native_coverage_finetuning', 'coverage-training-author-diagnostic',
                      'coverage-heldout-physical-evaluation', 'loading_pi0_for_visual_inference', None):
            self.status['stage'] = stage
            self.assertTrue(self.blockers())

    def test_terminal_stage_but_controller_or_child_alive_blocks(self):
        self.assertTrue(self.blockers(live=(42,)))
        self.status['active_pid'] = 43
        self.assertTrue(self.blockers(live=(43,)))
        self.pointer['controller_pid'] = 44
        self.assertTrue(self.blockers(live=(44,)))
        self.assertTrue(self.blockers())

    def test_new_job_pointer_cannot_be_ignored(self):
        self.pointer['campaign'] += '-new'
        self.assertTrue(self.blockers())

    def test_gpu1_occupied_blocks_but_reserved_gpu0_is_not_ours(self):
        self.assertTrue(self.blockers(apps=((ALLOWED_GPU_UUID, 100),)))
        self.assertEqual(self.blockers(apps=(('reserved-gpu0', 101),)), [])
        self.pointer['gpu_uuid'] = 'reserved-gpu0'
        self.assertTrue(self.blockers())

    def test_missing_controller_pid_fails_closed(self):
        del self.status['controller_pid']
        self.assertTrue(self.blockers())

    def test_pinned_archive_marker_is_required_and_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                pinned_upstream_revision(root)
            marker = root / 'UPSTREAM_REVISION'
            marker.write_text(OPENPI_REV+'\n')
            self.assertEqual(pinned_upstream_revision(root), OPENPI_REV)
            marker.write_text('unrelated-revision')
            with self.assertRaises(ValueError):
                pinned_upstream_revision(root)

    def test_module_import_does_not_import_numerical_runtime(self):
        path = Path(sys.modules[release_blockers.__module__].__file__)
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Import):
                self.assertFalse(any(x.name.split('.')[0] in ('jax', 'numpy', 'openpi', 'torch') for x in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], ('jax', 'numpy', 'openpi', 'torch'))


if __name__ == '__main__':
    unittest.main()
