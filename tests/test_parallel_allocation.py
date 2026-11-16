from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from resource_allocation import ALLOWED_GPU_UUID, GPU_UUIDS_BY_INDEX, resolve_gpu, require_dual_gpu_allocation
from run_paper_generation_shards import generation_run, unique_new_episodes


class ParallelAllocationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'runs').mkdir()
        self.path = self.root / 'runs/RESOURCE_ALLOCATION.json'
        self.config = {'max_gpu_count': 1, 'allowed_gpu_index': 1, 'allowed_gpu_uuid': ALLOWED_GPU_UUID,
                       'dual_gpu_authorized': True, 'allowed_gpu_uuids': list(GPU_UUIDS_BY_INDEX.values()),
                       'allowed_gpu_indices': [0, 1], 'total_authorized_gpu_count': 2,
                       'parallel_authorization_text': '两张卡都放开给用，跑满',
                       'parallel_authorized_at': '2026-11-16T23:05:00+08:00'}
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.config))

    def test_primary_unchanged_and_explicit_secondary_works(self):
        self.assertEqual(resolve_gpu(self.root), ALLOWED_GPU_UUID)
        self.assertEqual(resolve_gpu(self.root, '1'), ALLOWED_GPU_UUID)
        self.assertEqual(resolve_gpu(self.root, '0'), GPU_UUIDS_BY_INDEX[0])
        self.assertEqual(resolve_gpu(self.root, GPU_UUIDS_BY_INDEX[0]), GPU_UUIDS_BY_INDEX[0])
        for requested in ('0,1', 'GPU-foreign'):
            with self.assertRaises(ValueError):
                resolve_gpu(self.root, requested)
        with self.assertRaises(ValueError):
            require_dual_gpu_allocation(self.root)

    def test_incomplete_authorization_never_grants_gpu_zero(self):
        for key in ('dual_gpu_authorized', 'allowed_gpu_uuids', 'allowed_gpu_indices',
                    'total_authorized_gpu_count', 'parallel_authorization_text', 'parallel_authorized_at'):
            original = self.config.pop(key)
            self.save()
            with self.subTest(key=key), self.assertRaises(ValueError):
                resolve_gpu(self.root, '0')
            self.config[key] = original

    def test_legacy_primary_contract_not_rewritten_to_two(self):
        self.config['max_gpu_count'] = 2
        self.save()
        with self.assertRaises(ValueError):
            resolve_gpu(self.root, '0')

    def test_unique_episodes_are_not_double_counted(self):
        seen = {'old'}
        records = [{'fingerprint': value} for value in ['old', 'new', 'new', 'other']]
        self.assertEqual(unique_new_episodes(records, seen), [{'fingerprint': 'new'}, {'fingerprint': 'other'}])

    def test_run_path_parser_does_not_read_big_logs_or_allow_escape(self):
        log = self.root / 'launcher.log'
        log.write_text('RUN_DIR=' + str(self.root / 'runs/test') + '\nrest\n')
        self.assertEqual(generation_run(log, self.root), self.root / 'runs/test')
        log.write_text('RUN_DIR=/tmp/outside\n')
        with self.assertRaises(ValueError):
            generation_run(log, self.root)


if __name__ == '__main__':
    unittest.main()
