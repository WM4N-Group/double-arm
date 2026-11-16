from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_official_pool_snapshot import select_unique_records


class OfficialPoolSnapshotTests(unittest.TestCase):
    def test_cross_worker_duplicates_and_old_split_overlap_not_double_counted(self):
        records = [{'fingerprint': x * 64, 'source': src} for x, src in
                   [('a', 'worker0'), ('b', 'worker0'), ('a', 'worker1'), ('c', 'worker1')]]
        selected, duplicates, excluded = select_unique_records(records, {'b' * 64})
        self.assertEqual([e['fingerprint'] for e in selected], ['a' * 64, 'c' * 64])
        self.assertEqual((duplicates, excluded), (1, 1))
        self.assertEqual(selected[0]['source'], 'worker0')

    def test_missing_or_bad_fingerprint_is_not_an_independent_episode(self):
        for entry in ({}, {'fingerprint': 'not-a-sha'}, {'fingerprint': None}):
            with self.assertRaises(ValueError):
                select_unique_records([entry], set())


if __name__ == '__main__':
    unittest.main()
