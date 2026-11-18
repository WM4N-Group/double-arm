import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('random_raw', Path(__file__).parents[1] / 'scripts/export_random_raw_reviews.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RandomReviewTests(unittest.TestCase):
    def report(self):
        return {'mode': 'immutable_finalized_official_raw_pool_CPU_audit',
                'source_grasping_mode': 'assisted', 'eligible_unique_episodes': 10,
                'episodes': [{'fingerprint': f'{i:064x}', 'source_grasping_mode': 'assisted'} for i in range(10)]}

    def test_reproducible_unique_and_independent_of_entry_order(self):
        r = self.report()
        sample = m.select_entries(r, 5, 42)
        r['episodes'].reverse()
        self.assertEqual(sample, m.select_entries(r, 5, 42))
        self.assertEqual(len({e['fingerprint'] for e in sample}), 5)
        self.assertNotEqual(sample, m.select_entries(r, 5, 43))

    def test_reject_invalid_population(self):
        for key, value in [('mode', 'live_tmp'), ('source_grasping_mode', 'physical'), ('eligible_unique_episodes', 9)]:
            r = self.report(); r[key] = value
            with self.assertRaises(ValueError): m.select_entries(r, 5, 42)
        r = self.report(); r['episodes'][0] = r['episodes'][1]
        with self.assertRaises(ValueError): m.select_entries(r, 5, 42)
        with self.assertRaises(ValueError): m.select_entries(self.report(), 11, 42)
        r = self.report(); r['episodes'][0]['source_grasping_mode'] = 'unknown'
        with self.assertRaises(ValueError): m.select_entries(r, 5, 42)

    def test_frequency_from_metadata_not_assumed(self):
        self.assertEqual(m.recorded_fps({'env_kwargs': {'env': {'action_frequency': 20}}}), 20)
        for value in [0, -1, float('nan')]:
            with self.assertRaises(ValueError): m.recorded_fps({'env_kwargs': {'env': {'action_frequency': value}}})
