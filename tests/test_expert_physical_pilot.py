import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from plan_expert_physical_pilot import panel


class ExpertPhysicalPilotTest(unittest.TestCase):
    def fixture(self):
        shards = [f'/mnt/yuanyue/double-arm/runs/test-{i}' for i in range(3)]
        episodes = [dict(source=s+'/generated/data/demo.hdf5', fingerprint=f'{i:064x}',
                         source_grasping_mode='assisted', frames=20)
                    for i, s in enumerate(shards * 2)]
        return dict(mode='immutable_finalized_official_raw_pool_CPU_audit', source_grasping_mode='assisted',
                    dataset_modified=False, split_created=False, eligible_unique_episodes=len(episodes),
                    shards=dict.fromkeys(shards), episodes=episodes)

    def test_distinct_shard_reproducible_independent_of_input_order(self):
        report = self.fixture()
        a = panel(report, 'a'*64, 3)
        report['episodes'].reverse()
        self.assertEqual(a, panel(report, 'a'*64, 3))
        self.assertEqual(3, len({e['shard'] for e in a}))

    def test_require_clean_frozen_population_and_distinct_shards(self):
        report = self.fixture()
        with self.assertRaises(ValueError):
            panel(report, 'b'*64, 4)
        report['episodes'][0]['fingerprint'] = report['episodes'][1]['fingerprint']
        with self.assertRaises(ValueError):
            panel(report, 'b'*64, 2)
