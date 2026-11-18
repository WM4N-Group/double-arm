from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from freeze_pi0_multitrajectory_pilot import select
from plan_expert_physical_pilot import PANEL_NAMESPACE


class MultitrajectoryPilotTest(unittest.TestCase):
    def fixture(self):
        shards = [f'/mnt/yuanyue/double-arm/runs/{s}' for s in ('a', 'b', 'c', 'd')]
        groups = [[dict(source=f'{shard}/generated/task/demo.hdf5',
                        fingerprint=f'{3*i+j:064x}', seed=10+i,
                        source_grasping_mode='assisted', frames=15)
                   for j in range(3)] for i, shard in enumerate(shards)]
        entries = [entry for group in groups for entry in group]
        p = dict(namespace=PANEL_NAMESPACE, state='FROZEN_CPU_PANEL_NOT_LAUNCHED', episodes=[groups[0][0]])
        a = dict(mode='immutable_finalized_official_raw_pool_CPU_audit', source_grasping_mode='assisted', cleaning_applied=False,
                 episodes=entries, eligible_unique_episodes=len(entries), shards=dict.fromkeys(shards),
                 old_split_overlap_checks=[dict(split='train',matching_raw_fingerprints=[groups[1][0]['fingerprint']]),
                                           dict(split='validation',matching_raw_fingerprints=[])])
        return a,p

    def test_deterministic_separate_shard_seed_and_exclude_probe(self):
        a,p = self.fixture()
        t,v,shard = select(a,p,salt='fixed',n_train=6,n_val=3)
        self.assertEqual((len(t),len(v)),(6,3))
        self.assertTrue({e['seed'] for e in t}.isdisjoint({e['seed'] for e in v}))
        self.assertTrue(all(Path(e['source']).parents[2].as_posix() == shard for e in v))
        self.assertNotIn(p['episodes'][0]['fingerprint'],{e['fingerprint'] for e in t+v})
        self.assertNotIn(a['old_split_overlap_checks'][0]['matching_raw_fingerprints'][0],{e['fingerprint'] for e in t+v})
        a['episodes'].reverse()
        self.assertEqual((t,v,shard),select(a,p,salt='fixed',n_train=6,n_val=3))

    def test_reject_impossible_grouped_split_not_fill_across_boundary(self):
        a,p = self.fixture()
        for e in a['episodes']:
            e['seed']=10
        with self.assertRaises(ValueError):
            select(a,p,salt='fixed',n_train=6,n_val=3)
