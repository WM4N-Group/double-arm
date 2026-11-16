from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from continue_official_gpu1 import dependency_outcome, merge_pool_paths, COMPLETE


class OfficialGpu1FollowupTests(unittest.TestCase):
    def ready(self):
        return {'stage': COMPLETE, 'verified_unique_demos': 25, 'active_pid': 0,
                'policy_training': 'not_run', 'source_grasping_mode': 'assisted'}

    def test_in_progress_count_or_logs_are_not_completion(self):
        self.assertEqual(dependency_outcome({'stage': 'official_shard_generation', 'verified_unique_demos': 25}), 'pending')
        self.assertEqual(dependency_outcome(self.ready()), 'ready')

    def test_failure_or_incomplete_final_gate_cannot_launch_next_pool(self):
        for key, value in [('verified_unique_demos', 24), ('active_pid', 42), ('error', 'failed'),
                           ('policy_training', 'in_progress'), ('source_grasping_mode', 'physical')]:
            x = self.ready(); x[key] = value
            self.assertEqual(dependency_outcome(x), 'failed')
        self.assertEqual(dependency_outcome({'stage': 'generation_failed_or_interrupted'}), 'failed')

    def test_registry_keeps_previous_worker_paths_not_only_latest_pointer(self):
        self.assertEqual(merge_pool_paths(['/runs/old.json', '/runs/primary.json'],
                                         ['/runs/new.json', '/runs/primary.json']),
                         ['/runs/new.json', '/runs/old.json', '/runs/primary.json'])


if __name__ == '__main__':
    unittest.main()
