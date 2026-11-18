from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_B_paired_attachment_replay import DONE_RAW, release_blockers


class PairedAttachmentRunnerReleaseTests(unittest.TestCase):
    def ready(self):
        gpu = 'uuid1'
        owner = dict(controller_pid=11, gpu_uuid=gpu, max_gpu_count=1,
                     role='official_assisted_paper_data_generation_not_policy_training')
        state = dict(controller_pid=11, active_pid=0, stage=DONE_RAW, target_unique_demos=100,
                     verified_unique_demos=100, source_grasping_mode='assisted', policy_training='not_run')
        return gpu, owner, state

    def test_requires_completed_dead_matching_owner_and_empty_card(self):
        gpu, owner, state = self.ready()
        self.assertEqual(release_blockers(owner, state, gpu, set(), {'uuid0'}), [])
        self.assertTrue(release_blockers(owner, state, gpu, {11}, set()))
        self.assertTrue(release_blockers(owner, state, gpu, set(), {gpu}))
        self.assertTrue(release_blockers(owner, {**state, 'stage':'validating_finalized_shard'}, gpu, set(), set()))
        self.assertTrue(release_blockers(owner, {**state, 'verified_unique_demos':25}, gpu, set(), set()))
        self.assertTrue(release_blockers(owner, {**state, 'controller_pid':12}, gpu, set(), set()))
        self.assertTrue(release_blockers({**owner, 'role':'training'}, state, gpu, set(), set()))
