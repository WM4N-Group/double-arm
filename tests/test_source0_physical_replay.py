from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_source0_physical import dependency_released


class PhysicalSourceDependencyTests(unittest.TestCase):
    def good(self):return {'stage':'train-physical-diagnostic_FAILED','active_pid':0,'training':'completed'}
    def test_empty_gpu_between_training_and_eval_is_not_release(self):
        self.assertTrue(dependency_released(self.good(),False))
        self.assertFalse(dependency_released(self.good(),True))
        for field,value in [('stage','train-physical-diagnostic'),('active_pid',42),('training','in_progress')]:
            s=self.good();s[field]=value;self.assertFalse(dependency_released(s,False))
    def test_assisted_pass_alone_never_releases_B_chain(self):
        self.assertFalse(dependency_released({'stage':'train-author-diagnostic','active_pid':0,'training':'completed'},False))


if __name__=='__main__':unittest.main()
