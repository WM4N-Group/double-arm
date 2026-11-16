from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluation_split_contract import verify_evaluation_episode


class EvaluationSplitTests(unittest.TestCase):
    def test_source_and_demo_must_belong_to_declared_split(self):
        p={'split':'train','episodes':[{'source':'/run/demo.hdf5','source_demo':'demo_3','split':'train'}]}
        self.assertEqual(verify_evaluation_episode(p,'/run/demo.hdf5','demo_3','train'),p['episodes'][0])
        for dataset,demo,split in [('/run/demo.hdf5','demo_3','validation'),('/other/demo.hdf5','demo_3','train'),('/run/demo.hdf5','demo_0','train')]:
            with self.assertRaises(ValueError):verify_evaluation_episode(p,dataset,demo,split)
        p['episodes']*=2
        with self.assertRaises(ValueError):verify_evaluation_episode(p,'/run/demo.hdf5','demo_3','train')
