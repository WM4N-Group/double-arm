import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi0_initialization_contract import training_initialization


class InitializationTests(unittest.TestCase):
    def test_default_remains_official_base_without_requiring_a_download_at_config_time(self):
        path,info=training_initialization('/unused',None,'native')
        self.assertEqual(path,Path('/unused/data/pi0/checkpoints/pi0_base/params'))
        self.assertEqual(info['kind'],'official_pretrained_pi0_base')
    def test_warm_start_requires_complete_native_source_and_records_fresh_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);c=root/'runs/pi0_r1_tidy_table_lora32/old/29999';params=c/'params';params.mkdir(parents=True)
            norm=c/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json';norm.parent.mkdir(parents=True);norm.write_text('{}')
            meta=c/'_CHECKPOINT_METADATA';meta.write_text(json.dumps({'commit_timestamp_nsecs':1}))
            path,info=training_initialization(root,str(params),'native')
            self.assertEqual(path,params);self.assertFalse(info['optimizer_state_restored']);self.assertEqual(info['source_step_index'],29999)
            with self.assertRaises(ValueError):training_initialization(root,str(params),'absolute_base')
            meta.write_text('{}')
            with self.assertRaises(ValueError):training_initialization(root,str(params),'native')
            with self.assertRaises(ValueError):training_initialization(root,'/elsewhere/params','native')
