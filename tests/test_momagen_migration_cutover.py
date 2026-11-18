import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_momagen_migration_cutover import (ASSETS_COPIED, CANONICAL_LEAVES, COMPACT_VERIFIED, RUNTIME_COPIED, inspect)


class CutoverInventoryTests(unittest.TestCase):
    def test_staged_does_not_mean_active_and_detects_existing_data(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp); stage=root/'.migration-momagen-fixture'; stage.mkdir()
            for kind,file,value in [('phase','scope-status.json',RUNTIME_COPIED),('phase','asset-status.json',ASSETS_COPIED),
                                    ('state','compact-inputs-target-verification.json',COMPACT_VERIFIED)]:
                (stage/file).write_text(json.dumps({kind:value}))
            for name in CANONICAL_LEAVES:
                (stage/'payload'/name).mkdir(parents=True)
            (root/'code').mkdir()
            (root/'envs/momagen').mkdir(parents=True)
            d=inspect(root,stage)
            self.assertIn('canonical_path_collision:envs/momagen',d['blockers'])
            self.assertFalse(d['simulator_launched'])
            self.assertNotIn('canonical_path_collision:code', d['blockers'])
            self.assertIn('existing_authorized_dataset_access_key_not_securely_provisioned_on_target',d['blockers'])

    def test_rejects_nondedicated_scope(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(ValueError): inspect(p,p/'some-other-workflow')
