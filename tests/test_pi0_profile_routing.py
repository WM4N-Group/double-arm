from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_pi0_evaluation import inference_entrypoint,LEGACY_PROFILE,RELATIVE_FULL_PROFILE


class ProfileRoutingTests(unittest.TestCase):
    def test_legacy_default_paths_unchanged(self):
        self.assertEqual(inference_entrypoint(LEGACY_PROFILE,'native'),'serve_pi0_r1.py')
        self.assertEqual(inference_entrypoint(LEGACY_PROFILE,'absolute_base'),'serve_pi0_r1.py')

    def test_relative_profile_never_silently_reinterpreted_as_world_base_or_legacy(self):
        self.assertEqual(inference_entrypoint(RELATIVE_FULL_PROFILE,'native'),'serve_pi0_relative_full_r1.py')
        with self.assertRaises(ValueError):inference_entrypoint(RELATIVE_FULL_PROFILE,'absolute_base')
        with self.assertRaises(ValueError):inference_entrypoint('unknown','native')

    def test_legacy_service_rejects_versioned_contract_before_model_or_token_load(self):
        import importlib.util
        import json
        import os
        import tempfile
        from types import SimpleNamespace
        from unittest import mock
        from resource_allocation import ALLOWED_GPU_UUID
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cp=root/'runs/19999';(cp/'params').mkdir(parents=True);(cp/'assets').mkdir()
            (cp/'assets/r1-relative-profile-contract.json').write_text('{}')
            (root/'runs/RESOURCE_ALLOCATION.json').write_text(json.dumps({
                'max_gpu_count':1,'allowed_gpu_uuid':ALLOWED_GPU_UUID,'allowed_gpu_index':1}))
            source=Path(__file__).resolve().parents[1]/'scripts/serve_pi0_r1.py'
            spec=importlib.util.spec_from_file_location('_legacy_profile_guard_fixture',source)
            module=importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules,{'pi0_rpc':SimpleNamespace(decode_observation=lambda x:x)}):spec.loader.exec_module(module)
            argv=['serve_pi0_r1.py','--checkpoint',str(cp),'--ready-file',str(root/'ready.json'),
                '--token-file',str(root/'MISSING-token'),'--inference-platform','cpu']
            with mock.patch.dict(os.environ,{'DOUBLE_ARM_ROOT':str(root),'CUDA_VISIBLE_DEVICES':'1'}),mock.patch.object(sys,'argv',argv):
                with self.assertRaisesRegex(ValueError,'explicit profile service'):module.main()


if __name__=='__main__':unittest.main()
