"""Tests use only tiny synthetic files; no robotics assets, GPU or network."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wheels = load_tool('download_isaac_wheels')
assets = load_tool('prepare_asset_bundles')


class AssetToolTests(unittest.TestCase):
    def test_reads_exact_upstream_package_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'setup.sh'
            names = [f'package_{i}-4.5.0.0' for i in range(26)]
            script.write_text('local packages=(\n' + '\n'.join(f'"{n}"' for n in names) + '\n)')
            self.assertEqual(wheels.package_list(script), names)

    def test_rejects_changed_package_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'setup.sh'
            script.write_text('local packages=(\n"one-4.5.0.0"\n)')
            with self.assertRaises(ValueError):
                wheels.package_list(script)

    def test_manifest_has_fixed_revision_sizes_and_hashes(self):
        manifest = json.loads((ROOT / 'config/assets.json').read_text())
        self.assertRegex(manifest['revision'], r'^[a-f0-9]{40}$')
        self.assertEqual(len(manifest['files']), 3)
        for entry in manifest['files']:
            self.assertRegex(entry['sha256'], r'^[a-f0-9]{64}$')
            self.assertGreater(entry['size'], 0)
            self.assertNotIn('/', entry['filename'])

    def test_unpack_is_repeatable_and_uses_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'test.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('metadata/test.txt', 'synthetic test fixture')
            entry = {'target': 'bundle', 'filename': 'test.zip', 'sha256': 'a' * 64}
            assets.unpack_bundle(entry, archive, root)
            self.assertEqual((root / 'bundle/metadata/test.txt').read_text(), 'synthetic test fixture')
            assets.unpack_bundle(entry, archive, root)
            self.assertTrue((root / '.installed/bundle.json').is_file())

    def test_rejects_archive_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('../escape.txt', 'must not escape')
            entry = {'target': 'bundle', 'filename': 'bad.zip', 'sha256': 'b' * 64}
            with self.assertRaises(ValueError):
                assets.unpack_bundle(entry, archive, root)
            self.assertFalse((root / 'escape.txt').exists())
            self.assertFalse((root / 'bundle').exists())

    def test_refuses_unverified_existing_asset_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'bundle').mkdir()
            entry = {'target': 'bundle', 'filename': 'test.zip', 'sha256': 'c' * 64}
            with self.assertRaises(FileExistsError):
                assets.unpack_bundle(entry, root / 'unused.zip', root)

    def test_parallel_downloader_passes_expected_checksum(self):
        content = b'synthetic tiny download fixture'
        digest = hashlib.sha256(content).hexdigest()
        entry = {'filename': 'fixture.zip', 'size': len(content), 'sha256': digest}
        manifest = {'repo_id': 'test/fixture', 'revision': 'd' * 40}
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)

            def fake_download(arguments, check):
                self.assertIn(f'--checksum=sha-256={digest}', arguments)
                folder = Path(next(arg.split('=', 1)[1] for arg in arguments if arg.startswith('--dir=')))
                filename = next(arg.split('=', 1)[1] for arg in arguments if arg.startswith('--out='))
                (folder / filename).write_bytes(content)

            with patch.object(assets.subprocess, 'run', side_effect=fake_download) as run:
                result = assets.download_bundle(entry, manifest, cache)
                self.assertEqual(result.read_bytes(), content)
                assets.download_bundle(entry, manifest, cache)
                self.assertEqual(run.call_count, 1)
