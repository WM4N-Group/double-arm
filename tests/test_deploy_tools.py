"""Tiny archive deployment fixtures; no simulator, GPU or network."""
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def archive_bytes(name, content):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        member = tarfile.TarInfo(name)
        member.mode = 0o755
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return stream.getvalue()


class DeployToolTests(unittest.TestCase):
    def test_running_file_inode_and_private_configuration_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'code/scripts/example.sh'
            script.parent.mkdir(parents=True)
            script.write_bytes(b'old running script')
            private = root / 'code/config/licenses.local.env'
            private.parent.mkdir()
            private.write_text('synthetic private fixture')
            with script.open('rb') as running:
                subprocess.run(
                    ['python3', str(ROOT / 'scripts/deploy_code_archive.py'), '--root', str(root)],
                    input=archive_bytes('scripts/example.sh', b'new script'), check=True,
                )
                self.assertEqual(running.read(), b'old running script')
            self.assertEqual(script.read_bytes(), b'new script')
            self.assertEqual(private.read_text(), 'synthetic private fixture')
            self.assertEqual(script.stat().st_mode & 0o777, 0o755)
            self.assertFalse(list(root.glob('.code-sync-*')))

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = subprocess.run(
                ['python3', str(ROOT / 'scripts/deploy_code_archive.py'), '--root', str(root)],
                input=archive_bytes('../escape.txt', b'bad'), capture_output=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root.parent / 'escape.txt').exists())
            self.assertFalse(list(root.glob('.code-sync-*')))
