"""Lightweight workflow checks. No GPU, simulator, network or robotics imports."""
import ast
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_shell_syntax(self):
        for script in (ROOT / 'scripts').glob('*.sh'):
            with self.subTest(script=script.name):
                subprocess.run(['bash', '-n', str(script)], check=True)

    def test_python_syntax(self):
        for script in (ROOT / 'scripts').glob('*.py'):
            ast.parse(script.read_text(), filename=str(script))

    def test_upstream_revisions_are_pinned(self):
        text = (ROOT / 'config/upstream.env').read_text()
        for key in ('MOMAGEN_REV', 'BEHAVIOR_REV', 'ROBOMIMIC_REV', 'CUROBO_REV'):
            self.assertRegex(text, rf'(?m)^{key}=[0-9a-f]{{40}}$')

    def test_large_files_and_local_settings_are_ignored(self):
        for path in ('paper.pdf', 'model.ckpt', 'data/demo.hdf5',
                     'runs/test/video.mp4', 'config/server.local.env',
                     'config/licenses.local.env', '.env'):
            with self.subTest(path=path):
                result = subprocess.run(['git', 'check-ignore', '-q', path], cwd=ROOT)
                self.assertEqual(result.returncode, 0)

    def test_remote_command_preserves_argument_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            mock = tmp / 'ssh'
            mock.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
            mock.chmod(0o755)
            config = tmp / 'server.env'
            config.write_text('SSH_HOST=example.invalid\nSSH_PORT=2222\nSSH_USER=test\nREMOTE_ROOT=/remote\n')
            arguments = ['printf', '%s\\n', "spaces 'quotes'; $HOME", '']
            result = subprocess.run(
                [str(ROOT / 'scripts/remote.sh'), *arguments],
                env={**os.environ, 'SERVER_CONFIG': str(config), 'PATH': f'{tmp}:{os.environ["PATH"]}'},
                text=True, capture_output=True, check=True,
            )
            import json
            ssh_arguments = json.loads(result.stdout)
            self.assertEqual(ssh_arguments[-2], 'test@example.invalid')
            # The command uses bash printf %q: Bash, not POSIX sh, decodes it.
            decoded = subprocess.run(
                ['bash', '-c', 'set -- ' + ssh_arguments[-1] + '; printf "%s\\0" "$@"'],
                capture_output=True, check=True,
            ).stdout.decode().split('\0')[:-1]
            self.assertEqual(decoded, arguments)

    def test_no_large_files_tracked(self):
        files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
        for name in filter(None, files):
            self.assertLess((ROOT / name).stat().st_size, 1024 * 1024, name)


if __name__ == '__main__':
    unittest.main()
