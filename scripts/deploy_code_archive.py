"""Atomically replace lightweight code files without truncating running scripts.

Invoked only on the server, with a gzip tar archive streamed over stdin.
"""
import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, type=Path)
args = parser.parse_args()
root = args.root.resolve()
if not root.is_absolute() or root == Path('/'):
    raise SystemExit('Unsafe deployment root')
root.mkdir(parents=True, exist_ok=True)
staging = Path(tempfile.mkdtemp(prefix='.code-sync-', dir=root))
try:
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|gz') as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise ValueError(f'Unsupported archive member: {member.name}')
            source = archive.extractfile(member)
            target = staging / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('wb') as stream:
                shutil.copyfileobj(source, stream)
            target.chmod(member.mode & 0o777)
    for source in staging.rglob('*'):
        if source.is_file():
            dest = root / 'code' / source.relative_to(staging)
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, dest)
finally:
    shutil.rmtree(staging)
