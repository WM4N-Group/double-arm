#!/usr/bin/env python3
"""Emit NUL-separated lightweight code paths, never binary robot assets."""
from pathlib import Path
import subprocess
import sys

MAX_BYTES = 1024 * 1024
ASSET_SUFFIXES = {'.stl', '.obj', '.dae', '.fbx', '.ply', '.usd', '.usda', '.usdc',
                  '.h5', '.hdf5', '.npz', '.npy', '.pt', '.pth', '.ckpt', '.onnx',
                  '.zip', '.gz', '.tar', '.pdf', '.mp4', '.xlsx', '.xls'}


def lightweight(path):
    path = Path(path)
    return path.is_file() and not path.is_symlink() and path.suffix.lower() not in ASSET_SUFFIXES and path.stat().st_size <= MAX_BYTES


def main():
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'])
    kept = []; skipped = []
    for item in filter(None, raw.split(b'\0')):
        path = Path(item.decode())
        if lightweight(path): kept.append(item)
        else: skipped.append(str(path))
    if skipped:
        print(f'Skipping {len(skipped)} non-code/large paths (local files preserved)', file=sys.stderr)
    sys.stdout.buffer.write(b''.join(p + b'\0' for p in kept))


if __name__ == '__main__': main()
