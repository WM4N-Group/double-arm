"""Server-only: fetch pinned official assets, verify SHA-256, optionally unpack.

A reachable HF endpoint may be supplied, but expected hashes come from the
OFFICIAL upstream API, not from a third-party mirror. No asset/key enters Git.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


def download_bundle(entry, manifest, cache):
    bundles = cache / 'bundles'
    bundles.mkdir(exist_ok=True)
    path = bundles / entry['filename']
    old_blob = cache / 'datasets--behavior-1k--zipped-datasets' / 'blobs' / entry['sha256']
    if not path.exists() and old_blob.exists() and old_blob.stat().st_size == entry['size']:
        path = old_blob
    checked_by_aria = False
    if not path.exists():
        partial = bundles / (entry['filename'] + '.part')
        # Reuse the sequential partial downloaded by huggingface_hub, if present.
        old_parts = list(cache.glob(f'**/{entry["sha256"]}*.incomplete'))
        if not partial.exists() and len(old_parts) == 1:
            old_parts[0].rename(partial)
        endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co').rstrip('/')
        url = f'{endpoint}/datasets/{manifest["repo_id"]}/resolve/{manifest["revision"]}/{entry["filename"]}?download=true'
        print(f'DOWNLOAD {entry["filename"]} expected_bytes={entry["size"]}', flush=True)
        subprocess.run([
            'aria2c', '--continue=true', '--allow-overwrite=true',
            '--auto-file-renaming=false', '--file-allocation=none',
            '--max-connection-per-server=16', '--split=16', '--min-split-size=8M',
            '--retry-wait=10', '--max-tries=20', '--connect-timeout=30', '--timeout=120',
            '--summary-interval=30', '--console-log-level=warn', '--download-result=hide',
            '--check-certificate=true', f'--checksum=sha-256={entry["sha256"]}',
            f'--dir={bundles}', f'--out={partial.name}', url,
        ], check=True)
        partial.rename(path)
        checked_by_aria = True
    if path.stat().st_size != entry['size']:
        raise ValueError(f'Wrong file size: {entry["filename"]}')
    stamp = cache / f'{entry["sha256"]}.verified.json'
    fingerprint = {'path': str(path), 'size': path.stat().st_size,
                   'mtime_ns': path.stat().st_mtime_ns, 'sha256': entry['sha256']}
    cached = json.loads(stamp.read_text()) if stamp.exists() else None
    if cached != fingerprint:
        if not checked_by_aria:
            print(f'VERIFY_SHA256 {entry["filename"]}', flush=True)
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256']:
                raise ValueError(f'SHA-256 mismatch: {entry["filename"]}')
        stamp.write_text(json.dumps(fingerprint, indent=2))
    print(f'VERIFIED {entry["filename"]}', flush=True)
    return path


def unpack_bundle(entry, path, data_root):
    target = data_root / entry['target']
    stamps = data_root / '.installed'
    stamps.mkdir(exist_ok=True)
    marker = stamps / f'{entry["target"]}.json'
    expected = {'filename': entry['filename'], 'sha256': entry['sha256']}
    if marker.exists() and target.is_dir() and json.loads(marker.read_text()) == expected:
        print(f'ALREADY_INSTALLED {target}', flush=True)
        return
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite an unverified asset directory: {target}')
    staging = Path(tempfile.mkdtemp(prefix=f'.extract-{entry["target"]}-', dir=data_root))
    print(f'UNPACK {path.name} -> {target}', flush=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError(f'Unsafe ZIP member: {member.filename}')
        archive.extractall(staging)
    staging.rename(target)
    marker.write_text(json.dumps(expected, indent=2))
    print(f'INSTALLED {target}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--download-only', action='store_true')
    args = parser.parse_args()
    if os.environ.get('ACCEPT_BEHAVIOR_DATASET_TERMS') != 'YES':
        raise SystemExit('Explicit BEHAVIOR asset license confirmation required')
    root = Path(os.environ['DOUBLE_ARM_ROOT'])
    cache = root / 'cache' / 'hf-assets'
    cache.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (cache / 'download.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    manifest = json.loads(args.manifest.read_text())
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda entry: download_bundle(entry, manifest, cache), manifest['files']))
    if args.download_only:
        print('ALL_ASSET_BUNDLES_DOWNLOADED_AND_VERIFIED', flush=True)
        return
    data_root = Path(os.environ['OMNIGIBSON_DATA_PATH'])
    data_root.mkdir(parents=True, exist_ok=True)
    for entry, path in zip(manifest['files'], paths):
        unpack_bundle(entry, path, data_root)
    # Use the official project function. Never expose or redistribute the key.
    from omnigibson.utils.asset_utils import download_key, get_key_path
    import socket
    previous_timeout = socket.getdefaulttimeout()
    previous_umask = os.umask(0o077)
    try:
        socket.setdefaulttimeout(60)
        download_key()
        key = Path(get_key_path())
        key.chmod(0o600)
        # Validate format without logging, exporting or redistributing the key.
        from cryptography.fernet import Fernet
        Fernet(key.read_bytes())
    finally:
        socket.setdefaulttimeout(previous_timeout)
        os.umask(previous_umask)
    print('ALL_ASSETS_INSTALLED', flush=True)


if __name__ == '__main__':
    main()
