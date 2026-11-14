"""Server-only, resumable wheel download using the pinned upstream package list."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import subprocess
import zipfile


def package_list(setup_script):
    text = Path(setup_script).read_text()
    match = re.search(r'local packages=\((.*?)\n\s*\)', text, re.S)
    if not match:
        raise ValueError('Could not locate the pinned Isaac package list')
    packages = re.findall(r'"([a-z0-9_]+-[0-9.]+)"', match.group(1))
    if len(packages) != 26:
        raise ValueError(f'Unexpected upstream package count: {len(packages)}')
    return packages


def download(package, folder):
    name, version = package.rsplit('-', 1)
    filename = f'{name}-{version}-cp310-none-manylinux_2_34_x86_64.whl'
    dest = folder / filename
    if dest.exists() and zipfile.is_zipfile(dest):
        print(f'CACHED {filename} {dest.stat().st_size}', flush=True)
        return dest
    partial = folder / (filename + '.part')
    url = f'https://pypi.nvidia.com/{name.replace("_", "-")}/{filename}'
    print(f'DOWNLOAD {filename}', flush=True)
    subprocess.run([
        'curl', '--fail', '--location', '--silent', '--show-error',
        '--retry', '5', '--retry-all-errors', '--retry-delay', '5',
        '--connect-timeout', '25', '--speed-time', '120', '--speed-limit', '1024',
        '--continue-at', '-', '--output', str(partial), url,
    ], check=True)
    if not zipfile.is_zipfile(partial):
        raise ValueError(f'Not a valid wheel: {partial}')
    partial.replace(dest)
    print(f'COMPLETE {filename} {dest.stat().st_size}', flush=True)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup-script', required=True)
    parser.add_argument('--folder', required=True, type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    packages = package_list(args.setup_script)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda name: download(name, args.folder), packages))
    print('ALL_ISAAC_WHEELS_DOWNLOADED', flush=True)


if __name__ == '__main__':
    main()
