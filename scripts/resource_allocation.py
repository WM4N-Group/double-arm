"""Explicit GPU authorization; one UUID per job, legacy GPU1 default preserved.

2026-11-16: user explicitly authorized both cards. Keep the historical primary
fields intact so the ALREADY-RUNNING single-GPU controller does not abort when
it rereads allocation. New parallel authorization is a separate fail-closed
allowlist; max_gpu_count=1 is the legacy per-job limit, NOT the project total.
"""
import argparse
import json
from pathlib import Path

ALLOWED_GPU_UUID = 'GPU-50e5d38d-2825-e379-f537-7f0bf4609aba'
ALLOWED_GPU_INDEX = 1
GPU_UUIDS_BY_INDEX = {
    0: 'GPU-925e5f5c-3cbb-7e93-5a98-1f06d3a71c0d',
    1: ALLOWED_GPU_UUID,
}


def allocation(root):
    path = Path(root) / 'runs/RESOURCE_ALLOCATION.json'
    if not path.is_file():
        raise ValueError('Explicit RESOURCE_ALLOCATION.json is required; no GPU0 fallback')
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError('Invalid GPU allocation object')
    return config


def authorized_gpus(root):
    config = allocation(root)
    if (type(config.get('max_gpu_count')) is not int or config['max_gpu_count'] != 1
        or config.get('allowed_gpu_uuid') != ALLOWED_GPU_UUID
        or config.get('allowed_gpu_index') != ALLOWED_GPU_INDEX):
        raise ValueError('Legacy primary GPU1/per-job contract changed; refusing to interrupt existing workflow')
    if config.get('dual_gpu_authorized') is True:
        expected = [GPU_UUIDS_BY_INDEX[0], GPU_UUIDS_BY_INDEX[1]]
        if (config.get('allowed_gpu_uuids') != expected
            or config.get('allowed_gpu_indices') != [0, 1]
            or type(config.get('total_authorized_gpu_count')) is not int
            or config['total_authorized_gpu_count'] != 2
            or not isinstance(config.get('parallel_authorization_text'), str)
            or not config['parallel_authorization_text'].strip()
            or not config.get('parallel_authorized_at')):
            raise ValueError('Incomplete explicit two-GPU authorization; no implicit GPU0 access')
        return tuple(expected)
    if any(key in config for key in ('allowed_gpu_uuids', 'total_authorized_gpu_count')):
        raise ValueError('New allowlist exists without explicit dual-GPU authorization')
    return (ALLOWED_GPU_UUID,)


def resolve_gpu(root, requested=None):
    allowed = authorized_gpus(root)
    if requested in (None, ''):
        return ALLOWED_GPU_UUID  # Never migrate a live/default job from GPU1.
    for index, uuid in GPU_UUIDS_BY_INDEX.items():
        if requested in (str(index), uuid) and uuid in allowed:
            return uuid
    raise ValueError(f'GPU {requested!r} outside explicit allocation; each job must select ONE authorized UUID')


def require_dual_gpu_allocation(root):
    # This API belonged to the archived memory/state-policy controllers. A new
    # resource grant does not authorize substituting those for original pi0.
    raise ValueError('Archived dual-GPU controllers remain disabled; use explicit independent official/pi0 jobs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/yuanyue/double-arm'))
    parser.add_argument('--requested', default=None)
    parser.add_argument('--require-dual', action='store_true')
    args = parser.parse_args()
    if args.require_dual:
        require_dual_gpu_allocation(args.root)
    else:
        print(resolve_gpu(args.root, args.requested))
