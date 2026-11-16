"""Project GPU allocation checks. An idle reserved card is not permission to use it."""
import argparse
import json
from pathlib import Path

# Explicit current user allocation. A file disappearance or accidental config
# edit must not silently grant access to the reserved first card.
ALLOWED_GPU_UUID='GPU-50e5d38d-2825-e379-f537-7f0bf4609aba'
ALLOWED_GPU_INDEX=1


def allocation(root):
    path=Path(root)/'runs/RESOURCE_ALLOCATION.json'
    if not path.is_file():raise ValueError('Explicit RESOURCE_ALLOCATION.json is required; no GPU0 fallback')
    config=json.loads(path.read_text())
    if not isinstance(config,dict):raise ValueError('Invalid GPU allocation object')
    return config


def resolve_gpu(root, requested=None):
    config=allocation(root)
    if (type(config.get('max_gpu_count')) is not int or config['max_gpu_count']!=1
        or config.get('allowed_gpu_uuid')!=ALLOWED_GPU_UUID
        or config.get('allowed_gpu_index')!=ALLOWED_GPU_INDEX):
        raise ValueError('Allocation conflicts with explicitly authorized GPU1 UUID; refusing to launch')
    allowed=ALLOWED_GPU_UUID;index=str(ALLOWED_GPU_INDEX)
    if requested not in (None,'',allowed,index):
        raise ValueError(f'GPU {requested!r} is outside the user allocation; only GPU{index} is allowed')
    return allowed


def require_dual_gpu_allocation(root):
    raise ValueError('The old dual-GPU controller is disabled by current project policy; no implicit revival')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    parser.add_argument('--requested',default=None)
    parser.add_argument('--require-dual',action='store_true')
    args=parser.parse_args()
    if args.require_dual:require_dual_gpu_allocation(args.root)
    else:print(resolve_gpu(args.root,args.requested))
