"""Project GPU allocation checks. An idle reserved card is not permission to use it."""
import argparse
import json
from pathlib import Path


def allocation(root):
    path=Path(root)/'runs/RESOURCE_ALLOCATION.json'
    return json.loads(path.read_text()) if path.exists() else {}


def resolve_gpu(root, requested=None):
    config=allocation(root)
    if config.get('max_gpu_count')!=1:
        return requested or '0'
    allowed=config['allowed_gpu_uuid'];index=str(config['allowed_gpu_index'])
    if requested not in (None,'',allowed,index):
        raise ValueError(f'GPU {requested!r} is outside the user allocation; only GPU{index} is allowed')
    return allowed


def require_dual_gpu_allocation(root):
    if allocation(root).get('max_gpu_count')==1:
        raise ValueError('The old dual-GPU controller is disabled: user allocated one card only')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    parser.add_argument('--requested',default=None)
    parser.add_argument('--require-dual',action='store_true')
    args=parser.parse_args()
    if args.require_dual:require_dual_gpu_allocation(args.root)
    else:print(resolve_gpu(args.root,args.requested))
