#!/usr/bin/env python3
"""SERVER ONLY: register the R1 data adapter, then call official openpi routines."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['config','norms','train'])
    args,remaining=parser.parse_known_args()
    root=Path(os.environ.get('DOUBLE_ARM_ROOT','/mnt/yuanyue/double-arm'))
    os.environ.setdefault('OPENPI_DATA_HOME',str(root/'cache/openpi'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache/xdg'))
    from pi0_r1_config import register_config
    config=register_config()
    if args.mode=='config':print(config);return
    source=Path(os.environ['OPENPI_ROOT']);sys.path.insert(0,str(source/'scripts'))
    if args.mode=='norms':
        import compute_norm_stats
        limit=os.environ.get('PI0_NORM_MAX_FRAMES')
        compute_norm_stats.main(config.name,max_frames=int(limit) if limit else None)
    else:
        sys.argv=[str(source/'scripts/train.py'),config.name,*remaining]
        runpy.run_path(str(source/'scripts/train.py'),run_name='__main__')


if __name__=='__main__':main()
