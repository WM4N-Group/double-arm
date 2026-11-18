#!/usr/bin/env python3
"""Review the exact archived X7 BC checkpoint on a previously failed dev layout."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from generate_x7_pickup_anchor import PickupWorld, check_pose
from run_x7_mobile_bimanual import execute
from run_x7_turning_carry import review_turn
from train_x7_turning_trajectory_bc import DEV, digest, predict


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('pilot',type=Path); ap.add_argument('out',type=Path)
    ap.add_argument('--case',choices=DEV,required=True); args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':ap.error('Remote CPU only')
    args.out.mkdir(parents=True); case=args.case; book=args.pilot; report=json.loads((book/'report.json').read_text())
    archive=next(x for x in report['results'] if x['case']==case); assert not archive['success']
    ckpt=book/'trajectory-bc.npz'; old=book/(case+'.npz')
    assert digest(ckpt)==report['checkpoint_sha256'] and digest(old)==archive['trace_sha256']
    with np.load(ckpt,allow_pickle=False) as f:
        u,ph,_=predict({k:f[k] for k in f.files},DEV[case])
    with np.load(old,allow_pickle=False) as f:
        assert np.array_equal(u,f['action']) and np.array_equal(ph,f['phase'])
    homes=[SimpleNamespace(home=u[0,6+i*7:13+i*7]) for i in range(2)]
    video=args.out/(case+'-FAILED.mp4'); p.connect(p.DIRECT)
    try:
        world=PickupWorld(book/'model/x7_proxy_gripper.urdf',homes,DEV[case]);
        t=execute(world,u,ph,movie=video,reviewer=review_turn,review_label='CLOCKED BC   /   DEV FAILED')
        assert not check_pose(t,world,DEV[case])['success']
    finally:
        p.disconnect()
    keys=('action','object_post','arm_pre','base_post','gripper_post','contact_samples','station_clearance','on_floor','on_table')
    with np.load(old,allow_pickle=False) as f:
        assert all(np.array_equal(t[k],f[k]) for k in keys)
    out={'case':case,'result':'exact repeat of same frozen BC dev failure','checkpoint_sha256':digest(ckpt),
         'original_trace_sha256':digest(old),'repeat_exact_fields':list(keys),'video_sha256':digest(video)}
    (args.out/'review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
