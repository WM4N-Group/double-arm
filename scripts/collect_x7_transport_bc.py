#!/usr/bin/env python3
"""Remote X7 RGB/proprio dataset: immutable layouts and strict physical gates."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import FROZEN_CHECKPOINT, check_footprint
from generate_x7_pickup_anchor import PickupWorld, check_pose
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, capture_turn_rgb
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

SPLITS={'train':(72,2026111901),'dev':(12,2026111902),'heldout':(12,2026111903)}


def layouts():
    rows=[]
    yaw={'train':(23.6,27.35),'dev':(27.55,28.60),'heldout':(28.85,30.00)}
    for split,(n,seed) in SPLITS.items():
        rng=np.random.default_rng(seed)
        for i in range(n):
            sign=1 if i%2 else -1
            rows.append(dict(case=f'{split}-{i:03}',split=split,goal=[float(rng.uniform(3.275,3.350)),
                float(rng.uniform(.081,.161)),sign*float(rng.uniform(*yaw[split]))],bay_y=float(rng.uniform(.638,.718)),
                tote=list(map(float,[rng.uniform(-.0075,.0075),rng.uniform(-.0127,-.0080),rng.uniform(-1.5,1.5)]))))
    return rows


def sha_layout(row):
    return sha256(json.dumps({k:row[k] for k in ('goal','tote','bay_y')},sort_keys=True).encode()).hexdigest()


def validate_trace(t, scores, footprint):
    frames,actions=t['rgb_external'],t['action']
    if (frames.dtype!=np.uint8 or frames.shape!=(len(t['rgb_action_index']),128,192,3) or
            not np.array_equal(t['rgb_action_index'],np.arange(0,len(actions),6))):
        raise ValueError('Pre-action RGB is missing or misaligned')
    s=t['station_clearance']; init=t['station_initialization_clearance']
    if len(actions)!=len(t['phase']) or actions.shape[1]!=22:
        raise ValueError('Missing control horizon or wrong action dimensions')
    return bool(scores['success'] and footprint['full_bottom_supported'] and
                min(np.min(s[...,:2]),np.min(init[...,:2]))>=.005 and
                not np.any(s[...,2:4]) and not np.any(init[...,2:4]) and
                scores['object_attachment_constraints_seen']==0 and scores['state_writes_during_actions']==0)


def worker(pilot,manifest,out,index,workers):
    items=json.loads(manifest.read_text())
    if items['teacher_sha256']!=FROZEN_CHECKPOINT or items['layouts']!=layouts():
        raise ValueError('Unexpected layout or teacher manifest')
    ckpt=pilot/'anchored-hybrid.npz';asset=pilot/'model/x7_proxy_gripper.urdf'
    if digest(ckpt)!=FROZEN_CHECKPOINT or digest(asset)!=items['urdf_sha256']:
        raise ValueError('Drifted teacher or geometry')
    with np.load(ckpt,allow_pickle=False) as d: model={k:d[k] for k in d.files}
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    parts=out/'shards';parts.mkdir(parents=True,exist_ok=True)
    p.connect(p.DIRECT)
    try:
        collect=[r for r in items['layouts'] if r['split'] in ('train','dev')]
        for ordinal,row in enumerate(collect):
            if ordinal%workers!=index:continue
            name=row['case']; part=parts/name;part.mkdir()
            record=dict(case=name,split=row['split'],layout_sha256=sha_layout(row),physical_execution=False,
                        accepted=False,teacher_sha256=FROZEN_CHECKPOINT)
            try:
                u,ph,meta=make_action(model,asset,arms,row)
            except (ValueError,AssertionError,RuntimeError) as e:
                record['preflight_rejected']=str(e)
            else:
                homes=[SimpleNamespace(home=u[0,6+7*i:13+7*i]) for i in range(2)]
                w=PickupWorld(asset,homes,row)
                t=execute(w,u,ph,rgb=True,camera=capture_turn_rgb)
                m=check_pose(t,w,row); fp=check_footprint(t,row)
                passed=validate_trace(t,m,fp)
                dest=part/'trace.npz';np.savez_compressed(dest,**t,block_counts=meta['fitted_phase_durations'])
                record.update(physical_execution=True,accepted=passed,metrics=m,footprint=fp,
                              action_steps=len(u),rgb_frames=len(t['rgb_external']),trace_sha256=digest(dest))
            f=part/'report.tmp';f.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');f.replace(part/'report.json')
            print(json.dumps({key:record.get(key) for key in ('case','accepted','physical_execution','action_steps','rgb_frames','preflight_rejected')}),flush=True)
    finally:p.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':ap.error('remote CPU only')
    manifest=a.out/'layouts.json'
    if a.plan:
        a.out.mkdir(parents=True);rows=layouts()
        if len({sha_layout(r) for r in rows})!=len(rows):raise ValueError('Duplicate layout')
        manifest.write_text(json.dumps(dict(version='x7-transport-pilot-v1',teacher_sha256=FROZEN_CHECKPOINT,
            urdf_sha256=digest(a.pilot/'model/x7_proxy_gripper.urdf'),layouts=rows,
            heldout_teacher_traces_generated=False),indent=2,allow_nan=False)+'\n')
    elif a.worker is not None and 0<=a.worker<a.workers:
        worker(a.pilot,manifest,a.out,a.worker,a.workers)
    else:ap.error('Specify --plan or a valid --worker')


if __name__=='__main__':main()
