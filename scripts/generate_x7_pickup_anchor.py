#!/usr/bin/env python3
"""X7 pickup/dock SE(2) pilot: one calibrated tote transform shared by both arms."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from generate_x7_turning_se2 import digest, read_sources, transfer
from run_x7_mobile_bimanual import ArmIK, CONTROL_HZ, GRIP_Z, HZ, PARK_Y, SUPPORT_Z, create_tote, execute, make_model
from run_x7_turning_carry import (HANDLE_HALF_SPAN, TurningWorld, capture_turn_rgb, r2, turning_score)

# These boxes deliberately sit at different XY/yaw from the original template;
# B is also perturbed. +34-degree source-family holdout remains untouched.
PILOT = {
    'pickup-left': {'goal':(3.38,.12,-28.), 'bay_y':.69, 'tote':(-.009,-.010,-1.7)},
    'pickup-right': {'goal':(3.24,.14,27.), 'bay_y':.72, 'tote':(.008,-.012,1.8)},
    'pickup-yaw': {'goal':(3.30,.09,23.), 'bay_y':.64, 'tote':(-.004,-.013,-2.0)},
}


def object_anchor(pose, xyz, orn):
    dx,dy,degrees=pose; yaw=math.radians(degrees)
    new=np.array(xyz,copy=True)
    new[:2]=np.array([dx,PARK_Y+dy])+r2(yaw)@(new[:2]-[0.,PARK_Y])
    return new,Rotation.from_euler('z',yaw)*orn


class PickupWorld(TurningWorld):
    def __init__(self, model, arms, layout):
        self.pickup_delta=layout['tote']
        super().__init__(model,arms,layout)
        dx,dy,_=self.pickup_delta
        self.goal=np.r_[self.dock[:2]+r2(self.dock[2])@[dx,PARK_Y+dy],SUPPORT_Z]

    def setup_tote(self, handle_x):
        dx,dy,angle=self.pickup_delta
        self.start[:2]+=[dx,dy]
        return create_tote(self.start,handle_x,yaw=math.radians(angle))


def new_arm_tape(model, orig_arms, action, ph, pose):
    result=action.copy(); arms=[]; stats=[]
    # Use the same ordered SE(2) object transform for each arm, with IK
    # solved continuously from the corresponding source branch. Rotations
    # apply in the common chassis frame, not in either wrist-local frame.
    edges=np.r_[0,np.flatnonzero(ph[1:]!=ph[:-1])+1,len(ph)-1]
    knots=sorted(set(range(0,len(ph),4))|set(edges)|set(np.maximum(edges-1,0)))
    for i,src in enumerate(orig_arms):
        arm_slice=slice(6+7*i,13+7*i)
        xyz,orn,_=src.fk(src.home); home,R=object_anchor(pose,xyz,orn)
        tool=ArmIK(model,i,home)
        tool.orientation=R; tool.home=tool.solve(home,src.home)
        m=[]; previous=tool.home; last_q=None; vals=[]
        for k in knots:
            q=action[k,arm_slice]
            if last_q is None or not np.array_equal(q,last_q):
                xyz,r,_=src.fk(q); x,o=object_anchor(pose,xyz,r)
                tool.orientation=o; previous=tool.solve(x,previous,continuity=.010,max_eval=80)
                vals.append(tool.err(x,previous));last_q=q.copy()
            m.append(previous)
        m=np.array(m)
        result[:,arm_slice]=np.column_stack([np.interp(np.arange(len(result)),knots,m[:,j]) for j in range(7)])
        # Check realized FK for all interpolated targets, not only the solved knots.
        errors=[]
        for old,new in zip(action[:,arm_slice],result[:,arm_slice]):
            x,r,_=src.fk(old); x,r=object_anchor(pose,x,r)
            pos,rot,_=tool.fk(new); errors.append((np.linalg.norm(pos-x),(rot.inv()*r).magnitude()))
        err=np.max(errors,axis=0)
        speed=float(np.max(abs(np.diff(result[:,arm_slice],axis=0)))*CONTROL_HZ)
        if err[0]>.003 or err[1]>.024 or speed>1.25:
            raise ValueError(f'Unsafe/interpolated IK deviation arm {i}: error {err}, speed {speed}')
        stats.append({'max_xyz_error_m':float(err[0]),'max_rotation_error_rad':float(err[1]),'peak_joint_target_rad_s':speed})
        arms.append(tool)
    assert np.array_equal(result[:,:6],action[:,:6]) and np.array_equal(result[:,-2:],action[:,-2:])
    return result,arms,stats


def check_pose(t, w, layout):
    score=turning_score(t,w.initial_object,layout)
    dx,dy,yaw=layout['tote']
    act=Rotation.from_quat(t['object_post'][-1,3:7]).as_euler('zyx',degrees=True)[0]
    err=abs((act-layout['goal'][2]-yaw+180.)%360.-180.)
    ready=w.initial_object
    start_yaw=Rotation.from_quat(ready[3:7]).as_euler('zyx',degrees=True)[0]
    init_err=float(np.linalg.norm(ready[:2]-[dx,PARK_Y+dy]))
    yaw_error=float(abs((start_yaw-yaw+180.)%360.-180.))
    pass_anchor=init_err<.003 and yaw_error<.65 and err<2.5
    return score|{'success':bool(score['success'] and pass_anchor),'object_anchor_pass':bool(pass_anchor),
                  'actual_tote_pickup_xyz_m':ready[:3].tolist(),'actual_tote_pickup_yaw_deg':float(start_yaw),
                  'pickup_pose_xy_error_m':init_err,'pickup_pose_yaw_error_deg':yaw_error,
                  'tote_final_yaw_error_to_perturbed_pickup_plus_dock_deg':float(err)}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('asset',type=Path); ap.add_argument('out',type=Path)
    ap.add_argument('--cad-zip',type=Path,required=True);ap.add_argument('--sources',type=Path,required=True)
    ap.add_argument('--source-qa',type=Path,required=True);ap.add_argument('--rgb',action='store_true')
    ap.add_argument('--cases',nargs='+',choices=PILOT,default=list(PILOT)); args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or len(set(args.cases))!=len(args.cases):
        ap.error('Remote CPU only; set CUDA_VISIBLE_DEVICES="" and name distinct cases')
    args.out.mkdir(parents=True); model=make_model(args.asset,args.out/'model',args.cad_zip); tag=digest(model)
    pool=read_sources(args.sources,args.source_qa,tag); out=[]
    orig_arms=[ArmIK(model,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for case in args.cases:
            layout=PILOT[case]; src='source-ccw' if layout['goal'][2]>0 else 'source-cw'; rec=pool[src]
            u,ph,route,width=transfer(rec,layout)
            row={'case':case,'layout':layout,'source_case':src,'source_trace_sha256':rec['sha256'],
                 'route_segments':route,'source_and_target_route_steps':width}
            try:
                action,arms,ik=new_arm_tape(model,orig_arms,u,ph,layout['tote'])
            except (RuntimeError,ValueError) as e:
                row|={'success':False,'reason':'common-frame IK refused','detail':str(e)}
            else:
                world=PickupWorld(model,arms,layout); t=execute(world,action,ph,rgb=args.rgb,camera=capture_turn_rgb)
                t['dock_pose']=world.dock; t['pickup_anchor_delta']=np.asarray(layout['tote'])
                if args.rgb:
                    stamps=np.arange(0,len(ph),6)
                    if t['rgb_external'].shape!=(len(stamps),128,192,3) or not np.array_equal(t['rgb_action_index'],stamps):
                        raise RuntimeError('RGB/action alignment failure')
                result=check_pose(t,world,layout)
                folder=args.out/('accepted' if result['success'] else 'rejected'); folder.mkdir(exist_ok=True)
                path=folder/f'{case}.npz'; np.savez_compressed(path,**t)
                lift,place=[int(np.flatnonzero(ph==k)[0]) for k in (3,6)]
                row|=result|{'trace':str(path.relative_to(args.out)),'trace_sha256':digest(path),
                            'rgb_frames':len(t.get('rgb_external',[])),'max_ik_checks':ik,
                            'anchors':[['pickup_object',0,lift],['common_transport',lift,place],['target_station',place,len(ph)]]}
            out.append(row); print(json.dumps(row),flush=True)
            (args.out/'attempts.json').write_text(json.dumps(out,indent=2)+'\n')
    finally:
        p.disconnect()
    summary={'mode':'MoMaGen-inspired paired arm pickup re-anchoring and moving-base SE(2) retarget, not official upstream MoMaGen',
             'robot_model_sha256':tag,'pinned_qa_sha256':digest(args.source_qa),'preserved_holdout':['heldout-angle +34 deg'],
             'training_started':False,'control_hz':CONTROL_HZ,'physics_hz':HZ,'rgb_hz':5 if args.rgb else 0,
             'attempts':len(out),'accepted':sum(r['success'] for r in out),'attempts_sha256':digest(args.out/'attempts.json')}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
    return 0 if summary['accepted'] else 1


if __name__=='__main__':
    raise SystemExit(main())
