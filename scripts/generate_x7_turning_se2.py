#!/usr/bin/env python3
"""Restricted MoMaGen-style segmented SE(2) retargeting of X7 turning sources.

Read only the checksummed, physically qualified source partition. Reuse the recorded
both-arm/jaw commands in the common chassis frame; transform base-walk anchors
between task frames, time-scale the source motion, then rerun physics. A remains
fixed, object geometry is identical, and this is not upstream MoMaGen.
"""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path

import numpy as np
import pybullet as p

from run_x7_mobile_bimanual import ArmIK, CONTROL_HZ, GRIP_Z, HZ, PARK_Y, execute, make_model
from run_x7_turning_carry import (HANDLE_HALF_SPAN, LAYOUTS, WXY, TurningWorld, capture_turn_rgb,
                                  dock_pose, r2, turning_score, waypoints)

CANDIDATES = {
    'cw-wide-31': {'goal':(3.41,.14,-31.),'bay_y':.72},
    'cw-near-21': {'goal':(3.13,.03,-21.),'bay_y':.58},
    'ccw-offset-21': {'goal':(3.35,.20,21.),'bay_y':.69},
    'ccw-near-29': {'goal':(3.27,.15,29.),'bay_y':.75},
}
SOURCE_NAMES = ('source-cw','source-ccw')
PHASE_NAMES = ('pick_at_A','shared_held_tote','dock_and_release_at_B')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def compose(a,b):
    return np.r_[a[:2]+r2(a[2])@b[:2],a[2]+b[2]]


def invert(a):
    return np.r_[-r2(a[2]).T@a[:2],-a[2]]


def exp_jac(yaw):
    if abs(yaw)<1e-8:
        return np.array([[1.,-yaw/2],[yaw/2,1.]])
    return np.array([[math.sin(yaw),math.cos(yaw)-1], [1-math.cos(yaw),math.sin(yaw)]])/yaw


def blend_correction(c0,c1,u):
    """Interpolate rigid frame corrections on SE(2); fixed pivot stays fixed."""
    d=compose(invert(c0),c1)
    v=np.linalg.solve(exp_jac(d[2]),d[:2])
    return compose(c0,np.r_[exp_jac(d[2]*u)@(v*u),d[2]*u])


def resample_motion(path):
    """Use source step cadence if feasible, otherwise enlarge segment duration."""
    if path.shape[0]<3 or path.shape[1]!=3 or not np.isfinite(path).all():
        raise ValueError('Expected finite SE(2) pose path including initial boundary')
    original_n=len(path)-1; rate=np.array(WXY)*.86
    needed=int(np.ceil(original_n*np.max(abs(np.diff(path,axis=0))*CONTROL_HZ/rate)))
    n=max(needed,original_n); loc=np.linspace(0.,1.,original_n+1)
    ref=np.column_stack([np.interp(np.linspace(0.,1.,n+1),loc,path[:,k]) for k in range(3)])
    v=np.gradient(ref,1./CONTROL_HZ,axis=0); v[[0,-1],:]=0.
    if np.any(abs(v)>rate+1e-10):
        raise RuntimeError('Retarget reference exceeds planar speed envelope')
    return np.column_stack([ref[1:],v[1:]])


def read_sources(root, summary, model_sha):
    report=root/'report.json'; info=json.loads(report.read_text()); qa=json.loads(summary.read_text())
    if (not info['all_pass'] or info['asset_sha256']!=model_sha or
        qa['report_sha256']!=digest(report) or qa['source_count']!=2 or info['training_started'] is not False):
        raise ValueError('Expected verified archive made on this precise collision model')
    rows={r['case']:r for r in info['case_reports'] if r['partition']=='source'}
    if set(rows)!=set(SOURCE_NAMES):
        raise ValueError('Verified two-source partition required')
    pool={}
    for name in SOURCE_NAMES:
        r=rows[name]; path=root/'source'/'accepted'/f'{name}.npz'
        q=[entry for entry in qa['records'] if entry['case']==name and entry['split']=='source']
        sha=digest(path)
        if not (r['success'] and r['robot_station_clearance_pass'] and r['object_attachment_constraints_seen']==0 and
                r['trace']==f'source/accepted/{name}.npz' and len(q)==1 and r['trace_sha256']==q[0]['trace_sha256']==sha):
            raise ValueError(f'Unverified source {name}')
        with np.load(path,allow_pickle=False) as data:
            pool[name]={'row':r,'sha256':sha,'action':data['action'],'phase':data['phase']}
    return pool


def transfer(source, target):
    row,old,phase=source['row'],source['action'],source['phase']
    ws=row['waypoints']; rs=row['route_segments']; wt=waypoints(target)
    if (old.ndim!=2 or old.shape[1]!=22 or phase.shape!=(len(old),) or not np.isfinite(old).all() or
        len(rs)!=5 or len(ws)!=len(wt) or [x[0] for x in ws]!=[x[0] for x in wt]):
        raise ValueError('Source phase and command contract mismatch')
    fs,ls=np.flatnonzero(phase==4)[[0,-1]]
    if (rs[0]['start_inclusive']!=fs or rs[-1]['end_exclusive']!=ls+1 or
        any(l['end_exclusive']!=r['start_inclusive'] for l,r in zip(rs[:-1],rs[1:])) or
        not np.array_equal(old[fs:ls+1,6:],np.tile(old[fs,6:],(ls+1-fs,1)))):
        raise ValueError('Source does not contain one contiguous common held pose')
    correction=[]
    for (_,a),(_,b) in zip(ws,wt):
        correction.append(compose(np.array(b),invert(np.array(a))))
    snippets=[]; route=[]; k=int(fs); stats=[]
    for i,r in enumerate(rs):
        a,b=int(r['start_inclusive']),int(r['end_exclusive']); edge=np.array(ws[i][1]); end=np.array(ws[i+1][1]); direction=end-edge
        path=np.vstack([old[a-1,:3],old[a:b,:3]])
        if not np.allclose(path[0],edge,atol=1e-8) or not np.allclose(path[-1],end,atol=1e-8):
            raise ValueError('Source physical frame anchors and the action tape disagree')
        progress=(path-edge)@direction/(direction@direction)
        if (np.min(np.diff(progress))<-1e-8 or abs(progress[0])>1e-9 or abs(progress[-1]-1)>1e-9):
            raise ValueError('Unexpected source nonmonotonic segment')
        new=np.stack([compose(blend_correction(correction[i],correction[i+1],float(np.clip(u,0.,1.))),q)
                      for q,u in zip(path,progress)])
        if not np.allclose(new[[0,-1]],[wt[i][1],wt[i+1][1]],atol=1e-7):
            raise RuntimeError('Target start or end anchor mismatch')
        motion=resample_motion(new)
        snippets.append(np.column_stack([motion,np.tile(old[fs,6:],(len(motion),1))]))
        route.append({'name':r['name'],'start_inclusive':k,'end_exclusive':k+len(motion),
                      'source_interval':[a,b]})
        k+=len(motion); stats.append([b-a,len(motion)])
    after=old[ls+1:].copy(); after[:,:6]=np.r_[dock_pose(target),np.zeros(3)]
    drive=np.vstack(snippets); out=np.vstack([old[:fs],drive,after])
    tags=np.r_[phase[:fs],np.full(len(drive),4,dtype=phase.dtype),phase[ls+1:]]
    if np.array_equal(out,old):
        raise ValueError('A copied episode is not a retargeted candidate')
    return out,tags,route,stats


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('asset',type=Path); ap.add_argument('out',type=Path); ap.add_argument('--cad-zip',type=Path,required=True)
    ap.add_argument('--sources',type=Path,required=True); ap.add_argument('--source-qa',type=Path,required=True)
    ap.add_argument('--rgb',action='store_true'); args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': ap.error('Remote CPU only: set CUDA_VISIBLE_DEVICES to empty')
    args.out.mkdir(parents=True); model=make_model(args.asset,args.out/'model',args.cad_zip); model_sha=digest(model)
    pool=read_sources(args.sources,args.source_qa,model_sha)
    arms=[ArmIK(model,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[]; p.connect(p.DIRECT)
    try:
        for tag,layout in CANDIDATES.items():
            src='source-ccw' if layout['goal'][2]>0 else 'source-cw'; record=pool[src]
            u,phase,route,time_scale=transfer(record,layout)
            w=TurningWorld(model,arms,layout); trace=execute(w,u,phase,rgb=args.rgb,camera=capture_turn_rgb)
            trace['dock_pose']=w.dock
            if args.rgb:
                idx=trace['rgb_action_index']; rgb=trace['rgb_external']
                if rgb.shape!=(len(idx),128,192,3) or not np.array_equal(idx,np.arange(0,len(u),6)):
                    raise RuntimeError('RGB alignment mismatch')
            res=turning_score(trace,w.initial_object,layout)
            folder=args.out/('accepted' if res['success'] else 'rejected'); folder.mkdir(exist_ok=True)
            npz=folder/(tag+'.npz'); np.savez_compressed(npz,**trace)
            a,b=[int(np.flatnonzero(phase==k)[0]) for k in (3,6)]
            res|={'case':tag,'source_case':src,'source_trace_sha256':record['sha256'],'layout':layout,
                  'subtask_intervals':[[PHASE_NAMES[0],0,a],[PHASE_NAMES[1],a,b],[PHASE_NAMES[2],b,len(phase)]],
                  'route_segments':route,'source_and_target_steps':time_scale,'rgb_frames':len(trace.get('rgb_external', [])),
                  'trace':str(npz.relative_to(args.out)),'trace_sha256':digest(npz)}
            rows.append(res); (args.out/'attempts.json').write_text(json.dumps(rows,indent=2)+'\n'); print(json.dumps(res),flush=True)
    finally:
        p.disconnect()
    m={'algorithm':'X7 MoMaGen-style source segment retarget in SE(2); not upstream MoMaGen',
       'conditions':'A pose fixed, common rigid tote, different yaw and positions at dock B, source arm/jaw commands',
       'source_report_sha256':digest(args.sources/'report.json'),'source_qa_sha256':digest(args.source_qa),
       'asset_sha256':model_sha,'attempts_sha256':digest(args.out/'attempts.json'),
       'source_seeds':list(pool),'excluded_holdout':list(k for k,v in LAYOUTS.items() if v['split']=='heldout'),
       'task_training_started':False,'counts':{'attempts':len(rows),'accepted':sum(r['success'] for r in rows)},
       'physics_hz':HZ,'control_hz':CONTROL_HZ,'rgb_hz':5 if args.rgb else 0}
    (args.out/'summary.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2),flush=True)
    return 0 if m['counts']['accepted'] else 1


if __name__=='__main__':
    raise SystemExit(main())
