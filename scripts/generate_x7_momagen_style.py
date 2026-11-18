#!/usr/bin/env python3
"""MoMaGen-inspired X7 source-segment retarget preflight, not upstream MoMaGen.

Frozen source episodes -> phase anchors -> retrieve one paired-arm demo ->
object/station-relative targets -> simulate -> physical success gate. The
restricted first variant shares one tote geometry and translates XY only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pybullet as p

from run_x7_mobile_bimanual import (ArmIK, CONTROL_HZ, FRAME_NAMES, GRIP_Z, HZ, PARK_Y,
                                    World, execute, make_model, score)

SOURCE_REV = '5da621667669b15b97d7a220d086283b3c8ad0e5'
SPLITS = {'train': (6000, 6006), 'validation': (7000, 7002)}
HANDLE_X = .589
MOTION_PHASES = ('source_tote_approach_and_grip', 'coordinated_lift_and_transport', 'station_b_lower_and_release')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segments(t):
    """Keep exact source phases, and independently require physical milestones."""
    phase=t['phase']; obj=t['object_post']; base=t['base_post']; c=t['contact_samples']; z0=t['object_pre'][0,2]
    four=np.mean(np.all(c[:,:,:4]>.12,axis=2),axis=1)>=.75
    def first(mask):
        k=np.flatnonzero(mask)
        if len(k)==0: raise ValueError('Source or execution has no required phase event')
        return int(k[0])
    start=[first(phase==k) for k in (0,3,6)]
    boundaries=np.r_[start,len(phase)]
    events=dict(four_finger_grip=first((phase==2)&four),
                unsupported_lift=first((phase==3)&four&(obj[:,2]-z0>.055)&~np.any(t['on_table'],axis=1)),
                base_starts_transport=first((phase==4)&(abs(base[:,0,1])>.055)),
                station_b_release=first((phase>=7)&(np.min(t['gripper_post'][:,:,0],axis=1)>.030)&(t['on_table'][:,1]>0)))
    if list(events.values())!=sorted(events.values()): raise ValueError('Events out of order')
    return {'segments':[{"name":name,"start_inclusive":int(a),"end_exclusive":int(b),
        "arm_coordination":'dual' if n==1 else 'two_approaches_then_paired_grip' if n==0 else 'paired_release',
        "anchor":('tote_at_A','common_tote_transport_frame','B_station')[n]} for n,(name,a,b) in
        enumerate(zip(MOTION_PHASES,boundaries[:-1],boundaries[1:]))], 'physical_event_control_indices':events}


def read_sources(folders, model_sha):
    pool=[]
    for folder in folders:
        r=folder/'report.json'; report=json.loads(r.read_text())
        if (report['asset_sha256']!=model_sha or
            not report.get('collision_contract','').startswith('v2 fully collidable station compounds')):
            raise ValueError(f'Archived source model mismatch: {folder}')
        for row in report['case_reports']:
            if row['mode']!='mobile' or not row['success'] or row.get('robot_station_clearance_pass') is not True:
                continue
            path=folder/row['trace']
            if digest(path)!=row['trace_sha256']:
                raise ValueError(f'Source trace hash mismatch: {path}')
            with np.load(path,allow_pickle=False) as f:
                t={k:f[k] for k in f.files}
            geom=(HANDLE_X,float(t['initial_object_xyz'][1]),float(t['initial_object_xyz'][0]),float(t['action'][-1,0]))
            pool.append(dict(seed=row['seed'],path=str(path),sha256=row['trace_sha256'],geometry=geom,
                             trace=t,events=segments(t)))
    if len(pool)<2 or len({r['sha256'] for r in pool})!=len(pool) or len({r['seed'] for r in pool})!=len(pool):
        raise ValueError('Need at least two unique successful source demonstrations')
    return pool


def target_geometry(seed, split):
    rng=np.random.default_rng(seed)
    if split=='validation':
        x=float(rng.uniform(2.651,2.689))
    else:
        x=float(rng.uniform(*( (2.565,2.632) if seed%2==0 else (2.71,2.80) )))
    return HANDLE_X,float(PARK_Y+rng.uniform(-.008,.007)),float(rng.uniform(-.018,.018)),x


def retrieve(pool, target):
    return min(pool,key=lambda s:(np.linalg.norm((np.array(s['geometry'][1:])-target[1:])/[.020,.035,.30]),s['seed']))


def retarget(source, target, urdf):
    s=source['trace']; phase=s['phase']; old=s['action']; action=old.copy()
    _,sy,sx,travel_s=source['geometry']; hx,ty,tx,travel_t=target
    action[:,:2]*=travel_t/travel_s
    # Capture a single object-/station-frame translation in world space for each
    # phase. In transit both end effectors use the same moving reference, never
    # two independent object trajectories. This restricted task has no yaw.
    source_anchor=np.column_stack([sx+old[:,0],np.full(len(phase),sy),np.zeros(len(phase))])
    target_anchor=np.column_stack([tx+action[:,0],np.full(len(phase),ty),np.zeros(len(phase))])
    delta=target_anchor-source_anchor
    delta[:,0]+=old[:,0]-action[:,0]  # convert from world poses to chassis-local IK
    arms=[]; errors=[]
    for arm,sign in enumerate((-1,1)):
        home=np.array([tx+sign*hx,ty+.015,GRIP_Z+.105]); kin=ArmIK(urdf,arm,home)
        default_orientation=kin.orientation
        kin.home=kin.solve(home,s['arm_pre'][0,arm*7:(arm+1)*7,0])
        old_joints=old[:,2+arm*7:9+arm*7]
        ix=sorted(set(range(0,len(old),4))|{0,len(old)-1}|set(np.flatnonzero(np.r_[True,phase[1:]!=phase[:-1]])))
        knots=[]; previous=kin.home.copy(); prev_ref=None
        for k in ix:
            xyz,r,_=kin.fk(old_joints[k]); xyz+=delta[k]
            ref=np.r_[xyz,r.as_quat()]
            if prev_ref is None or np.max(abs(ref-prev_ref))>1e-10:
                kin.orientation=r
                previous=kin.solve(xyz,previous)
                errors.append(kin.err(xyz,previous)[0]); prev_ref=ref
            knots.append(previous)
        kin.orientation=default_orientation
        action[:,2+arm*7:9+arm*7]=np.column_stack([np.interp(np.arange(len(old)),ix,np.asarray(knots)[:,j]) for j in range(7)])
        arms.append(kin)
    return arms, action, phase, max(errors)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('asset',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cad-zip',type=Path,required=True)
    ap.add_argument('--sources',type=Path,required=True,nargs='+')
    ap.add_argument('--rgb',action='store_true');ap.add_argument('--review-seed',type=int,choices=[6000])
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': ap.error('Remote CPU only: set CUDA_VISIBLE_DEVICES to empty')
    args.out.mkdir(parents=True)
    urdf=make_model(args.asset,args.out/'model',args.cad_zip); model_sha=digest(urdf); pool=read_sources(args.sources,model_sha)
    (args.out/'source-segments.json').write_text(json.dumps([{k:r[k] for k in ('seed','sha256','geometry','events')} for r in pool],indent=2)+'\n')
    result=[]
    p.connect(p.DIRECT)
    try:
        for split,(lo,hi) in SPLITS.items():
            for seed in range(lo,hi):
                target=target_geometry(seed,split); src=retrieve(pool,target)
                meta={'seed':seed,'split':split,'source_seed':src['seed'],'source_trace_sha256':src['sha256'],
                      'new_geometry':target}
                try:
                    arms,action,phase,err=retarget(src,target,urdf)
                except RuntimeError as exc:
                    meta|={'success':False,'reason':'IK rejected candidate','detail':str(exc)}
                else:
                    world=World(urdf,arms,target,'mobile')
                    video=args.out/f'retarget-{seed}.mp4' if seed==args.review_seed else None
                    t=execute(world,action,phase,rgb=args.rgb,movie=video,review_label="RETARGETED EXPERT")
                    s=score(t,'mobile',world.initial_object)
                    # A completed motion with missing physical segment evidence is not valid training data.
                    try:
                        events=segments(t)
                    except ValueError:
                        events=None
                    s['success']=bool(s['success'] and events is not None)
                    if args.rgb and not (np.array_equal(t['rgb_action_index'],np.arange(0,len(phase),6)) and
                                         t['rgb_external'].shape==(len(t['rgb_action_index']),128,192,3) and
                                         t['rgb_external'].dtype==np.uint8):
                        raise RuntimeError('RGB schema/alignment mismatch')
                    folder=args.out/split/('accepted' if s['success'] else 'rejected'); folder.mkdir(parents=True,exist_ok=True)
                    f=folder/f'episode-{seed}.npz'; np.savez_compressed(f,**t)
                    meta|=s|{'phase_evidence':events,'max_retarget_ik_position_error_m':err,
                             'file':str(f.relative_to(args.out)),'sha256':digest(f),'rgb_frames':len(t.get('rgb_external',[]))}
                result.append(meta)
                print(json.dumps({k:v for k,v in meta.items() if k!='phase_evidence'}),flush=True)
                (args.out/'attempts.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:
        p.disconnect()
    m={'algorithm':'X7 MoMaGen-inspired source retrieval and common-anchor retargeting; not upstream MoMaGen execution',
       'upstream_reference_revision':SOURCE_REV,'fitted_policy':False,'task':'straight sideways tote carry, fixed geometry, translation-only',
       'asset_sha256':model_sha,'physics_hz':HZ,'control_hz':CONTROL_HZ,'camera_hz':5 if args.rgb else 0,
       'split_rule':'disjoint seeds and chassis-travel bands; one scene family, not generalization',
       'sources_manifest_sha256':digest(args.out/'source-segments.json'),'attempts_manifest_sha256':digest(args.out/'attempts.json'),
       'counts':{s:{'attempts':sum(r['split']==s for r in result),'accepted':sum(r['split']==s and r['success'] for r in result)} for s in SPLITS},
       'dataset':'accepted per-episode NPZs; RGB before indexed action; rejected candidates never enter accepted directory',
       'collision_contract':'v2 fully collidable station compounds; per-substep closest distance and contact gate'}
    (args.out/'summary.json').write_text(json.dumps(m,indent=2)+'\n')
    return 0 if all(group['accepted'] for group in m['counts'].values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
