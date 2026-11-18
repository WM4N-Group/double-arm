#!/usr/bin/env python3
"""Frozen X7 geometry-conditioned controller under known pose-estimate biases.

World geometry and all success labels always use the true configuration. Only
arguments to the already fitted command generator are perturbed. No refit,
heldout replay, real hardware, or change to physically modeled collision.
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from generate_x7_pickup_anchor import PickupWorld, check_pose
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, TABLE_Z, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, dock_pose, r2, review_turn
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest
from x7_station_collision import station_parts

FROZEN_CHECKPOINT='0bf62bec7ef5cc7c14caebf248d3536ff8df9df1dbbe2819a8a433ea9fab3e37'
LAYOUTS={
    'north': {'goal':(3.37,.178,27.5),'bay_y':.70,'tote':(.005,-.010,.9)},
    'south': {'goal':(3.30,.113,-26.5),'bay_y':.665,'tote':(-.006,-.010,-1.0)},
}
# XY in each declared frame; angles in degrees. The B normal is its local Y.
ZERO=(0.,0.,0.)
CASES={
    'exact': (ZERO,ZERO),
    'pickup-x+5mm':((.005,0.,0.),ZERO),
    'pickup-x-5mm':((-.005,0.,0.),ZERO),
    'pickup-x+12mm':((.012,0.,0.),ZERO),
    'pickup-yaw+1deg':((0.,0.,1.),ZERO),
    'pickup-yaw+3deg':((0.,0.,3.),ZERO),
    'dock-in-5mm':(ZERO,(0.,-.005,0.)),
    'dock-in-10mm':(ZERO,(0.,-.010,0.)),
    'dock-in-20mm':(ZERO,(0.,-.020,0.)),
    'dock-out+10mm':(ZERO,(0.,.010,0.)),
    'dock-yaw+1.5deg':(ZERO,(0.,0.,1.5)),
    'combined':((.005,0.,1.),(0.,-.005,1.)),
}
SOUTH=('exact','pickup-x+5mm','pickup-yaw+1deg','dock-in-10mm','dock-yaw+1.5deg','combined')


def perceived_pose(true, pickup, station):
    l=dict(true)
    l['tote']=tuple(np.asarray(true['tote'])+pickup)
    dock=dock_pose(true)
    xy=dock[:2]+r2(dock[2])@np.asarray(station[:2])
    l['goal']=(*map(float,xy),float(true['goal'][2]+station[2]))
    return l


def check_footprint(t, true):
    # Intentionally measures the full *flat bottom* of the tote, not the
    # overhanging handles; deck support after release is the only intent.
    bottom=np.array([[-.50,-.095,0.],[-.50,.095,0.],[.50,-.095,0.],[.50,.095,0.]])
    pose=t['object_post']; dock=dock_pose(true); local_yaw=r2(dock[2]);
    _,deck,dims,_=station_parts(0.,PARK_Y,TABLE_Z,[0.,0.,0.,1.])[0]
    last=(t['phase']==9)&(t['on_table'][:,1]>0)&(np.min(t['gripper_post'][:,:,0],axis=1)>.03)
    if not np.any(last): return {'full_bottom_supported':False,'bottom_support_margin_m':None,'bottom_support_frames':0}
    margins=[]
    for row in pose[last]:
        xy=(Rotation.from_quat(row[3:7]).apply(bottom)[:,:2]+row[:2]-dock[:2])@local_yaw
        margin=np.min(np.asarray(dims[:2])*.5-abs(xy-deck[:2]));margins.append(margin)
    return {'full_bottom_supported':np.count_nonzero(np.asarray(margins)>.001)>=25,
            'bottom_support_margin_m':float(min(margins)),'bottom_support_frames':int(np.count_nonzero(np.asarray(margins)>.001))}


def failure_evidence(t):
    ph=t['phase']; a=t['station_clearance']; force=t['contact_samples']; pose=t['base_post']
    frame=np.flatnonzero(np.any(a[...,2:4]>0,axis=(1,2)))
    moving_turn=(ph==4)&(np.abs(pose[:,2,1])>.06)
    return {'first_robot_station_contact_control_index':int(frame[0]) if len(frame) else None,
            'last_robot_station_contact_control_index':int(frame[-1]) if len(frame) else None,
            'turning_finger_contact_rates':np.mean(force[moving_turn,:,:4]>.12,axis=(0,1)).tolist() if np.any(moving_turn) else None}


def case_run(checkpoint,asset,arms,name,true,verify_against=None,video=None,window=None):
    pickup,station=CASES[name];obs=perceived_pose(true,pickup,station)
    rec={'case':name,'true_geometry':true,'perceived_geometry':obs,'pickup_bias_xy_m_yaw_deg':pickup,
         'dock_local_bias_xy_m_yaw_deg':station,'checkpoint_sha256':FROZEN_CHECKPOINT}
    try:
        u,ph,contract=make_action(checkpoint,asset,arms,obs)
    except (ValueError,RuntimeError,AssertionError) as exc:
        return None,rec|{'success':False,'physical_execution':False,'setup_rejection':str(exc)}
    if verify_against is not None:
        assert np.array_equal(u,verify_against['action']) and np.array_equal(ph,verify_against['phase'])
    init=[SimpleNamespace(home=u[0,6+i*7:13+i*7]) for i in range(2)]
    w=PickupWorld(asset,init,true)
    def review_window(path,mode,label):
        enc,draw=review_turn(path,mode,label)
        def frame(k,*values):
            if window is None or window[0]<=k<window[1]:
                draw(k,*values)
        return enc,frame
    trace=execute(w,u,ph,movie=video,reviewer=review_window,review_label='ANCHOR BIAS TEST / FAILED')
    r=check_pose(trace,w,true); footprint=check_footprint(trace,true)
    result=rec|r|contract|footprint|failure_evidence(trace)|{'existing_task_pass':bool(r['success']),
             'success':bool(r['success'] and footprint['full_bottom_supported']),'physical_execution':True}
    result['full_bottom_supported']=bool(result['full_bottom_supported'])
    trace['perceived_dock_pose']=dock_pose(obs);trace['true_dock_pose']=w.dock
    trace['true_tote_delta']=np.asarray(true['tote']);trace['perceived_tote_delta']=np.asarray(obs['tote'])
    if verify_against is not None:
        fields=('base_post','arm_pre','gripper_post','object_post','contact_samples','station_clearance','on_table','on_floor')
        assert all(np.array_equal(trace[k],verify_against[k]) for k in fields)
        result['repeat_exact_fields']=list(fields)
        result['review_window_control_frames']=window
    return trace,result


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--layout',choices=LAYOUTS,required=True);ap.add_argument('--cases',choices=CASES,nargs='+')
    ap.add_argument('--review-failure',type=Path,help='Exact repeat a single archived failed trace for video')
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':ap.error('CPU only')
    cases=args.cases or (list(CASES) if args.layout=='north' else list(SOUTH))
    if len(set(cases))!=len(cases) or args.review_failure and len(cases)!=1:ap.error('Select distinct cases; review uses one case')
    args.out.mkdir(parents=True);path=args.pilot/'anchored-hybrid.npz';m=json.loads((args.pilot/'report.json').read_text());
    assert digest(path)==m['checkpoint_sha256']==FROZEN_CHECKPOINT and m['heldout']['success']
    with np.load(path,allow_pickle=False) as f: model={k:f[k] for k in f.files}
    asset=args.pilot/'model'/'x7_proxy_gripper.urdf'; true=LAYOUTS[args.layout]
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    (args.out/'manifest.json').write_text(json.dumps({'layout':args.layout,'true':true,'cases':{c:CASES[c] for c in cases},
        'fixed_checkpoint':FROZEN_CHECKPOINT,'heldout_reused':False,'measured_target_only':True},indent=2)+'\n')
    p.connect(p.DIRECT);rows=[]
    try:
        for c in cases:
            if args.review_failure:
                src=args.review_failure; archived=json.loads((src.parent/'report.json').read_text()); ref=next(r for r in archived['cases'] if r['case']==c)
                assert not ref['success'] and ref['physical_execution'] and ref['trace_sha256']==digest(src)
                with np.load(src,allow_pickle=False) as old:
                    idx=ref['first_robot_station_contact_control_index']
                    if idx is None:
                        idx=int(np.flatnonzero(old['phase']==2)[0])
                    window=(max(0,idx-60),min(len(old['phase']),idx+180))
                    trace,score=case_run(model,asset,arms,c,true,verify_against=old,video=args.out/(c+'-FAILED.mp4'),window=window)
                assert trace is not None and not score['success']
            else:
                trace,score=case_run(model,asset,arms,c,true)
            if trace is not None:
                f=args.out/(c+'.npz'); np.savez_compressed(f,**trace);score['trace_sha256']=digest(f)
            rows.append(score);print(json.dumps({k:score.get(k) for k in ('case','success','physical_execution','robot_station_clearance_pass','station_min_clearance_m','both_grip_substep_fraction_turn','bottom_support_margin_m','setup_rejection')},allow_nan=False),flush=True)
            (args.out/'attempts.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    finally:
        p.disconnect()
    (args.out/'report.json').write_text(json.dumps({'cases':rows,'checkpoint_sha256':FROZEN_CHECKPOINT,'layout':args.layout,
        'trained_new_weights':False,'heldout_reused':False},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
