#!/usr/bin/env python3
"""Remote X7 simulation: latch a dock-distance hazard, brake, retreat, never release.

The proximity observation is Bullet's independently measured signed robot-to-
station distance. It does not use the goal bias supplied to the trajectory.
No hardware safety claim: this is an oracle-aided, substep-only physics test.
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import CASES, FROZEN_CHECKPOINT, LAYOUTS, check_footprint, perceived_pose
from generate_x7_pickup_anchor import PickupWorld, check_pose
from run_x7_mobile_bimanual import ArmIK, CONTROL_HZ, GRIP_Z, HZ, PARK_Y, execute, smooth
from run_x7_turning_carry import HANDLE_HALF_SPAN, dock_pose, r2, review_turn
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest
from x7_station_collision import STATION_CLEARANCE_M

TRAIN_CASES={
 'north':('exact','dock-in-5mm','dock-in-10mm','dock-in-20mm','fault-dropout'),
 'south':('exact','dock-in-10mm','combined'),
}
MIN_TRIGGER_M=.0076
GUARD_SCAN_M=.32
RETREAT_M=.014
RETREAT_SUBSTEPS=168
STABLE_SUBSTEPS=30
CODE={'pass':0,'brake':1,'retreat':2,'hold':3}


def clearance_trigger(velocity, angle):
    # Estimated approach speed along the B dock normal in world coordinates.
    # Nominal values here are test tuning, not a validated stop-distance model.
    close=max(0.,-float(np.dot(velocity[:2],r2(angle)[:,1])))
    return max(MIN_TRIGGER_M, STATION_CLEARANCE_M+.0012+.026*close+close**2/(2.*.60))


class DockGuard:
    def __init__(self, angle):
        self.angle=float(angle);self.state='pass';self.events=[];self.reason=None
        self.origin=None;self.hold=None;self.held_cmd=None;self.still=0;self.s=0
        self.retreat_origin=None;self.best_clear=-np.inf;self.cannot_retreat=False

    def switch(self,k,state):
        self.state=state;self.events.append({'substep':k, 'state':state})

    def update(self,k,qv,dist,hits,a):
        good=bool(dist is not None and len(dist)==2 and np.isfinite(dist).all() and max(dist)<=GUARD_SCAN_M+1e-6)
        trigger=clearance_trigger(qv[:,1],self.angle)
        minimum=float(min(dist)) if good else -1.
        if self.state=='pass':
            if good and minimum>trigger and not sum(hits):return a,trigger
            self.origin=qv[:,0].copy();self.hold=self.origin.copy();self.held_cmd=a.copy()
            self.reason=('station_clearance' if good else 'proximity_unavailable')
            self.cannot_retreat=(not good or dist[0]<.030 or dist[1]!=minimum or dist[1]>=GUARD_SCAN_M)
            self.events.append({'substep':k,'type':'latch','reason':self.reason,
                'distance_m':dist,'trigger_m':trigger,'actual_base':self.origin.tolist()})
            self.switch(k,'brake')
        if not good:
            self.cannot_retreat=True
            if self.state=='retreat':
                self.hold=qv[:,0].copy();self.switch(k,'hold')
        b=self.held_cmd.copy();b[:3]=self.hold;b[3:6]=0.
        if self.state=='brake':
            stopped=np.linalg.norm(qv[:2,1])<.004 and abs(qv[2,1])<.012
            self.still=self.still+1 if stopped else 0
            if self.still>=STABLE_SUBSTEPS:
                if self.cannot_retreat or not good or dist[0]<.030 or sum(hits) or minimum<.005:
                    self.switch(k,'hold')
                else:
                    self.retreat_origin=qv[:,0].copy();self.best_clear=dist[1]
                    self.switch(k,'retreat')
        if self.state=='retreat':
            self.best_clear=max(self.best_clear,dist[1])
            if sum(hits) or minimum<STATION_CLEARANCE_M or dist[1]<self.best_clear-.0015:
                self.hold=qv[:,0].copy();self.switch(k,'hold')
            else:
                self.s+=1
                u=min(1.,self.s/RETREAT_SUBSTEPS)
                move=np.r_[RETREAT_M*r2(self.angle)[:,1],0.]
                b[:3]=self.retreat_origin+move*smooth(u)
                b[3:6]=move*30*u*u*(1.-u)*(1.-u)*HZ/RETREAT_SUBSTEPS
                if self.s>=RETREAT_SUBSTEPS:
                    self.hold=b[:3].copy();self.switch(k,'hold')
        if self.state=='hold':b[:3]=self.hold;b[3:6]=0.
        return b,trigger


class GuardWorld(PickupWorld):
    def __init__(self,asset,arms,true,theta,arm_stage_frame,dropout):
        self.armed=False;self.tick=0;self.guard=DockGuard(theta);self.timelog=[];self.applied=[]
        self.arm_stage=arm_stage_frame;self.dropout=dropout
        super().__init__(asset,arms,true)
        self.armed=True

    def drive(self,a):
        # World drives the original physics step with this returned command.
        if not self.armed:
            return super().drive(a)
        k=len(self.applied); input_applied=a; dist=None; hits=[0,0]; trigger=-1.
        active=self.tick>=self.arm_stage or self.guard.state!='pass'
        if active:
            d=[min((float(c[8]) for c in p.getClosestPoints(self.robot,t,GUARD_SCAN_M)),default=GUARD_SCAN_M)
               for t in self.tables]
            hits=[len(p.getContactPoints(self.robot,t)) for t in self.tables]
            # Inject exactly one absent reading, without changing the real geometry.
            dist=d if k!=self.dropout else None
            input_applied,trigger=self.guard.update(k,self.qv(self.base),dist,hits,a)
        self.applied.append(input_applied.copy());self.timelog.append([k,int(active),CODE[self.guard.state],trigger,
                     *([-1.,-1.] if dist is None else dist),int(dist is not None)])
        return super().drive(input_applied)

    def step(self,a):
        y=super().step(a);self.tick+=1
        return y


def guard_metrics(w,t,true):
    r=check_pose(t,w,true); full=check_footprint(t,true)
    log=np.asarray(w.timelog);a=np.asarray(w.applied);guard=w.guard;active=log[:,2]>0
    stopped_final=(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
    if np.any(active):
        frozen_close=bool(np.array_equal(a[active,-2:],np.tile(guard.held_cmd[-2:],(np.count_nonzero(active),1)))
                          and np.max(a[active,-2:])<.012)
        gripper_still_holds=np.all(t['contact_samples'][-32:,:,:4]>.12)
    else:
        frozen_close=False;gripper_still_holds=False
    safe=bool(np.any(active) and r['robot_station_clearance_pass'] and frozen_close and gripper_still_holds and stopped_final
              and r['state_writes_during_actions']==0 and not r['touched_floor'])
    if guard.retreat_origin is None:
        retreat=0.
    else:
        retreat=float(np.dot(t['base_post'][-1,:2,0]-guard.retreat_origin[:2],r2(guard.angle)[:,1]))
    success=bool(not np.any(active) and r['success'] and full['full_bottom_supported'])
    return r|full|{'success':success,'normal_task_pass_not_guarded':success,
          'guard_triggered':bool(np.any(active)),'guard_reason':guard.reason,'guard_states':guard.events,
          'safe_holding_interrupt':safe,'latched_without_opening':frozen_close,'retreat_measured_m':retreat,
          'final_base_stopped':bool(stopped_final),'held_box_after_stop':bool(gripper_still_holds),
          'guard_distance_source':'simulator signed robot-station proximity (oracle), not goal estimate or a deployable sensor'}


def run_case(weights,asset,arms,l,name,out,archived=None):
    case=name if name!='fault-dropout' else 'exact';obs=perceived_pose(l,*CASES[case]);u,ph,meta=make_action(weights,asset,arms,obs)
    counts=meta['fitted_phase_durations']; arm=int(sum(counts[:8])); dropout=8*(arm+int(.85*counts[8])) if name=='fault-dropout' else -1
    home=[SimpleNamespace(home=u[0,6+i*7:13+i*7]) for i in range(2)];w=GuardWorld(asset,home,l,dock_pose(obs)[2],arm,dropout)
    if archived is None:
        t=execute(w,u,ph);video=None
    else:
        video=out/(name+'-guarded.mp4');k=int(archived['guard_states'][0]['substep']//8)
        begin,end=max(0,k-90),min(len(u),k+180)
        def window(path,mode,label):
            enc,draw=review_turn(path,mode,label)
            def f(i,phase,*args):
                if begin<=i<end:draw(i,phase if w.guard.state=='pass' else 5,*args)
            return enc,f
        t=execute(w,u,ph,movie=video,reviewer=window,review_label='DOCK GUARD / STOP & RETREAT')
    r=guard_metrics(w,t,l);t['guard_substep_telemetry']=np.asarray(w.timelog);t['actual_motor_references_by_substep']=np.asarray(w.applied)
    if archived:
        assert r['safe_holding_interrupt'] and r['guard_states']==archived['guard_states']
    npz=out/(name+'.npz');np.savez_compressed(npz,**t)
    r|={'case':name,'trace_sha256':digest(npz),'simulated_dropped_sensor_substep':dropout if dropout>=0 else None,
        'true_geometry':l,'perceived_geometry':obs,'guard_armed_at_control_frame':arm}
    if archived:
        r['review_video_sha256']=digest(video);r['review_control_interval']=[begin,end]
    return r


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--layout',choices=LAYOUTS,required=True);ap.add_argument('--cases',choices=CASES.keys()|{'fault-dropout'},nargs='+')
    ap.add_argument('--review-archive',type=Path);args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':ap.error('remote CPU only')
    cases=args.cases or TRAIN_CASES[args.layout];assert len(set(cases))==len(cases)
    if args.review_archive:assert len(cases)==1
    args.out.mkdir(parents=True);weights_path=args.pilot/'anchored-hybrid.npz';assert digest(weights_path)==FROZEN_CHECKPOINT
    with np.load(weights_path,allow_pickle=False) as f: weights={k:f[k] for k in f.files}
    asset=args.pilot/'model/x7_proxy_gripper.urdf';l=LAYOUTS[args.layout]
    arms=[ArmIK(asset,i,np.array([sign*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,sign in enumerate((-1,1))]
    archived=None
    if args.review_archive:
        original=json.loads(args.review_archive.read_text());assert original['layout']==args.layout and original['checkpoint_sha256']==FROZEN_CHECKPOINT
        archived=next(r for r in original['cases'] if r['case']==cases[0]);assert archived['guard_triggered'] and archived['safe_holding_interrupt']
    results=[];p.connect(p.DIRECT)
    try:
        for c in cases:
            r=run_case(weights,asset,arms,l,c,args.out,archived)
            if archived:assert r['trace_sha256']==archived['trace_sha256']
            results.append(r);print(json.dumps({k:r[k] for k in ('case','success','guard_reason','safe_holding_interrupt','station_min_clearance_m','guard_states','retreat_measured_m','robot_station_contact_substeps')},allow_nan=False),flush=True)
            (args.out/'attempts.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    finally:
        p.disconnect()
    (args.out/'report.json').write_text(json.dumps({'cases':results,'layout':args.layout,'checkpoint_sha256':FROZEN_CHECKPOINT,
        'trained_new_weights':False,'heldout_reused':False,'guard_scan_radius_m':GUARD_SCAN_M,
        'sensor':'simulation oracle, evaluated at each physics substep'},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
