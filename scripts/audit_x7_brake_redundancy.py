#!/usr/bin/env python3
"""Compare truthful derating and a modeled independent backup brake in X7 physics."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_brake_monitor import ExerciseWorld,PAUSE,READY_FRAMES,exercise_plan
from audit_x7_preslow_transport import STRETCH,retime
from audit_x7_terminal_permit import BRAKE_MAX_N,PROOF,V2
from audit_x7_transport_front_stop import CLEARANCE
from collect_x7_transport_bc import sha_layout
from stress_x7_rgb_transport import (ArmIK,GRIP_Z,HANDLE_HALF_SPAN,PARK_Y,POSES,SCENARIOS,calc,digest,execute,
    load_episode_assets,make_action,transport_prior,write_json)

DERATED_N=100.
BACKUP_MAX_N=1100.
FRESH={'north-east-brake':dict(goal=(3.345,.140,29.63),bay_y=.699,tote=(.0021,-.0116,-.73)),
       'north-west-brake':dict(goal=(3.321,.132,-29.57),bay_y=.676,tote=(-.0052,-.0109,1.21))}
LAYOUTS=POSES|FRESH
CASES=('clean-redundant','late-healthy-single','late-derated-single','late-derated-redundant',
       'late-open-redundant','late-common-loss')


def limits(case,k,exposure):
    if case not in CASES:raise ValueError('Unknown actuator fixture')
    first=BRAKE_MAX_N;second=0. if case.endswith('-single') else BACKUP_MAX_N
    if k>=exposure:
        if 'derated' in case:first=DERATED_N
        if case in ('late-open-redundant','late-common-loss'):first=0.
        if case=='late-common-loss':second=0.
    return first,second


class RedundantWorld(ExerciseWorld):
    def __init__(self,*a,fixture,**kw):
        self.capacity_fixture=fixture;self.backup_anchor=None;self.backup_log=[]
        # Keep the previous synthetic transducer honest. Only the physical
        # force limit changes; no new fault-dependent controller branch.
        super().__init__(*a,case='late-healthy',**kw)

    def fixture_brake_limit(self,k):return limits(self.capacity_fixture,k,self.exposure)[0]

    def drive(self,a):
        frame=self.physical_k
        if frame is None:return super().drive(a)
        tick=8*frame+self.sub;q=self.qv(self.base);pos,vel=q[:2,0],q[:2,1]
        super().drive(a)
        available=limits(self.capacity_fixture,frame,self.exposure)[1]
        command=np.zeros(2);effect=np.zeros(2)
        if self.hold is not None:
            if self.backup_anchor is None:self.backup_anchor=pos.copy()
            # Simulate a parallel independent force-producing branch on every
            # stop, sharing the same modeled pose/rate signal in this bench.
            command=-150000.*(pos-self.backup_anchor)-23000.*vel
            effect=command*min(1.,available/max(np.linalg.norm(command),1e-12))
            if np.any(effect):
                center=p.getLinkState(self.robot,self.base_link,computeForwardKinematics=True)[0]
                p.applyExternalForce(self.robot,self.base_link,forceObj=[*effect,0.],posObj=center,flags=p.WORLD_FRAME)
        self.backup_log.append([tick,int(self.hold is not None),available,*command,*effect])


def plan():return [(l,c) for l in LAYOUTS for c in CASES]


def run(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU physics only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anch,net,norm,h,_=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    known=json.loads((a.fit/'dataset/layouts.json').read_text())['layouts']
    if h!=V2 or {sha_layout(x) for x in known}&{sha_layout(x) for x in FRESH.values()} or \
            json.loads((a.out/'plan.json').read_text())!=[list(x) for x in plan()]:raise ValueError('Layout/weight drift')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,(layout,case) in enumerate(plan()):
            if i%a.shards!=a.worker:continue
            geom=LAYOUTS[layout];old,oldph,meta=make_action(anch,asset,arms,geom);counts=meta['fitted_phase_durations']
            prior,lo,_=transport_prior(geom,counts);u,ph,terminal,_,end=retime(old,oldph,counts,STRETCH)
            insert=terminal+round(.30*(end-terminal));u,ph=exercise_plan(u,ph,insert);end+=PAUSE+READY_FRAMES
            onset=end-44;params=SCENARIOS['clean' if case=='clean-redundant' else 'push-limited-drive']
            home=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            w=RedundantWorld(asset,home,geom,prior,(lo,terminal),net,norm,external_end=end,arm_frame=terminal,
                             drop=False,params=params,insert=insert,fixture=case,exposure=onset)
            t=execute(w,u,ph);ok,near,m,foot=calc(t,w,geom);event=w.events[0] if w.events else None
            br=np.asarray(w.brake_log);ex=np.asarray(w.exercise);backup=np.asarray(w.backup_log);e=np.asarray(w.effects);motor=np.asarray(w.motor_log)
            force=np.flatnonzero(np.any(e[:,4:7]!=0,axis=1));want=np.arange(onset,onset+24) if params['force_n'] else np.zeros(0,dtype=int)
            if w.proof_steps!=8*PROOF or not np.array_equal(force,want) or not np.all(e[force,8]==8):raise ValueError('Missing calibrated stimulus')
            contacts=np.any(t['station_clearance'][...,2:4]>0,axis=2).reshape(-1)
            held=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
            still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
            k=event['substep'] if event else None
            locked=bool(event and np.all(motor[k:,1:17]==motor[k,1:17]) and np.max(motor[k:,15:17])<.015)
            t.update(front_sensor=np.asarray(w.front_log),arm_jaw_and_hold=motor,primary_auxiliary_brake=br,
                     primary_brake_exercise=ex,independent_backup_model=backup,disturbance=e)
            d=a.out/'episodes'/f'{layout}__{case}';d.mkdir(parents=True);f=d/'trace.npz';np.savez_compressed(f,**t)
            r=dict(layout=layout,case=case,checkpoint_sha256=h,trace_sha256=digest(f),force_onset_frame=onset,push_frames=len(force),
                proof_events=w.proof_events,feedback_events=w.feedback_events,guard_event=event,
                normal_pass=bool(ok and w.announced and not event),qualified_hold=bool(event and near>=CLEARANCE and
                    held and still and locked and not contacts.any()),station_minimum_m=near,contact_substeps=int(contacts.sum()),
                strongest_backup_force_n=float(np.linalg.norm(backup[:,5:7],axis=1).max()),
                separate_force_actuator_model=True,shared_simulated_encoder=True,hardware_claim=False)
            write_json(d/'report.json',r);print(json.dumps(r),flush=True)
    finally:p.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('pilot','fit','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',plan())
    elif a.worker is not None and 0<=a.worker<a.shards:run(a)
    else:ap.error('Choose plan or worker')


if __name__=='__main__':main()
