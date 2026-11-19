#!/usr/bin/env python3
"""Opt-in periodic brake exercise with delayed synthetic load-cell feedback."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_preslow_transport import STRETCH,retime
from audit_x7_standoff_wait import FRESH
from audit_x7_terminal_permit import BRAKE_MAX_N,PAUSE,PROOF,PermitWorld,V2,insert_proof
from audit_x7_transport_front_stop import CLEARANCE
from stress_x7_rgb_transport import (ArmIK, EFFORT, GRIP_Z, HANDLE_HALF_SPAN, PARK_Y, POSES, SCENARIOS,
    calc,digest,execute,load_episode_assets,make_action,transport_prior,write_json)

LAYOUTS=POSES|FRESH
SCENES=('monitor-clean','early-loss','late-loss','late-healthy','early-dropout','late-false-feedback')
SAMPLE=8
DELAY=8
MAX_AGE=18
READY_FRAMES=6
TOLERANCE_N=8.


def probe(tick):return 48. if (tick//SAMPLE)%2==0 else 80.


def feedback_valid(packet,tick):
    if packet is None:return False
    stamped,raw=packet
    return bool(raw is not None and np.isfinite(raw) and DELAY<=tick-stamped<=MAX_AGE and abs(raw-probe(stamped))<=TOLERANCE_N)


class ExerciseWorld(PermitWorld):
    def __init__(self,*a,insert,case,exposure,**kw):
        self.scenario=case;self.exposure=exposure;self.feedback_queue=deque();self.feedback=None;self.feedback_events=[]
        self.exercise=[];self.announced=False
        super().__init__(*a,insert=insert,fault_case='proof-clean',**kw)
        self.motion_release=insert+PAUSE+READY_FRAMES

    def fixture_brake_limit(self,k):
        # Only the actuator model reads the created physical fault.
        return 0. if k>=self.exposure and self.scenario in ('early-loss','late-loss','late-false-feedback') else BRAKE_MAX_N

    def front_decision(self,raw,age,travel,rate):
        frame=self.physical_k;tick=8*frame+self.sub
        if self.feedback_queue and self.feedback_queue[0][0]<=tick:
            _,s,z=self.feedback_queue.popleft();self.feedback=(s,z)
        reason,lower,required=super().front_decision(raw,age,travel,rate)
        if not self.granted or self.hold is not None:return reason,lower,required
        valid=feedback_valid(self.feedback,tick)
        if frame>=self.motion_release and not valid:
            stamped,value=self.feedback if self.feedback is not None else (-1,None)
            self.feedback_events.append(dict(substep=tick,sample_substep=stamped,reported_force_n=value,kind='invalid_feedback'))
            return 'brake_feedback_invalid',lower,required
        if frame>=self.motion_release and valid and not self.announced and reason is None:
            stamped,value=self.feedback;self.announced=True
            self.feedback_events.append(dict(substep=tick,sample_substep=stamped,reported_force_n=value,kind='terminal_motion_allowed'))
        return reason,lower,required

    def drive(self,a):
        frame=self.physical_k
        if frame is None:return super().drive(a)
        tick=8*frame+self.sub
        super().drive(a)
        req=applied=0.; sensed=None
        if self.granted and self.hold is None and frame>=self.end_proof:
            req=probe(tick);applied=min(req,self.fixture_brake_limit(frame))
            center=p.getLinkState(self.robot,self.base_link,computeForwardKinematics=True)[0]
            # An independent low-force exercise physically opposes approach.
            # The transducer is a *synthetic* readback of this applied force.
            if applied>0.:p.applyExternalForce(self.robot,self.base_link,forceObj=-self.force_direction*applied,posObj=center,flags=p.WORLD_FRAME)
            if self.end_proof<=frame<self.motion_release:
                p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,targetVelocities=[0.]*3,forces=EFFORT)
            if tick%SAMPLE==0:
                if not (self.scenario=='early-dropout' and frame>=self.exposure):
                    sensed=(req if self.scenario=='late-false-feedback' and frame>=self.exposure else applied)+np.sin(tick*.31)
                self.feedback_queue.append((tick+DELAY,tick,sensed))
        delivered_f,delivered_raw=self.feedback if self.feedback is not None else (-1,None)
        self.exercise.append([tick,req,applied,int(req>0 and tick%SAMPLE==0),-1. if sensed is None else sensed,
                              delivered_f,-1. if delivered_raw is None else delivered_raw,int(self.announced)])


def plan():return [(l,c) for l in LAYOUTS for c in SCENES]


def exercise_plan(u,ph,insert):
    u,ph=insert_proof(u,ph,insert);end=insert+PAUSE
    block=u[end-1].copy();block[3:6]=0.
    return np.concatenate((u[:end],np.tile(block,(READY_FRAMES,1)),u[end:])),np.r_[ph[:end],np.full(READY_FRAMES,4),ph[end:]]


def run(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anchor,net,norm,h,_=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    if h!=V2 or json.loads((a.out/'plan.json').read_text())!=[list(item) for item in plan()]:raise ValueError('Input drift')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for index,(layout,case) in enumerate(plan()):
            if index%a.shards!=a.worker:continue
            geom=LAYOUTS[layout];old,ph0,meta=make_action(anchor,asset,arms,geom);counts=meta['fitted_phase_durations']
            prior,lo,_=transport_prior(geom,counts);u,ph,terminal,_,end=retime(old,ph0,counts,STRETCH)
            insert=terminal+round(.30*(end-terminal));u,ph=exercise_plan(u,ph,insert);end+=PAUSE+READY_FRAMES
            # Both early and late pulses have fixed global clocks chosen before
            # any controller or diagnostic result. They run through a hold.
            event_frame=insert+PAUSE+16 if case.startswith('early-') else end-44
            params=SCENARIOS['clean' if case=='monitor-clean' else 'push-limited-drive']
            homes=[SimpleNamespace(home=u[0,6+7*i:13+7*i]) for i in range(2)]
            w=ExerciseWorld(asset,homes,geom,prior,(lo,terminal),net,norm,external_end=event_frame+44,arm_frame=terminal,
                            drop=False,params=params,insert=insert,case=case,exposure=event_frame)
            t=execute(w,u,ph);task_ok,near,m,foot=calc(t,w,geom);stop=w.events[0] if w.events else None
            actuator=np.asarray(w.brake_log);exercise=np.asarray(w.exercise);dist=np.asarray(w.effects);motor=np.asarray(w.motor_log)
            actual=np.flatnonzero(np.any(dist[:,4:7]!=0,axis=1));expect=np.arange(event_frame,event_frame+24) if params['force_n'] else np.zeros(0,dtype=int)
            if w.proof_steps!=8*PROOF or not np.array_equal(actual,expect) or not np.all(dist[actual,8]==8):
                raise ValueError('Independent proof/force aperture lost')
            contact=np.any(t['station_clearance'][...,2:4]>0,axis=2).reshape(-1)
            held=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
            still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
            locked=bool(stop and np.all(motor[stop['substep']:,1:17]==motor[stop['substep'],1:17]) and np.max(motor[stop['substep']:,15:17])<.015)
            t.update(front_sensor=np.asarray(w.front_log),arm_jaw_and_hold=motor,brake_proof=actuator,brake_exercise=exercise,
                     disturbance=dist,original_frozen_route=old,actuator_bounds_are_simulated=np.array([1],dtype=np.uint8))
            d=a.out/'episodes'/f'{layout}__{case}';d.mkdir(parents=True);f=d/'trace.npz';np.savez_compressed(f,**t)
            r=dict(layout=layout,case=case,checkpoint_sha256=h,trace_sha256=digest(f),fixture_clock=event_frame,
                   proof_events=w.proof_events,feedback_events=w.feedback_events,guard_event=stop,external_push_frames=len(actual),
                   normal_pass=bool(task_ok and w.announced and not stop),qualified_hold=bool(stop and near>=CLEARANCE and held and still and locked and not contact.any()),
                   minimum_station_m=near,first_contact_substep=int(np.flatnonzero(contact)[0]) if contact.any() else None,
                   contact_substeps=int(contact.sum()),monitored_readback_is_synthetic=True,modeled_auxiliary_brake=True,
                   targeted_training_data=False)
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
