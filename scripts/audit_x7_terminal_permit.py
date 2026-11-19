#!/usr/bin/env python3
"""Simulated independent brake proof before terminal entry, with fail-closed permit."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_preslow_transport import STRETCH, PreslowWorld, retime
from audit_x7_transport_front_stop import CLEARANCE,V2,trip
from audit_x7_standoff_wait import decide_standoff
from run_x7_turning_carry import r2
from stress_x7_rgb_transport import (ArmIK, EFFORT, GRIP_Z, HANDLE_HALF_SPAN, PARK_Y, POSES, SCENARIOS,
    calc,digest,execute,load_episode_assets,make_action,transport_prior,write_json)

CASES=('proof-clean','proof-reject','permit-revoked','permit-late-push','permit-late-dual-fault')
SETTLE=18
PROOF=18
RECOVER=14
PAUSE=SETTLE+PROOF+RECOVER
PROOF_N=620.
BRAKE_MAX_N=1100.
MAX_TRAVEL_M=.008
MAX_SETTLED_M_S=.006


def permission(travel,rate,front_lower,proof_steps):
    return bool(np.isfinite([travel,rate,front_lower]).all() and 0<=travel<=MAX_TRAVEL_M and
                0<=rate<=MAX_SETTLED_M_S and front_lower>=.24 and proof_steps==8*PROOF)


def insert_proof(u,ph,insert):
    if ph[insert-1]!=4 or ph[insert]!=4:raise ValueError('Proof must start while in transport')
    hold=u[insert-1].copy();hold[3:6]=0.
    return np.concatenate((u[:insert],np.tile(hold,(PAUSE,1)),u[insert:])),np.r_[ph[:insert],np.full(PAUSE,4,dtype=ph.dtype),ph[insert:]]


class PermitWorld(PreslowWorld):
    def __init__(self,*a,insert,fault_case,**kwargs):
        self.insert=insert;self.end_proof=insert+PAUSE;self.fixture_case=fault_case
        self.granted=False;self.proof_ref=None;self.proof_steps=0;self.proof_max=0.;self.brake_ref=None
        self.proof_events=[];self.brake_log=[];self.revocation_frame=self.end_proof+40
        super().__init__(*a,**kwargs)
        self.force_direction=np.r_[r2(self.challenge_yaw)@[0.,-1.],0.]

    def fixture_heartbeat(self,k):
        # Bench independent signal, separate from both actuator and pushes.
        return self.fixture_case!='permit-revoked' or k<self.revocation_frame

    def fixture_brake_limit(self,k):
        # The controller never branches on this artificial failure flag.
        if self.fixture_case=='proof-reject':return 0.
        if self.fixture_case=='permit-late-dual-fault' and k>=self.external_end-44:return 0.
        return BRAKE_MAX_N

    def front_decision(self,raw,age,travel,rate):
        k=self.physical_k
        _,lower,_=trip(raw,age,travel,rate)
        if not self.granted and k>=self.end_proof and self.hold is None:
            ok=lower is not None and permission(self.proof_max,rate,lower,self.proof_steps)
            self.proof_events.append(dict(kind='grant' if ok else 'reject',frame=k,travel_m=self.proof_max,
                                          rate_m_s=rate,front_lower_m=lower,steps=self.proof_steps))
            if not ok:return 'proof_rejected',lower,None
            self.granted=True
        if self.granted:
            if not self.fixture_heartbeat(k):return 'permit_revoked',lower,None
            return trip(raw,age,travel,rate)
        return decide_standoff(raw,age,travel,rate)

    def drive(self,a):
        k=self.physical_k
        if k is None:return super().drive(a)
        tick=8*k+self.sub;fixture_limit=self.fixture_brake_limit(k)
        q=self.qv(self.base);pos=q[:2,0];vel=q[:2,1]
        at_proof=self.insert<=k<self.end_proof
        loading=self.insert+SETTLE<=k<self.insert+SETTLE+PROOF
        if loading and self.proof_ref is None:self.proof_ref=pos.copy()
        if self.proof_ref is not None and k<=self.end_proof:
            self.proof_max=max(self.proof_max,float(np.linalg.norm(pos-self.proof_ref)))
        super().drive(a)
        brake_on=at_proof or self.hold is not None
        primary_zero=bool(loading)
        if brake_on:
            if self.brake_ref is None:self.brake_ref=pos.copy()
            _,_,eff=self.applied_disturbance(k)
            p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,targetVelocities=[0.]*3,
                                        forces=np.zeros(3) if primary_zero else np.asarray(EFFORT)*eff)
            requested=-150000.*(pos-self.brake_ref)-23000.*vel
            norm=float(np.linalg.norm(requested));act=requested*min(1.,fixture_limit/max(norm,1e-12))
            center=p.getLinkState(self.robot,self.base_link,computeForwardKinematics=True)[0]
            if np.any(act):p.applyExternalForce(self.robot,self.base_link,forceObj=[*act,0.],posObj=center,flags=p.WORLD_FRAME)
            if loading:
                p.applyExternalForce(self.robot,self.base_link,forceObj=self.force_direction*PROOF_N,posObj=center,flags=p.WORLD_FRAME)
                self.proof_steps+=1
        else:
            act=np.zeros(2);self.brake_ref=None
        self.brake_log.append([tick,int(at_proof),int(loading),int(primary_zero),int(brake_on),float(np.linalg.norm(act)),
                               fixture_limit,int(self.granted),int(self.fixture_heartbeat(k)),self.proof_max])


def plan():return [(l,c) for l in POSES for c in CASES]


def run(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU physics only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anchor,net,norm,h,_=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    if h!=V2 or json.loads((a.out/'plan.json').read_text())!=[list(x) for x in plan()]:raise ValueError('Checkpoint/plan drift')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,(layout,case) in enumerate(plan()):
            if i%a.shards!=a.worker:continue
            geom=POSES[layout];orig,ph0,meta=make_action(anchor,asset,arms,geom);counts=meta['fitted_phase_durations']
            prior,lo,_=transport_prior(geom,counts);u,ph,terminal,_,end=retime(orig,ph0,counts,STRETCH)
            insert=terminal+round(.30*(end-terminal));u,ph=insert_proof(u,ph,insert);end+=PAUSE
            params=SCENARIOS['push-limited-drive' if case in ('permit-revoked','permit-late-push','permit-late-dual-fault') else 'clean']
            homes=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            w=PermitWorld(asset,homes,geom,prior,(lo,terminal),net,norm,external_end=end,arm_frame=terminal,drop=False,
                          params=params,insert=insert,fault_case=case)
            t=execute(w,u,ph);normal,near,m,foot=calc(t,w,geom);event=w.events[0] if w.events else None
            e=np.asarray(w.effects);br=np.asarray(w.brake_log);motor=np.asarray(w.motor_log)
            force=np.flatnonzero(np.any(e[:,4:7]!=0,axis=1));want=np.arange(end-44,end-20) if params['force_n'] else np.zeros(0,dtype=int)
            if not np.array_equal(force,want) or not np.all(e[force,8]==8) or w.proof_steps!=PROOF*8:
                raise ValueError('External or proof stimulus was not fully physically applied')
            contacts=int(np.count_nonzero(np.any(t['station_clearance'][...,2:4]>0,axis=2)))
            held=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
            still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
            if event:
                k=event['substep'];frozen=bool(np.all(motor[k:,1:17]==motor[k,1:17]) and np.max(motor[k:,15:17])<.015)
            else:frozen=False
            t.update(front_sensor=np.asarray(w.front_log),arm_jaw_and_hold=motor,secondary_brake=br,disturbance=e,
                     original_action=orig,original_phase=ph0,pre_guard_selection=np.asarray(w.applied))
            r=dict(layout=layout,case=case,checkpoint_sha256=h,proof_start_frame=insert,proof_end_frame=insert+PAUSE,
                   independent_push_end_clock=end,permit_events=w.proof_events,guard_event=event,
                   normal_task_pass=bool(normal and w.granted and not event),qualified_hold=bool(event and contacts==0 and near>=CLEARANCE and held and still and frozen),
                   minimum_station_m=near,contact_substeps=contacts,external_push_frames=len(force),
                   modeled_separate_brake=True,proof_load_source_is_independent=True,hardware_claim=False)
            out=a.out/'episodes'/f'{layout}__{case}';out.mkdir(parents=True);f=out/'trace.npz';np.savez_compressed(f,**t)
            r['trace_sha256']=digest(f);write_json(out/'report.json',r);print(json.dumps(r),flush=True)
    finally:p.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('pilot','fit','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',plan())
    elif a.worker is not None and 0<=a.worker<a.shards:run(a)
    else:ap.error('Choose plan or worker')


if __name__=='__main__':main()
