#!/usr/bin/env python3
"""Opt-in X7 terminal-transport front range stop; never releases or reverses on a fault."""
import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_chassis_range_proxy import PROXY_GUARD_M, RADIUS_M, gaps_from_fractions, make_mounts, transform_mounts
from audit_x7_dock_sensor import BIAS, GUARD_SCAN_M, LAG, MAX_AGE, PERIOD, POS_ERROR_BOUND, WAVE
from run_x7_mobile_bimanual import HZ
from stress_x7_rgb_transport import (ArmIK, ChallengeWorld, EFFORT, GRIP_Z, HANDLE_HALF_SPAN, PARK_Y, POSES,
    SCENARIOS, calc, digest, execute, load_episode_assets, make_action, transport_prior, write_json)

V2='015f9adc8fbad453c4dfcae875160f2e71948abfb55c8bf68ea4b4977e0133c7'
CLEARANCE=.005
BUFFER=.001
RESPONSE_S=.055
DECEL_ASSUMPTION=.55  # Bench design assumption, not an actuator guarantee under external loads.
FAULT_INTO_PHASE=48
SETTINGS=tuple(SCENARIOS)+('range-dropout',)


def packet_bound(raw,age,travel):
    if raw is None or not 0<=age<=MAX_AGE or not np.isfinite([raw,travel]).all() or travel<0:
        return None
    return raw*math.cos(math.radians(3))-travel-POS_ERROR_BOUND-PROXY_GUARD_M


def trip(raw,age,travel,v):
    lower=packet_bound(raw,age,travel)
    if lower is None or not math.isfinite(v) or v<0:return 'unknown',lower,None
    need=CLEARANCE+BUFFER+v*RESPONSE_S+v*v/(2*DECEL_ASSUMPTION)
    return ('near' if lower<=need else None),lower,need


class FrontStopWorld(ChallengeWorld):
    def __init__(self,*a,arm_frame,drop,params):
        self.arm_frame=arm_frame;self.drop=drop;self.sub=0;self.travel=0.
        self.queue=deque();self.delivered=None;self.front_log=[];self.motor_log=[]
        self.events=[];self.hold=None;self.last_physics_action=None
        # Pose serves only to physically attach this virtual scanner to the
        # chassis. The decision consumes fractions and a separate rate channel.
        asset=a[0]; self.front_mounts=make_mounts(json.loads((asset.parent/'base-envelope.json').read_text()))[0]
        super().__init__(*a,params=params,layout=a[2])

    def drive(self,a):
        if self.physical_k is None:return super().drive(a)
        k=self.physical_k; tick=8*k+self.sub
        q=self.qv(self.base);v=float(np.linalg.norm(q[:2,1])+RADIUS_M*abs(q[2,1]))
        self.travel+=v/HZ
        if k>=self.arm_frame-10 and tick%PERIOD==0:
            lo,hi=transform_mounts(self.front_mounts,q[:,0],-1)
            rs=p.rayTestBatch(lo.tolist(),hi.tolist(),numThreads=1)
            try:
                raw=float(min(np.min(gaps_from_fractions([r[2] for r in rs]))+
                              np.sin(.77*(tick//PERIOD)+.43)*WAVE+BIAS,GUARD_SCAN_M)) if len(rs)==len(lo) else None
            except ValueError:raw=None
            if self.drop and self.arm_frame+FAULT_INTO_PHASE<=k<self.arm_frame+FAULT_INTO_PHASE+3:raw=None
            self.queue.append((tick+LAG,tick,raw,self.travel))
        if self.queue and self.queue[0][0]<=tick:self.delivered=self.queue.popleft()
        if self.delivered is None:stamp,raw,motion=-1,None,0.
        else:_,stamp,raw,s=self.delivered;motion=max(0.,self.travel-s)
        reason,lower,need=trip(raw,tick-stamp,motion,v)
        if self.hold is None and k>=self.arm_frame and reason:
            self.hold=(self.last_physics_action if self.last_physics_action is not None else a).copy()
            self.events.append(dict(substep=tick,frame=k,reason=reason,sample_substep=stamp,
                lower_bound_m=lower,braking_threshold_m=need,sampled_range_m=raw,measured_motion_rate=v))
        actual=a if self.hold is None else self.hold
        super().drive(actual)
        if self.hold is not None:
            # Preserve the independent force and restricted authority from
            # ChallengeWorld. Stop never increases available actuator effort.
            _,_,eff=self.applied_disturbance(k)
            p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,targetVelocities=[0.]*3,
                                        forces=np.asarray(EFFORT)*eff)
        self.last_physics_action=actual.copy();self.sub+=1
        self.front_log.append([tick,stamp,v,-1. if raw is None else raw,motion,
                               -1. if lower is None else lower,-1. if need is None else need,int(self.hold is not None)])
        self.motor_log.append([tick,*actual[6:],int(self.hold is not None)])

    def step(self,a):
        self.sub=0
        return super().step(a)


def panel():return [(name,s) for name in POSES for s in SETTINGS]


def run(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Remote CPU only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anchor,net,normal,h,_=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    if h!=V2 or json.loads((a.out/'plan.json').read_text())!=[list(r) for r in panel()]:raise ValueError('Panel/checkpoint changed')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,(name,scenario) in enumerate(panel()):
            if i%a.shards!=a.worker:continue
            layout=POSES[name];params=SCENARIOS['clean' if scenario=='range-dropout' else scenario]
            u,ph,meta=make_action(anchor,asset,arms,layout);n=meta['fitted_phase_durations'];p0,lo,hi=transport_prior(layout,n)
            homes=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            w=FrontStopWorld(asset,homes,layout,p0,(lo,hi),net,normal,arm_frame=sum(n[:8]),drop=scenario=='range-dropout',params=params)
            t=execute(w,u,ph);motion=np.asarray(w.effects);motor=np.asarray(w.motor_log);sensor=np.asarray(w.front_log)
            selected=np.asarray(w.applied);t.update(action=selected,anchor_action=u,disturbance=motion,front_sensor=sensor,arm_jaw_and_hold=motor)
            _,near,m,foot=calc(t,w,layout); clean=near>=CLEARANCE and not m['robot_station_contact_substeps'];latched=bool(w.events)
            hold_start=w.events[0]['substep'] if latched else None
            held=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
            still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
            jaw_lock=bool(latched and np.all(motor[hold_start:,1:17]==motor[hold_start,1:17]) and np.max(motor[hold_start:,15:17])<.015)
            interrupt=bool(latched and clean and held and still and jaw_lock)
            result=dict(layout=name,scenario=scenario,checkpoint_sha256=h,latched=latched,event=w.events[0] if latched else None,
                normal_task_pass=bool(not latched and m['success'] and foot['full_bottom_supported'] and clean),
                qualified_hold=interrupt,held_box=held,final_base_still=still,arms_jaws_frozen_closed=jaw_lock,
                minimum_station_m=near,contact_substeps=m['robot_station_contact_substeps'],metrics=m,
                disturbance_substeps=int(motion[:,8].sum()),simulated_front_cad_rays_only=True,inference_calls=w.calls)
            folder=a.out/'episodes'/f'{name}__{scenario}';folder.mkdir(parents=True);f=folder/'trace.npz';np.savez_compressed(f,**t)
            result['trace_sha256']=digest(f);write_json(folder/'report.json',result)
            print(json.dumps({k:result[k] for k in ('layout','scenario','normal_task_pass','qualified_hold','minimum_station_m','contact_substeps','event')}),flush=True)
    finally:p.disconnect()


def summarize(a):
    rows=[]; table={}
    for n,c in panel():
        d=a.out/'episodes'/f'{n}__{c}';r=json.loads((d/'report.json').read_text());f=d/'trace.npz'
        if r['checkpoint_sha256']!=V2 or r['trace_sha256']!=digest(f):raise ValueError('Trace hash drift')
        if c!='range-dropout':
            base=a.previous/'episodes'/f'{n}__{c}__candidate'/'trace.npz'
            with np.load(f,allow_pickle=False) as t,np.load(base,allow_pickle=False) as b:
                for x,y in (('disturbance','disturbance'),('phase','phase'),('anchor_action','frozen_anchor')):
                    if not np.array_equal(t[x],b[y]):raise ValueError('Comparison no longer matched')
        rows.append(r);s=table.setdefault(c,dict(cases=0,normal_pass=0,holding_interrupt=0,latched=0,min_m=1.,contact_cases=0))
        s['cases']+=1;s['normal_pass']+=r['normal_task_pass'];s['holding_interrupt']+=r['qualified_hold'];s['latched']+=r['latched']
        s['min_m']=min(s['min_m'],r['minimum_station_m']);s['contact_cases']+=r['contact_substeps']>0
    write_json(a.out/'qa-front-stop.json',dict(checkpoint_sha256=V2,scenarios=table,cases=rows,no_retreat=True,no_hardware_claim=True))
    print(json.dumps(table),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('pilot','fit','previous','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--summarize',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',panel())
    elif a.summarize:summarize(a)
    elif a.worker is not None and 0<=a.worker<a.shards:run(a)
    else:ap.error('Choose a mode')


if __name__=='__main__':main()
