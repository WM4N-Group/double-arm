#!/usr/bin/env python3
"""Carried-load braking bench and retimed terminal approach with independent external clocks."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_transport_front_stop import CLEARANCE, DECEL_ASSUMPTION, FrontStopWorld, V2
from run_x7_mobile_bimanual import HZ
from run_x7_turning_carry import r2
from stress_x7_rgb_transport import (ArmIK, ChallengeWorld, EFFORT, GRIP_Z, HANDLE_HALF_SPAN, PARK_Y,
    POSES, SCENARIOS, calc, digest, disturbance, execute, load_episode_assets, make_action, transport_prior, write_json)

STRETCH=2.5
BENCH=('bench-original-speed','bench-slow-speed')
PANEL=('slow-clean','slow-push-fixed','slow-push-late','slow-bias16-late')


def retime(u,ph,n,factor):
    n=np.asarray(n,dtype=int);a,b=int(n[:8].sum()),int(n[:9].sum())
    if factor<1 or b<=a or a==0 or not np.all(ph[a:b]==4):raise ValueError('Invalid terminal phase')
    width=math.ceil((b-a)*factor);actual=width/(b-a)
    if width==b-a:return u.copy(),ph.copy(),a,b,b
    knot=np.concatenate((u[a-1:a],u[a:b]));ts=np.arange(width,dtype=float)+1.
    old=np.arange(len(knot),dtype=float);ref=np.column_stack([np.interp(ts/actual,old,knot[:,j]) for j in range(22)])
    ref[:,3:6]/=actual
    return np.concatenate((u[:a],ref,u[b:])),np.r_[ph[:a],np.full(width,4,dtype=ph.dtype),ph[b:]],a,b,a+width


def spec(name,original_end,new_end):
    if name in BENCH or name=='slow-push-fixed':return SCENARIOS['push-limited-drive'],original_end
    if name=='slow-push-late':return SCENARIOS['push-limited-drive'],new_end
    if name=='slow-bias16-late':return SCENARIOS['odom-inward-16mm'],new_end
    if name=='slow-clean':return SCENARIOS['clean'],original_end
    raise ValueError(name)


class IndependentClock:
    def applied_disturbance(self,k):
        return disturbance(self.challenge_params,k,self.external_end,self.challenge_yaw)


class PreslowWorld(IndependentClock,FrontStopWorld):
    def __init__(self,*a,external_end,**kwargs):
        self.external_end=external_end;super().__init__(*a,**kwargs)


class BrakeBench(IndependentClock,ChallengeWorld):
    def __init__(self,*a,external_end,**kwargs):
        self.external_end=external_end;self.saved=None;self.base_substeps=[];super().__init__(*a,**kwargs)

    def setup_stations(self,ignored):
        super().setup_stations(ignored)
        # Measurement fixture is absent from the braking corridor. The real
        # station remains at its unchanged location in the integrated panel.
        p.resetBasePositionAndOrientation(self.tables[1],[-5.,2.,0.],[0.,0.,0.,1.])

    def drive(self,a):
        k=self.physical_k
        if k is not None:
            self.base_substeps.append(self.qv(self.base).copy())
            if k>=self.external_end-44:
                if self.saved is None:self.saved=a.copy()
                super().drive(self.saved)
                _,_,eff=self.applied_disturbance(k)
                p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,targetVelocities=[0.]*3,
                                            forces=np.asarray(EFFORT)*eff)
                return
        super().drive(a)


def plan(group):return [(l,c) for l in POSES for c in (BENCH if group=='bench' else PANEL)]


def run(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Remote CPU physics only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anchor,model,norm,h,_=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    if h!=V2 or json.loads((a.out/'plan.json').read_text())!=[list(t) for t in plan(a.group)]:raise ValueError('Fitted model or plan changed')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,(layout_id,scenario) in enumerate(plan(a.group)):
            if i%a.shards!=a.worker:continue
            layout=POSES[layout_id];orig,p_orig,metadata=make_action(anchor,asset,arms,layout);counts=metadata['fitted_phase_durations']
            prior,lo,_=transport_prior(layout,counts);speed=1. if scenario=='bench-original-speed' else STRETCH
            u,ph,terminal,old_end,new_end=retime(orig,p_orig,counts,speed);params,end=spec(scenario,old_end,new_end)
            home=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            common=(asset,home,layout,prior,(lo,terminal),model,norm)
            # The same frozen model is used strictly before the terminal
            # approach. The retimed terminal block is geometric, not learned.
            if a.group=='bench':
                w=BrakeBench(*common,external_end=end,layout=layout,params=params)
                t=execute(w,u[:end+65],ph[:end+65]);q=np.asarray(w.base_substeps)
                first=8*(end-44);forward=r2(np.radians(layout['goal'][2]))@[0.,-1.]
                pre_speed=float(q[first,:2,1]@forward);dx=(q[first:,:2,0]-q[first,:2,0])@forward
                settled=np.linalg.norm(q[:, :2,1],axis=1)<.003
                # Require 8 consecutive control frames' worth of stable substeps after the scheduled push.
                ks=[k for k in range(8*(end-20),len(q)-64) if settled[k:k+64].all() and np.all(abs(q[k:k+64,2,1])<.01)]
                contact=np.count_nonzero(t['station_clearance'][...,2:4]);held=bool(np.all(t['contact_samples'][-16:,:,:4]>.12) and not np.any(t['on_floor']))
                r=dict(bench_only=True,forward_speed_at_zero_m_s=pre_speed,peak_forward_displacement_m=float(dx.max()),
                    old_unopposed_v2_over_2a_m=pre_speed**2/(2*DECEL_ASSUMPTION),first_stable_substep=ks[0] if ks else None,
                    retained_load=held,robot_station_contact_substeps=int(contact),ground_contact=bool(np.any(t['on_floor'])))
                if r['ground_contact'] or not held or contact or not ks:raise ValueError('Bench not qualified')
                t['bench_base_substeps']=q
            else:
                w=PreslowWorld(*common,external_end=end,arm_frame=terminal,drop=False,params=params)
                t=execute(w,u,ph);normal,near,m,foot=calc(t,w,layout);event=w.events[0] if w.events else None
                motor=np.asarray(w.motor_log);held=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
                still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
                cl=np.any(t['station_clearance'][...,2:4]>0,axis=2).reshape(-1);contacts=int(cl.sum())
                locked=bool(event and np.all(motor[event['substep']:,1:17]==motor[event['substep'],1:17]) and np.max(motor[event['substep']:,15:17])<.015)
                speed_window=t['base_pre'][terminal:new_end,:2,1]@r2(np.radians(layout['goal'][2]))[:,1]
                r=dict(bench_only=False,normal_task_pass=bool(normal and event is None),qualified_holding_interrupt=bool(
                    event and near>=CLEARANCE and contacts==0 and held and still and locked),latched=bool(event),event=event,
                    minimum_station_m=near,robot_station_contact_substeps=contacts,first_contact_substep=int(np.flatnonzero(cl)[0]) if contacts else None,
                    terminal_peak_approach_speed_m_s=float(-speed_window.min()),actual_policy_calls_before_terminal=w.calls,
                    held_box=held,final_base_still=still,arms_jaws_locked=locked,metrics=m)
                t['front_sensor']=np.asarray(w.front_log);t['arm_jaw_and_hold']=motor
            e=np.asarray(w.effects);selected=np.asarray(w.applied)
            active_force=np.flatnonzero(np.linalg.norm(e[:,4:7],axis=1)>0);active_bias=np.flatnonzero(np.linalg.norm(e[:,1:4],axis=1)>0)
            if params['force_n'] and not (np.array_equal(active_force,np.arange(end-44,end-20)) and np.all(e[active_force,8]==8)):
                raise ValueError('Physical push schedule changed')
            if params['bias_m'] and not (np.array_equal(active_bias,np.arange(end-71,end)) and np.all(e[active_bias,8]==8)):
                raise ValueError('Measured fault schedule changed')
            t.update(policy_selected_before_guard=selected,disturbance=e,reference_action_before_time_scaling=orig,
                     original_phase=p_orig)
            d=a.out/'episodes'/f'{layout_id}__{scenario}';d.mkdir(parents=True);f=d/'trace.npz';np.savez_compressed(f,**t)
            r.update(layout=layout_id,scenario=scenario,checkpoint_sha256=h,terminal_start_frame=terminal,
                original_end_frame=old_end,retimed_end_frame=new_end,disturbance_clock_end_frame=end,
                force_frames=len(active_force),bias_frames=len(active_bias),trace_sha256=digest(f),terminal_scale=speed)
            write_json(d/'report.json',r)
            print(json.dumps({k:v for k,v in r.items() if k not in ('metrics','checkpoint_sha256','trace_sha256')}),flush=True)
    finally:p.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('pilot','fit','out'):ap.add_argument(key,type=Path)
    ap.add_argument('--group',choices=('bench','panel'),required=True);ap.add_argument('--plan',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',plan(a.group))
    elif a.worker is not None and 0<=a.worker<a.shards:run(a)
    else:ap.error('Choose plan or valid worker')


if __name__=='__main__':main()
