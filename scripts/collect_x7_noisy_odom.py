#!/usr/bin/env python3
"""X7 v2 development data: physically qualified odometry-recovery demonstrations.

Oracle teacher reads the injected bias with a two-control-frame delay. Learner
inputs are the biased odometry and pre-action RGB, never the oracle or true pose.
The live traces are not heldout and are not policy validation.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import check_footprint
from collect_x7_transport_bc import FROZEN_CHECKPOINT, layouts, validate_trace
from generate_x7_pickup_anchor import PickupWorld, check_pose
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import EFFORT, HANDLE_HALF_SPAN, WXY, capture_turn_rgb, r2
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

LAG=2


def task_plan():
    rng=np.random.default_rng(202611191421); out=[]
    train=[r for r in layouts() if r['split']=='train']
    # Same geometry repeated under opposing profile signs; neither sign can
    # be recovered solely from route direction or geometry identity.
    for row in train[:24]:
        for s in (-1,1):
            out.append(dict(case=f'noisy-{len(out):03}',parent_train_case=row['case'],layout=row,
                bias_m=float(s*rng.uniform(.008,.019)), onset_from_end=int(rng.integers(56,90)),
                ramp=int(rng.integers(14,21)),taper=int(rng.integers(11,17))))
    return out


def make_bias(n,lo,hi,row):
    b=np.zeros((n,3)); k=np.arange(n);t=k-(hi-row['onset_from_end'])
    s=np.minimum(np.clip(t/row['ramp'],0.,1.),np.clip((hi-LAG-1-k)/row['taper'],0.,1.))
    d=r2(np.radians(row['layout']['goal'][2]))@[0.,row['bias_m']]
    b[:,:2]=s[:,None]*d
    if np.any(b[:lo]) or np.any(b[hi-LAG:]):raise ValueError('Bias envelope crosses contact phase')
    return b


class NoisyOdomWorld(PickupWorld):
    def __init__(self,asset,homes,layout,bias):
        self.bias=bias;self.k=None;self.k_next=0;self.applied=[]
        super().__init__(asset,homes,layout)

    def step(self,a):
        self.k=self.k_next;self.k_next+=1;self.hit=0
        result=super().step(a);self.applied.append(self.hit);self.k=None
        return result

    def drive(self,a):
        super().drive(a)
        if self.k is None or not self.bias[self.k].any():return
        measured=self.qv(self.base)[:,0]+self.bias[self.k];err=a[:3]-measured
        err[2]=(err[2]+np.pi)%(2*np.pi)-np.pi
        v=np.clip(a[3:6]+3.5*err,-np.asarray(WXY),np.asarray(WXY))
        p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,targetVelocities=v,forces=EFFORT)
        self.hit+=1


def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def collect(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Remote CPU only')
    f=a.pilot/'anchored-hybrid.npz';asset=a.pilot/'model/x7_proxy_gripper.urdf';planfile=a.out/'plan.json'
    if digest(f)!=FROZEN_CHECKPOINT:raise ValueError('Teacher hash drift')
    if a.plan:
        a.out.mkdir();atomic(planfile,dict(teacher_sha256=FROZEN_CHECKPOINT,urdf_sha256=digest(asset),
            cases=task_plan(),teacher_observer_lag_frames=LAG,provenance='old TRAIN geometries only; synthetic odometry disturbance'))
        return
    entries=json.loads(planfile.read_text());rows=task_plan()
    if rows!=entries['cases'] or entries['urdf_sha256']!=digest(asset):raise ValueError('Plan/asset drift')
    with np.load(f,allow_pickle=False) as w:model={k:w[k] for k in w.files}
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,row in enumerate(rows):
            if i%a.shards!=a.worker:continue
            layout=row['layout'];u,ph,meta=make_action(model,asset,arms,layout);n=np.asarray(meta['fitted_phase_durations'])
            lo,hi=int(n[:4].sum()),int(n[:9].sum());b=make_bias(len(ph),lo,hi,row)
            correction=np.zeros_like(b);correction[LAG:]=b[:-LAG];actions=u.copy();actions[:,:3]+=correction
            if not np.any(b) or np.any(correction[ph!=4]) or np.array_equal(actions,u):
                raise ValueError('Invalid teacher intervention')
            homes=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            w=NoisyOdomWorld(asset,homes,layout,b);t=execute(w,actions,ph,rgb=True,camera=capture_turn_rgb)
            applied=np.asarray(w.applied);active=np.any(b!=0,axis=1)
            if not np.array_equal(applied,8*active):raise ValueError('Actuator disturbance not fully applied')
            measured=t['base_pre'].copy();measured[:,:,0]+=b
            t.update(base_policy_pre=measured,odom_bias_for_audit_only=b,teacher_compensation_for_audit_only=correction,
                original_anchor_action=u,block_counts=n,bias_drive_substeps=applied)
            m=check_pose(t,w,layout);foot=check_footprint(t,layout);ok=validate_trace(t,m,foot)
            dst=a.out/'episodes'/row['case'];dst.mkdir(parents=True);trace=dst/'trace.npz';np.savez_compressed(trace,**t)
            r=row|dict(accepted=ok,steps=len(ph),frames=len(t['rgb_external']),active_bias_frames=int(active.sum()),
                metrics=m,footprint=foot,trace_sha256=digest(trace),teacher_sha256=FROZEN_CHECKPOINT)
            atomic(dst/'report.json',r)
            print(json.dumps({k:r[k] for k in ('case','accepted','active_bias_frames','frames')}),flush=True)
    finally:p.disconnect()


def summarize(a):
    records=[]
    for row in task_plan():
        path=a.out/'episodes'/row['case'];r=json.loads((path/'report.json').read_text());f=path/'trace.npz'
        if any(r[k]!=v for k,v in row.items()) or digest(f)!=r['trace_sha256']:raise ValueError('Provenance mismatch')
        with np.load(f,allow_pickle=False) as t:
            expect=t['base_pre'].copy();expect[:,:,0]+=t['odom_bias_for_audit_only']
            if not np.array_equal(expect,t['base_policy_pre']) or validate_trace(t,r['metrics'],r['footprint'])!=r['accepted']:
                raise ValueError('Physical/observation contract mismatch')
        records.append(r)
    kept=[r for r in records if r['accepted']]
    report=dict(attempts=len(records),accepted=len(kept),action_steps=sum(r['steps'] for r in kept),
                rgb_frames=sum(r['frames'] for r in kept),active_bias_frames=sum(r['active_bias_frames'] for r in kept),
                trained=False,learned_policy_validation=False,records=records)
    atomic(a.out/'qa-noisy-odom.json',report)
    print(json.dumps({k:report[k] for k in ('attempts','accepted','action_steps','rgb_frames','active_bias_frames')}))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--summarize',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.summarize:summarize(a)
    elif a.plan or a.worker is not None and 0<=a.worker<a.shards:collect(a)
    else:ap.error('Choose plan, worker or summarize mode')


if __name__=='__main__':main()
