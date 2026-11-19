#!/usr/bin/env python3
"""Paired physics challenge of frozen X7 base residuals; no fitting or heldout reuse."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_anchor_tolerance import check_footprint
from collect_x7_transport_bc import sha_layout
from evaluate_x7_rgb_transport import LiveTransport, load_episode_assets
from generate_x7_pickup_anchor import check_pose
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import EFFORT, HANDLE_HALF_SPAN, WXY, r2
from train_x7_anchored_turning_bc import make_action
from train_x7_rgb_transport import transport_prior, write_json
from train_x7_turning_trajectory_bc import digest

CHECKPOINT='0da95775f33f3b3975d48c01dcf30f3693ea69e28c48baecf70bbcca10ab5065'
POSES={
    'west-low':dict(goal=(3.294,.094,-26.2),bay_y=.671,tote=(-.006,-.0115,-1.29)),
    'east-low':dict(goal=(3.301,.139,26.4),bay_y=.692,tote=(.005,-.0108,1.33)),
    'west-high':dict(goal=(3.329,.119,-28.45),bay_y=.703,tote=(.003,-.0097,.71)),
    'east-high':dict(goal=(3.322,.110,28.43),bay_y=.661,tote=(-.004,-.0121,-.83)),
}
SCENARIOS={
    'clean':dict(bias_m=0.,force_n=0.,motor_fraction=1.),
    'odom-inward-8mm':dict(bias_m=.008,force_n=0.,motor_fraction=1.),
    'odom-inward-16mm':dict(bias_m=.016,force_n=0.,motor_fraction=1.),
    'push-limited-drive':dict(bias_m=0.,force_n=620.,motor_fraction=.30),
}

def plan():
    return [dict(case=f'{layout}__{scenario}__{mode}',layout=layout,scenario=scenario,mode=mode)
            for layout in POSES for scenario in SCENARIOS for mode in ('candidate','nominal')]


def disturbance(params, k, end, yaw):
    # Positive local Y measurement error drives the physical base toward B
    # (negative local Y). Affect both policy input and velocity-servo feedback;
    # the scoring/qv/physics state is never overwritten.
    t=int(k)-(int(end)-72)
    amount=params['bias_m']*min(1.,max(0.,t/18.)) if 0<=t<72 else 0.
    bias=np.r_[r2(yaw)@[0.,amount],0.]
    push_active=end-44<=k<end-20 and params['force_n']>0
    f=np.r_[r2(yaw)@[0.,-params['force_n']],0.] if push_active else np.zeros(3)
    return bias,f,params['motor_fraction'] if push_active else 1.


class ChallengeWorld(LiveTransport):
    def __init__(self,*args,params,layout):
        self.challenge_params=params;self.effects=[];self.physical_k=None;self.drive_hits=0
        self.challenge_yaw=np.radians(layout['goal'][2]);super().__init__(*args)

    def applied_disturbance(self,k):
        return disturbance(self.challenge_params,k,self.hi,self.challenge_yaw)

    def policy_base_observation(self):
        q=super().policy_base_observation().copy()
        b,_,_=self.applied_disturbance(self.physical_k)
        q[:,0]+=b
        return q

    def step(self,a):
        self.physical_k=len(self.applied)
        bias,force,fraction=self.applied_disturbance(self.physical_k)
        self.effects.append([self.physical_k,*bias,*force,fraction]);self.drive_hits=0
        result=super().step(a)
        self.effects[-1].append(self.drive_hits)
        self.physical_k=None
        return result

    def drive(self,a):
        super().drive(a)
        if self.physical_k is None:return
        b,f,effort=self.applied_disturbance(self.physical_k)
        if not (np.any(b) or np.any(f)):return
        q=self.qv(self.base)[:,0];err=a[:3]-(q+b)
        err[2]=(err[2]+np.pi)%(2*np.pi)-np.pi
        v=np.clip(a[3:6]+3.5*err,-np.asarray(WXY),np.asarray(WXY))
        p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,targetVelocities=v,
                                    forces=np.asarray(EFFORT)*effort)
        if np.any(f):
            center=p.getLinkState(self.robot,self.base_link,computeForwardKinematics=True)[0]
            p.applyExternalForce(self.robot,self.base_link,forceObj=f,posObj=center,flags=p.WORLD_FRAME)
        self.drive_hits+=1


def calc(t,w,layout):
    scores=check_pose(t,w,layout);foot=check_footprint(t,layout)
    st=t['station_clearance'];init=t['station_initialization_clearance'];near=float(min(st[...,:2].min(),init[...,:2].min()))
    passed=bool(scores['success'] and foot['full_bottom_supported'] and near>=.005 and not np.any(st[...,2:4])
                and not np.any(init[...,2:4]) and scores['state_writes_during_actions']==0 and
                scores['object_attachment_constraints_seen']==0)
    return passed,near,scores,foot


def worker(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Remote CPU only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anchor,net,norm,h,epoch=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    if h!=a.checkpoint or json.loads((a.out/'plan.json').read_text())!=plan():raise ValueError('Unfrozen inputs')
    old=json.loads((a.fit/'dataset/layouts.json').read_text())['layouts']
    if {sha_layout(l) for l in POSES.values()}&{sha_layout(l) for l in old}:raise ValueError('Layout reuse')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for j,row in enumerate(plan()):
            if a.candidate_only and row['mode']!='candidate':continue
            index=j//2 if a.candidate_only else j
            if index%a.shards!=a.worker:continue
            layout=POSES[row['layout']];params=SCENARIOS[row['scenario']];learned=row['mode']=='candidate'
            u,ph,meta=make_action(anchor,asset,arms,layout);prior,lo,hi=transport_prior(layout,meta['fitted_phase_durations'])
            homes=[SimpleNamespace(home=u[0,6+7*i:13+7*i]) for i in range(2)]
            w=ChallengeWorld(asset,homes,layout,prior,(lo,hi),net if learned else None,norm,params=params,layout=layout)
            t=execute(w,u,ph);actions=np.asarray(w.applied);effect=np.asarray(w.effects);mask=np.asarray(w.mask)
            if not (np.array_equal(actions[:,6:],u[:,6:]) and np.array_equal(actions[~mask],u[~mask])
                    and np.all(ph[mask]==4) and len(effect)==len(u) and np.all(effect[effect[:,8]>0,0]>=lo)
                    and np.all(effect[effect[:,8]>0,0]<hi) and (w.calls*6==mask.sum() if learned else w.calls==0)):
                raise ValueError('Changed command scope or incomplete disturbance trace')
            changed=effect[:,8]>0; counts=effect[changed,8];expected=0 if row['scenario']=='clean' else 24 if params['force_n'] else 71
            if changed.sum()!=expected or not np.all(counts==8):raise ValueError('Unqualified disturbance injection')
            t.update(action=actions,frozen_anchor=u,override_mask=mask,disturbance=effect,live_rgb_idx=np.asarray(w.indices))
            ok,near,metrics,foot=calc(t,w,layout)
            d=a.out/'episodes'/row['case'];d.mkdir(parents=True);path=d/'trace.npz';np.savez_compressed(path,**t)
            r=row|dict(checkpoint_sha256=h,epoch=epoch,passed=ok,minimum_station_m=near,
                metrics=metrics,footprint=foot,first_disturbed_frame=int(effect[changed,0][0]) if changed.any() else None,
                disturbing_frames=int(changed.sum()),scope='transport only; calibrated geometry for nontransport',
                applied_force_n=float(np.linalg.norm(effect[:,4:7],axis=1).max()),
                inferred=w.calls,frames=len(u),trace_sha256=digest(path))
            write_json(d/'report.json',r)
            print(json.dumps({k:r[k] for k in ('case','passed','minimum_station_m','disturbing_frames','inferred')}),flush=True)
    finally:p.disconnect()


def summarize(a):
    results={}; pairs={}
    for row in plan():
        d=a.out/'episodes'/row['case'];r=json.loads((d/'report.json').read_text());f=d/'trace.npz'
        if any(r[k]!=v for k,v in row.items()) or r['checkpoint_sha256']!=CHECKPOINT or r['trace_sha256']!=digest(f):
            raise ValueError('Unmatched paired archive')
        with np.load(f,allow_pickle=False) as t:
            s=t['station_clearance'];init=t['station_initialization_clearance'];minimum=float(min(s[...,:2].min(),init[...,:2].min()))
            if minimum!=r['minimum_station_m'] or r['passed'] and (minimum<.005 or np.any(s[...,2:4])):
                raise ValueError('Strict station audit disagreement')
            results[row['case']]=r
            group=pairs.setdefault((row['layout'],row['scenario']),{});group[row['mode']]=(r,t['disturbance'],t['frozen_anchor'],t['phase'])
    table={};all_pairs=[]
    for (layout,scenario),pair in pairs.items():
        c,n=pair['candidate'],pair['nominal']
        if not all(np.array_equal(x,y) for x,y in zip(c[1:],n[1:])):raise ValueError('Pair is not matched')
        a1,a2=c[0],n[0];series=table.setdefault(scenario,dict(pairs=0,candidate_pass=0,nominal_pass=0,
             candidate_only=0,nominal_only=0,candidate_min_station_m=1.,nominal_min_station_m=1.))
        series['pairs']+=1
        for mode,r in [('candidate',a1),('nominal',a2)]:
            series[mode+'_pass']+=r['passed'];series[mode+'_min_station_m']=min(series[mode+'_min_station_m'],r['minimum_station_m'])
        series['candidate_only']+=bool(a1['passed'] and not a2['passed']);series['nominal_only']+=bool(a2['passed'] and not a1['passed'])
        all_pairs.append(dict(layout=layout,scenario=scenario,candidate_pass=a1['passed'],nominal_pass=a2['passed'],
            candidate_min_station_m=a1['minimum_station_m'],nominal_min_station_m=a2['minimum_station_m']))
    write_json(a.out/'qa-stress.json',dict(checkpoint_sha256=CHECKPOINT,no_training=True,no_old_heldout_rollouts=True,
        scenario_table=table,pairs=all_pairs,reports=results));print(json.dumps(table),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','fit','out'):ap.add_argument(name,type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--summarize',action='store_true')
    ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8)
    ap.add_argument('--checkpoint',default=CHECKPOINT);ap.add_argument('--candidate-only',action='store_true')
    a=ap.parse_args()
    if a.summarize and (a.checkpoint!=CHECKPOINT or a.candidate_only):ap.error('Original paired summary only')
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',plan())
    elif a.summarize:summarize(a)
    elif a.worker is not None and 0<=a.worker<a.shards:worker(a)
    else:ap.error('Mandatory plan, worker or summarize mode')


if __name__=='__main__':main()
