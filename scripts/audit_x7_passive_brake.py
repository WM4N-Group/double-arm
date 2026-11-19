#!/usr/bin/env python3
"""CPU-only test of a sensorless, bounded joint-friction stop model."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_brake_encoder_fault import EncoderWorld,LAYOUTS
from audit_x7_brake_monitor import PAUSE,READY_FRAMES,exercise_plan
from audit_x7_preslow_transport import STRETCH,retime
from audit_x7_terminal_permit import PROOF,V2
from audit_x7_transport_front_stop import CLEARANCE
from stress_x7_rgb_transport import (ArmIK,GRIP_Z,HANDLE_HALF_SPAN,PARK_Y,SCENARIOS,calc,digest,execute,
    load_episode_assets,make_action,transport_prior,write_json)

STRONG=(900.,900.,320.)
WEAK=(160.,160.,100.)
EPSILON_CAP=1e-4  # Positive floor keeps the solver motor readback live when isolated.
CASES={
    'clear-armed':(0,STRONG),
    'lost-no-passive':(0,(0.,0.,0.)),
    'lost-immediate':(0,STRONG),
    'lost-delay100ms':(24,STRONG),
    'lost-delay250ms':(60,STRONG),
    'lost-weak':(0,WEAK),
}


def passive_capacity(tick,latch,setting):
    delay,capacity=setting
    return capacity if latch is not None and tick>=latch+delay else (0.,0.,0.)


class PassiveWorld(EncoderWorld):
    def __init__(self,*a,setting,**kw):
        self.setting=setting;self.passive_log=[];
        super().__init__(*a,**kw)

    def drive(self,a):
        frame=self.physical_k
        if frame is None:return super().drive(a)
        tick=8*frame+self.sub
        # Readback of the *previous* integration step for audit only. It is
        # intentionally excluded from the passive engagement decision.
        prev=[z[3] for z in p.getJointStates(self.robot,self.base)]
        super().drive(a)
        latch=self.events[0]['substep'] if self.hold is not None else None
        caps=passive_capacity(tick,latch,self.setting)
        if latch is not None:
            caps=tuple(max(EPSILON_CAP,c) for c in caps)
            # Isolate the existing drive on both sides of this comparison.
            # The same simulated joint then carries only the bounded zero-
            # relative-velocity resisting constraint. No encoder data are
            # supplied to its engage/force-cap logic. This is an idealized
            # joint-friction model in Bullet, not two validated actuators.
            p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,
                                        targetVelocities=[0.,0.,0.],forces=caps)
        self.passive_log.append([tick,int(latch is not None),*caps,*prev])


def plan():return [(l,c) for l in LAYOUTS for c in CASES]


def worker(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU physics only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anch,net,norm,h,_=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    if h!=V2 or json.loads((a.out/'plan.json').read_text())!=[list(x) for x in plan()]:raise ValueError('Frozen inputs changed')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,(layout,case) in enumerate(plan()):
            if i%a.shards!=a.worker:continue
            geom=LAYOUTS[layout];old,ph0,meta=make_action(anch,asset,arms,geom);counts=meta['fitted_phase_durations']
            prior,lo,_=transport_prior(geom,counts);u,ph,terminal,_,end=retime(old,ph0,counts,STRETCH)
            insert=terminal+round(.30*(end-terminal));u,ph=exercise_plan(u,ph,insert);end+=PAUSE+READY_FRAMES
            onset=end-44;clean=case=='clear-armed';params=SCENARIOS['clean' if clean else 'push-limited-drive']
            home=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            w=PassiveWorld(asset,home,geom,prior,(lo,terminal),net,norm,external_end=end,arm_frame=terminal,
                           drop=False,params=params,insert=insert,condition='clear-separated' if clean else 'push-frozen-backup-drop',
                           exposure=onset,setting=CASES[case])
            t=execute(w,u,ph);ok,near,m,foot=calc(t,w,geom);stop=w.events[0] if w.events else None
            e=np.asarray(w.effects);motor=np.asarray(w.motor_log);passive=np.asarray(w.passive_log)
            force=np.flatnonzero(np.any(e[:,4:7]!=0,axis=1));wanted=np.zeros(0,dtype=int) if clean else np.arange(onset,onset+24)
            if not np.array_equal(force,wanted) or not np.all(e[force,8]==8) or w.proof_steps!=8*PROOF:raise ValueError('Exposure/proof mismatch')
            hits=np.any(t['station_clearance'][...,2:4]>0,axis=2).reshape(-1)
            still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
            kept=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
            k=stop['substep'] if stop else None
            locked=bool(stop and np.all(motor[k:,1:17]==motor[k,1:17]) and np.max(motor[k:,15:17])<.015)
            active=np.flatnonzero(np.any(passive[:,2:5]>EPSILON_CAP,axis=1))
            t.update(front_sensor=np.asarray(w.front_log),arm_jaw_and_hold=motor,disturbance=e,
                brake_signals=np.asarray(w.read_log),primary_auxiliary_brake=np.asarray(w.brake_log),backup_brake=np.asarray(w.backup_log),
                passive_joint_constraint=passive)
            d=a.out/'episodes'/f'{layout}__{case}';d.mkdir(parents=True);f=d/'trace.npz';np.savez_compressed(f,**t)
            r=dict(layout=layout,case=case,checkpoint_sha256=h,trace_sha256=digest(f),external_onset_frame=onset,
                push_frames=len(force),guard_event=stop,engage_tick=int(active[0]) if len(active) else None,
                normal_pass=bool(ok and w.announced and not stop),holding_pass=bool(stop and near>=CLEARANCE and not hits.any() and kept and still and locked),
                minimum_station_m=near,first_contact_substep=int(np.flatnonzero(hits)[0]) if hits.any() else None,
                station_contact_substeps=int(hits.sum()),stop_path_does_not_use_encoder=True,bench_friction_constraint=True,
                hardware_claim=False)
            write_json(d/'report.json',r);print(json.dumps(r),flush=True)
    finally:p.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('pilot','fit','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',plan())
    elif a.worker is not None and 0<=a.worker<a.shards:worker(a)
    else:ap.error('Choose plan or worker')


if __name__=='__main__':main()
