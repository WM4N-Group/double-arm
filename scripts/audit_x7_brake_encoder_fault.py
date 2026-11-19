#!/usr/bin/env python3
"""Paired common brake-encoder fault versus a separate sampled backup signal."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_brake_monitor import PAUSE,READY_FRAMES,exercise_plan
from audit_x7_brake_redundancy import LAYOUTS,RedundantWorld
from audit_x7_preslow_transport import STRETCH,retime
from audit_x7_terminal_permit import PROOF,V2
from audit_x7_transport_front_stop import CLEARANCE
from stress_x7_rgb_transport import (ArmIK,GRIP_Z,HANDLE_HALF_SPAN,PARK_Y,SCENARIOS,calc,digest,execute,
    load_episode_assets,make_action,transport_prior,write_json)

CASES=('clear-separated','push-valid-separated','push-frozen-shared','push-frozen-separated','push-frozen-backup-drop')
SAMPLE=2
LAG=2
MAX_AGE=4


def take_packet(packet,tick):
    if packet is None:return None
    sample,q=packet
    if q is None or not LAG<=tick-sample<=MAX_AGE or np.shape(q)!=(2,2) or not np.isfinite(q).all():return None
    return q


class EncoderWorld(RedundantWorld):
    def __init__(self,*a,condition,**kw):
        self.condition=condition;self.frozen_q=None;self.packet=None;self.backup_packet_queue=deque();self.read_log=[]
        self.measured_main=None;self.measured_backup=None
        super().__init__(*a,fixture='clean-redundant',**kw)

    def primary_brake_encoder(self):return self.measured_main.copy() if self.measured_main is not None else super().primary_brake_encoder()

    def backup_brake_encoder(self):return None if self.measured_backup is None else self.measured_backup.copy()

    def drive(self,a):
        frame=self.physical_k
        if frame is None:return super().drive(a)
        tick=8*frame+self.sub;true=self.qv(self.base)[:2].copy()
        frozen=frame>=self.exposure and self.condition.startswith('push-frozen-')
        mode_shared=self.condition=='push-frozen-shared';drop=self.condition=='push-frozen-backup-drop' and frame>=self.exposure
        if frozen and self.frozen_q is None:
            self.frozen_q=np.column_stack((true[:,0],np.zeros(2)))
        main=self.frozen_q if frozen else true
        sent=False
        if tick%SAMPLE==0:
            # This is an idealized, separately faulted position/rate sensor
            # sampled from the same physical chassis, not URDF brake hardware.
            self.backup_packet_queue.append((tick+LAG,tick,None if drop else true.copy()));sent=not drop
        if self.backup_packet_queue and self.backup_packet_queue[0][0]<=tick:
            _,s,q=self.backup_packet_queue.popleft();self.packet=(s,q)
        backup=main if mode_shared else take_packet(self.packet,tick)
        stamp=tick if mode_shared else -1 if self.packet is None else self.packet[0]
        self.measured_main=main;self.measured_backup=backup
        self.read_log.append([tick,int(frozen),int(mode_shared),int(drop),int(sent),stamp,int(backup is not None),
            *true.ravel(),*main.ravel(),*(backup.ravel() if backup is not None else [-1.]*4)])
        super().drive(a)


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
            onset=end-44;params=SCENARIOS['clean' if case=='clear-separated' else 'push-limited-drive']
            home=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            w=EncoderWorld(asset,home,geom,prior,(lo,terminal),net,norm,external_end=end,arm_frame=terminal,
                           drop=False,params=params,insert=insert,condition=case,exposure=onset)
            t=execute(w,u,ph);ok,near,m,foot=calc(t,w,geom);stop=w.events[0] if w.events else None
            e=np.asarray(w.effects);motor=np.asarray(w.motor_log);backup=np.asarray(w.backup_log)
            force=np.flatnonzero(np.any(e[:,4:7]!=0,axis=1));want=np.arange(onset,onset+24) if params['force_n'] else np.zeros(0,dtype=int)
            if not np.array_equal(force,want) or not np.all(e[force,8]==8) or w.proof_steps!=8*PROOF:raise ValueError('Disturbance/proof incomplete')
            contact=np.any(t['station_clearance'][...,2:4]>0,axis=2).reshape(-1)
            kept=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
            still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
            tick=stop['substep'] if stop else None
            freeze=bool(stop and np.all(motor[tick:,1:17]==motor[tick,1:17]) and np.max(motor[tick:,15:17])<.015)
            t.update(front_sensor=np.asarray(w.front_log),arm_jaw_and_hold=motor,disturbance=e,
                     brake_signals=np.asarray(w.read_log),primary_auxiliary_brake=np.asarray(w.brake_log),backup_brake=backup)
            out=a.out/'episodes'/f'{layout}__{case}';out.mkdir(parents=True);f=out/'trace.npz';np.savez_compressed(f,**t)
            r=dict(layout=layout,case=case,checkpoint_sha256=h,trace_sha256=digest(f),guard_event=stop,
                proof_events=w.proof_events,normal_pass=bool(ok and w.announced and not stop),
                holding_pass=bool(stop and near>=CLEARANCE and not contact.any() and kept and still and freeze),
                station_minimum_m=near,first_contact_substep=int(np.flatnonzero(contact)[0]) if contact.any() else None,
                station_contact_substeps=int(contact.sum()),external_onset_frame=onset,push_frames=len(force),
                backup_peak_n=float(np.linalg.norm(backup[:,5:7],axis=1).max()),
                shared_or_separate_signal_is_modeled=True,hardware_claim=False)
            write_json(out/'report.json',r);print(json.dumps(r),flush=True)
    finally:p.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('pilot','fit','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',plan())
    elif a.worker is not None and 0<=a.worker<a.shards:worker(a)
    else:ap.error('Choose plan or worker')


if __name__=='__main__':main()
