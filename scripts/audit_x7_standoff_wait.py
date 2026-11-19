#!/usr/bin/env python3
"""Staging-only pilot: uncertified terminal approach stays locked, even on a clear path."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_preslow_transport import STRETCH, PreslowWorld, retime
from audit_x7_transport_front_stop import CLEARANCE, V2, trip
from collect_x7_transport_bc import sha_layout
from run_x7_turning_carry import capture_turn_rgb
from stress_x7_rgb_transport import (ArmIK, GRIP_Z, HANDLE_HALF_SPAN, PARK_Y, POSES, SCENARIOS,
    calc, digest, execute, load_episode_assets, make_action, transport_prior, write_json)

# Measured bench results motivate this trial value, but do not certify it
# for untested entry speeds, actuator failures, or external force histories.
RESERVE_M=.240
FRESH={
    'west-fresh':dict(goal=(3.312,.153,-29.32),bay_y=.682,tote=(.0011,-.0102,-.6)),
    'east-fresh':dict(goal=(3.304,.097,29.38),bay_y=.704,tote=(-.0021,-.0112,1.1)),
}
ALL=POSES|FRESH
CASES=('unverified-clear','push-early','push-staging','push-late','missing-range','push-with-missing-range')


def decide_standoff(raw,age,travel,rate):
    reason,lower,need=trip(raw,age,travel,rate)
    if need is None:return reason,lower,need
    # There is deliberately no fixture-controlled "verified" branch. A
    # clear optical beam alone cannot grant a physical braking guarantee.
    required=need+RESERVE_M
    return ('standoff_clearance' if lower<=required else None),lower,required


def spec(case,terminal,old_end,new_end):
    horizon=new_end-terminal
    if case=='push-early':force_start=terminal+math.ceil(.32*horizon)
    elif case=='push-staging':force_start=terminal+math.ceil(.52*horizon)
    elif case=='push-late':force_start=new_end-44
    elif case=='push-with-missing-range':force_start=terminal+48
    elif case in ('unverified-clear','missing-range'):force_start=None
    else:raise ValueError(case)
    drop=case in ('missing-range','push-with-missing-range')
    return (SCENARIOS['clean'],old_end,drop) if force_start is None else (SCENARIOS['push-limited-drive'],force_start+44,drop)


class StandoffWorld(PreslowWorld):
    def front_decision(self,raw,age,travel,rate):return decide_standoff(raw,age,travel,rate)


def plan():return [(l,c) for l in ALL for c in CASES]


def worker(a):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU physics only')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    asset,anch,net,norm,h,_=load_episode_assets(a.pilot,a.fit/'dataset',a.fit/'training')
    published=json.loads((a.fit/'dataset/layouts.json').read_text())['layouts']
    known={sha_layout(x) for x in published}|{sha_layout(x) for x in POSES.values()}
    if h!=V2 or {sha_layout(x) for x in FRESH.values()}&known or json.loads((a.out/'plan.json').read_text())!=[list(x) for x in plan()]:
        raise ValueError('Geometry, checkpoint or plan drift')
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,(layout,case) in enumerate(plan()):
            if i%a.shards!=a.worker:continue
            geom=ALL[layout];old,ph0,meta=make_action(anch,asset,arms,geom);counts=meta['fitted_phase_durations']
            prior,lo,_=transport_prior(geom,counts);u,ph,terminal,old_end,new_end=retime(old,ph0,counts,STRETCH)
            params,schedule,drop=spec(case,terminal,old_end,new_end)
            home=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
            w=StandoffWorld(asset,home,geom,prior,(lo,terminal),net,norm,external_end=schedule,
                            arm_frame=terminal,drop=drop,params=params)
            t=execute(w,u,ph,rgb=True,camera=capture_turn_rgb);_,near,m,foot=calc(t,w,geom)
            e=np.asarray(w.effects);motor=np.asarray(w.motor_log);sensor=np.asarray(w.front_log);rgb_i=t['rgb_action_index'];
            if (not np.array_equal(rgb_i,np.arange(0,len(ph),6)) or t['rgb_external'].shape!=(len(rgb_i),128,192,3)
                    or len(sensor)!=8*len(ph) or not np.array_equal(np.arange(len(sensor)),sensor[:,0])):
                raise ValueError('Perception/actuation grid alignment failed')
            event=w.events[0] if w.events else None; contacts=int(np.count_nonzero(np.any(t['station_clearance'][...,2:4]>0,axis=2)))
            held=bool(np.all(t['contact_samples'][-32:,:,:4]>.12) and not m['touched_floor'])
            still=bool(np.linalg.norm(t['base_post'][-1,:2,1])<.003 and abs(t['base_post'][-1,2,1])<.01)
            k=event['substep'] if event else None
            freeze=bool(event and np.all(motor[k:,1:17]==motor[k,1:17]) and np.max(motor[k:,15:17])<.015)
            physical=bool(event and near>=CLEARANCE and contacts==0 and held and still and freeze and
                          m['state_writes_during_actions']==0 and m['object_attachment_constraints_seen']==0)
            good=bool(physical and event['reason']==('unknown' if drop else 'standoff_clearance'))
            force=np.flatnonzero(np.any(e[:,4:7]!=0.,axis=1));want=np.arange(schedule-44,schedule-20) if params['force_n'] else np.zeros(0,dtype=int)
            if not np.array_equal(force,want) or not np.all(e[force,8]==8):raise ValueError('Fixture clock was not maintained')
            # Explicit labels for observation-only interruption records. The
            # intended action trace is not a demonstration of actual commands.
            t.update(nominal_plan_before_guard=t.pop('action'),selected_plan_before_guard=np.asarray(w.applied),
                front_sensor=sensor,arm_jaw_and_hold=motor,disturbance=e,original_action=old,original_phase=ph0,
                rgb_range_packet_at_pre_step=np.where(rgb_i>0,sensor[np.maximum(rgb_i*8-1,0),1],-1).astype(np.int64),
                rgb_hold_already_latched=((rgb_i*8)>k) if k is not None else np.zeros(len(rgb_i),dtype=bool))
            out=a.out/'episodes'/f'{layout}__{case}';out.mkdir(parents=True);f=out/'trace.npz';np.savez_compressed(f,**t)
            r=dict(layout=layout,fresh_layout=layout in FRESH,case=case,checkpoint_sha256=h,
                terminal_start=terminal,original_terminal_end=old_end,retimed_terminal_end=new_end,scenario_clock=schedule,
                event=event,minimum_station_m=near,contact_substeps=contacts,held_box=held,final_still=still,frozen_closed=freeze,
                qualified_holding_interrupt=good,physical_hold_ignoring_trigger_class=physical,
                task_completed=bool(m['success'] and foot['full_bottom_supported'] and not event),
                active_force_frames=len(force),rgb_observations=len(rgb_i),training_data=False,trace_sha256=digest(f))
            write_json(out/'report.json',r);print(json.dumps(r),flush=True)
    finally:p.disconnect()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for k in ('pilot','fit','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--plan',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--shards',type=int,default=8);a=ap.parse_args()
    if a.plan:a.out.mkdir();write_json(a.out/'plan.json',plan())
    elif a.worker is not None and 0<=a.worker<a.shards:worker(a)
    else:ap.error('Choose a mode')


if __name__=='__main__':main()
