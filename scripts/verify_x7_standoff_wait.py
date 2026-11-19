#!/usr/bin/env python3
"""Reconstruct standoff decisions and export strictly causal validation indices."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_standoff_wait import RESERVE_M, V2, decide_standoff, plan, spec
from audit_x7_transport_front_stop import LAG, PERIOD
from train_x7_turning_trajectory_bc import digest


def check_run(r,t):
    scenario=r['case']; st=t['station_clearance']; initial=t['station_initialization_clearance'];
    near=float(min(st[...,:2].min(),initial[...,:2].min()));contacts=int(np.count_nonzero(np.any(st[...,2:4]>0,axis=2)))
    if near!=r['minimum_station_m'] or contacts!=r['contact_substeps']:raise ValueError('Physical clearance mismatch')
    im=t['rgb_external']; idx=t['rgb_action_index']; sensor=t['front_sensor'];motor=t['arm_jaw_and_hold'];e=t['disturbance']
    if im.dtype!=np.uint8 or im.shape!=(len(idx),128,192,3) or not np.array_equal(idx,np.arange(0,len(st),6)):
        raise ValueError('Pre-action capsule mismatch')
    tick=idx*8
    before=np.maximum(tick-1,0)
    causal_packet=np.where(tick>0,sensor[before,1],-1).astype(np.int64)
    observed=t['rgb_range_packet_at_pre_step']
    if np.array_equal(observed,causal_packet):encoding='causal-prior-drive'
    elif np.array_equal(observed,sensor[tick,1].astype(np.int64)):encoding='legacy-same-drive-corrected'
    else:raise ValueError('Unrecognized archive range-latch indexing')
    valid=(tick>0)&(sensor[before,3]>=0)
    hold_before=(tick>0)&(sensor[before,-1]>0)
    if not np.array_equal(hold_before,t['rgb_hold_already_latched']):raise ValueError('RGB hold label mismatch')
    params,clock,drop=spec(scenario,r['terminal_start'],r['original_terminal_end'],r['retimed_terminal_end'])
    force=np.flatnonzero(np.any(e[:,4:7]!=0.,axis=1))
    if clock!=r['scenario_clock'] or len(force)!=r['active_force_frames']:
        raise ValueError('Fixture clock mismatch')
    if params['force_n'] and (not np.array_equal(force,np.arange(clock-44,clock-20)) or not np.all(e[force,8]==8)):
        raise ValueError('Wrong actuator exposure')
    event=r['event']; force_before_latch=False
    if event is not None:
        k=event['substep'];rs=sensor[k];age=k-int(rs[1]);raw=None if rs[3]<0 else float(rs[3]);
        reason,lower,need=decide_standoff(raw,age,rs[4],rs[2])
        if not (LAG<=age<LAG+PERIOD and reason==event['reason'] and lower==event['lower_bound_m'] and
                need==event['braking_threshold_m'] and np.all(motor[:k,-1]==0) and np.all(motor[k:,-1]==1)
                and np.all(motor[k:,1:17]==motor[k,1:17]) and np.max(motor[k:,15:17])<.015):
            raise ValueError('Guard decision/throttling not reconstructible')
        force_before_latch=bool(len(force) and 8*force[0]<k)
        if drop!=bool(reason=='unknown') and r['qualified_holding_interrupt']:
            raise ValueError('Unexpected safe trigger classification')
    passed=bool(r['qualified_holding_interrupt'])
    if passed and (event is None or near<.005 or contacts or r['task_completed']):raise ValueError('Unsafe positive')
    d=dict(scene=r['layout'],case=scenario,hold_pass=passed,task_completed=r['task_completed'],min_station_m=near,
           min_terminal_b_m=float(st[r['terminal_start']:, :,1].min()),force_began_before_latched=force_before_latch,
           has_closed_box_end=r['held_box'],rgb_frames=len(idx),raw_rgb_range_encoding=encoding)
    return d,dict(rgb_action_index=idx,range_packet_before_rgb=causal_packet,range_sample_available_before_rgb=valid,
        hold_already_latched_before_rgb=hold_before)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);a=ap.parse_args()
    rows=[];table={}
    for layout,scenario in plan():
        path=a.run/'episodes'/f'{layout}__{scenario}';report=json.loads((path/'report.json').read_text());raw=path/'trace.npz'
        if report['checkpoint_sha256']!=V2 or report['trace_sha256']!=digest(raw):raise ValueError('Archive drift')
        with np.load(raw,allow_pickle=False) as t:r,idx=check_run(report,t)
        ix=path/'causal-observation-indices.npz'
        with ix.open('xb') as f:np.savez_compressed(f,**idx)
        r['causal_indices_sha256']=digest(ix);r['trace_sha256']=report['trace_sha256'];rows.append(r)
        group=table.setdefault(scenario,dict(attempts=0,qualified_holds=0,completed_tasks=0,min_station_m=1.,min_terminal_b_m=1.,
                                             pushes_before_latch=0))
        group['attempts']+=1;group['qualified_holds']+=r['hold_pass'];group['completed_tasks']+=r['task_completed'];
        group['min_station_m']=min(group['min_station_m'],r['min_station_m']);group['min_terminal_b_m']=min(group['min_terminal_b_m'],r['min_terminal_b_m'])
        group['pushes_before_latch']+=r['force_began_before_latched']
    result=dict(reserve_trial_m=RESERVE_M,checkpoint_sha256=V2,table=table,records=rows,
                validation_rgb_frames=sum(r['rgb_frames'] for r in rows),training_data=False,clamped_external_authority=False)
    with (a.run/'qa-standoff.json').open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(table))


if __name__=='__main__':main()
