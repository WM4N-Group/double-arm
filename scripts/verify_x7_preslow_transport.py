#!/usr/bin/env python3
"""Audits clearance, exact fixed-clock actuator disturbance, and bench displacement."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_preslow_transport import V2, plan, spec
from run_x7_turning_carry import r2
from stress_x7_rgb_transport import POSES
from train_x7_turning_trajectory_bc import digest


def verify(root,baseline):
    all_rows=[];table={}
    for group in ('bench','panel'):
        for layout,scenario in plan(group):
            d=root/group/'episodes'/f'{layout}__{scenario}';r=json.loads((d/'report.json').read_text());p=d/'trace.npz'
            if r['checkpoint_sha256']!=V2 or r['trace_sha256']!=digest(p):raise ValueError('Unfrozen result')
            old,new=r['original_end_frame'],r['retimed_end_frame'];params,clock=spec(scenario,old,new)
            if clock!=r['disturbance_clock_end_frame']:raise ValueError('Unexpected disturbance clock')
            with np.load(p,allow_pickle=False) as t:
                e=t['disturbance'];force=np.flatnonzero(np.any(e[:,4:7]!=0,axis=1));score={}
                if params['force_n']:
                    if not np.array_equal(force,np.arange(clock-44,clock-20)) or not np.all(e[force,8]==8):raise ValueError('Push not applied at every substep')
                    if clock==old:
                        ref=baseline/'episodes'/f'{layout}__push-limited-drive'/'trace.npz'
                        with np.load(ref,allow_pickle=False) as f:
                            if (not np.array_equal(e[force],f['disturbance'][force]) or
                                    not np.array_equal(t['reference_action_before_time_scaling'],f['anchor_action'])):
                                raise ValueError('Fixed-clock run does not match the original external force and timing')
                if group=='bench':
                    q=t['bench_base_substeps'];start=8*force[0];ori=r2(np.radians(POSES[layout]['goal'][2]))@[0.,-1.]
                    delta=(q[start:,:2,0]-q[start,:2,0])@ori
                    if float(delta.max())!=r['peak_forward_displacement_m'] or not r['retained_load']:
                        raise ValueError('Bench drift did not reconstruct')
                    score=dict(bench_drift_m=r['peak_forward_displacement_m'],bench_start_speed_m_s=r['forward_speed_at_zero_m_s'])
                else:
                    s=t['station_clearance'];init=t['station_initialization_clearance'];near=float(min(s[...,:2].min(),init[...,:2].min()))
                    contact=np.any(s[...,2:4]>0,axis=2).reshape(-1)
                    if near!=r['minimum_station_m'] or int(contact.sum())!=r['robot_station_contact_substeps']:
                        raise ValueError('Clearance differs from archived physics')
                    if (r['normal_task_pass'] or r['qualified_holding_interrupt']) and (near<.005 or contact.any()):
                        raise ValueError('Unsafe pass')
                    if r['latched']:
                        k=r['event']['substep'];m=t['arm_jaw_and_hold']
                        if not (np.all(m[:k,-1]==0) and np.all(m[k:,-1]==1) and np.all(m[k:,1:17]==m[k,1:17])
                                and np.max(m[k:,15:17])<.015):raise ValueError('Hold does not physically freeze jaws and arms')
                    onset=None if not len(force) else float(s[max(0,force[0]-1),-1,:2].min())
                    score=dict(normal_pass=r['normal_task_pass'],holding_interrupt=r['qualified_holding_interrupt'],
                               minimum_station_m=near,contacts=int(contact.sum()),force_onset_clearance_m=onset)
                all_rows.append(dict(layout=layout,scenario=scenario,**score,trace_sha256=r['trace_sha256']))
            table.setdefault(scenario,[]).append(score)
    return dict(no_hardware_claim=True,checkpoint_sha256=V2,scenario_rows=table,records=all_rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('baseline',type=Path);a=ap.parse_args()
    result=verify(a.run,a.baseline)
    with (a.run/'qa-preslow-brake.json').open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['scenario_rows'],allow_nan=False))


if __name__=='__main__':main()
