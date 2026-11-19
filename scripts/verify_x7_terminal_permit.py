#!/usr/bin/env python3
"""Verify billed proof load, interlock sequencing, and physical outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_terminal_permit import BRAKE_MAX_N, PROOF, SETTLE, V2, permission, plan
from train_x7_turning_trajectory_bc import digest


def verify(run):
    grouped={};rows=[]
    for layout,case in plan():
        d=run/'episodes'/f'{layout}__{case}';r=json.loads((d/'report.json').read_text());trace=d/'trace.npz'
        if r['trace_sha256']!=digest(trace) or r['checkpoint_sha256']!=V2:raise ValueError('Trace/checkpoint changed')
        with np.load(trace,allow_pickle=False) as t:
            b=t['secondary_brake'];s=t['station_clearance'];init=t['station_initialization_clearance'];motor=t['arm_jaw_and_hold'];e=t['disturbance']
            if (not np.array_equal(b[:,0],np.arange(len(b))) or np.any(b[:,5]>BRAKE_MAX_N+1e-9) or np.any(b[:,5]>b[:,6]+1e-9)):
                raise ValueError('Brake telemetry invalid')
            where=np.flatnonzero(b[:,2]);expected=8*(r['proof_start_frame']+SETTLE)
            if not np.array_equal(where,np.arange(expected,expected+8*PROOF)) or not np.all(b[where,3]==1) or not np.all(b[where,4]==1):
                raise ValueError('Primary actuator not isolated during authentic proof pulse')
            if r['case']=='proof-reject' and np.any(b[where,5:7]):raise ValueError('Rejection fixture not physically failed')
            force=np.flatnonzero(np.any(e[:,4:7]!=0,axis=1))
            if len(force)!=r['external_push_frames'] or len(force) and (not np.array_equal(force,np.arange(r['independent_push_end_clock']-44,r['independent_push_end_clock']-20)) or np.any(e[force,8]!=8)):
                raise ValueError('Independent late external exposure changed')
            expected_drop=case=='permit-late-dual-fault'
            if len(force) and not np.all(b[8*force[0]:8*(force[-1]+1),6]==(0. if expected_drop else BRAKE_MAX_N)):
                raise ValueError('No audited late brake-loss control')
            events=r['permit_events'];stop=r['guard_event'];grant=False
            if len(events)==1:
                ev=events[0];low=ev['front_lower_m'];actual=low is not None and permission(ev['travel_m'],ev['rate_m_s'],low,ev['steps'])
                grant=actual
                if ev['frame']!=r['proof_end_frame'] or (ev['kind']=='grant')!=actual or bool(b[ev['frame']*8,7])!=actual:
                    raise ValueError('Permit inconsistent with observed motion proof')
            elif len(events)>1 or np.any(b[:,7]):raise ValueError('Permit silently granted')
            near=float(min(s[...,:2].min(),init[...,:2].min()));contacts=np.any(s[...,2:4]>0,axis=2).reshape(-1)
            if near!=r['minimum_station_m'] or contacts.sum()!=r['contact_substeps']:raise ValueError('Independent station scoring mismatch')
            if stop:
                k=stop['substep'];
                if not (np.all(motor[k:,1:17]==motor[k,1:17]) and np.max(motor[k:,15:17])<.015):raise ValueError('Claws opened under fault')
                if stop['reason']=='permit_revoked' and (not grant or b[k,8]!=0 or np.any(b[:k,8]==0)):
                    raise ValueError('Revocation not traced to independent signal')
            if (near<.005 or contacts.any()) and (r['normal_task_pass'] or r['qualified_hold']):raise ValueError('Unsafe positive')
            small=dict(layout=layout,case=case,granted=grant,reason=stop['reason'] if stop else None,
                completion=r['normal_task_pass'],hold=r['qualified_hold'],min_m=near,contact_case=bool(contacts.any()),
                proof_travel_m=events[0]['travel_m'] if events else None,trace_sha256=r['trace_sha256'])
        grouped.setdefault(case,[]).append(small);rows.append(small)
    comparisons=[]
    for layout in {r['layout'] for r in rows}:
        healthy=run/'episodes'/f'{layout}__permit-late-push'/'trace.npz'
        failed=run/'episodes'/f'{layout}__permit-late-dual-fault'/'trace.npz'
        with np.load(healthy,allow_pickle=False) as good,np.load(failed,allow_pickle=False) as bad:
            start=int(np.flatnonzero(np.any(good['disturbance'][:,4:7]!=0,axis=1))[0])
            if not (np.array_equal(good['disturbance'],bad['disturbance']) and
                    np.array_equal(good['action'],bad['action']) and
                    np.array_equal(good['base_pre'][:start],bad['base_pre'][:start])):
                raise ValueError('Late-fault pair confounded before independent force')
        comparisons.append(layout)
    return dict(cases=rows,by_case=grouped,checkpoint_sha256=V2,secondary_brake_is_hypothetical=True,
                equal_pre_force_late_pairs=sorted(comparisons),hardware_claim=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);a=ap.parse_args()
    r=verify(a.run)
    with (a.run/'qa-terminal-permit.json').open('x') as f:f.write(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r['by_case']))


if __name__=='__main__':main()
