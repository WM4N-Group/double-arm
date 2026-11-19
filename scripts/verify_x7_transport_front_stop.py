#!/usr/bin/env python3
"""Independent audit: actual range-gate timing, hold commands and station contact trace."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_transport_front_stop import CLEARANCE, LAG, PERIOD, V2, panel, trip
from train_x7_turning_trajectory_bc import digest


def check_case(report, t, baseline, scenario):
    contact=np.any(t['station_clearance'][...,2:4]>0,axis=2).reshape(-1)
    distances=t['station_clearance'][...,:2].min(axis=2).reshape(-1)
    near=float(min(distances.min(),t['station_initialization_clearance'][...,:2].min()))
    event=report['event']; s=t['front_sensor']; m=t['arm_jaw_and_hold']; effect=t['disturbance']
    if (near!=report['minimum_station_m'] or int(contact.sum())!=report['contact_substeps'] or
            not np.array_equal(s[:,0],np.arange(len(s))) or not np.array_equal(s[:,0],m[:,0])):
        raise ValueError('Archive discrepancy between substep aggregates')
    if baseline is not None:
        for k,j in (('disturbance','disturbance'),('anchor_action','frozen_anchor'),('phase','phase')):
            if not np.array_equal(t[k],baseline[j]):raise ValueError('Baseline mismatch')
    first=int(np.flatnonzero(contact)[0]) if contact.any() else None
    result=dict(first_contact_substep=first,minimum_station_m=near,contact_substeps=int(contact.sum()),
                guard_substep=None,first_contact_during_force=None,clean_trace_exact=None)
    if event is None:
        if np.any(s[:,-1]) or np.any(m[:,-1]):raise ValueError('Unrecorded hold')
        if scenario=='clean' and baseline is not None:
            result['clean_trace_exact']=all(np.array_equal(t[k],baseline[k]) for k in ('base_post','station_clearance','action'))
            if not result['clean_trace_exact']:raise ValueError('Unlatched control changed')
    else:
        k=event['substep'];rs=s[k]
        if rs[1]!=event['sample_substep'] or not LAG<=k-int(rs[1])<LAG+PERIOD:raise ValueError('Unexpected range age')
        reason,bound,need=trip(None if rs[3]<0 else rs[3],k-int(rs[1]),rs[4],rs[2])
        if reason!=event['reason'] or bound!=event['lower_bound_m'] or need!=event['braking_threshold_m']:
            raise ValueError('Decision not reconstructible from allowed sensors')
        if not (np.all(s[:k,-1]==0) and np.all(s[k:,-1]==1) and np.array_equal(s[:,-1],m[:,-1]) and
                np.all(m[k:,1:17]==m[k,1:17]) and np.max(m[k:,15:17])<.015):
            raise ValueError('Later release/unlock or command changed after fault')
        result['guard_substep']=k
        if first is not None:
            result['first_contact_during_force']=bool(np.any(effect[first//8,4:7]))
            result['dt_hold_to_first_contact_s']=(first-k)/240.
        if scenario=='range-dropout' and (reason!='unknown' or event['sampled_range_m'] is not None):
            raise ValueError('Dropout was not independently responsible for hold')
    if report['qualified_hold'] and (event is None or contact.any() or near<CLEARANCE):
        raise ValueError('Wrongly accepted hold')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('baseline',type=Path);a=ap.parse_args()
    checks=[]
    for layout,scenario in panel():
        p=a.run/'episodes'/f'{layout}__{scenario}';r=json.loads((p/'report.json').read_text());fn=p/'trace.npz'
        if r['checkpoint_sha256']!=V2 or r['trace_sha256']!=digest(fn):raise ValueError('Output changed')
        with np.load(fn,allow_pickle=False) as t:
            if scenario=='range-dropout':summary=check_case(r,t,None,scenario)
            else:
                ref=a.baseline/'episodes'/f'{layout}__{scenario}__candidate'/'trace.npz'
                with np.load(ref,allow_pickle=False) as b:summary=check_case(r,t,b,scenario)
        checks.append(dict(layout=layout,scenario=scenario,trace_sha256=r['trace_sha256'],**summary))
    with (a.run/'independent-front-audit.json').open('x') as f:
        f.write(json.dumps({'cases':checks,'no_physical_hardware_claim':True},indent=2,allow_nan=False)+'\n')
    print(json.dumps([r for r in checks if r['scenario']=='push-limited-drive']))


if __name__=='__main__':main()
