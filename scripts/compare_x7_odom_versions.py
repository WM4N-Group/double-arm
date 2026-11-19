#!/usr/bin/env python3
"""Compare v2 against archived v1 and zero-residual matched development runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from stress_x7_rgb_transport import CHECKPOINT, SCENARIOS, plan
from train_x7_turning_trajectory_bc import digest


def one(out,case,h):
    d=out/'episodes'/case;r=json.loads((d/'report.json').read_text());p=d/'trace.npz'
    if r['case']!=case or r['checkpoint_sha256']!=h or r['trace_sha256']!=digest(p):raise ValueError(f'Unmatched {case}')
    with np.load(p,allow_pickle=False) as t:
        s=t['station_clearance']; init=t['station_initialization_clearance'];sub=s[...,:2].min(axis=2)
        k,j=np.unravel_index(sub.argmin(),sub.shape);phi=int(t['phase'][k]);m=float(min(sub.min(),init[...,:2].min()))
        breach=np.argwhere(sub<.005);contact=np.argwhere(np.any(s[...,2:4]>0,axis=2))
        if m!=r['minimum_station_m'] or r['passed'] and (m<.005 or len(contact)):raise ValueError('Independent clearance mismatch')
        conditions=[t[key] for key in ('frozen_anchor','phase','disturbance')]
        return (dict(passed=r['passed'],minimum_station_m=m,first_below_5mm_control=int(breach[0,0]) if len(breach) else None,
            station_contact_substeps=len(contact),first_contact_control=int(contact[0,0]) if len(contact) else None,
            minimum_phase=phi,minimum_control=int(k),minimum_physics_substep=int(8*k+j),
            on_disturbance_at_execution_minimum=bool(t['disturbance'][k,8]>0),trace_sha256=digest(p)),conditions)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('baseline',type=Path);ap.add_argument('run',type=Path)
    args=ap.parse_args();h=digest(args.run/'training'/'best.pt');st=json.loads((args.run/'training'/'status.json').read_text())
    if st['checkpoint_sha256']!=h or st['stage']!='train_completed' or st['optimizer_updates']!=640:
        raise ValueError('Invalid trained v2 checkpoint')
    pairs=[];groups={s:{} for s in SCENARIOS}
    for row in plan():
        if row['mode']!='candidate':continue
        case=row['case'];m=row['scenario']
        v1=one(args.baseline,case,CHECKPOINT); nominal=one(args.baseline,case.removesuffix('candidate')+'nominal',CHECKPOINT)
        v2=one(args.run/'dev-v2',case,h)
        if not all(np.array_equal(a,b) and np.array_equal(a,c) for a,b,c in zip(v1[1],v2[1],nominal[1])):
            raise ValueError('Exact disturbance/phase pairing changed')
        pair=dict(layout=row['layout'],scenario=m,candidate_v1=v1[0],candidate_v2=v2[0],nominal=nominal[0]);pairs.append(pair)
        for label,score in [('candidate_v1',v1[0]),('candidate_v2',v2[0]),('nominal',nominal[0])]:
            stats=groups[m].setdefault(label,dict(cases=0,passes=0,minimum_m=1.,contact_cases=0))
            stats['cases']+=1;stats['passes']+=score['passed'];stats['minimum_m']=min(stats['minimum_m'],score['minimum_station_m'])
            stats['contact_cases']+=score['station_contact_substeps']>0
    gate=all(groups[s]['candidate_v2']['passes']==4 for s in SCENARIOS if s!='push-limited-drive')
    groups_out=dict(station_threshold_m=.005,checkpoint_sha256=h,v1_sha256=CHECKPOINT,scenario_table=groups,pairs=pairs,
        development_only=True,odom_development_gate_passed=gate,heldout_executed=False)
    with (args.run/'qa-v2-transport-stress.json').open('x') as f:f.write(json.dumps(groups_out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(table=groups,odom_development_gate_passed=gate),allow_nan=False))


if __name__=='__main__':main()
