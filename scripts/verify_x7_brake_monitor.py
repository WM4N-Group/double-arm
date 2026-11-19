#!/usr/bin/env python3
"""Recompute feedback decisions from delivered samples and audit actual physics."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_brake_monitor import DELAY, LAYOUTS, MAX_AGE, SAMPLE, V2, feedback_valid, plan, probe
from train_x7_turning_trajectory_bc import digest


def audit(run):
    summary={};rows=[]
    for layout,case in plan():
        folder=run/'episodes'/f'{layout}__{case}';r=json.loads((folder/'report.json').read_text());f=folder/'trace.npz'
        if r['trace_sha256']!=digest(f) or r['checkpoint_sha256']!=V2:raise ValueError('Trace/model mismatch')
        with np.load(f,allow_pickle=False) as t:
            x=t['brake_exercise'];e=t['disturbance'];s=t['station_clearance'];i=t['station_initialization_clearance'];m=t['arm_jaw_and_hold']
            if not (np.array_equal(x[:,0],np.arange(len(x))) and np.all(x[:,2]<=x[:,1]) and np.all(x[:,2]>=0)):
                raise ValueError('Invalid exercise telemetry')
            # Sample and delivery must be different points in time; the
            # actuator's current force is never substituted for delayed data.
            got=x[:,5]>=0; k=np.flatnonzero(got);stamp=x[got,5].astype(int)
            if not (np.all(k-stamp>=DELAY) and np.array_equal(x[k,6],x[stamp,4]) and np.all(x[stamp,3]==1)):
                raise ValueError('Readback bypassed sample clock')
            events=r['feedback_events'];enable=[z for z in events if z['kind']=='terminal_motion_allowed']
            invalid=[z for z in events if z['kind']=='invalid_feedback']
            if len(enable)!=1 or len(invalid)>1:raise ValueError('Unqualified permit episode')
            for z in events:
                kk=z['substep'];st=int(x[kk,5]);val=None if x[kk,6]==-1. else float(x[kk,6]);
                if z['sample_substep']!=st or z['reported_force_n']!=val or feedback_valid((st,val),kk)!=(z['kind']=='terminal_motion_allowed'):
                    raise ValueError('Decision not reproducible from independent feedback')
            onset=r['fixture_clock']*8;active=np.flatnonzero(np.any(e[:,4:7]!=0.,axis=1));
            if len(active)!=r['external_push_frames'] or len(active) and (not np.array_equal(active,np.arange(onset//8,onset//8+24)) or not np.all(e[active,8]==8)):
                raise ValueError('Mis-timed independent pulse')
            if case in ('early-loss','late-loss','early-dropout','late-false-feedback'):
                probes=x[onset:];probes=probes[probes[:,3]>0]
                if len(probes)==0:raise ValueError('No independently exercised fault')
                if case=='early-dropout' and not (np.all(probes[:,4]<0) and np.all(probes[:,2]>0)):
                    raise ValueError('Missing-packet case changed actuator')
                if case in ('early-loss','late-loss') and not (np.all(probes[:,2]==0) and np.all(abs(probes[:,4])<=1.)):
                    raise ValueError('Truthful complete brake loss missing')
                if case=='late-false-feedback' and (np.any(probes[:,2]) or not np.all(abs(probes[:,4]-probes[:,1])<=1.) or invalid):
                    raise ValueError('False-readback control accidentally honest or detected')
            contact=np.any(s[...,2:4]>0,axis=2).reshape(-1);near=float(min(s[...,:2].min(),i[...,:2].min()))
            first=int(np.flatnonzero(contact)[0]) if contact.any() else None
            if near!=r['minimum_station_m'] or first!=r['first_contact_substep'] or contact.sum()!=r['contact_substeps']:
                raise ValueError('Physical audit does not match')
            stop=r['guard_event'];
            if stop:
                k0=stop['substep'];
                if not (np.all(m[:k0,-1]==0) and np.all(m[k0:,-1]==1) and np.all(m[k0:,1:17]==m[k0,1:17]) and np.max(m[k0:,15:17])<.015):
                    raise ValueError('Unauthorized resumed motion/release')
            row=dict(layout=layout,case=case,normal=r['normal_pass'],holding=r['qualified_hold'],contact=bool(contact.any()),
                     first_contact_substep=first,min_m=near,invalid_feedback_substep=invalid[0]['substep'] if invalid else None,
                     stop_reason=stop['reason'] if stop else None,force_onset_substep=onset if len(active) else None,
                     trace_sha256=r['trace_sha256'])
            if (row['normal'] or row['holding']) and (row['contact'] or near<.005):raise ValueError('False pass')
            rows.append(row);summary.setdefault(case,[]).append(row)
    for layout in LAYOUTS:
        refs=[run/'episodes'/f'{layout}__{case}'/'trace.npz' for case in ('late-healthy','late-loss','late-false-feedback')]
        with np.load(refs[0],allow_pickle=False) as a,np.load(refs[1],allow_pickle=False) as b,np.load(refs[2],allow_pickle=False) as c:
            start=int(np.flatnonzero(np.any(a['disturbance'][:,4:7]!=0,axis=1))[0]);
            for t in (b,c):
                if not np.array_equal(a['disturbance'],t['disturbance']) or not np.array_equal(a['base_pre'][:start],t['base_pre'][:start]):
                    raise ValueError('Late challenge pairs changed pre-exposure')
    return dict(cases=rows,groups=summary,checkpoint_sha256=V2,idealized_readback=True,hardware_claim=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);a=ap.parse_args()
    r=audit(a.run)
    with (a.run/'qa-brake-monitor.json').open('x') as f:f.write(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:(len(v),sum(x['normal'] for x in v),sum(x['holding'] for x in v),sum(x['contact'] for x in v),
                        sum(x['invalid_feedback_substep'] is not None for x in v)) for k,v in r['groups'].items()}))


if __name__=='__main__':main()
