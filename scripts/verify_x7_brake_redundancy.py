#!/usr/bin/env python3
"""Independently qualify commanded/applied force, truthful low-load feedback, and contacts."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_brake_monitor import feedback_valid
from audit_x7_brake_redundancy import DERATED_N,LAYOUTS,V2,limits,plan
from train_x7_turning_trajectory_bc import digest


def audit(root):
    results=[];grouped={}
    for layout,case in plan():
        d=root/'episodes'/f'{layout}__{case}';r=json.loads((d/'report.json').read_text());f=d/'trace.npz'
        if r['trace_sha256']!=digest(f) or r['checkpoint_sha256']!=V2:raise ValueError('Archive drift')
        with np.load(f,allow_pickle=False) as t:
            b=t['independent_backup_model'];p=t['primary_auxiliary_brake'];x=t['primary_brake_exercise'];m=t['arm_jaw_and_hold'];e=t['disturbance']
            at=r['force_onset_frame'];idx=at*8;cl=np.linalg.norm(b[:,5:7],axis=1)
            if not (np.array_equal(b[:,0],np.arange(len(b))) and np.array_equal(b[:,1],m[:,-1]) and
                    np.max(cl)<=np.max(b[:,2])+1e-9 and np.all(cl<=b[:,2]+1e-9) and not np.any(cl[b[:,1]==0])):
                raise ValueError('Unstated backup braking or exceeded capacity')
            for lo,hi,limits_t in ((0,idx,limits(case,at-1,at)),(idx,len(b),limits(case,at,at))):
                if not (np.all(p[lo:hi,6]==limits_t[0]) and np.all(b[lo:hi,2]==limits_t[1])):raise ValueError('Wrong actuation fixture')
            applied=np.flatnonzero(np.any(e[:,4:7]!=0.,axis=1))
            expected=np.zeros(0,dtype=int) if case=='clean-redundant' else np.arange(at,at+24)
            if not np.array_equal(applied,expected) or not np.all(e[applied,8]==8) or len(applied)!=r['push_frames']:
                raise ValueError('Force clock mismatch')
            sensed=x[idx:];sensed=sensed[sensed[:,3]>0];invalid=[y for y in r['feedback_events'] if y['kind']=='invalid_feedback']
            if 'derated' in case:
                if not (len(sensed)>0 and not invalid and np.array_equal(sensed[:,1],sensed[:,2]) and
                        np.all(abs(sensed[:,4]-sensed[:,2])<=1.) and np.all(p[idx:,5]<=DERATED_N+1e-9)):
                    raise ValueError('Derating must be physically present and honest under low-force proof')
            if case in ('late-open-redundant','late-common-loss'):
                if not (len(invalid)==1 and len(sensed)>0 and not np.any(sensed[:,2])):raise ValueError('Complete loss not diagnosed')
                ev=invalid[0]
                if feedback_valid((ev['sample_substep'],ev['reported_force_n']),ev['substep']):raise ValueError('Wrong feedback rejection')
            if r['guard_event']:
                k=r['guard_event']['substep'];
                if not (np.all(m[k:,1:17]==m[k,1:17]) and np.max(m[k:,15:17])<.015):raise ValueError('Latch did not freeze closed grasp')
            st,init=t['station_clearance'],t['station_initialization_clearance'];min_d=float(min(st[...,:2].min(),init[...,:2].min()))
            hit=int(np.count_nonzero(np.any(st[...,2:4]>0,axis=2)))
            if min_d!=r['station_minimum_m'] or hit!=r['contact_substeps'] or (r['normal_pass'] or r['qualified_hold']) and (hit or min_d<.005):
                raise ValueError('Collision scoring mismatch')
            row=dict(layout=layout,case=case,min_m=min_d,contacts=hit,normal_pass=r['normal_pass'],hold=r['qualified_hold'],
                     invalid_feedback=bool(invalid),backup_peak_n=float(cl.max()),
                     brake_probe_samples_after_fault=len(sensed),trace_sha256=r['trace_sha256'])
            results.append(row);grouped.setdefault(case,[]).append(row)
    for layout in LAYOUTS:
        root1=root/'episodes';baseline=root1/f'{layout}__late-healthy-single'/'trace.npz'
        with np.load(baseline,allow_pickle=False) as raw:
            start=int(np.flatnonzero(np.any(raw['disturbance'][:,4:7]!=0,axis=1))[0]);
            for _,case in [item for item in plan() if item[0]==layout and item[1].startswith('late-')]:
                with np.load(root1/f'{layout}__{case}'/'trace.npz',allow_pickle=False) as t:
                    if not (np.array_equal(raw['disturbance'],t['disturbance']) and np.array_equal(raw['action'],t['action']) and
                            np.array_equal(raw['base_pre'][:start],t['base_pre'][:start])):
                        raise ValueError('Pre-exposure state confounded')
    table={k:dict(attempts=len(v),normal=sum(x['normal_pass'] for x in v),hold=sum(x['hold'] for x in v),
        contact_cases=sum(x['contacts']>0 for x in v),feedback_reject=sum(x['invalid_feedback'] for x in v),
        min_m=min(x['min_m'] for x in v)) for k,v in grouped.items()}
    return dict(summary=table,records=results,checkpoint_sha256=V2,shared_simulated_encoder=True,hardware_claim=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);a=ap.parse_args();r=audit(a.run)
    with (a.run/'qa-brake-redundancy.json').open('x') as f:f.write(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r['summary']))


if __name__=='__main__':main()
