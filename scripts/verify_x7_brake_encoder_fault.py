#!/usr/bin/env python3
"""Audit physically applied brakes, readback causality, and matched fault exposures."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_brake_encoder_fault import LAG,LAYOUTS,MAX_AGE,SAMPLE,V2,plan
from train_x7_turning_trajectory_bc import digest


def verify(a):
    cases=[];groups={}
    for layout,case in plan():
        folder=a/'episodes'/f'{layout}__{case}';r=json.loads((folder/'report.json').read_text());f=folder/'trace.npz'
        if r['trace_sha256']!=digest(f) or r['checkpoint_sha256']!=V2:raise ValueError('Trace/checkpoint drift')
        onset=r['external_onset_frame'];t0=onset*8
        with np.load(f,allow_pickle=False) as t:
            x=t['brake_signals'];b=t['backup_brake'];p=t['primary_auxiliary_brake'];m=t['arm_jaw_and_hold'];e=t['disturbance'];
            tick=np.arange(len(x));shared=case=='push-frozen-shared';frozen=case.startswith('push-frozen-');drop=case=='push-frozen-backup-drop'
            valid=x[:,6]>0;stamp=x[:,5].astype(np.int64);bf=np.linalg.norm(b[:,5:7],axis=1)
            if not (np.array_equal(tick,x[:,0]) and np.array_equal(x[:,1]>0,frozen&(tick>=t0)) and
                np.array_equal(x[:,3]>0,drop&(tick>=t0)) and np.all(x[:,2]==shared) and not np.any(bf[~valid])):
                raise ValueError('Unqualified encoder fixture')
            if frozen:
                if not np.all(x[t0:,11:15]==x[t0,11:15]) or np.any(x[t0:,12:15:2]):raise ValueError('No real frozen measurement')
            if not np.array_equal(x[:t0,7:11],x[:t0,11:15]):raise ValueError('Primary pre-exposure changed')
            if shared:
                if not np.all(valid) or not np.array_equal(stamp,tick) or not np.array_equal(x[:,11:15],x[:,15:19]):raise ValueError('No fresh-looking shared freeze')
            else:
                good=np.flatnonzero(valid);s=stamp[good];ages=good-s
                if not (np.all(s%SAMPLE==0) and np.all((ages>=LAG)&(ages<=MAX_AGE)) and np.all(x[s,4]==1)
                        and np.array_equal(x[good,15:19],x[s,7:11])):raise ValueError('Noncausal backup signal or substitution')
                if drop and (np.any(valid[t0+LAG:]) or np.any(x[t0:,4])):raise ValueError('Backup signal not actually lost')
            force=np.flatnonzero(np.any(e[:,4:7]!=0,axis=1));wanted=np.zeros(0,dtype=int) if case=='clear-separated' else np.arange(onset,onset+24)
            if not np.array_equal(force,wanted) or not np.all(e[force,8]==8):raise ValueError('External force exposure changed')
            stop=r['guard_event'];k=stop['substep'] if stop else None
            if stop and not (np.all(m[k:,1:17]==m[k,1:17]) and np.max(m[k:,15:17])<.015 and np.all(m[k:,-1])):
                raise ValueError('Stop failed to freeze arm and jaw commands')
            if frozen and stop and not (k>=t0 and np.all(p[k:,5]==0.)):raise ValueError('Frozen primary loop nevertheless produced force')
            st=t['station_clearance'];init=t['station_initialization_clearance'];near=float(min(st[...,:2].min(),init[...,:2].min()))
            contacts=int(np.count_nonzero(np.any(st[...,2:4]>0,axis=2)))
            if near!=r['station_minimum_m'] or contacts!=r['station_contact_substeps'] or (contacts or near<.005) and (r['normal_pass'] or r['holding_pass']):
                raise ValueError('Physical score discrepancy')
            dock=float(st[onset:,:,1].min()) if len(force) else None
            row=dict(layout=layout,case=case,completed=r['normal_pass'],hold=r['holding_pass'],contacts=contacts,min_m=near,
                     minimum_b_after_force_m=dock,backup_peak_n=float(bf.max()),trace_sha256=r['trace_sha256'])
            cases.append(row);groups.setdefault(case,[]).append(row)
    for layout in LAYOUTS:
        baseline=a/'episodes'/f'{layout}__push-valid-separated'/'trace.npz'
        with np.load(baseline,allow_pickle=False) as q:
            onset=int(np.flatnonzero(np.any(q['disturbance'][:,4:7]!=0,axis=1))[0])
            for name in ('push-frozen-shared','push-frozen-separated','push-frozen-backup-drop'):
                with np.load(a/'episodes'/f'{layout}__{name}'/'trace.npz',allow_pickle=False) as r:
                    if not all((np.array_equal(q['disturbance'],r['disturbance']),np.array_equal(q['action'],r['action']),
                        np.array_equal(q['base_pre'][:onset],r['base_pre'][:onset]))):raise ValueError('Pre-fault dynamics unmatched')
    table={k:dict(attempts=len(v),complete=sum(x['completed'] for x in v),held=sum(x['hold'] for x in v),
                  contact_cases=sum(x['contacts']>0 for x in v),min_m=min(x['min_m'] for x in v),
                  backup_peak_n=max(x['backup_peak_n'] for x in v)) for k,v in groups.items()}
    return dict(scenarios=table,records=cases,simulated_channels=True,hardware_claim=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);a=ap.parse_args();r=verify(a.out)
    with (a.out/'qa-brake-encoder-fault.json').open('x') as f:f.write(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r['scenarios']))


if __name__=='__main__':main()
