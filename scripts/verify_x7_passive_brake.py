#!/usr/bin/env python3
"""Audit physically isolated drive, timed brake engagement, and station clearance."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_passive_brake import CASES,EPSILON_CAP,LAYOUTS,V2,passive_capacity
from sweep_x7_passive_delay import DELAYS
from train_x7_turning_trajectory_bc import digest


def verify(a,settings=None,comparison=None):
    settings=CASES if settings is None else settings
    comparison=a if comparison is None else comparison
    rows=[];groups={}
    for layout,case in ((l,c) for l in LAYOUTS for c in settings):
        d=a/'episodes'/f'{layout}__{case}';r=json.loads((d/'report.json').read_text());p=d/'trace.npz'
        if r['checkpoint_sha256']!=V2 or r['trace_sha256']!=digest(p):raise ValueError('Artifact identity drift')
        with np.load(p,allow_pickle=False) as t:
            f=t['passive_joint_constraint'];mot=t['arm_jaw_and_hold'];x=t['brake_signals'];e=t['disturbance'];
            br=t['primary_auxiliary_brake'];bk=t['backup_brake'];s=t['station_clearance'];init=t['station_initialization_clearance']
            stop=r['guard_event'];k=stop['substep'] if stop else None;onset=r['external_onset_frame'];tick=np.arange(len(f))
            mask=(tick>=k) if stop else np.zeros(len(f),dtype=bool)
            wanted=np.array([passive_capacity(int(j),k,settings[case]) for j in tick])
            active=np.flatnonzero(np.any(wanted>0,axis=1));wanted[mask]=np.maximum(wanted[mask],EPSILON_CAP)
            if not (np.array_equal(tick,f[:,0]) and np.array_equal(f[:,1],mask) and np.array_equal(mot[:,-1],mask)
                    and np.array_equal(f[:,2:5],wanted)):
                raise ValueError('Unqualified passive command')
            if r['engage_tick']!=(int(active[0]) if len(active) else None):raise ValueError('Actuator delay mismatch')
            force=np.flatnonzero(np.any(e[:,4:7]!=0.,axis=1));expected=np.zeros(0,dtype=int) if case=='clear-armed' else np.arange(onset,onset+24)
            if not np.array_equal(force,expected) or not np.all(e[force,8]==8):raise ValueError('Independent force clip mismatch')
            act=0.
            if stop:
                if not (np.all(x[k:,1]==1) and np.all(x[k:,6]==0) and np.all(br[k:,5]==0) and np.all(bk[k:,5:7]==0) and
                        np.all(mot[k:,1:17]==mot[k,1:17]) and np.max(mot[k:,15:17])<.015):
                    raise ValueError('Feedback brake/encoder not isolated or claw opened')
                prev=abs(f[k+1:,5:8]);caps=f[k:-1,2:5]
                if np.any(prev>caps+.005):raise ValueError('Measured solver reaction exceeded declared capacity')
                act=float(prev.max())
                if len(active)>0 and act<=0.:raise ValueError('Proposed constraint never measurably applied')
            dist=float(min(s[...,:2].min(),init[...,:2].min()));contacts=np.any(s[...,2:4]>0,axis=2).reshape(-1)
            first=int(np.flatnonzero(contacts)[0]) if contacts.any() else None
            if (dist!=r['minimum_station_m'] or int(contacts.sum())!=r['station_contact_substeps'] or
                    first!=r['first_contact_substep'] or (contacts.any() or dist<.005) and (r['holding_pass'] or r['normal_pass'])):
                raise ValueError('Physical audit mismatch')
            dock=float(s[onset:,:,1].min()) if len(force) else None
            row=dict(layout=layout,case=case,complete=r['normal_pass'],hold=r['holding_pass'],min_m=dist,
                     min_b_after_force_m=dock,contact=bool(contacts.any()),first_contact=first,
                     peak_axis_reaction_n_or_nm=act,engage_tick=r['engage_tick'],latch=k,trace_sha256=r['trace_sha256'])
            rows.append(row);groups.setdefault(case,[]).append(row)
    for layout in LAYOUTS:
        base=comparison/'episodes'/f'{layout}__lost-no-passive'/'trace.npz'
        with np.load(base,allow_pickle=False) as t:
            start=int(np.flatnonzero(np.any(t['disturbance'][:,4:7]!=0,axis=1))[0]);
            for name in settings:
                if name in ('clear-armed','lost-no-passive'):continue
                with np.load(a/'episodes'/f'{layout}__{name}'/'trace.npz',allow_pickle=False) as r:
                    if not all((np.array_equal(r['disturbance'],t['disturbance']),np.array_equal(r['action'],t['action']),
                                np.array_equal(r['base_pre'][:start],t['base_pre'][:start]))):
                        raise ValueError('Unmatched pre-disturbance history')
    table={k:dict(cases=len(v),tasks=sum(r['complete'] for r in v),holds=sum(r['hold'] for r in v),
        contacts=sum(r['contact'] for r in v),minimum_m=min(r['min_m'] for r in v),
        min_b_after_force_m=min((r['min_b_after_force_m'] for r in v if r['min_b_after_force_m'] is not None),default=None))
        for k,v in groups.items()}
    return dict(table=table,rows=rows,simulated_friction_joint_only=True,hardware_claim=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('out',type=Path);ap.add_argument('--baseline',type=Path,help='Qualify only intermediate-delay outputs against the original v2 panel');a=ap.parse_args()
    r=verify(a.out,DELAYS,a.baseline) if a.baseline else verify(a.out)
    with (a.out/'qa-passive-brake.json').open('x') as out:out.write(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps(r['table']))


if __name__=='__main__':main()
