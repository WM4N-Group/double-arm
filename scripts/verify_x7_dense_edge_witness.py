#!/usr/bin/env python3
"""Audit actual clearance and packet decisions in the dense-edge pilot."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_dense_edge_witness import CASES, FIXTURES, FROZEN_CHECKPOINT
from audit_x7_handle_rear_budget import ACTOR_DELAY_SUBSTEPS, OBSTACLE_TRAVEL, RETREAT_M, STATION_CLEARANCE_M, digest, entry_margin
from audit_x7_split_rear_rays import GAP_FIXTURE_TRAVEL_M

BATCHES=('clear-edge','qualified-obstacles','missing')
OLD_BATCH={'clear-with-handles':'western','fresh-clear':'coverage-fresh',
           'main-wall-with-budget':'western','handle-with-coverage':'western',
           'inner-gap-actor':'western','outer-cad-actor':'western','masked-handle':'coverage-fresh',
           'masked-gap':'coverage-fresh','masked-exterior':'coverage-fresh',
           'edge-right-cube':'western','edge-right-thin':'western','masked-edge-right':'western'}


def assess(row, directory, reference):
    name=row['case'];is_actor=CASES[name][-1] is not None
    expect_occupied=name in FIXTURES
    expect_mask=name.startswith('masked-')
    is_clear=not is_actor and not expect_mask
    main=directory/(name+'.npz'); witness=directory/(name+'-witness.npz')
    if digest(main)!=row['trace_sha256'] or digest(witness)!=row['witness_sha256']:
        raise ValueError(f'Trace hash changed: {name}')
    checks={'motion_is_interrupt_not_task_success':row['held_box_after_stop'] and row['safe_holding_interrupt'] and
                not row['success'] and row['guard_triggered'] and not row['object_attachment_constraints_seen'] and
                row['state_writes_during_actions']==0,
            'no_truth_identity_in_control':not row['hit_body_ids_used_for_control']}
    with np.load(main,allow_pickle=False) as a, np.load(witness,allow_pickle=False) as w:
        rays=w['witness']; events=row['rear_events']; sample=[]; event=None
        box, robot, station, init=(a[k] for k in ('payload_audit','robot_rear_audit','station_clearance','station_initialization_clearance'))
        checks['no_rear_contacts']=not bool(np.any(box[:,3]) or np.any(robot[:,1]))
        checks['robot_station_trace_clear']=bool(min(np.min(station[...,:2]),np.min(init[...,:2]))>=STATION_CLEARANCE_M and
                                                not np.any(station[...,2:4]) and not np.any(init[...,2:4]))
        if name in OLD_BATCH:
            base=name if name not in ('edge-right-cube','edge-right-thin','masked-edge-right') else 'clear-with-handles'
            with np.load(reference/OLD_BATCH[name]/(base+'.npz'),allow_pickle=False) as b:
                checks['same_frozen_action_phase_as_reference']=all(np.array_equal(a[k],b[k]) for k in ('action','phase'))
        if is_clear:
            checks['measured_14mm_retreat_without_veto']=bool(abs(row['retreat_measured_m']-RETREAT_M)<.001 and not events)
            checks['no_side_unknown_or_obstacle']=not bool(np.any(rays[:,6]>=2))
            if name=='clear-with-handles':
                checks['original_west_echo_actually_released']=bool(np.any((rays[:,6]==1)&(rays[:,2]==1)))
        else:
            checks['zero_retreat']=abs(row['retreat_measured_m'])<.001
            event_name='rear_coverage_lost' if expect_mask or expect_occupied else 'bounded_rear_travel_veto'
            relevant=[e for e in events if e['event']==event_name]
            checks['expected_gate_latched']=len(relevant)==1
            if checks['expected_gate_latched']:
                event=relevant[0];t=event['sample_k'];raw=a['sensor_telemetry']
                delivered=raw[(raw[:,0]==event['substep'])&(raw[:,3]==t)]
                if len(delivered)!=1:
                    raise ValueError(f'Missing delivery: {name}')
                sample=rays[rays[:,0]==t]
                low=float(delivered[0,11])
                if expect_mask or expect_occupied:
                    checks['delivered_boundary_packet_inhibited']=bool(delivered[0,8]==low==-1.)
                else:
                    checks['unchanged_entry_budget']=bool(event['required_m']==entry_margin() and
                        0.<=event['rear_lower_bound_m']==low<entry_margin())
                if name.startswith('masked-edge-'):
                    on_side=sample[:,2]==(-1 if name.endswith('left') else 1)
                    checks['real_edge_dropout_not_other_channel']=bool(np.any(on_side&(sample[:,6]==2)) and
                                                                       not np.any(sample[:,6]==3))
                if expect_occupied:
                    sign=-1 if 'left' in name else 1
                    checks['edge_occupancy_on_expected_side_and_height']=bool(np.any((sample[:,2]==sign)&(sample[:,6]==3)&
                            (abs(sample[:,3]-FIXTURES[name][0][2])<=FIXTURES[name][1][2]+1e-12)&(sample[:,5]>0)))
        if is_actor:
            checks['rear_5mm_from_trace']=bool(min(np.min(box[:,2]),np.min(robot[:,0]))>=STATION_CLEARANCE_M and
                                               row['rear_min_box_m']==np.min(box[:,2]) and row['rear_min_robot_m']==np.min(robot[:,0]))
            birth=row['obstacle_reset'];motion=a['rear_obstacle_motion'];start=birth['substep']+ACTOR_DELAY_SUBSTEPS
            travel=GAP_FIXTURE_TRAVEL_M if expect_occupied or name=='inner-gap-actor' else OBSTACLE_TRAVEL
            checks['independent_actor_motion']=bool(np.max(abs(motion[motion[:,0]<start,1]))<.0001 and
                                                   abs(float(np.ptp(motion[:,1]))-travel)<.0005)
    return dict(pass_case=all(checks.values()),checks=checks,retreat_m=row['retreat_measured_m'],
                rear_min_box_m=row['rear_min_box_m'],rear_min_robot_m=row['rear_min_robot_m'],event=event,
                witness_at_event=sample.tolist() if isinstance(sample,np.ndarray) else sample)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('previous',type=Path)
    ap.add_argument('--out',default='qa-dense-edge.json')
    a=ap.parse_args();cases={}
    for batch in BATCHES:
        if (a.run/(batch+'.exit')).read_text().strip()!='0':
            raise ValueError(f'Incomplete simulator batch: {batch}')
        folder=a.run/batch;r=json.loads((folder/'report.json').read_text())
        if r['checkpoint_sha256']!=FROZEN_CHECKPOINT or r['trained'] or r['heldout_angle_reused']:
            raise ValueError(f'Unexpected checkpoint/training/holdout: {batch}')
        for row in r['cases']:
            name=row['case']
            if name in cases:raise ValueError(f'Duplicate case: {name}')
            cases[name]=assess(row,folder,a.previous)
    if cases.keys()!=CASES.keys():raise ValueError('Incomplete case panel')
    # An injected dropout may be delivered and still not isolate blocking:
    # the otherwise matched, unmasked geometry must itself permit retreat.
    for name,base in [('masked-edge-right','clear-with-handles'),('masked-edge-left','mirror-clear')]:
        row=cases[name]
        row['checks']['matched_unmasked_path_allows_retreat'] = cases[base]['checks']['measured_14mm_retreat_without_veto']
        row['pass_case']=all(row['checks'].values())
    ok=all(row['pass_case'] for row in cases.values())
    with (a.run/a.out).open('x') as f:
        f.write(json.dumps(dict(pass_all=ok,bench_only=True,checkpoint_sha256=FROZEN_CHECKPOINT,cases=cases,
                                  hardware_or_task_success=False),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:list(n for n,v in r['checks'].items() if not v) for k,r in cases.items()}))
    return 0 if ok else 1


if __name__=='__main__':raise SystemExit(main())
