#!/usr/bin/env python3
"""Audit the complementary-ray panel; a simulation exit alone is not a pass."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_x7_handle_rear_budget import (
    ACTOR_DELAY_SUBSTEPS, FROZEN_CHECKPOINT, OBSTACLE_TRAVEL, RETREAT_M, STATION_CLEARANCE_M,
    digest, entry_margin,
)
from audit_x7_split_rear_rays import GAP_FIXTURE_TRAVEL_M, SCENARIOS

REFERENCE = {'clear-with-handles': ('controls', 'clear-with-handles'),
             'main-wall-with-budget': ('wall', 'main-wall-with-budget'),
             'handle-with-coverage': ('handles', 'handle-with-coverage'),
             'masked-handle': ('mask-qualified-budget', 'masked-handle'),
             **{name: ('controls', 'clear-with-handles') for name in
                ('inner-gap-actor', 'outer-cad-actor', 'masked-gap', 'masked-exterior')}}


def evaluate_case(row, out, original):
    name = row['case']
    trace = out/(name+'.npz')
    split = out/(name+'-split.npz')
    if digest(trace) != row['trace_sha256'] or digest(split) != row['split_trace_sha256']:
        raise ValueError(f'Archive hash mismatch: {name}')
    is_actor = SCENARIOS[name][-1] is not None
    is_mask = name.startswith('masked-')
    checks = {
        'qualified_holding_interrupt': row['safe_holding_interrupt'] and row['held_box_after_stop'] and
            row['guard_triggered'] and row['robot_station_clearance_pass'] and not row['success'],
        'no_contacts_and_rear_clears_5mm': row['box_contact_samples'] == row['robot_contact_samples'] == 0 and
            (not is_actor or min(row['rear_min_box_m'],row['rear_min_robot_m']) >= STATION_CLEARANCE_M),
        'decision_does_not_use_body_ids': not row['hit_body_ids_used_for_control'],
    }
    event = None
    event_packets = []
    with np.load(trace,allow_pickle=False) as a, np.load(split,allow_pickle=False) as s:
        packets = s['packets']
        box, robot = a['payload_audit'], a['robot_rear_audit']
        checks['recomputed_no_rear_contacts'] = not bool(np.any(box[:,3]) or np.any(robot[:,1]))
        if is_actor:
            checks['recomputed_rear_minima'] = bool(np.min(box[:,2]) == row['rear_min_box_m'] and
                np.min(robot[:,0]) == row['rear_min_robot_m'])
        station, init = a['station_clearance'], a['station_initialization_clearance']
        checks['all_audited_robot_station_substeps_clear'] = bool(
            min(np.min(station[...,:2]),np.min(init[...,:2])) >= STATION_CLEARANCE_M and
            not np.any(station[...,2:4]) and not np.any(init[...,2:4]))
        checks['main_box_has_split_and_direct_cells'] = bool(np.all(packets[packets[:,1] == 0,2:4]>0))
        if name in REFERENCE:
            folder, original_case = REFERENCE[name]
            with np.load(original/folder/(original_case+'.npz'),allow_pickle=False) as b:
                checks['same_original_action_and_phase'] = all(np.array_equal(a[k],b[k]) for k in ('action','phase'))
        if is_mask or is_actor:
            expected = 'rear_coverage_lost' if is_mask else 'bounded_rear_travel_veto'
            events = [e for e in row['rear_events'] if e['event'] == expected]
            checks['qualified_no_retreat'] = len(events) == 1 and abs(row['retreat_measured_m']) < .001
            if len(events) == 1:
                event = events[0]
                event_packets = packets[packets[:,0]==event['sample_k']].tolist()
                delivered = a['sensor_telemetry']
                delivered = delivered[(delivered[:,0]==event['substep']) & (delivered[:,3]==event['sample_k'])]
                if len(delivered) != 1:
                    raise ValueError(f'Missing delivered event: {name}')
                if is_mask:
                    checks['delivered_unknown_implies_no_retreat'] = bool(delivered[0,8]==delivered[0,11]==-1.)
                    if name != 'masked-handle':
                        checks['split_reading_lost'] = any(p[1]==0 and p[7]>0 for p in event_packets)
                else:
                    checks['unchanged_entry_budget'] = bool(event['required_m']==entry_margin() and
                        0. <= event['rear_lower_bound_m']==delivered[0,11]<entry_margin())
                if name == 'inner-gap-actor':
                    checks['gap_actually_seen_at_veto'] = any(p[1]==0 and p[5]>0 and p[6]==0 for p in event_packets)
                if name == 'outer-cad-actor':
                    checks['exterior_actually_seen_at_veto'] = any(p[1]==0 and p[6]>0 and p[5]==0 for p in event_packets)
        else:
            checks['complete_14mm_retreat_without_packet_veto'] = (
                abs(row['retreat_measured_m']-RETREAT_M) < .001 and not row['rear_events'])
            checks['no_false_split_returns_or_unknown'] = not bool(np.any(packets[:,5:]))
        if is_actor:
            motion = a['rear_obstacle_motion']
            birth = row['obstacle_reset']
            start = birth['substep']+ACTOR_DELAY_SUBSTEPS
            target_travel = GAP_FIXTURE_TRAVEL_M if name=='inner-gap-actor' else OBSTACLE_TRAVEL
            checks['actor_started_separately_and_executed_travel'] = bool(
                np.max(abs(motion[motion[:,0]<start,1])) < .0001 and abs(row['measured_obstacle_travel_m']-target_travel)<.0005)
            checks['initial_actor_clearance'] = min(birth['initial_box_gap_m'],birth['initial_robot_gap_m']) >= STATION_CLEARANCE_M
    return {'checks':checks,'pass':all(checks.values()),'retreat_m':row['retreat_measured_m'],
            'min_actor_to_box_m':row['rear_min_box_m'],'min_actor_to_robot_m':row['rear_min_robot_m'],
            'actor_travel_m':row['measured_obstacle_travel_m'],'event':event,'event_split_packets':event_packets}


def verify_original_controls(run, original):
    controls = {}
    for label, name in (('original-west','clear-with-handles'),('original-fresh','fresh-clear')):
        if (run/(label+'.exit')).read_text().strip()!='0':
            raise ValueError(f'Matching original evaluator not complete: {label}')
        out = run/label
        row = json.loads((out/'report.json').read_text())
        path = out/(name+'.npz')
        if row['case']!=name or digest(path)!=row['trace_sha256']:
            raise ValueError(f'Original evaluator trace mismatch: {label}')
        events = [e for e in row['rear_events'] if e['event']=='bounded_rear_travel_veto']
        checks = {'same_budget_and_original_veto': len(events)==1 and events[0]['required_m']==entry_margin(),
                  'held_without_retreat': row['safe_holding_interrupt'] and abs(row['retreat_measured_m'])<.001}
        if label=='original-west':
            checks['byte_identical_to_pre_refactor_original'] = digest(path)==digest(original/'controls'/(name+'.npz'))
        else:
            with np.load(path,allow_pickle=False) as a, np.load(run/'coverage-fresh'/(name+'.npz'),allow_pickle=False) as b:
                checks['same_fresh_motion_plan'] = all(np.array_equal(a[k],b[k]) for k in ('action','phase'))
        controls[label] = {'checks':checks,'pass':all(checks.values()),'retreat_m':row['retreat_measured_m'],
                           'trace_sha256':row['trace_sha256'],'rear_events':row['rear_events']}
    return controls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path)
    ap.add_argument('original',type=Path)
    a = ap.parse_args()
    cases = {}
    for batch in ('western','coverage-fresh'):
        if (a.run/(batch+'.exit')).read_text().strip()!='0':
            raise ValueError(f'Simulation not complete: {batch}')
        out = a.run/batch
        r = json.loads((out/'report.json').read_text())
        if r['checkpoint_sha256']!=FROZEN_CHECKPOINT or r['trained'] or r['heldout_angle_reused']:
            raise ValueError(f'Unexpected checkpoint/training/holdout: {batch}')
        for row in r['cases']:
            name = row['case']
            if name in cases:
                raise ValueError(f'Duplicate case: {name}')
            cases[name] = evaluate_case(row,out,a.original)
    if cases.keys()!=SCENARIOS.keys():
        raise ValueError('Incomplete panel')
    controls = verify_original_controls(a.run,a.original)
    all_pass = all(r['pass'] for r in (*cases.values(),*controls.values()))
    document = dict(pass_all=all_pass,cases=cases,matched_original_controls=controls,checkpoint_sha256=FROZEN_CHECKPOINT,
                    normal_task_success_claimed=False,hardware_safety_claimed=False)
    with (a.run/'qa-split-rays.json').open('x') as f:
        f.write(json.dumps(document,indent=2,allow_nan=False)+'\n')
    print(json.dumps({name:{k:v for k,v in r['checks'].items() if not v} for name,r in cases.items()}))
    return 0 if all_pass else 1


if __name__=='__main__':
    raise SystemExit(main())
