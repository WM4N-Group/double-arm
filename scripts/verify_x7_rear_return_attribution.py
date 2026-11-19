#!/usr/bin/env python3
"""Compare the diagnostic replay to frozen traces and summarize latch evidence."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_x7_certified_retreat import REAR_OBSTACLE_SPEED_BOUND
from audit_x7_chassis_range_proxy import PROXY_GUARD_M
from audit_x7_dock_sensor import POS_ERROR_BOUND
from diagnose_x7_rear_returns import FROZEN_CHECKPOINT, digest
from run_x7_mobile_bimanual import HZ

REFERENCES = {
    'clear-with-handles': 'controls',
    'main-wall-with-budget': 'wall',
    'handle-with-coverage': 'handles',
    'masked-handle': 'mask-qualified-budget',
}


def compare_trace(before, after):
    """Require exact agreement, including raw packets and every actuator reference."""
    with np.load(before, allow_pickle=False) as a, np.load(after, allow_pickle=False) as b:
        if a.files != b.files or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError(f'Diagnostic changed archived trace: {after.name}')
        return list(a.files), b['sensor_telemetry']


def explain_event(packet, event, telemetry):
    k, sample = event['substep'], event['sample_k']
    rows = telemetry[(telemetry[:, 0] == k) & (telemetry[:, 3] == sample)]
    if len(rows) != 1:
        raise ValueError(f'Missing or ambiguous delivered sensor row: {k}/{sample}')
    row = rows[0]
    raw, low, odom = map(float, row[[8, 11, 12]])
    if raw != packet['fused_raw_m']:
        raise ValueError('Diagnostic/telemetry raw packet mismatch')
    evidence = dict(event=event, delivered_raw_m=raw, delivered_lower_m=low, age_substeps=int(row[4]),
                    channels=packet['channels'])
    if raw < 0:
        if low != -1. or event['event'] != 'rear_coverage_lost':
            raise ValueError('Unknown diagnostic packet did not fail closed')
        return evidence | {'known': False}
    age = float(row[4]) / HZ * REAR_OBSTACLE_SPEED_BOUND
    projection = raw * math.cos(math.radians(3.))
    rebuilt = projection - odom - POS_ERROR_BOUND - PROXY_GUARD_M - age
    if not math.isclose(rebuilt, low, rel_tol=0., abs_tol=1e-12) or low != event['rear_lower_bound_m']:
        raise ValueError('Lower-bound reconstruction mismatch')
    winner = min(packet['channels'], key=lambda ch: packet['channels'][ch]['calibrated_m'])
    return evidence | dict(known=True, limiting_channel=winner, projection_loss_m=raw - projection,
                           odometry_loss_m=odom, range_error_loss_m=POS_ERROR_BOUND,
                           proxy_loss_m=PROXY_GUARD_M, age_loss_m=age)


def verify(reference, replay):
    rows = {}
    for name, subdir in REFERENCES.items():
        before, after = reference / subdir / (name + '.npz'), replay / (name + '.npz')
        baseline = json.loads((before.parent / 'report.json').read_text())
        expected = next(row for row in baseline['cases'] if row['case'] == name)
        report = json.loads(after.with_suffix('.json').read_text())
        actual = report['result']
        if (report['checkpoint_sha256'] != FROZEN_CHECKPOINT or report['diagnostic_ids_used_for_control'] is not False
                or actual != expected or digest(before) != expected['trace_sha256'] or digest(after) != actual['trace_sha256']):
            raise ValueError(f'Unexpected archive/checkpoint/result: {name}')
        fields, telemetry = compare_trace(before, after)
        events = [explain_event(report['packets'][str(e['sample_k'])], e, telemetry) for e in actual['rear_events']
                  if e['event'] in ('rear_coverage_lost', 'bounded_rear_travel_veto')]
        if len(events) != 1:
            raise ValueError(f'Unexpected number of qualified latch events: {name}')
        rows[name] = dict(reference_sha256=expected['trace_sha256'], replay_sha256=actual['trace_sha256'],
                          identical_arrays=fields, attribution_sha256=digest(after.with_suffix('.json')),
                          measured_retreat_m=actual['retreat_measured_m'], payload_clearance_m=actual['rear_min_box_m'],
                          robot_clearance_m=actual['rear_min_robot_m'], payload_contact_samples=actual['box_contact_samples'],
                          robot_contact_samples=actual['robot_contact_samples'], events=events)
    return dict(controller_changed=False, checkpoint_sha256=FROZEN_CHECKPOINT, cases=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference', type=Path)
    ap.add_argument('replay', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    content = verify(args.reference, args.replay)
    with args.out.open('x') as out:
        out.write(json.dumps(content, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
