#!/usr/bin/env python3
"""CPU-only contract for the planned same-tape assisted/physical replay.

Does not load Isaac or a model. Each fresh simulator subprocess must supply
its own read-only initial snapshot and ordered boundary evidence. The result
is a bounded open-loop comparison, never learned-policy acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

RUNS = Path('/mnt/yuanyue/double-arm/runs')
BOUNDARY_FIELDS = ('left_joint', 'right_joint', 'left_task_cup', 'right_task_cup',
                   'left_freeze', 'right_freeze', 'left_usd_joint', 'right_usd_joint')
INITIAL_FIELDS = {'initial_proprio': (21,), 'initial_root_pose': (7,),
                  'initial_cup_pose': (7,), 'initial_left_eef_pose': (7,)}
RESET = 20
# Fail the causal comparison if the realized post-refresh scenes differ by
# more than these predeclared tolerances; don't repair by reloading mid-run.
MAX_INITIAL_TRANSLATION_M = .0005
MAX_INITIAL_ROTATION_RAD = .005
MAX_INITIAL_PROPRIO_COORDINATE_DIFF = .001  # mixed native units


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_tape(plan, data):
    if (plan['status'] != 'INPUT_PREFLIGHT_PASSED_NOT_LAUNCHED' or plan['refresh_steps'] != RESET
            or plan['initialization_only_source_frame'] != 0 or data.dtype != np.float32
            or data.shape != (plan['original_steps'], 21) or not np.isfinite(data).all()):
        raise ValueError('Wrong planned replay initialization or tape layout')
    n = plan['planned_prefix_steps']
    if not 0 < plan['first_cup_specific_report_in_original_trace'] < n <= len(data):
        raise ValueError('Planned prefix does not contain original grasp marker')
    prefix = data[:n]
    if sha(prefix.tobytes()) != plan['planned_prefix_action_float32_sha256']:
        raise ValueError('Applied action prefix differs from frozen tape')
    return prefix


def finite(trace, key, shape):
    value = np.asarray(trace[key], dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'Missing or invalid {key}')
    return value


def unit_quat(pose):
    q = pose[..., 3:]
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(np.abs(norms - 1) > .001):
        raise ValueError('Non-unit XYZW quaternion')
    return q / norms


def rotation_error(a, b):
    return 2 * np.arccos(np.clip(np.abs((unit_quat(a) * unit_quat(b)).sum(axis=-1)), 0, 1))


def validate_case(tape, mode, result, trace):
    n = len(tape)
    if (result.get('grasping_mode') != mode or result.get('completed_actions') != n
            or result.get('aborted') is not False or result.get('error')
            or result.get('initial_state_load_count') != 1 or result.get('reset_physics_steps') != RESET
            or result.get('explicit_object_pose_writes') != 0 or result.get('policy_calls') != 0
            or result.get('recorded_boundary_fields') != list(BOUNDARY_FIELDS)):
        raise ValueError(f'{mode}: incomplete replay or unexpected control path')
    if (trace['actions'].dtype != np.float32 or trace['actions'].shape != tape.shape
            or trace['actions'].tobytes() != tape.tobytes()):
        raise ValueError(f'{mode}: actually executed actions differ from frozen tape')
    initial = {k: finite(trace, k, shape) for k, shape in INITIAL_FIELDS.items()}
    cup = finite(trace, 'post_step_cup_poses', (n, 7))
    left = finite(trace, 'post_step_left_eef_poses', (n, 7))
    for x in (cup, left):
        unit_quat(x)
    contact = trace['post_step_left_cup_contacts']
    ag = trace['grasp_boundaries']
    if (contact.shape != (n,) or contact.dtype.kind not in 'iu' or (contact < 0).any()
            or ag.shape != (1 + RESET + 2*n, 8) or ag.dtype.kind not in 'iu' or not np.isin(ag, [0, 1]).all()):
        raise ValueError(f'{mode}: incomplete or unknown read-only evidence')
    if (ag[:1+RESET].any() or (mode == 'physical' and ag.any())):
        raise ValueError(f'{mode}: unexpected assistance at observed boundary')
    # A Python joint without its USD counterpart (or vice versa) is ambiguous.
    if not np.array_equal(ag[:, :2], ag[:, 6:8]):
        raise ValueError(f'{mode}: inconsistent Python/USD joint observations')
    return initial, cup, left, contact, ag


def compare(tape, assisted_result, assisted, physical_result, physical):
    ai, ac, ae, at, ag = validate_case(tape, 'assisted', assisted_result, assisted)
    pi, pc, pe, pt, _ = validate_case(tape, 'physical', physical_result, physical)
    initial = {'proprio_max_native_abs': float(np.max(np.abs(ai['initial_proprio'] - pi['initial_proprio'])))}
    if initial['proprio_max_native_abs'] > MAX_INITIAL_PROPRIO_COORDINATE_DIFF:
        raise ValueError('Realized post-refresh proprio diverges before commands')
    for field in ('initial_root_pose', 'initial_cup_pose', 'initial_left_eef_pose'):
        a, p = ai[field], pi[field]
        pos, rot = float(np.linalg.norm(a[:3]-p[:3])), float(rotation_error(a,p))
        initial[field] = {'translation_m': pos, 'rotation_rad': rot}
        if pos > MAX_INITIAL_TRANSLATION_M or rot > MAX_INITIAL_ROTATION_RAD:
            raise ValueError(f'{field}: realized post-refresh states are not comparable')
    boundaries = ag[1+RESET:]
    left_cup_joint = (boundaries[:, 0] == 1) & (boundaries[:, 2] == 1) & (boundaries[:, 6] == 1)
    onset = np.flatnonzero(left_cup_joint)
    post = np.flatnonzero(left_cup_joint[1::2])
    first = int(onset[0]) if len(onset) else None
    marker = first // 2 if first is not None else None
    delta = np.linalg.norm(ac[:, :3]-pc[:, :3],axis=1)

    def summarize(cup, eef, contact, start):
        both = np.flatnonzero(contact >= 2)
        return {'first_post_step_with_two_or_more_cup_contact_links': int(both[0]) if len(both) else None,
                'max_cup_lift_m': float(np.max(cup[:, 2] - start[2])),
                'end_eef_origin_to_cup_origin_m': float(np.linalg.norm(eef[-1,:3]-cup[-1,:3])),
                'left_contact_samples': int(np.count_nonzero(contact))}

    window = None if marker is None else [
        {'step': k, 'cup_world_separation_m': float(delta[k]),
         'cup_orientation_separation_rad': float(rotation_error(ac[k], pc[k])),
         'assisted_contacts': int(at[k]), 'physical_contacts': int(pt[k])}
        for k in range(max(marker-3,0), min(marker+4,len(tape))) ]
    return {'same_action_bytes': True, 'actions': len(tape), 'initial_alignment': initial,
            'first_assisted_left_cup_joint_observed_boundary': None if first is None else
                {'step': marker, 'phase': 'post' if first % 2 else 'pre'},
            'first_assisted_left_cup_joint_observed_post_step': int(post[0]) if len(post) else None,
            'pre_attachment_max_cup_separation_m': float(np.max(delta[:marker])) if marker else None,
            'max_cup_separation_m': float(np.max(delta)), 'onset_window': window,
            'assisted': summarize(ac,ae,at,ai['initial_cup_pose']),
            'physical': summarize(pc,pe,pt,pi['initial_cup_pose']),
            'scope': 'Replay of identical commands, not live policy. Boundary observations, not every physics substep. No grasp/placement success claim.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--case-root', type=Path, help='Contains assisted/ and physical/, each with result.json and trace.npz')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    paths = (args.plan, args.case_root, args.output)
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(x is not None and not x.resolve().is_relative_to(RUNS) for x in paths)
            or bool(args.case_root) != bool(args.output)):
        raise ValueError('CPU only and project runs only; supply both replay root and output or neither')
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    tape_path = Path(plan['tape']).resolve()
    if not tape_path.is_relative_to(RUNS) or sha(tape_path.read_bytes()) != plan['tape_sha256']:
        raise ValueError('Recorded full trace hash changed')
    with np.load(tape_path, allow_pickle=False) as t:
        tape = verify_tape(plan, t['actions'])
    if args.case_root is None:
        print(json.dumps({'state': 'CPU_TAPE_CONTRACT_PASSED_NOT_LAUNCHED', 'actions': len(tape), 'plan_sha256': sha(plan_bytes)}))
        return
    a, b = (args.case_root / mode for mode in ('assisted','physical'))
    ra, rb = ((x/'result.json').read_bytes() for x in (a,b))
    with np.load(a/'trace.npz', allow_pickle=False) as ta, np.load(b/'trace.npz', allow_pickle=False) as tb:
        report = compare(tape, json.loads(ra), ta, json.loads(rb), tb)
    report.update(plan_sha256=sha(plan_bytes), assisted_result_sha256=sha(ra), physical_result_sha256=sha(rb),
                  assisted_trace_sha256=sha((a/'trace.npz').read_bytes()), physical_trace_sha256=sha((b/'trace.npz').read_bytes()))
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
