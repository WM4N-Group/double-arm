#!/usr/bin/env python3
"""Server CPU-only comparison of saved cup motion around first reported grasp.

The marker is an observation, not the exact physics-substep time of joint
creation. This diagnostic never restores a scene, changes a label, or runs π₀.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

RUNS = Path('/mnt/yuanyue/double-arm/runs')


def first_true(flags):
    indices = np.flatnonzero(flags)
    return int(indices[0]) if len(indices) else None


def root_relative_to_world(local, root):
    local = np.asarray(local, np.float64)
    root = np.asarray(root, np.float64)
    if (local.ndim != 2 or local.shape[1] != 3 or root.shape != (len(local), 7)
            or not np.isfinite(local).all() or not np.isfinite(root).all()):
        raise ValueError('Expected finite aligned XYZ and root XYZ+XYZW')
    q = root[:, 3:]
    norm = np.linalg.norm(q, axis=1)
    if np.any(abs(norm - 1) > .001):
        raise ValueError('Expected unit XYZW quaternions')
    q = q / norm[:, None]
    t = 2 * np.cross(q[:, :3], local)
    return root[:, :3] + local + q[:, 3:] * t + np.cross(q[:, :3], t)


def onset_metrics(cup, marker, *, eef=None, contacts=None):
    cup = np.asarray(cup, np.float64)
    if cup.ndim != 2 or cup.shape[1] != 3 or not len(cup) or not np.isfinite(cup).all():
        raise ValueError('Expected finite XYZ samples')
    if marker is None:
        return {'first_grasp_sample': None}
    if not 0 <= marker < len(cup):
        raise ValueError('Grasp index outside trace')
    if eef is not None and (eef.shape != cup.shape or not np.isfinite(eef).all()):
        raise ValueError('Expected aligned finite EEF origins')
    if contacts is not None and contacts.shape != (len(cup),):
        raise ValueError('Expected aligned contact counts')
    rows = []
    for i in range(max(marker - 3, 0), min(marker + 4, len(cup))):
        delta = (cup[i] - cup[i - 1]) * 1000 if i > 0 else None
        rows.append({'sample': i, 'relative_sample': i - marker,
                     'cup_world_step_mm': float(np.linalg.norm(delta)) if delta is not None else None,
                     'cup_world_step_z_mm': float(delta[2]) if delta is not None else None,
                     'left_eef_origin_to_cup_origin_mm': float(np.linalg.norm(eef[i] - cup[i]) * 1000)
                     if eef is not None else None,
                     'left_contact_links': int(contacts[i]) if contacts is not None and contacts[i] >= 0 else None})
    return {'first_grasp_sample': marker,
            'cup_step_at_marker_mm': next(r['cup_world_step_mm'] for r in rows if r['sample'] == marker),
            'max_cup_world_step_mm_within_3_samples': max((r['cup_world_step_mm'] for r in rows
                                                         if r['cup_world_step_mm'] is not None), default=None),
            'window': rows}


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_raw(entry):
    import h5py
    p = Path(entry['source']).resolve()
    if not p.is_relative_to(RUNS) or p.name != 'demo.hdf5' or p.parent.parent.name != 'generated':
        raise ValueError('Require finalized in-project HDF5')
    with h5py.File(p, 'r') as f:
        data = f['data']
        env = json.loads(data.attrs['env_args'])
        g = data[entry['source_demo']]
        obs = g['obs']
        if (env['env_name'] != 'r1_tidy_table_D0' or env['env_kwargs']['robots'][0]['grasping_mode'] != 'assisted'
                or bool(g.attrs.get('partial', False))):
            raise ValueError('Expected full official assisted D0 episode')
        state = np.asarray(obs['prop_state'][0], dtype=np.float32)
        act = np.asarray(g['actions'][()], dtype=np.float32)
        if hashlib.sha256(act.tobytes() + state.tobytes()).hexdigest() != entry['fingerprint']:
            raise ValueError('Source fingerprint mismatch')
        root = obs['base_link_pose'][()]
        cup = root_relative_to_world(obs['object::teacup'][:, :3], root)
        flags = obs['grasp_left'][()].reshape(-1)
        if (len(flags) != len(cup) or len(act) != len(cup) or len(act) != entry['frames']
                or not np.isin(flags, (-1, 0, 1)).all()):
            raise ValueError('Unaligned source samples or unknown grasp encoding')
        fps = float(env['env_kwargs']['env']['action_frequency'])
        return {'source': str(p), 'demo': entry['source_demo'], 'fingerprint': entry['fingerprint'],
                'seed': entry.get('seed'), 'mode': 'assisted', 'frames': len(cup), 'fps': fps,
                'marker_scope': 'source left arm is_grasping(any object), saved observation, not event log',
                **onset_metrics(cup, first_true(flags == 1))}


def read_rollout(label, p):
    opts = {'B_assisted': ('assisted', 'demo_0-trace.npz'), 'B_physical_execute10': ('physical', 'demo_0-trace.npz'),
            'B_physical_execute5': ('physical', 'demo_0-trace.npz'), 'source_physical_replay': ('physical', 'trace.npz')}
    mode, trace_name = opts[label]
    result, result_sha = load_json(p / 'result.json')
    if (result['grasping_mode'] != mode or result.get('aborted') or any(e.get('aborted') for e in result.get('episodes', []))
            or (label.startswith('B_') and result.get('mode') != 'closed_loop_neural_policy')):
        raise ValueError('Unexpected runtime mode or incomplete rollout')
    path = p / trace_name
    with np.load(path, allow_pickle=False) as x:
        cup = x['post_step_cup_positions']
        flags = x['post_step_cup_grasp_flags']
        if flags.shape != (len(cup),) or flags.dtype != np.bool_:
            raise ValueError('Unaligned or unknown rollout grasp flags')
        m = onset_metrics(cup, first_true(flags), eef=x['post_step_eef_world_poses'][:, 0, :3],
                          contacts=x['post_step_cup_contact_finger_counts'][:, 0])
        noassist = x['no_assistance_observations'] if 'no_assistance_observations' in x else None
        evidence = (int(len(noassist)) if noassist is not None and noassist.shape == (21 + 2 * len(cup), 8)
                    and np.all(noassist == 0) else None)
        return {'label': label, 'path': str(p), 'result_sha256': result_sha,
                'trace_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'mode': mode,
                'checkpoint': result.get('checkpoint'), 'assisted_constraint_seen': result.get('assisted_constraint_seen'),
                'object_pose_write_count': result.get('object_pose_write_count'), 'all_zero_no_assistance_boundaries': evidence,
                'frames': len(cup), 'marker_scope': 'cup-specific backend grasp report, after action; not exact joint creation timestamp',
                **m}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--B', type=Path, required=True)
    p.add_argument('--B-execute5', type=Path, required=True)
    p.add_argument('--source-replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists()
            or not all(x.resolve().is_relative_to(RUNS) for x in vars(a).values())):
        raise ValueError('Require CPU only, in-project paths, and new report')
    selection, selection_sha = load_json(a.selection)
    replay = a.source_replay / 'replay'
    replay_result, _ = load_json(replay / 'result.json')
    matched = replay_result['source']
    if (replay_result['mode'] != 'SOURCE_ACTION_REPLAY_PHYSICAL_NOT_LEARNED_POLICY'
            or replay_result['teacher_source_actions_executed'] != matched['frames']
            or replay_result['policy_inference_calls'] != 0):
        raise ValueError('Expected completed expert replay with no policy inference')
    paths = {'B_assisted': a.B / 'train-author-diagnostic/physical-rollout',
             'B_physical_execute10': a.B / 'train-physical-diagnostic/physical-rollout',
             'B_physical_execute5': a.B_execute5 / 'train-physical-chunk5/physical-rollout',
             'source_physical_replay': replay}
    report = {'selection_sha256': selection_sha, 'audit_sha256': selection['audit_sha256'],
              'scope': '30 Hz observation-level descriptive comparison; action sequences and viewpoints differ; not a physics intervention',
              'units': 'mm; object pose origin, not center of mass or fingertip clearance',
              'raw_random_samples': [read_raw(e) for e in selection['episodes']],
              'raw_matched_train_source': read_raw(matched),
              'rollouts': {k: read_rollout(k, v) for k, v in paths.items()}}
    if len({row['checkpoint'] for key, row in report['rollouts'].items() if key.startswith('B_')}) != 1:
        raise ValueError('B comparisons require the same checkpoint')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(str(a.output))
    for label, row in [(f'raw_{i}', e) for i, e in enumerate(report['raw_random_samples'], 1)] + [
            ('raw_matched', report['raw_matched_train_source']), *report['rollouts'].items()]:
        print(label, {k: row.get(k) for k in ('first_grasp_sample', 'cup_step_at_marker_mm',
              'max_cup_world_step_mm_within_3_samples', 'assisted_constraint_seen', 'all_zero_no_assistance_boundaries')})


if __name__ == '__main__':
    main()
