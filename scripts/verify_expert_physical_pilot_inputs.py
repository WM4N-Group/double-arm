#!/usr/bin/env python3
"""Read-only CPU preflight of the eight frozen physical expert replay inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from plan_expert_physical_pilot import RUNS, panel

PANEL_SHA = 'ea90461c1c9d7d6193eda55957e6837779146dde9df7698434cada4e5c6ed96a'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def path_under_runs(value):
    p = Path(value).resolve()
    if not p.is_relative_to(RUNS):
        raise ValueError('Expected project run path')
    return p


def verify_bound_source(entry, shard_info):
    """Return original arrays; perform no action clipping or normalization."""
    import h5py
    import numpy as np
    from momagen_pi0_schema import camera_keys, native_actions, rgb_image
    from pi0_base_targets import root_rotation_xy_and_yaw

    src = path_under_runs(entry['source'])
    shard = path_under_runs(entry['shard'])
    if src.name != 'demo.hdf5' or src.parent.parent != shard/'generated':
        raise ValueError('Expected finalized shard data, not staging or converted training data')
    for key, name in [('generation_result_sha256','generation-result.json'),('manifest_sha256','manifest.json')]:
        blob = (shard/name).read_bytes()
        if digest(blob) != shard_info[key]:
            raise ValueError('Finalized shard evidence changed after audit')
        if name == 'generation-result.json':
            summary = json.loads(blob)
            if summary['state'] != 'passed' or summary['generator_exit_status'] != 0 or path_under_runs(summary['dataset']) != src:
                raise ValueError('Finalized generator verdict not passed')
    n = entry['frames']
    with h5py.File(src, 'r') as f:
        d = f['data']; meta = json.loads(d.attrs['env_args']); g = d[entry['source_demo']]; obs = g['obs']
        env = meta['env_kwargs']['env']
        if (meta['env_name'] != 'r1_tidy_table_D0' or meta['env_kwargs']['robots'][0]['grasping_mode'] != 'assisted'
                or bool(g.attrs.get('partial', False)) or env['action_frequency'] != 30):
            raise ValueError('Unexpected task, frequency, grasping mode or partial source')
        if g['actions'].dtype != np.float32 or g['states'].dtype != np.float32:
            raise ValueError('Original FP32 expert tape and frame0 state required')
        actions = native_actions(g['actions'][()]); init = g['states'][0]; proprio = obs['prop_state'][0]
        keys = camera_keys(obs)
        if (actions.shape != (n,21) or g['states'].shape[0] != n or init.ndim != 1 or not len(init)
                or obs['prop_state'].shape != (n,21) or not np.isfinite(init).all() or not np.isfinite(proprio).all()
                or entry['cameras'] != keys or any(obs[k].shape[:3] != (n,256,256) for k in keys.values())):
            raise ValueError('Expected aligned complete action/state/camera source')
        grasp_zero = [int(obs[f'grasp_{side}'][0].reshape(-1)[0]) for side in ('left','right')]
        if grasp_zero != [-1,-1]:
            raise ValueError('Source frame0 does not report both hands empty')
        root_rotation_xy_and_yaw(obs['base_link_pose'][0])
        images = {k: rgb_image(obs[v][0]) for k,v in keys.items()}
    if digest(actions.tobytes()+proprio.tobytes()) != entry['fingerprint']:
        raise ValueError('Raw actions/proprio fingerprint mismatch')
    return actions, init, proprio, images, meta


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists():
        raise ValueError('CPU only and fresh report required')
    report_path = path_under_runs(a.output)
    raw = path_under_runs(a.panel).read_bytes()
    if digest(raw) != PANEL_SHA:
        raise ValueError('Not the original frozen expert panel')
    plan = json.loads(raw)
    audit_bytes = path_under_runs(plan['audit']).read_bytes()
    audit_sha = digest(audit_bytes)
    audit = json.loads(audit_bytes)
    if audit_sha != plan['audit_sha256'] or panel(audit,audit_sha,8) != plan['episodes']:
        raise ValueError('Panel is not reproducible from frozen QA')
    rows = []
    for e in plan['episodes']:
        actions, init, proprio, images, meta = verify_bound_source(e, audit['shards'][e['shard']])
        rows.append({'seed':e['seed'], 'source':e['source'], 'demo':e['source_demo'], 'fingerprint':e['fingerprint'],
                     'frames':len(actions), 'state_width':len(init), 'frame0_rgb_sha256':{k:digest(v.tobytes()) for k,v in images.items()},
                     'frame0_state_float32_sha256':digest(init.tobytes()), 'original_actions_float32_sha256':digest(actions.tobytes()),
                     'env_config_canonical_sha256':digest(json.dumps(meta,sort_keys=True).encode()),
                     'observed_native_action_max_abs_per_axis':np.max(np.abs(actions),axis=0).tolist(),
                     'left_native_gripper_values':np.unique(actions[:,13]).tolist(), 'source_frame0_reports_empty':True})
    result = {'state':'FROZEN_PANEL_INPUTS_VERIFIED_CPU_NOT_LAUNCHED','panel_sha256':PANEL_SHA,'audit_sha256':audit_sha,
              'physical_outcome_known':False,'gpu_jobs_started':0,'policy_calls':0,
              'source_frames':sum(row['frames'] for row in rows),'episodes':rows,
              'scope':'Provenance and array preflight only. No controller clipping or normalization. Empty-hand runtime and physical feasibility not yet verified.'}
    report_path.parent.mkdir(parents=True,exist_ok=True)
    payload = (json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
    with report_path.open('xb') as f:
        f.write(payload)
    print(report_path,digest(payload),len(rows),result['source_frames'])


if __name__ == '__main__':
    main()
