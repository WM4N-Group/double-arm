#!/usr/bin/env python3
"""Verify and read the migrated compact eight-episode replay bundle on CPU.

No simulator imports, credential handling, GPU work, model or training code.
The public data contract preserves frame0 and the original action bytes, not
later RGB frames or original full HDF5 observations.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

BUNDLE_SHA = '2a6244f5ca194cbbdef5f5865fde6122f021a864bb04e5ad71bb146e2a8d0fa0'
PREFLIGHT_SHA = '26758174a7dea626ca74dcba4df29f3c3b3d80db1395e53454d32e4a27825a2d'
PANEL_SHA = 'ea90461c1c9d7d6193eda55957e6837779146dde9df7698434cada4e5c6ed96a'
CAMERAS = ('head_image', 'left_wrist_image', 'right_wrist_image')


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read_verified(directory, filename, expected):
    path = directory / filename
    if path.parent != directory or path.is_symlink() or not path.is_file():
        raise ValueError('Expected a regular file in the compact bundle directory')
    data = path.read_bytes()
    if sha(data) != expected:
        raise ValueError(f'Compact input checksum drift: {filename}')
    return data


def load_panel(directory, original_preflight):
    directory = Path(directory).resolve()
    original = json.loads(read_verified(original_preflight.parent,original_preflight.name,PREFLIGHT_SHA))
    m = json.loads(read_verified(directory,'manifest.json',BUNDLE_SHA))
    if (m.get('mode') != 'MINIMAL_EIGHT_EXPERT_REPLAY_INPUTS_NOT_FULL_HDF5_NOT_TRAINING_DATA'
            or m.get('panel_sha256') != PANEL_SHA or m.get('source_cpu_preflight_sha256') != PREFLIGHT_SHA
            or m.get('source_files_changed') is not False
            or m['replay_must_run'] != {'grasping_mode':'physical','restore_scene_state_count':1,
                                      'refresh_steps':20,'no_assistance_audit':True}
            or original['panel_sha256'] != PANEL_SHA or len(m['cases']) != 8 or len(original['episodes']) != 8):
        raise ValueError('Invalid frozen compact panel contract')
    return directory, m['cases'], original['episodes']


def load_case(directory, entry, reference):
    """Return (native actions, serialized init, first proprio, RGB, source config)."""
    import io
    import numpy as np
    i = entry['index']
    expected = {'original_actions_float32_sha256','frame0_state_float32_sha256',
                'frame0_rgb_sha256','env_config_canonical_sha256'}
    if (entry['source'] != reference['source'] or entry['source_demo'] != reference['demo']
            or entry['fingerprint'] != reference['fingerprint'] or entry['frames'] != reference['frames']
            or set(entry['input_arrays']) != expected or any(entry['input_arrays'][k] != reference[k] for k in expected)
            or entry['npz'] != f'case-{i:02d}.npz' or entry['env_json'] != f'case-{i:02d}-env.json'):
        raise ValueError('Compact case is not bound to the archived source episode')
    with np.load(io.BytesIO(read_verified(directory,entry['npz'],entry['npz_sha256'])),allow_pickle=False) as z:
        if set(z.files) != {'actions','serialized_frame0','proprio_frame0',*CAMERAS}:
            raise ValueError('Unexpected compact fields')
        actions, state, proprio = (z[k] for k in ('actions','serialized_frame0','proprio_frame0'))
        rgb = {k:z[k] for k in CAMERAS}
    if (actions.shape != (entry['frames'],21) or state.shape != (reference['state_width'],) or proprio.shape != (21,)
            or not all(x.dtype == np.float32 and np.isfinite(x).all() for x in (actions,state,proprio))
            or any(a.shape!=(256,256,3) or a.dtype!=np.uint8 for a in rgb.values())):
        raise ValueError('Bad raw compact frame/action layout')
    if (sha(actions.tobytes()+proprio.tobytes()) != entry['fingerprint']
            or sha(actions.tobytes()) != reference['original_actions_float32_sha256']
            or sha(state.tobytes()) != reference['frame0_state_float32_sha256']
            or {k:sha(v.tobytes()) for k,v in rgb.items()} != reference['frame0_rgb_sha256']):
        raise ValueError('Migrated arrays changed from source preflight')
    env = json.loads(read_verified(directory,entry['env_json'],entry['env_json_sha256']))
    if (env['env_name'] != 'r1_tidy_table_D0' or env['env_kwargs']['init_curobo'] is not False
            or env['env_kwargs']['env']['action_frequency'] != 30
            or env['env_kwargs']['robots'][0]['grasping_mode'] != 'assisted'):
        raise ValueError('Need unchanged serialized source layout; physical is applied after verified empty-hand reset')
    return actions,state,proprio,rgb,env


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--original-preflight',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CPU-only verification required')
    directory,entries,refs=load_panel(a.bundle,a.original_preflight.resolve())
    n=0
    for index,(e,r) in enumerate(zip(entries,refs),1):
        if e['index'] != index:
            raise ValueError('Case order changed')
        n+=len(load_case(directory,e,r)[0])
    if n!=13567:
        raise ValueError('Action count differs from frozen expert panel')
    print(json.dumps({'state':'COMPACT_EXPERT_ARRAYS_VERIFIED_ON_TARGET_NOT_LAUNCHED','cases':len(entries),'actions':n,
                      'bundle_manifest_sha256':BUNDLE_SHA,'simulator_launched':False,'model_or_gpu_used':False,
                      'replay_must_disable_assistance_before_any_action':True},indent=2))


if __name__=='__main__':
    main()
