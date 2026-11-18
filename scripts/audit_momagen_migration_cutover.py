#!/usr/bin/env python3
"""Read-only target inventory for a scoped MoMaGen migration, never activate it."""
import argparse
import json
import os
from pathlib import Path
import stat

RUNTIME_COPIED = 'scoped_runtime_staged_assets_dependency_check_pending'
ASSETS_COPIED = 'scoped_assets_staged_not_simulator_verified'
COMPACT_VERIFIED = 'COMPACT_EXPERT_ARRAYS_VERIFIED_ON_TARGET_NOT_LAUNCHED'
# Intentionally omit code/ (shared with X7), old run pointers, tools/ and licenses.
CANONICAL_LEAVES = ('envs/momagen', 'src/MoMaGen', 'src/curobo', 'data/behavior-1k-assets',
                    'data/omnigibson-robot-assets', 'data/2025-challenge-task-instances')


def safe_private(path):
    try:
        if path.is_symlink():
            return False
        info = path.stat()
        return stat.S_ISREG(info.st_mode) and not (stat.S_IMODE(info.st_mode) & 0o077) and info.st_uid == os.getuid()
    except FileNotFoundError:
        return False


def file_status(path, key):
    try:
        obj=json.loads(path.read_bytes())
        value=obj.get(key)
        return value if isinstance(value,str) else None
    except (OSError, ValueError, AttributeError):
        return None


def inspect(root, stage):
    root=Path(root).resolve()
    stage=Path(stage).resolve()
    if not stage.is_relative_to(root) or stage==root or not stage.name.startswith('.migration-momagen-'):
        raise ValueError('Explicit isolated MoMaGen staging directory under target project required')
    payload=stage/'payload'
    runtime=file_status(stage/'scope-status.json','phase')
    assets=file_status(stage/'asset-status.json','phase')
    compact=file_status(stage/'compact-inputs-target-verification.json','state')
    blockers=[]
    for ok, label in ((runtime==RUNTIME_COPIED,'runtime_transfer_incomplete_or_unverified'),
                     (assets==ASSETS_COPIED,'scene_asset_transfer_incomplete_or_unverified'),
                     (compact==COMPACT_VERIFIED,'compact_input_arrays_unverified')):
        if not ok: blockers.append(label)
    leaves={}
    for name in CANONICAL_LEAVES:
        local=payload/name
        dest=root/name
        ready=local.is_dir() and not local.is_symlink() and local.resolve().is_relative_to(payload)
        collision=os.path.lexists(dest) and not (dest.is_symlink() and dest.resolve()==local.resolve() and ready)
        leaves[name]={'staged_directory_seen':ready, 'target_path_already_exists':os.path.lexists(dest),
                      'target_collision_do_not_overwrite':collision}
        if collision: blockers.append(f'canonical_path_collision:{name}')
        if not ready: blockers.append(f'staged_directory_missing_or_unsafe:{name}')
    # Presence and file mode only. No read, reuse by assumption, or license acceptance.
    key=safe_private(root/'data/omnigibson.key')
    if not key: blockers.append('existing_authorized_dataset_access_key_not_securely_provisioned_on_target')
    # No auto-launch even if all paths are staged. Runtime, license confirmation
    # and a named job/GPU authorization are separate review gates.
    return {'state':'READ_ONLY_PREFLIGHT_NOT_ACTIVATED','runtime_phase':runtime,'assets_phase':assets,
            'compact_phase':compact,'required_leaf_paths':leaves,'blockers':blockers,
            'shared_project_code_not_an_install_target':True,'license_confirmation_read_or_changed':False,
            'old_pids_or_gpu_authorizations_reused':False,'filesystem_changes':False,'simulator_launched':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--staging',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(inspect(a.root,a.staging),indent=2))


if __name__=='__main__':
    main()
