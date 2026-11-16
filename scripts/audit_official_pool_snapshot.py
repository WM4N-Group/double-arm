#!/usr/bin/env python3
"""SERVER CPU ONLY: immutable raw-pool snapshot, dedup and endpoint-cleaning audit.

Reads ONLY finalized official shards. Never creates train/validation splits,
changes HDF5/LeRobot/labels/norms, or controls a model/simulator. Joint qpos is
post-hoc audit data, NOT a VLA input. Author joint-subset mapping stays unverified.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path('/mnt/yuanyue/double-arm')


def select_unique_records(records, excluded):
    seen = set()
    selected = []
    duplicate_count = excluded_count = 0
    for entry in records:
        fingerprint = entry.get('fingerprint')
        if not isinstance(fingerprint, str) or re.fullmatch(r'[0-9a-f]{64}', fingerprint) is None:
            raise ValueError('Explicit SHA256 episode fingerprint required')
        if fingerprint in seen:
            duplicate_count += 1
            continue
        seen.add(fingerprint)
        if fingerprint in excluded:
            excluded_count += 1
        else:
            selected.append(entry)
    return selected, duplicate_count, excluded_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-provenance', type=Path, action='append', required=True)
    parser.add_argument('--exclude-provenance', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CPU-only audit required')
    if not args.output.resolve().is_relative_to(ROOT / 'runs') or args.output.exists():
        raise ValueError('New remote audit directory under runs/ required')
    import h5py
    import numpy as np
    from frozen_segment_audit import frozen_segment_mask
    from momagen_pi0_schema import camera_keys, native_actions

    snapshots = []
    exclusions = []
    excluded = set()
    records = []
    for path in args.raw_provenance:
        raw = path.read_bytes()
        meta = json.loads(raw)
        if not meta.get('scope', '').startswith('Raw official author-assisted source pool;'):
            raise ValueError('Expected unsplit official raw-pool provenance')
        snapshots.append({'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                          'episodes': len(meta['episodes']), 'snapshot': meta})
        records.extend(meta['episodes'])
    for path in args.exclude_provenance:
        raw = path.read_bytes()
        meta = json.loads(raw)
        if meta.get('split') not in ('train', 'validation'):
            raise ValueError('Explicit existing split required for overlap exclusion')
        keys = {entry['fingerprint'] for entry in meta['episodes']}
        exclusions.append({'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
                           'split': meta['split'], 'episodes': len(keys),
                           'matching_raw_fingerprints': sorted(keys & {e['fingerprint'] for e in records})})
        excluded.update(keys)
    selected, duplicates, overlap = select_unique_records(records, excluded)
    if not selected:
        raise ValueError('No finalized unique eligible source episodes')
    episodes = []
    shard_hashes = {}
    for entry in selected:
        source = Path(entry['source'])
        if (not source.resolve().is_relative_to(ROOT / 'runs') or source.name != 'demo.hdf5'
            or source.parent.parent.name != 'generated'):
            raise ValueError('Finalized in-project official shard HDF5 required; never read tmp')
        run = source.parents[2]
        verdict_path = run / 'generation-result.json'
        manifest_path = run / 'manifest.json'
        verdict = json.loads(verdict_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if (verdict.get('state') != 'passed' or verdict.get('generator_exit_status') != 0
            or Path(verdict['dataset']).resolve() != source.resolve()
            or manifest.get('mode') != 'official_upstream_entrypoint_unmodified_config'
            or manifest.get('task') != 'tidy_table' or manifest.get('randomization') != 'D0'):
            raise ValueError('Official finalized shard gate failed')
        shard_hashes[str(run)] = {
            'generation_result_sha256': hashlib.sha256(verdict_path.read_bytes()).hexdigest(),
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        with h5py.File(source, 'r') as file:
            env = json.loads(file['data'].attrs['env_args'])
            if env['env_name'] != 'r1_tidy_table_D0' or env['env_kwargs']['robots'][0].get('grasping_mode') != 'assisted':
                raise ValueError('Unexpected task/source grasp mode')
            group = file['data'][entry['source_demo']]
            if bool(group.attrs.get('partial', False)):
                raise ValueError('Partial episode')
            actions = native_actions(group['actions'][()])
            state = group['obs/prop_state'][()]
            qpos = group['obs/joint_qpos'][()]
            cameras = camera_keys(group['obs'])
            if (len(actions) != entry['frames'] or state.shape != (len(actions), 21)
                or len(qpos) != len(actions) or not np.isfinite(state).all()
                or any(group['obs'][key].shape[:3] != (len(actions), 256, 256) for key in cameras.values())):
                raise ValueError('Frame/proprio/joint/camera header alignment mismatch')
        fingerprint = hashlib.sha256(actions.tobytes() + state[0].tobytes()).hexdigest()
        if fingerprint != entry['fingerprint']:
            raise ValueError('Finalized source actions/initial proprio changed')
        mask, windows = frozen_segment_mask(qpos)
        transitions = np.zeros(len(actions), bool)
        transitions[1:] = np.any(np.abs(np.diff(actions[:, [13, 20]], axis=0)) > 1e-3, axis=1)
        episodes.append({**entry, 'qpos_sha256': hashlib.sha256(qpos.tobytes()).hexdigest(),
                         'joint_width': qpos.shape[1], 'candidate_removed_frames': int(mask.sum()),
                         'endpoint_quiet_windows': int(windows.sum()),
                         'candidate_removed_gripper_transitions': int((mask & transitions).sum())})
    report = {
        'mode': 'immutable_finalized_official_raw_pool_CPU_audit',
        'created_at': datetime.now(timezone.utc).isoformat(), 'gpu_initialized': False,
        'dataset_modified': False, 'split_created': False, 'normalization_changed': False,
        'cleaning_applied': False, 'policy_task_success': False,
        'source_grasping_mode': 'assisted', 'author_joint_subset_verified': False,
        'source_field': 'obs/joint_qpos', 'lag': 5, 'epsilon': 1e-3,
        'predicate': 'strict per-coordinate endpoint abs(q[i+5]-q[i])<1e-3; inclusive[i,i+5] union',
        'candidate_frames_from_verified_qpos_field_only': True,
        'raw_episode_records': len(records), 'duplicates_within_snapshot': duplicates,
        'unique_episode_overlap_excluded': overlap, 'eligible_unique_episodes': len(episodes),
        'eligible_raw_frames': sum(e['frames'] for e in episodes),
        'candidate_removed_frames': sum(e['candidate_removed_frames'] for e in episodes),
        'candidate_removed_gripper_transitions': sum(e['candidate_removed_gripper_transitions'] for e in episodes),
        'shards': shard_hashes, 'old_split_overlap_checks': exclusions,
        'provenance_snapshots': snapshots, 'episodes': episodes,
        'fingerprint_scope': 'native FP32 action bytes + original initial proprio bytes; not semantic diversity or unseen-scene proof',
        'rgb_scope': 'Camera headers checked; pixels NOT re-decoded here. Generator controller sampled RGB when validating finalized shard.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.mkdir(parents=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('mode', 'eligible_unique_episodes', 'eligible_raw_frames',
        'candidate_removed_frames', 'duplicates_within_snapshot', 'unique_episode_overlap_excluded',
        'author_joint_subset_verified', 'dataset_modified', 'split_created')}, indent=2))


if __name__ == '__main__':
    main()
