#!/usr/bin/env python3
"""Freeze 225/25 official raw episodes for a non-paper-scale π₀ pilot.

The physical expert diagnostic panel is excluded. No training or conversion;
the optional readout is descriptive native quantiles from TRAIN only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from audit_official_pool_snapshot import select_unique_records
from plan_expert_physical_pilot import RUNS, PANEL_NAMESPACE

AUDIT_SHA = 'a92f91aac9ab41e1d37c6ab68e65b367ab56ebac9fbf1fea06c83684a49243a2'
PANEL_SHA = 'ea90461c1c9d7d6193eda55957e6837779146dde9df7698434cada4e5c6ed96a'
NAMESPACE = 'r1_tidy_table_D0_pi0_225train_25val_native_raw_candidate_v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def select(audit, panel, *, salt, n_train=225, n_val=25):
    if (audit.get('mode') != 'immutable_finalized_official_raw_pool_CPU_audit'
            or audit.get('source_grasping_mode') != 'assisted' or audit.get('cleaning_applied') is not False
            or len(audit['episodes']) != audit['eligible_unique_episodes']
            or panel.get('namespace') != PANEL_NAMESPACE or panel.get('state') != 'FROZEN_CPU_PANEL_NOT_LAUNCHED'):
        raise ValueError('Need official raw audit and frozen expert diagnostic panel')
    excluded = {e['fingerprint'] for e in panel['episodes']}
    old = audit['old_split_overlap_checks']
    if ({e['split'] for e in old} != {'train','validation'} or len(excluded) != len(panel['episodes'])
            or not excluded <= {e['fingerprint'] for e in audit['episodes']}):
        raise ValueError('Cannot prove panel and prior split exclusions')
    excluded.update(f for record in old for f in record['matching_raw_fingerprints'])
    eligible, duplicates, _ = select_unique_records(audit['episodes'], excluded)
    if duplicates:
        raise ValueError('Duplicate official fingerprint in immutable audit')
    groups = defaultdict(list)
    shard_seeds = defaultdict(set)
    for item in eligible:
        path = Path(item['source'])
        if (not path.is_relative_to(RUNS) or '..' in path.parts or path.name != 'demo.hdf5'
                or path.parent.parent.name != 'generated' or item['source_grasping_mode'] != 'assisted'
                or str(path.parents[2]) not in audit['shards'] or type(item['seed']) is not int):
            raise ValueError('Unexpected unverified episode identity')
        shard = str(path.parents[2]); groups[shard].append(item); shard_seeds[shard].add(item['seed'])
    if any(len(seeds) != 1 for seeds in shard_seeds.values()):
        raise ValueError('Expected one generator seed per finalized shard')

    def order(kind, value):
        return sha(f'{NAMESPACE}:{salt}:{kind}:{value}'.encode()), value

    # Preserve complete shards and generator seeds across the split boundary.
    # A one-shard validation split is a debugging check, not unseen-scene evidence.
    candidates = [s for s, xs in groups.items() if len(xs) == n_val]
    candidates.sort(key=lambda s: order('validation_shard',s))
    for held in candidates:
        val_seed,=shard_seeds[held]
        pool = [x for s, xs in groups.items() if val_seed not in shard_seeds[s] for x in xs]
        if len(pool) >= n_train:
            train = sorted(pool, key=lambda e: order('train_episode',e['fingerprint']))[:n_train]
            validation = sorted(groups[held],key=lambda e: e['fingerprint'])
            return ([{**e,'split':'train'} for e in train], [{**e,'split':'validation'} for e in validation], held)
    raise ValueError('Could not bind one complete 25-episode validation shard and 225 disjoint train episodes')


def native_train_quantiles(train):
    import h5py
    import numpy as np
    from momagen_pi0_schema import ACTION_NAMES, STATE_NAMES, native_actions
    states, actions = [], []
    for e in train:
        if e.get('split') != 'train':
            raise ValueError('Validation rows may not enter TRAIN quantiles')
        with h5py.File(e['source'], 'r') as f:
            g = f['data'][e['source_demo']]; a = native_actions(g['actions'][()]); s = g['obs/prop_state'][()]
        if (a.shape != (e['frames'],21) or s.shape != a.shape or s.dtype != np.float32 or not np.isfinite(s).all()
                or sha(a.tobytes()+s[0].tobytes()) != e['fingerprint']):
            raise ValueError('Source actions or initial proprio identity changed')
        states.append(s); actions.append(a)
    def describe(values,names):
        x = np.concatenate(values, axis=0)
        low, high = np.quantile(x, [.01,.99], axis=0)
        return {'axes':list(names),'q01':low.tolist(),'q99':high.tolist(),
                'exactly_zero_range_axes':[name for name,q0,q1 in zip(names,low,high) if q0==q1],
                'near_zero_nonzero_range_axes':[name for name,q0,q1 in zip(names,low,high) if 0 < q1-q0 < 1e-6]}
    return {'mode':'EXACT_EMPIRICAL_NATIVE_RAW_TRAIN_ONLY_QUANTILES_NOT_RUNTIME_NORM_ASSET',
            'train_frames':sum(len(a) for a in actions),'validation_rows_read':0,
            'action_space':'native delta-base / absolute body joints / native grippers; not assumed author encoding',
            'cleaning_applied':False,'runtime_transform_ready':False,'actions':describe(actions,ACTION_NAMES),
            'state':describe(states,STATE_NAMES)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--expert-panel',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists() or not all(
            x.resolve().is_relative_to(RUNS) for x in (a.audit,a.expert_panel,a.output)):
        raise ValueError('Server CPU and fresh run path required')
    audit_blob, panel_blob = a.audit.read_bytes(), a.expert_panel.read_bytes()
    if sha(audit_blob)!=AUDIT_SHA or sha(panel_blob)!=PANEL_SHA:
        raise ValueError('Use the predeclared audit and expert panel; do not reselect after physical outcomes')
    train,val,shard = select(json.loads(audit_blob),json.loads(panel_blob),salt=AUDIT_SHA)
    manifest = {'version':NAMESPACE,'scope':'Frozen non-paper-scale candidate; raw assisted sources only; no physical feasibility claim',
                'audit':str(a.audit.resolve()),'audit_sha256':AUDIT_SHA,'excluded_physical_panel_sha256':PANEL_SHA,
                'train_count':len(train),'validation_count':len(val),'validation_shard':shard,
                'train_frames':sum(e['frames'] for e in train),'validation_frames':sum(e['frames'] for e in val),
                'exact_fingerprint_overlap':0,'seed_and_shard_overlap':0,'cleaning_applied':False,
                'heldout_closed_loop_policy_evaluation':False,'formal_training_started':False,
                'episodes':train+val}
    stats = native_train_quantiles(train)
    if stats['train_frames'] != manifest['train_frames']:
        raise ValueError('Statistics and split row counts differ')
    a.output.mkdir(parents=True)
    m=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    (a.output/'split-candidate.json').write_bytes(m)
    stats['split_candidate_sha256']=sha(m)
    z=(json.dumps(stats,indent=2,allow_nan=False)+'\n').encode()
    (a.output/'native-raw-train-quantiles.json').write_bytes(z)
    print(a.output,len(train),len(val),manifest['train_frames'],manifest['validation_frames'],sha(m),sha(z))
    print('zero range',stats['actions']['exactly_zero_range_axes'],stats['state']['exactly_zero_range_axes'])


if __name__=='__main__':
    main()
