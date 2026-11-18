#!/usr/bin/env python3
"""CPU-only per-lag label audit for the frozen 225-episode π₀ pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path

RUNS = Path('/mnt/yuanyue/double-arm/runs')
SPLIT_SHA = '1a0517801299e46107a2f58ece715be3f8a0e46d1bb94be4475c6c4571beb2be'
LAGS = (0, 9, 19, 49)
HORIZON = 50


def episode_indices(n, lag):
    import numpy as np
    if not 0 < n or not 0 <= lag < HORIZON:
        raise ValueError('Positive episode length and in-horizon lag required')
    target = np.arange(n) + lag
    return np.minimum(target, n - 1), target >= n


def boundary_anchors(n, lag):
    return sorted({0, max(n-lag-1,0), min(max(n-lag,0),n-1), n-1})


def audit(train):
    import h5py
    import numpy as np
    from momagen_pi0_schema import ACTION_NAMES, native_actions
    from pi0_relative_joint_diagnostic import JOINT_ACTION_INDICES, JOINT_STATE_INDICES, joint_encoding
    aa = list(JOINT_ACTION_INDICES); ss = list(JOINT_STATE_INDICES)
    copied = [i for i in range(21) if i not in aa]
    buffers = {lag: [] for lag in LAGS}
    pad_counts = {lag: 0 for lag in LAGS}
    left_close_pad = {lag: 0 for lag in LAGS}
    errors = {lag: 0.0 for lag in LAGS}
    total_frames = 0
    for entry in train:
        if entry['split'] != 'train' or not Path(entry['source']).resolve().is_relative_to(RUNS):
            raise ValueError('Only finalized TRAIN raw HDF5 paths may be read')
        with h5py.File(entry['source'], 'r') as f:
            grp = f['data'][entry['source_demo']]
            actions = native_actions(grp['actions'][()]); state = grp['obs/prop_state'][()]
        n = len(actions)
        if (n != entry['frames'] or state.dtype != np.float32 or state.shape != (n,21) or not np.isfinite(state).all()
                or hashlib.sha256(actions.tobytes()+state[0].tobytes()).hexdigest() != entry['fingerprint']):
            raise ValueError('Original TRAIN actions/frame/first proprio identity changed')
        total_frames += n
        for lag in LAGS:
            index,padded = episode_indices(n,lag)
            target = actions[index]
            rel = target[:,aa] - state[:,ss]
            restored = rel + state[:,ss]
            errors[lag] = max(errors[lag],float(np.max(np.abs(target[:,aa]-restored))))
            for anchor in boundary_anchors(n,lag):
                enc = joint_encoding(target[anchor:anchor+1],state[anchor])[0]
                if not np.array_equal(enc[aa],rel[anchor]) or not np.array_equal(enc[copied],target[anchor,copied]):
                    raise ValueError('Vectorized reference disagrees with existing relative contract')
            if not np.isfinite(rel).all():
                raise ValueError('Nonfinite relative labels')
            buffers[lag].append(rel)
            pad_counts[lag] += int(padded.sum())
            left_close_pad[lag] += int(np.count_nonzero(padded & (target[:,13] < 0)))
    results = {}
    for lag, arrays in buffers.items():
        x = np.concatenate(arrays)
        low,high = np.quantile(x,[.01,.99],axis=0)
        results[str(lag)] = {'label_values_per_joint':len(x),'repeated_episode_tail_rows':pad_counts[lag],
                             'repeat_tail_raw_left_jaw_negative_rows':left_close_pad[lag],
                             'relative_q01_diagnostic':low.tolist(),'relative_q99_diagnostic':high.tolist(),
                             'relative_joint_inverse_max_abs_rad':errors[lag]}
    all_horizon_padded = sum(sum(min(lag,e['frames']) for lag in range(HORIZON)) for e in train)
    return {'train_episodes':len(train),'train_anchor_frames':total_frames,'validation_rows_read':0,
            'lags':results,'native_joint_axes':[ACTION_NAMES[i] for i in aa],
            'unchanged_base_and_jaw_action_axes':copied, 'horizon':HORIZON,
            'all_horizon_window_targets':HORIZON*total_frames,
            'all_horizon_episode_tail_repeats':all_horizon_padded,
            'scope':'CPU descriptive lag comparison. Original action targets repeat last step within each episode. Joint labels subtract the CURRENT anchor observation; no training sampler, official replay, physical success or author action-encoding claim.',
            'runtime_normalization_written':False,'new_training_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists()
            or not all(x.resolve().is_relative_to(RUNS) for x in (a.split,a.output))):
        raise ValueError('Explicit CPU and new project report required')
    raw=a.split.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SPLIT_SHA:
        raise ValueError('Frozen split changed')
    plan=json.loads(raw)
    train=[entry for entry in plan['episodes'] if entry['split']=='train']
    if len(train)!=225 or len({e['fingerprint'] for e in train})!=225:
        raise ValueError('Expected fixed 225 unique training episodes')
    result=audit(train)
    if result['train_anchor_frames']!=plan['train_frames']:
        raise ValueError('Unexpected training frame count')
    result['split_sha256']=SPLIT_SHA
    blob=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode()
    with a.output.open('xb') as f:f.write(blob)
    print(a.output, hashlib.sha256(blob).hexdigest(),
          {k:result[k] for k in ('train_anchor_frames','all_horizon_window_targets','all_horizon_episode_tail_repeats')})
    print('inverse max', {k:v['relative_joint_inverse_max_abs_rad'] for k,v in result['lags'].items()})


if __name__=='__main__':
    main()
