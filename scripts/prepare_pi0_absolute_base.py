#!/usr/bin/env python3
"""SERVER ONLY: derive absolute-base labels; reuse all existing real RGB data.

Only current robot base odometry and native motor commands form targets. No task
object poses, future observations or teacher actions enter inference inputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from pi0_base_targets import native_to_absolute_base,absolute_base_to_native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,default=Path('/mnt/yuanyue/double-arm/data/pi0/lerobot/momagen'))
    parser.add_argument('--output',type=Path,default=Path('/mnt/yuanyue/double-arm/data/pi0/absolute-base-v2'))
    args=parser.parse_args()
    if not str(args.output.resolve()).startswith('/mnt/yuanyue/double-arm/data/'):raise ValueError('Remote output required')
    args.output.mkdir(exist_ok=False)
    report={'label_encoding':'world XYZ-plane XY + wrapped world yaw [-pi, pi) base targets, remaining native arm/trunk/gripper commands',
        'model_inputs_unchanged':'Real head + both wrist RGB, original 21D proprioception, instruction; NO odometry added to VLA inputs',
        'odometry_scope':'Current robot root pose used only for offline label construction and low-level target tracking',
        'future_observations_used':False,'object_poses_used':False,'source_grasping':'author assisted attachment',
        'source_rgb_reused_without_reencoding':True,'splits':{}}
    for split,repo in (('train','r1_tidy_table_train_v1'),('validation','r1_tidy_table_holdout_v1')):
        provenance_path=args.dataset_root/repo/'momagen-provenance.json';provenance=json.loads(provenance_path.read_text())
        targets=[];ends=[];episodes=[];offset=0;max_error=0
        for entry in provenance['episodes']:
            with h5py.File(entry['source']) as file:
                group=file['data'][entry['source_demo']];actions=group['actions'][()];poses=group['obs/base_link_pose'][()]
            converted=native_to_absolute_base(actions,poses)
            recovered=absolute_base_to_native(converted,poses)
            # Wrapped heading equivalence is expected for delta yaw; real source
            # local commands are much smaller than pi. Check ALL 21 coordinates.
            error=float(np.max(np.abs(recovered-actions)));max_error=max(max_error,error)
            if error>2e-5:raise ValueError(f'Native/absolute roundtrip mismatch: {error}')
            targets.append(converted);ends.extend([offset+len(actions)-1]*len(actions))
            episodes.append({'source_demo':entry['source_demo'],'source':entry['source'],'start':offset,'end_exclusive':offset+len(actions)})
            offset+=len(actions)
        values=np.concatenate(targets);boundaries=np.asarray(ends,np.int64)
        if len(values)!=provenance['total_frames']:raise ValueError('RGB/label frame counts differ')
        np.save(args.output/(split+'-actions.npy'),values);np.save(args.output/(split+'-episode-ends.npy'),boundaries)
        report['splits'][split]={'frames':len(values),'episodes':episodes,'source_provenance':str(provenance_path),
            'source_provenance_sha256':hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            'label_sha256':hashlib.sha256(values.tobytes()).hexdigest(),'max_native_roundtrip_error':max_error,
            'base_target_min':values[:,:3].min(0).tolist(),'base_target_max':values[:,:3].max(0).tolist()}
    (args.output/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
