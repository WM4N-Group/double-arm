#!/usr/bin/env python3
"""SERVER CPU: dry-run paper endpoint cleaning on recorded joint_qpos only.

Does not alter HDF5, LeRobot data, labels, normalization or a running trainer.
Raw qpos is for offline auditing, never an extra VLA observation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from frozen_segment_audit import frozen_segment_mask


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path('/mnt/yuanyue/double-arm')
    if not args.output.resolve().is_relative_to(root/'runs') or args.output.exists():raise ValueError('New remote output required')
    args.output.mkdir()
    report={'mode':'offline_frozen_segment_dry_run','active_training_data_modified':False,
        'source_field':'obs/joint_qpos','field_author_mapping_verified':False,
        'paper_reference':'arXiv2510.18316v2 AppendixC.1','lag':5,'epsilon':1e-3,
        'predicate':'all(abs(q[i+5]-q[i])<1e-3); union of inclusive [i,i+5] windows',
        'not_adjacent_five_step_rule':True,'not_applied_to_prop_state_or_native_base_delta_actions':True,'splits':{}}
    for split,repo in (('train','r1_tidy_table_train_v1'),('validation','r1_tidy_table_holdout_v1')):
        provenance=root/'data/pi0/lerobot/momagen'/repo/'momagen-provenance.json';meta=json.loads(provenance.read_text())
        masks=[];episodes=[];offset=0
        for entry in meta['episodes']:
            with h5py.File(entry['source']) as file:
                g=file['data/'+entry['source_demo']];q=g['obs/joint_qpos'][()];actions=g['actions'][()]
            if len(q)!=len(actions) or len(q)!=entry['frames']:raise ValueError('Joint/action/provenance length mismatch')
            mask,windows=frozen_segment_mask(q)
            adjacent=np.max(np.abs(np.diff(q,axis=0)),axis=1)
            excursion=sum(bool(np.any(adjacent[i:i+5]>=1e-3)) for i in np.flatnonzero(windows))
            transitions=np.zeros(len(q),bool);transitions[1:]=np.any(np.abs(np.diff(actions[:,[13,20]],axis=0))>1e-3,axis=1)
            episodes.append({'source':entry['source'],'source_demo':entry['source_demo'],'joint_width':q.shape[1],
                'original_frames':len(q),'candidate_removed_frames':int(mask.sum()),'candidate_kept_frames':int((~mask).sum()),
                'endpoint_quiet_windows':int(windows.sum()),'quiet_endpoints_with_nonquiet_interior_windows':excursion,
                'gripper_command_transitions':int(transitions.sum()),'transitions_inside_candidate_removed_frames':int((transitions&mask).sum()),
                'qpos_sha256':hashlib.sha256(q.tobytes()).hexdigest(),'global_start':offset,'global_end_exclusive':offset+len(q)})
            masks.append(mask);offset+=len(q)
        removed=np.concatenate(masks);np.save(args.output/(split+'-candidate-remove-mask.npy'),removed)
        np.save(args.output/(split+'-candidate-keep-indices.npy'),np.flatnonzero(~removed))
        report['splits'][split]={'episodes':episodes,'original_frames':len(removed),
            'candidate_removed_frames':int(removed.sum()),'candidate_kept_frames':int((~removed).sum()),
            'candidate_removed_fraction':float(removed.mean()),'provenance_sha256':hashlib.sha256(provenance.read_bytes()).hexdigest()}
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
