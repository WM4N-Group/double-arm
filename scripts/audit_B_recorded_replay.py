#!/usr/bin/env python3
"""CPU compare completed live-B, exact tape replay and source pre-close poses.

No model or simulator calls. Expert pre-close pose is a descriptive reference,
NOT a unique reachable/graspable pose or causal base-only failure proof.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def yaw_xyzw(pose):
    import math
    x,y,z,w=pose[3:];return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--B-campaign',type=Path,required=True);p.add_argument('--recorded-replay',type=Path,required=True);p.add_argument('--source-replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists() or not a.output.resolve().is_relative_to(Path('/mnt/yuanyue/double-arm/runs')):raise ValueError('Fresh remote CPU evidence required')
    import numpy as np
    from analyze_pi0_local_grasp import close_runs
    orig=a.B_campaign/'train-physical-diagnostic/physical-rollout/demo_0-trace.npz';tape=a.recorded_replay/'replay/trace.npz';source=a.source_replay/'replay/trace.npz'
    with np.load(orig) as b,np.load(tape) as t,np.load(source) as s:
        same={}
        for field in ('observations','actions','post_step_cup_positions','post_step_eef_world_poses','post_step_cup_contact_finger_counts','post_step_cup_grasp_flags'):
            x=b[field];y=t[field]
            if x.shape!=y.shape:raise ValueError('Recorded/live trace alignment mismatch')
            same[field]={'array_equal_including_nan':bool(np.array_equal(x,y,equal_nan=True)),'max_abs_difference':float(np.nanmax(np.abs(x.astype(float)-y.astype(float))))}
        rows=[]
        for label,f,root_key in (('expert_source_NOT_PI0',s,'root_poses'),('B_original_live_policy',b,'pre_step_root_poses')):
            threshold=.5;closed=f['actions'][:,13]<threshold;first=int(np.flatnonzero(closed)[0]);i=first-1
            if i<0:raise ValueError('Pre-close snapshot unavailable')
            pose=f[root_key][i];cup=f['post_step_cup_positions'][i];eef=f['post_step_eef_world_poses'][i,0,:3]
            rows.append({'label':label,'first_effective_close_step':first,'pre_close_step':i,
                'root_world_xyz_xyzw':pose.tolist(),'left_EEF_world_xyz':eef.tolist(),'cup_world_xyz':cup.tolist(),
                'left_EEF_origin_minus_cup_xyz_m':(eef-cup).tolist(),
                'scope':'Last PRE-CLOSE control index, recorded EEF/cup is POST-STEP for that index; not identical elapsed time or a uniquely valid target'})
        one,two=rows[0]['root_world_xyz_xyzw'],rows[1]['root_world_xyz_xyzw'];dyaw=(yaw_xyzw(two)-yaw_xyzw(one)+np.pi)%(2*np.pi)-np.pi
        result={'mode':'recorded_B_physical_replay_equivalence_and_preclose_pose_audit','model_or_simulator_control_calls':0,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'trace_sha256':{'live_B':hashlib.sha256(orig.read_bytes()).hexdigest(),'recorded_B_replay':hashlib.sha256(tape.read_bytes()).hexdigest(),'source':hashlib.sha256(source.read_bytes()).hexdigest()},
            'same_action_replay_checks':same,'all_six_arrays_exact_equal':all(v['array_equal_including_nan'] for v in same.values()),
            'recorded_B_contact_intervals':close_runs(t['post_step_cup_contact_finger_counts'][:,0]>0),
            'recorded_B_grasp_intervals':close_runs(t['post_step_cup_grasp_flags']),
            'no_assistance_samples_shape':list(t['no_assistance_observations'].shape),
            'no_assistance_all_recorded_zero':bool((t['no_assistance_observations']==0).all()),
            'preclose_references':rows,'B_minus_source_root_xy_m':[float(two[0]-one[0]),float(two[1]-one[1])],
            'B_minus_source_root_xy_distance_m':float(np.linalg.norm(np.asarray(two[:2])-one[:2])),
            'B_minus_source_root_yaw_rad':float(dyaw),'B_minus_source_root_yaw_deg':float(np.degrees(dyaw)),
            'caveat':'Only these two completed deterministic executions compared. No universal numerical reproducibility claim. Source preclose root/joints/EEF differ jointly; cannot assign unique causation to base or pick a unique correct grasp pose.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
