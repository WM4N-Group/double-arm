#!/usr/bin/env python3
"""SERVER CPU: forensic rollout/reference statistics, NOT teacher control.

Reference timing/poses are examples, not unique valid solutions. Statistics do
not prove a single cause. Missing contact/EEF telemetry stays unknown.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import h5py
import numpy as np
from evaluation_split_contract import verify_evaluation_episode
from momagen_pi0_schema import camera_keys,rgb_image


def percentile(values,q):return float(np.quantile(values,q)) if np.size(values) else None


def action_metrics(actions,observations,proposals,source_actions,source_states,execute_steps):
    a=np.asarray(actions);s=np.asarray(observations);n=len(a)
    if a.shape!=(n,21) or s.shape!=(n,21) or n<1:raise ValueError('Invalid aligned R1 trace')
    close=a[:,13]<0;starts=np.flatnonzero(close&~np.r_[False,close[:-1]]);ends=np.flatnonzero(close&~np.r_[close[1:],False])
    jump=np.max(np.abs(np.diff(a[:,7:13],axis=0)),axis=1);boundary=(np.arange(1,n)%execute_steps)==0
    lag=a[:-1,7:13]-s[1:,7:13];src_lag=source_actions[:-1,7:13]-source_states[1:,7:13]
    report={'steps':n,'negative_jaw_command_steps':int(close.sum()),'close_command_runs':[[int(x),int(y)] for x,y in zip(starts,ends)],
        'first_close_step':int(starts[0]) if len(starts) else None,'first_close_video_seconds':float(starts[0]/30) if len(starts) else None,
        'left_arm_jump_rad':{'boundary_median':percentile(jump[boundary],.5),'boundary_p95':percentile(jump[boundary],.95),
            'inside_chunk_median':percentile(jump[~boundary],.5),'max':percentile(jump,1),
            'source_all_steps_p95':percentile(np.max(np.abs(np.diff(source_actions[:,7:13],axis=0)),axis=1),.95)},
        'one_step_tracking_rmse_rad':float(np.sqrt(np.mean(lag**2))) if len(lag) else None,
        'reference_tracking_rmse_rad':float(np.sqrt(np.mean(src_lag**2))) if len(src_lag) else None,
        'tracking_scope':'left-arm commanded absolute q at step i vs actual q at i+1, not world EEF error; trajectories not speed-paired'}
    if len(starts):
        i=int(starts[0]);report.update(jaw_state_at_close=float(s[i,19]),jaw_state_one_second_later=float(s[min(i+30,n-1),19]),
            jaw_state_at_end=float(s[-1,19]),jaw_scope='sum of recorded finger joint positions, not calibrated fingertip gap')
    if proposals is not None:report['raw_native_base_output_clipped_steps']=int(np.any(np.abs(np.asarray(proposals)[:,:3]-a[:,:3])>1e-8,axis=1).sum())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--source-provenance',type=Path,required=True);p.add_argument('--source-dataset',type=Path,required=True)
    p.add_argument('--demo',default='demo_0');p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only analysis required')
    root=Path('/mnt/yuanyue/double-arm/runs')
    if not args.output.resolve().is_relative_to(root) or args.output.exists():raise ValueError('New remote report required')
    rollout=args.evaluation/'physical-rollout';result=json.loads((rollout/'result.json').read_text());episode=result['episodes'][0]
    provenance=json.loads(args.source_provenance.read_text());verify_evaluation_episode(provenance,args.source_dataset,args.demo,episode['split'])
    with h5py.File(args.source_dataset) as file:
        g=file['data/'+args.demo];source_actions=g['actions'][()];source_states=g['obs/prop_state'][()]
        if (rollout/'initial-observation.npz').is_file():
            with np.load(rollout/'initial-observation.npz') as snap:
                camera_mae={key:float(np.abs(snap[key].astype(float)-rgb_image(g['obs'][name][0])).mean()) for key,name in camera_keys(g['obs']).items()}
        else:camera_mae=None
    trace_path=rollout/(episode['demo']+'-trace.npz')
    with np.load(trace_path,allow_pickle=False) as trace:
        raw=trace['model_proposed_targets'] if result.get('action_representation')=='native' and 'model_proposed_targets' in trace else None
        report=action_metrics(trace['actions'],trace['observations'],raw,source_actions,source_states,result['execute_steps'])
        report['eef_contact_telemetry_available']='post_step_eef_world_poses' in trace and 'post_step_cup_contact_finger_counts' in trace
        if report['eef_contact_telemetry_available']:
            poses=trace['post_step_eef_world_poses'];cups=trace['post_step_cup_positions'];contact=trace['post_step_cup_contact_finger_counts'][:,0]
            valid=np.isfinite(poses[:,0,:3]).all(axis=1);distance=np.linalg.norm(poses[:,0,:3]-cups,axis=1)
            report.update(min_left_eef_to_cup_centroid_m=float(distance[valid].min()) if valid.any() else None,
                known_left_contact_frames=int((contact>=0).sum()),left_cup_finger_contact_frames=int((contact>0).sum()),
                two_or_more_finger_link_contact_frames=int((contact>=2).sum()),unknown_left_contact_frames=int((contact<0).sum()),
                contact_scope='geometric link contact, NOT calibrated forces or proof of a stable physical grasp; EEF origin distance is not fingertip clearance')
    report.update(mode='offline_pi0_rollout_forensics',evaluation=str(args.evaluation),evaluation_split=episode['split'],
        grasping_mode=result['grasping_mode'],task_success=result['task_success'],actual_initial_rgb_mae_0_255=camera_mae,
        teacher_reference_used_only_for_scoring=True,model_or_simulator_control_calls=0,active_training_untouched=True,
        trace_sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_caveat='Successful reference is not a unique valid trajectory; non-paired temporal/spatial distances alone do not establish failure cause',
        absent_telemetry_caveat='If EEF/contact fields are absent, no exact fingertip/cup distance or contact-force cause is claimed')
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
