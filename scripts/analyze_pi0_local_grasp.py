#!/usr/bin/env python3
"""SERVER CPU forensics for the explicitly privileged-init LOCAL grasp diagnostic.

No source HDF5, model or simulator calls. Source replay trace is ONLY a labelled
post-hoc reference, never control; temporal differences are not unique causes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def close_runs(flags):
    result=[];start=None
    for i,flag in enumerate(flags):
        if flag and start is None:start=i
        if not flag and start is not None:result.append([start,i-1]);start=None
    if start is not None:result.append([start,len(flags)-1])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/mnt/yuanyue/double-arm/runs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Explicit CPU-only analysis')
    if a.output.exists() or not a.output.resolve().is_relative_to(root):raise ValueError('Fresh project report')
    import numpy as np
    from motor_contract import choose_close_threshold,binary_close_flags
    c=a.campaign;r=json.loads((c/'local/result.json').read_text());rows=[];replay=None
    if r['stage'] not in ('completed_local_grasp_diagnostic_NOT_full_task_success','source_local_replay_gate_failed_NO_learned_case','local_initialization_or_control_aborted_NOT_full_task'):
        raise ValueError('Actual local transitions must be finalized before analysis')
    for ep in r['episodes']:
        path=c/'local'/(ep['mode']+'-trace.npz')
        if not path.exists():continue
        with np.load(path) as f:
            n=ep['steps'];actions=f['actions'];obs=f['observations'];proposed=f['model_proposed_targets']
            if actions.shape!=(n,21) or obs.shape!=(n,21) or not np.isfinite(actions).all() or not np.isfinite(obs).all():raise ValueError('Actual finite aligned transitions only')
            threshold,scope=choose_close_threshold(ep['runtime_motor_contract'])
            closed=np.asarray(binary_close_flags(actions[:,13],threshold)) if threshold is not None else None
            contact=f['post_step_cup_contact_finger_counts'][:,0];eef=f['post_step_eef_world_poses'][:,0,:3];cup=f['post_step_cup_positions']
            valid=np.isfinite(eef).all(1)&np.isfinite(cup).all(1);distance=np.linalg.norm(eef-cup,axis=1)
            clipped=np.abs(actions-proposed)>1e-6
            row={'mode':ep['mode'],'steps':n,'teacher_action_reads':ep['teacher_action_reads'],
                 'local_cup_grasp_report_seen':ep['local_cup_grasp_report_seen'],'assisted_constraint_seen':ep['assisted_constraint_seen'],
                 'max_cup_lift_m':ep['max_cup_lift_m'],'object_pose_write_count':ep['object_pose_write_count'],
                 'effective_binary_threshold':threshold,'threshold_evidence':scope,
                 'effective_close_steps':int(closed.sum()) if closed is not None else None,
                 'effective_close_contiguous_runs':close_runs(closed) if closed is not None else None,
                 'first_effective_close_step':int(np.flatnonzero(closed)[0]) if closed is not None and closed.any() else None,
                 'jaw_width_min_m':float(f['post_step_gripper_widths'][:,0].min()),
                 'known_left_contact_frames':int((contact>=0).sum()),'unknown_left_contact_frames':int((contact<0).sum()),
                 'left_contact_frames':int((contact>0).sum()),'two_finger_link_contact_frames':int((contact>=2).sum()),
                 'min_left_EEF_origin_to_cup_centroid_m':float(distance[valid].min()) if valid.any() else None,
                 'clipped_gt1e_minus6_steps_by_coordinate':clipped.sum(0).tolist(),
                 'first_action':actions[0].tolist(),'first10_left_action_minus_observed_q_RMS_rad':float(np.sqrt(np.mean((actions[:10,7:13]-obs[:10,7:13])**2))),
                 'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            chunks=f['predicted_chunks']
            row['predicted_chunk_first_close_index']=[int(np.flatnonzero(x[:,13]<threshold)[0]) if (x[:,13]<threshold).any() else None for x in chunks] if threshold is not None else None
            if ep['mode']=='source_action_replay':replay=actions.copy()
            elif replay is not None:
                count=min(10,n,len(replay));difference=actions[:count]-replay[:count]
                row['first10_vs_replay_left_target_RMSE_rad']=float(np.sqrt(np.mean(difference[:,7:13]**2)))
                row['first10_vs_replay_left_target_mean_error_rad']=difference[:,7:13].mean(0).tolist()
                row['comparison_scope']='Same initialization setup and first elapsed steps, NOT a unique valid trajectory or isolated causal factor'
            rows.append(row)
    result={'mode':'CPU_local_grasp_forensics_NOT_full_task','campaign':str(c),'initialization_source_frame':635,
        'task_success_claim':False,'physical_task_acceptance':False,'model_or_simulator_control_calls':0,
        'source_replay_used_only_as_posthoc_reference':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'episodes':rows,'caveat':'Contact is geometric link evidence, not calibrated force. EEF-origin distance is not fingertip clearance. Close STEPS form contiguous intervals, not separate attempts. No full task/heldout success.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
