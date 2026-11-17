#!/usr/bin/env python3
"""CPU-only retained-cup loss timeline from completed B traces, not control.

No model/source-dataset/simulator reads. Grasp/contact are backend evidence, not
force calibration. Keep physical and assisted episodes separately labelled.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from analyze_pi0_local_grasp import close_runs


def first_persistent_contact_loss(contact_flags,after,min_steps=10):
    for start,end in close_runs([not bool(x) for x in contact_flags]):
        candidate=max(start,after)
        if end-candidate+1>=min_steps:return candidate
    return None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/mnt/yuanyue/double-arm/runs')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists() or not a.output.resolve().is_relative_to(root):raise ValueError('New remote CPU-only report')
    import numpy as np
    from motor_contract import choose_close_threshold
    report={'mode':'completed_B_assisted_vs_physical_retention_forensics','model_or_simulator_control_calls':0,
        'policy_task_success_claim':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'episodes':[],
        'caveat':'One TRAIN episode per mode; same weights, not guaranteed paired random outputs/physics. Contact counts are not calibrated forces; EEF-origin distance is not fingertip clearance. No unique-cause proof.'}
    for condition in ('train-author-diagnostic','train-physical-diagnostic'):
        e=a.campaign/condition;r=json.loads((e/'physical-rollout/result.json').read_text());episode=r['episodes'][0]
        if episode['aborted'] or not episode['steps']:raise ValueError('Completed unaborted episode required')
        trace=e/'physical-rollout/demo_0-trace.npz'
        with np.load(trace) as f:
            n=episode['steps'];cups=f['post_step_cup_positions'];initial=f['initial_cup_position'];contact=f['post_step_cup_contact_finger_counts'][:,0]
            grasp=f['post_step_cup_grasp_flags'];actions=f['actions'];eef=f['post_step_eef_world_poses'][:,0,:3];q=f['observations']
            if len(cups)!=n or not np.isfinite(cups).all():raise ValueError('Aligned finite actual cup samples required')
            threshold,evidence=choose_close_threshold(r.get('runtime_motor_contract'))
            if threshold is None:raise ValueError('Actual runtime gripper threshold unknown')
            close=actions[:,13]<threshold;lift=cups[:,2]-initial[2];grasp_indices=np.flatnonzero(grasp)
            start=int(grasp_indices[0]) if len(grasp_indices) else None
            # Missing contact samples cannot be equated with zero contact.
            lost=first_persistent_contact_loss(contact>0,start,10) if start is not None and (contact>=0).all() else None
            row={'evaluation':str(e),'grasping_mode':r['grasping_mode'],'assisted_constraint_seen':r['assisted_constraint_seen'],
                 'task_success':r['task_success'],'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
                 'steps':n,'backend_grasp_true_intervals':close_runs(grasp),'known_contact_positive_intervals':close_runs(contact>0),
                 'unknown_contact_samples':int((contact<0).sum()),'effective_close_intervals':close_runs(close),'threshold_evidence':evidence,
                 'cup_max_lift_m':float(lift.max()),'cup_max_lift_step':int(lift.argmax()),
                 'final_cup_position':cups[-1].tolist(),'initial_cup_position':initial.tolist(),
                 'first_persistent_contact_loss_after_first_grasp_step':lost,
                 'loss_definition':'first10-or-more consecutive NO left finger contact steps after first backend grasp; unknowns block this inference',
                 'release_or_recovery_scope':'Repeated later gripper transitions alone are not proof of goal-directed regrasp attempts'}
            if lost is not None:
                last=min(n-1,lost+15);valid=np.isfinite(eef[lost]).all()
                row['loss_event']={'step':lost,'video_seconds':lost/30,'commanded_close':bool(close[lost]),
                    'next30_commands_all_close':bool(close[lost:min(lost+30,n)].all()),
                    'lift_at_loss_m':float(lift[lost]),'cup_z_at_loss_m':float(cups[lost,2]),
                    'cup_z_half_second_later_m':float(cups[last,2]),'half_second_later_step':last,
                    'eef_origin_to_cup_centroid_m':float(np.linalg.norm(eef[lost]-cups[lost])) if valid else None,
                    'recorded_width_next_observation_m':float(q[min(lost+1,n-1),19]),
                    'finger_contact_positive_after_loss_steps':int((contact[lost:]>0).sum()),
                    'backend_grasp_true_after_loss_steps':int(grasp[lost:].sum()),
                    'nearby_left_targets_max_diff_rad':float(np.abs(np.diff(actions[max(0,lost-10):min(n,lost+11),7:13],axis=0)).max()),
                    'nearby_samples':[{'step':i,'cup_z_m':float(cups[i,2]),'left_finger_count':int(contact[i]),
                        'grasp_flag':bool(grasp[i]),'jaw_raw':float(actions[i,13])} for i in range(max(0,lost-6),min(n,lost+9))]}
            report['episodes'].append(row)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
