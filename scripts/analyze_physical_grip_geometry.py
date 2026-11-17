#!/usr/bin/env python3
"""SERVER CPU: expert replay vs failed B cup-center offsets in EEF frames.

Post-hoc geometry only, not force evidence, unique grasp pose or a controller.
No source HDF5/model/simulator calls; do not mix these metrics into VLA inputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def cup_in_eef(cup,pose):
    import numpy as np
    cup=np.asarray(cup);pose=np.asarray(pose)
    if cup.ndim!=2 or cup.shape[1]!=3 or pose.shape!=(len(cup),7):raise ValueError('Aligned XYZ/XYZ+XYZW samples required')
    if not np.isfinite(cup).all() or not np.isfinite(pose).all():raise ValueError('Unknown geometry not zero-filled')
    q=pose[:,3:];norm=np.linalg.norm(q,axis=1)
    if np.any(np.abs(norm-1)>.001):raise ValueError('Recorded XYZW quaternion not unit length')
    q=q/norm[:,None];v=cup-pose[:,:3];xyz=-q[:,:3];w=q[:,3:4];t=2*np.cross(xyz,v)
    return v+w*t+np.cross(xyz,t)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-replay',type=Path,required=True);p.add_argument('--B-campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.output.exists() or not a.output.resolve().is_relative_to(Path('/mnt/yuanyue/double-arm/runs')):raise ValueError('New remote CPU evidence required')
    import numpy as np
    report={'mode':'posthoc_expert_vs_B_physical_grip_geometry','model_or_simulator_control_calls':0,'policy_task_success':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'episodes':[],
        'caveat':'One expert replay and one learned episode; each centered on OWN first30 backend-grasp samples, not time/pose-paired. Cup centroid in EEF frame is not fingertip clearance, grip force, unique valid placement or causal proof.'}
    entries=[('expert_source_replay_NOT_PI0',a.source_replay/'replay/trace.npz'),('failed_B_live_policy',a.B_campaign/'train-physical-diagnostic/physical-rollout/demo_0-trace.npz')]
    for label,path in entries:
        with np.load(path) as f:
            cup=f['post_step_cup_positions'];pose=f['post_step_eef_world_poses'][:,0];grasp=f['post_step_cup_grasp_flags'];local=cup_in_eef(cup,pose)
            indices=np.flatnonzero(grasp)
            if not len(indices):raise ValueError('No recorded backend-grasp sample to center')
            first=int(indices[0]);initial=local[first:first+30];anchor=np.median(initial,axis=0);drift=np.linalg.norm(local-anchor,axis=1)
            report['episodes'].append({'label':label,'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'first_backend_grasp_step':first,'initial30_local_centroid_median_m':anchor.tolist(),
                'backend_grasp_sample_count':int(grasp.sum()),'own_anchor_centroid_drift_during_backend_grasp_p50_p95_max_m':np.quantile(drift[grasp],[.5,.95,1]).tolist(),
                'first_grasp_EEF_origin_to_cup_centroid_m':float(np.linalg.norm(cup[first]-pose[first,:3])),
                'scope':'Centroid translation only; cup orientation/contact normals/force/torque unavailable'})
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
