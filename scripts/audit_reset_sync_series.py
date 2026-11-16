#!/usr/bin/env python3
"""SERVER ONLY: compare real reset snapshots to source RGB, no policy/control."""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from PIL import Image,ImageDraw
from momagen_pi0_schema import camera_keys,rgb_image
from pi0_base_targets import root_rotation_xy_and_yaw
from reset_sync_contract import CAMERAS,LIMITS,select_reset_sync_candidate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--series',type=Path,required=True)
    p.add_argument('--source-dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if not args.output.resolve().is_relative_to(Path('/mnt/yuanyue/double-arm/runs')):raise ValueError('Remote output required')
    if args.output.exists():raise FileExistsError(args.output)
    raw=json.loads((args.series/'result.json').read_text())
    if raw.get('mode')!='reset_observation_series_audit_only' or not raw.get('snapshot_series_completed'):raise ValueError('Incomplete snapshot series')
    with h5py.File(args.source_dataset) as file:
        obs=file['data/demo_0/obs'];keys=camera_keys(obs)
        source={key:rgb_image(obs[value][0]) for key,value in keys.items()}
        source['state']=obs['prop_state'][0];pose=obs['base_link_pose'][0]
    _,yaw=root_rotation_xy_and_yaw(pose)
    shots=raw['reset_snapshots'];sheet=Image.new('RGB',(256*3,280*(len(shots)+1)),'#161616');draw=ImageDraw.Draw(sheet)
    for i,key in enumerate(CAMERAS):
        sheet.paste(Image.fromarray(source[key]),(i*256,24));draw.text((i*256+4,4),'SOURCE '+key,fill='yellow')
    report={'mode':'reset_sync_image_audit_only','task_success_claim':False,'grasping_mode':raw['grasping_mode'],
        'initial_state_load_count':raw['initial_state_load_count'],'teacher_action_reads':raw['teacher_action_reads'],
        'simulator_control_calls':raw['simulator_control_calls'],'inference_calls':raw['inference_calls'],
        'assisted_constraint_seen':raw['assisted_constraint_seen'],'object_pose_write_count':raw['object_pose_write_count'],
        'series_completed':raw['snapshot_series_completed'],'limits':LIMITS,'snapshots':[],
        'gate_scope':'Conservative source/live reset image+robot alignment integration check, not physical task success or hardware calibration'}
    for row,shot in enumerate(shots,1):
        path=Path(shot['path'])
        if not path.resolve().is_relative_to(args.series.resolve()):raise ValueError('Snapshot escaped audit directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=shot['sha256']:raise ValueError('Snapshot hash mismatch')
        with np.load(path,allow_pickle=False) as loaded:live={key:loaded[key].copy() for key in (*CAMERAS,'state')}
        root=np.asarray(shot['robot_root_pose']);q=root[3:];norm=float(np.linalg.norm(q))
        if not .99<=norm<=1.01:raise ValueError('Invalid audited quaternion')
        # Audit an unaligned pose without weakening the production converter's
        # strict planarity check. The gate below rejects its determinant.
        x,y,z,w=q/norm
        rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w)],[2*(x*y+z*w),1-2*(x*x+z*z)]])
        live_yaw=np.arctan2(rotation[1,0],rotation[0,0])
        entry={'physics_steps':shot['physics_steps'],'path':str(path),'sha256':shot['sha256'],
            'rgb_mae_0_255':{},'root_translation_m':float(np.linalg.norm(root[:3]-pose[:3])),
            'root_yaw_abs_difference_rad':float(abs((live_yaw-yaw+np.pi)%(2*np.pi)-np.pi)),
            'joint_max_abs_difference_rad':float(np.abs(live['state'][3:19]-source['state'][3:19]).max()),
            'state_max_abs_difference':float(np.abs(live['state']-source['state']).max()),
            'xy_rotation_determinant':float(np.linalg.det(rotation))}
        for i,key in enumerate(CAMERAS):
            image=rgb_image(live[key]);entry['rgb_mae_0_255'][key]=float(np.abs(image.astype(float)-source[key]).mean())
            sheet.paste(Image.fromarray(image),(i*256,row*280+24));draw.text((i*256+4,row*280+4),f"PHYSICS {shot['physics_steps']} {key}",fill='yellow')
        report['snapshots'].append(entry)
    report['selected_reset_settle_steps']=select_reset_sync_candidate(report)
    args.output.mkdir();sheet.save(args.output/'source-live-reset-series.jpg',quality=91)
    (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
