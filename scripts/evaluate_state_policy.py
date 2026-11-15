#!/usr/bin/env python3
"""SERVER ONLY: closed-loop neural policy rollout. No teacher or state replay.

The recorded simulator state is restored ONCE per episode to define a reset;
each subsequent state is produced by physics following the neural actions.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import traceback
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from policy_action_codecs import decode_world_base_goals
from state_policy_features import add_unique_task_poses, vectorize, as_numpy
from train_state_policy import ChunkPolicy, JOINT, GRIP, joint_anchor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--demo',action='append',default=[])
    p.add_argument('--max-steps',type=int,default=4000)
    p.add_argument('--stagnation-window',type=int,default=600)
    p.add_argument('--video-skip',type=int,default=2)
    p.add_argument('--camera-layout',choices=['follow_ego','global_ego'],default='follow_ego')
    p.add_argument('--seed',type=int,default=123)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    policy_label='LEARNED BC / ONE-DEMO SMOKE' if ckpt.get('single_demo_smoke') else 'LEARNED BC'
    if ckpt.get('algorithm')=='denoising_world_goal_MLP_BC':
        from train_goal_policy import GoalPolicy
        net=GoalPolicy(ckpt['obs_dim'],ckpt['history'],ckpt['hidden']).cuda().eval()
    elif ckpt.get('algorithm')=='nearest_state_memory_BC':
        from train_memory_policy import NearestStatePolicy
        policy_label='MEMORY-BASED BC / NOT NEURAL'
        net=NearestStatePolicy(ckpt['model']['centers'],ckpt['model']['targets']).cuda().eval()
    elif ckpt.get('algorithm')=='recurrent_GRU_BC':
        from train_recurrent_policy import RecurrentPolicy
        net=RecurrentPolicy(ckpt['obs_dim'],ckpt['hidden'],ckpt['layers']).cuda().eval()
    elif ckpt.get('algorithm')=='rbf_kernel_BC':
        from train_rbf_policy import RBFPolicy
        state=ckpt['model'];net=RBFPolicy(state['centers'],state['weights'],ckpt['bandwidth'],normalized=ckpt.get('normalized_basis',False)).cuda().eval()
    else:
        net=ChunkPolicy(ckpt['obs_dim'],ckpt['history'],ckpt['horizon'],ckpt['hidden'],ckpt.get('mixtures',1)).cuda().eval()
    net.load_state_dict(ckpt['model'])
    norm={k:np.asarray(v,dtype=np.float32) for k,v in ckpt['norm'].items()}
    demos=args.demo or ckpt['val_demos']
    manifest={'mode':('closed_loop_learned_policy' if ckpt.get('algorithm')=='nearest_state_memory_BC' else 'closed_loop_neural_policy'),'algorithm':ckpt.get('algorithm','state_BC'),'planner_enabled':False,'teacher_action_reads':0,
              'checkpoint':str(args.checkpoint),'training_step':ckpt['step'],'camera_layout':args.camera_layout,
              'dataset':str(args.dataset),'privileged_state':True,'mixtures':ckpt.get('mixtures',1),'single_demo_smoke':ckpt.get('single_demo_smoke',False),'action_mode':ckpt.get('action_mode','delta_joint'),'episodes':[]}
    (args.output/'result.json').write_text(json.dumps(manifest,indent=2))
    # Read prior explicit server-side consent; never accept new terms silently.
    flags={}
    license_file=Path(__file__).resolve().parent.parent/'config/licenses.local.env'
    if license_file.is_file():
        for line in license_file.read_text().splitlines():
            if '=' in line:
                key,value=line.split('=',1);flags[key.strip()]=value.strip().strip("\"'")
    for key in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS'):
        if os.environ.get(key,flags.get(key,'NO'))!='YES':
            raise RuntimeError(f'Explicit previously recorded consent required: {key}')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    except OSError: font=ImageFont.load_default()
    env=None; writer=None; failed=False
    try:
        meta=ckpt['env_args']; meta['env_kwargs']['init_curobo']=False
        # False avoids the upstream point-cloud policy adapter. No wrapper.reset()
        # is called: its generation-only implementation assumes a planner exists.
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs'])
        from pxr import Gf
        actual_grip=[int(as_numpy(env.robot.gripper_action_idx[a]).reshape(-1)[0]) for a in ('left','right')]
        if actual_grip!=GRIP: raise ValueError(f'R1 action layout mismatch: {actual_grip}')
        if env.robot.controllers['base'].motor_type!='position':
            raise ValueError('Expected MoMaGen local-delta position base controller')
        camera_pos=torch.tensor([8.7,3.0,4.5])
        mat=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*camera_pos.tolist()),Gf.Vec3d(5.5,0.0,0.8),Gf.Vec3d(0,0,1)).GetInverse()
        quat=mat.ExtractRotationQuat(); imag=quat.GetImaginary()
        camera_quat=torch.tensor([imag[0],imag[1],imag[2],quat.GetReal()],dtype=torch.float32)
        for demo in demos:
            with h5py.File(args.dataset,'r') as f:
                initial=f[f'data/{demo}/states'][0]
                ref_len=len(f[f'data/{demo}/states'])
            env.env.reset()
            if int(og.sim.dump_state(serialized=True).numel())!=len(initial):
                raise ValueError('Serialized scene state layout mismatch')
            og.sim.load_state(torch.tensor(initial,dtype=torch.float32),serialized=True)
            for _ in range(3): og.sim.render()
            predicate=env.env.task._termination_conditions["predicate"]
            predicate.reset(env.env.task,env.env)
            if env.is_success()["task"]:
                raise ValueError("Initial state already satisfies goal; invalid evaluation reset")
            obs,_=env.get_obs_IL(); obs=add_unique_task_poses(env,obs)
            if hasattr(net,'reset_memory'):net.reset_memory()
            history=deque(maxlen=(ckpt['history']-1)*ckpt['history_stride']+1)
            predictions=deque(maxlen=ckpt['horizon'])
            trace_x=[]; trace_a=[]; base_path=[]; gripped=[False,False]
            success=False; stable=0; termination_reason='horizon'
            writer=imageio.get_writer(str(args.output/f'{demo}.mp4'),fps=30/args.video_skip,
                                       codec='libx264',quality=7,macro_block_size=2)
            report={'demo':demo,'split':'validation' if demo in ckpt['val_demos'] else 'training',
                    'initial_state_load_count':1,'reference_length':ref_len,'success':False}
            for step in range(args.max_steps):
                x=vectorize(obs,ckpt['keys']); history.append(x)
                hs=list(history); past=[hs[max(0,len(hs)-1-j*ckpt['history_stride'])] for j in reversed(range(ckpt['history']))]
                nx=(np.stack(past)-norm['x_mean'])/norm['x_std']
                with torch.inference_mode(): out=net(torch.tensor(nx[None],device='cuda')).cpu().numpy()[0]
                chunk=out*norm['y_std']+norm['y_mean']
                if ckpt.get('action_mode')=='world_base_goal':
                    chunk=decode_world_base_goals(chunk,as_numpy(obs['base_link_pose']),float(as_numpy(obs['base_qpos'])[2]))
                    jaw_scores=out[:,[14,21]]
                else:jaw_scores=out[:,GRIP]
                prop=joint_anchor(as_numpy(obs['prop_state']))
                if ckpt.get('action_mode','delta_joint')=='delta_joint':
                    chunk[:,JOINT]+=prop[None,JOINT]
                # Threshold learned signed gripper scores; never average jaw widths.
                chunk[:,GRIP]=np.where(jaw_scores>0,norm['action_max'][GRIP],norm['action_min'][GRIP])
                predictions.append((step,chunk))
                candidates=[]; weights=[]
                for origin,prediction in predictions:
                    age=step-origin
                    if age<len(prediction): candidates.append(prediction[age]); weights.append(np.exp(-0.15*age))
                action=np.average(np.stack(candidates),axis=0,weights=weights).astype('float32')
                action[GRIP]=np.where(action[GRIP]>0,norm['action_max'][GRIP],norm['action_min'][GRIP])
                action=np.clip(action,norm['action_min'],norm['action_max'])
                if not np.isfinite(action).all(): raise ValueError('Nonfinite network action')
                trace_x.append(x); trace_a.append(action)
                base_path.append(as_numpy(env.robot.get_position_orientation()[0]).copy())
                raw_obs,reward,done,truncated,info=env.env.step(torch.tensor(action))
                obs,_=env.get_obs_IL(di=(raw_obs, {})); obs=add_unique_task_poses(env,obs)
                for side,key in enumerate(('grasp_left','grasp_right')):
                    gripped[side] |= int(as_numpy(obs[key]).reshape(-1)[0])==1
                success=bool(env.is_success()['task']); stable=stable+1 if success else 0
                if step%args.video_skip==0:
                    og.sim.viewer_camera.set_position_orientation(position=camera_pos,orientation=camera_quat)
                    og.sim.render()
                    if args.camera_layout=='follow_ego':
                        primary=env.env._external_sensors['external_sensor2'].get_obs()[0]['rgb']
                        panel_width=480
                    else:
                        primary=og.sim.viewer_camera._get_obs()[0]['rgb']; panel_width=854
                    global_img=as_numpy(primary)[:,:,:3].astype('uint8')
                    ego_keys=[k for k in raw_obs if ':eyes:Camera:0::rgb' in k]
                    panels=[Image.fromarray(global_img).resize((panel_width,480))]
                    if ego_keys:
                        panels.append(Image.fromarray(as_numpy(raw_obs[ego_keys[0]])[:,:,:3].astype('uint8')).resize((480,480)))
                    frame=Image.new('RGB',(sum(i.width for i in panels),508),'#171717')
                    cursor=0
                    for im in panels: frame.paste(im,(cursor,28)); cursor+=im.width
                    ImageDraw.Draw(frame).text((8,8),f'{policy_label} | {demo} | step {step} | success: {success}',fill='white',font=font)
                    writer.append_data(np.asarray(frame))
                    if step%200==0:frame.resize((960,508)).save(args.output/f'progress-{demo}.jpg',quality=85)
                if step%100==0:
                    print(json.dumps({'demo':demo,'step':step,'success':success,'grasped_ever':gripped,
                          'base_position':base_path[-1].tolist()}),flush=True)
                # Require sustained predicate success, not a single transient frame.
                if stable>=30:
                    termination_reason='sustained_success';break
                if truncated or (done and not success):
                    termination_reason='environment_termination';break
                window=args.stagnation_window
                if not success and window>0 and len(trace_x)>=window and step%100==0:
                    if float(np.ptp(np.stack(trace_x[-window:]),axis=0).max())<0.003:
                        termination_reason='stagnation';break
            writer.close(); writer=None
            path=np.stack(base_path)
            report.update(steps=step+1,success=success,stable_success_frames=stable,termination_reason=termination_reason,
                  sustained_success=stable>=30,grasped_ever=gripped,
                  base_path_length_m=float(np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1).sum()),
                  base_net_displacement_m=float(np.linalg.norm(path[-1,:2]-path[0,:2])),
                  final_goal_status=str(env.env.task._termination_conditions['predicate'].goal_status))
            np.savez_compressed(args.output/f'{demo}-trace.npz',observations=np.stack(trace_x),actions=np.stack(trace_a),base_positions=path)
            manifest['episodes'].append(report)
            (args.output/'result.json').write_text(json.dumps(manifest,indent=2))
            print('EPISODE_RESULT',json.dumps(report),flush=True)
    except BaseException:
        failed=True; traceback.print_exc()
        manifest['error']='Evaluation failed; inspect evaluation.log; no task success claimed.'
        (args.output/'result.json').write_text(json.dumps(manifest,indent=2))
    finally:
        if writer: writer.close()
        try: og.shutdown()
        except SystemExit: pass
    if failed: raise SystemExit(1)


if __name__=='__main__': main()
