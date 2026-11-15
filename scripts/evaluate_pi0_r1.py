#!/usr/bin/env python3
"""SERVER ONLY: real visual pi0 closed-loop R1 rollout, no teacher / state replay.

Runs in the existing Python3.10 simulator env, calls the isolated Python3.11
openpi service on localhost. Only real RGB + proprio + prompt enter pi0.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import time
import traceback
import h5py
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from grasping_validation import activate_physical_after_reset
from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
from object_pose_audit import ObjectPoseAudit
from pi0_rpc import infer_local
from state_policy_features import as_numpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--demo',default='demo_0')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--action-bounds',type=Path,required=True)
    p.add_argument('--port',type=int,default=18731)
    p.add_argument('--max-steps',type=int,default=3000)
    p.add_argument('--execute-steps',type=int,default=5)
    p.add_argument('--grasping-mode',choices=['configured','physical'],default='physical')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not 1<=args.execute_steps<=50:raise ValueError('Invalid receding horizon')
    token=args.token_file.read_text().strip();bounds=json.loads(args.action_bounds.read_text())
    lower=np.asarray(bounds['min'],np.float32);upper=np.asarray(bounds['max'],np.float32)
    if lower.shape!=(21,) or upper.shape!=(21,) or np.any(lower>upper):raise ValueError('Invalid dataset native action limits')
    root=Path('/mnt/yuanyue/double-arm');flags={}
    for line in (root/'code/config/licenses.local.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key,value=line.split('=',1);flags[key.strip()]=value.strip().strip('"\x27')
    for name in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS'):
        if os.environ.get(name,flags.get(name))!='YES':raise ValueError('Previously accepted server license flags required')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES';torch.set_num_threads(4);torch.set_num_interop_threads(1)
    with h5py.File(args.dataset) as file:
        meta=json.loads(file['data'].attrs['env_args']);g=file['data/'+args.demo];initial=g['states'][0]
    meta['env_kwargs']['init_curobo']=False
    args.output.mkdir(parents=True)
    result={'mode':'closed_loop_neural_policy','algorithm':'official_openpi_pi0_lora32','privileged_state':False,
            'checkpoint':str(args.checkpoint),'planner_enabled':False,'teacher_action_reads':0,'initial_state_load_count':1,
            'input_modalities':['head_rgb','left_wrist_rgb','right_wrist_rgb','proprioception','instruction'],
            'execute_steps':args.execute_steps,'predicted_horizon':50,'grasping_mode':args.grasping_mode,
            'assisted_constraint_seen':False,'object_pose_write_count':0,'upstream_finger_friction':4.0,
            'physical_parameters_calibrated_to_real_robot':False,'task_success':False,'episodes':[]}
    def save():
        tmp=args.output/'result.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(args.output/'result.json')
    save();writer=None;audit=None;failed=False
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    try:
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs']);env.env.reset()
        if env.robot.controllers['base'].motor_type!='position':raise ValueError('Expected native local-displacement base controller')
        if int(og.sim.dump_state(serialized=True).numel())!=len(initial):raise ValueError('Serialized reset layout mismatch')
        og.sim.load_state(torch.tensor(initial,dtype=torch.float32),serialized=True)
        if args.grasping_mode=='physical':activate_physical_after_reset(env.robot)
        result['grasping_mode']=env.robot.grasping_mode
        for _ in range(3):og.sim.render()
        predicate=env.env.task._termination_conditions['predicate'];predicate.reset(env.env.task,env.env)
        if env.is_success()['task']:raise ValueError('Initial state already satisfies task')
        cup=env.env.task.object_scope['teacup.n.02_1'].unwrapped
        initial_cup_z=float(as_numpy(cup.get_position_orientation()[0])[2])
        max_cup_lift=0.0;cup_grasped=False
        obs,_=env.get_obs_IL();cameras=camera_keys(obs);audit=ObjectPoseAudit(env.env.task.object_scope)
        writer=imageio.get_writer(str(args.output/(args.demo+'.mp4')),fps=15,codec='libx264',quality=7,macro_block_size=2)
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        except OSError:font=ImageFont.load_default()
        queued=deque();states=[];actions=[];path=[];grasp=[False,False];stable=0;rpc=[];clipped=0
        for step in range(args.max_steps):
            audit.step=step
            if not queued:
                inputs={key:rgb_image(obs[name]) for key,name in cameras.items()}
                inputs.update(state=as_numpy(obs['prop_state']).copy(),prompt=PROMPT)
                prediction,info=infer_local(inputs,token,args.port)
                queued.extend(prediction[:args.execute_steps]);rpc.append({'step':step,'inference_seconds':info['inference_seconds']})
            proposed=np.asarray(queued.popleft(),np.float32);action=np.clip(proposed,lower,upper)
            clipped+=int(np.any(action!=proposed))
            states.append(as_numpy(obs['prop_state']).copy());actions.append(action.copy());path.append(as_numpy(env.robot.get_position_orientation()[0]).copy())
            raw,reward,done,truncated,info=env.env.step(torch.tensor(action))
            obs,_=env.get_obs_IL(di=(raw,{}))
            assisted=any(value is not None for value in env.robot._ag_obj_constraints.values())
            result['assisted_constraint_seen']|=assisted
            if args.grasping_mode=='physical' and assisted:raise ValueError('Assisted attachment appeared in physical eval')
            if audit.events:raise ValueError('Task-object explicit pose write during closed-loop eval')
            for i,name in enumerate(('grasp_left','grasp_right')):grasp[i]|=int(as_numpy(obs[name]).reshape(-1)[0])==1
            cup_grasped|=int(env.robot.is_grasping(arm='left',candidate_obj=cup))==1
            max_cup_lift=max(max_cup_lift,float(as_numpy(cup.get_position_orientation()[0])[2])-initial_cup_z)
            success=bool(env.is_success()['task']);stable=stable+1 if success else 0
            if step%2==0:
                primary=rgb_image(env.env._external_sensors['external_sensor2'].get_obs()[0]['rgb'])
                head=rgb_image(obs[cameras['head_image']])
                frame=Image.new('RGB',(960,508),'#161616');frame.paste(Image.fromarray(primary).resize((480,480)),(0,28));frame.paste(Image.fromarray(head).resize((480,480)),(480,28))
                ImageDraw.Draw(frame).text((5,5),f'PI0 LoRA32 / {env.robot.grasping_mode.upper()} / step {step} / success {success}',font=font,fill='white')
                writer.append_data(np.asarray(frame))
                if step%200==0:frame.save(args.output/'progress.jpg')
            if step%100==0:print(json.dumps({'step':step,'success':success,'grasped_ever':grasp}),flush=True)
            if stable>=30 or truncated or (done and not success):break
        writer.close();writer=None
        positions=np.stack(path);episode={'demo':args.demo,'split':'validation','initial_state_load_count':1,'steps':step+1,
            'sustained_success':stable>=30,'stable_success_frames':stable,'grasped_ever':grasp,
            'base_path_length_m':float(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1).sum()),
            'final_goal_status':str(predicate.goal_status),'object_pose_write_count':len(audit.events),
            'cup_specific_grasp_seen':cup_grasped,'max_cup_lift_m':max_cup_lift}
        result['episodes'].append(episode);result.update(object_pose_write_count=len(audit.events),inference_calls=len(rpc),clipped_action_steps=clipped)
        result['task_success']=stable>=30 and any(grasp) and cup_grasped and max_cup_lift>.03 and episode['base_path_length_m']>.5
        np.savez_compressed(args.output/(args.demo+'-trace.npz'),observations=np.stack(states),actions=np.stack(actions),base_positions=positions)
        (args.output/'inference-times.json').write_text(json.dumps(rpc,indent=2));save()
    except BaseException as error:
        failed=True;result.update(error=repr(error),task_success=False)
        if audit:result['object_pose_write_count']=len(audit.events)
        traceback.print_exc();save()
    finally:
        if audit:audit.close()
        if writer:writer.close()
        try:og.shutdown()
        except SystemExit:pass
    if failed:raise SystemExit(7)


if __name__=='__main__':main()
