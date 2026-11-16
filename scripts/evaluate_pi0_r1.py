#!/usr/bin/env python3
"""SERVER ONLY: real visual pi0 closed-loop R1 rollout, no teacher / state replay.

Runs in the existing Python3.10 simulator env, calls the isolated Python3.11
openpi service on localhost. Only real RGB + proprio + prompt enter pi0.
"""
import argparse
from collections import deque
import json
import hashlib
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
from pi0_base_targets import absolute_base_to_native
from rollout_motion_metrics import cup_motion_metrics
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
    p.add_argument('--snapshot-only',action='store_true',help='Audit reset RGB/proprio; no model inference or env.step')
    p.add_argument('--snapshot-settle-series',default='',help='Snapshot-only cumulative physics steps, e.g. 0,1,2,5,20; max20')
    p.add_argument('--reset-settle-steps',type=int,default=0,help='Explicit physics/Fabric sync candidate; default keeps original evaluation behavior')
    p.add_argument('--execute-steps',type=int,default=5)
    p.add_argument('--grasping-mode',choices=['configured','physical'],default='physical')
    p.add_argument('--action-representation',choices=['native','absolute_base'],default='native')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if not 1<=args.execute_steps<=50:raise ValueError('Invalid receding horizon')
    if not 0<=args.reset_settle_steps<=20:raise ValueError('Invalid bounded reset settling')
    series=[]
    if args.snapshot_settle_series:
        series=[int(value) for value in args.snapshot_settle_series.split(',')]
        if not args.snapshot_only or args.reset_settle_steps!=0 or series!=sorted(set(series)) or not all(0<=n<=20 for n in series):
            raise ValueError('Snapshot series requires ordered unique steps0..20, snapshot-only and no extra settling')

    token=args.token_file.read_text().strip();bounds=json.loads(args.action_bounds.read_text())
    lower=np.asarray(bounds['min'],np.float32);upper=np.asarray(bounds['max'],np.float32)
    if lower.shape!=(21,) or upper.shape!=(21,) or np.any(lower>upper):raise ValueError('Invalid dataset native action limits')
    if args.action_representation=='absolute_base':
        base_lower=np.asarray(bounds['native_base_min'],np.float32);base_upper=np.asarray(bounds['native_base_max'],np.float32)
        if base_lower.shape!=(3,) or base_upper.shape!=(3,):raise ValueError('Missing native base step limits')
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
    result={'mode':'reset_observation_audit_only' if args.snapshot_only else 'closed_loop_neural_policy','algorithm':'official_openpi_pi0_lora32','privileged_state':False,
            'checkpoint':str(args.checkpoint),'planner_enabled':False,'teacher_action_reads':0,'initial_state_load_count':1,
            'input_modalities':['head_rgb','left_wrist_rgb','right_wrist_rgb','proprioception','instruction'],
            'execute_steps':args.execute_steps,'predicted_horizon':50,'grasping_mode':args.grasping_mode,
            'assisted_constraint_seen':False,'object_pose_write_count':0,'upstream_finger_friction':4.0,
            'physical_parameters_calibrated_to_real_robot':False,'task_success':False,'episodes':[],
            'inference_platform':None,'real_time_control_claim':False,'action_representation':args.action_representation,
            'low_level_robot_odometry_used':args.action_representation=='absolute_base',
            'requested_reset_physics_steps':args.reset_settle_steps,'reset_physics_steps':0,
            'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save():
        tmp=args.output/'result.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(args.output/'result.json')
    save();writer=None;audit=None;failed=False;started=time.monotonic()
    # Keep completed transitions even when a safety check aborts the rollout.
    # Failed/in-flight commands are never padded into a completed trace.
    cup_positions=[];post_base_positions=[];cup_grasp_flags=[]
    queued=deque();states=[];actions=[];targets=[];proposals=[];root_poses=[];base_clipped=0;path=[]
    grasp=[False,False];stable=0;rpc=[];clipped=0;max_cup_lift=0.0;cup_grasped=False
    step=-1;pose=None;initial_cup_position=None
    def finalize_episode(aborted=False):
        n=len(cup_positions)
        if not n:return
        positions=np.stack(path[:n])
        episode={'demo':args.demo,'split':'validation','initial_state_load_count':1,'steps':n,
            'aborted':aborted,'sustained_success':not aborted and stable>=30,'stable_success_frames':stable,
            'grasped_ever':grasp,'base_path_length_m':float(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1).sum()),
            'final_goal_status':str(predicate.goal_status),'object_pose_write_count':len(audit.events),
            'cup_specific_grasp_seen':cup_grasped,'max_cup_lift_m':max_cup_lift}
        episode.update(cup_motion_metrics(cup_positions,post_base_positions,cup_grasp_flags,initial_cup_position))
        result['episodes']=[episode]
        result.update(object_pose_write_count=len(audit.events),inference_calls=len(rpc),clipped_action_steps=clipped,
            clipped_native_base_steps=base_clipped,wall_clock_seconds=time.monotonic()-started)
        result['task_success']=bool(not aborted and stable>=30 and any(grasp) and cup_grasped
            and max_cup_lift>.03 and episode['base_path_length_m']>.5)
        np.savez_compressed(args.output/(args.demo+'-trace.npz'),observations=np.stack(states[:n]),
            actions=np.stack(actions[:n]),base_positions=positions,model_action_targets=np.stack(targets[:n]),
            model_proposed_targets=np.stack(proposals[:n]),
            pre_step_root_poses=np.stack(root_poses[:n]),post_step_cup_positions=np.stack(cup_positions),
            post_step_base_positions=np.stack(post_base_positions),post_step_cup_grasp_flags=np.asarray(cup_grasp_flags,dtype=bool),
            initial_cup_position=initial_cup_position)
        (args.output/'inference-times.json').write_text(json.dumps(rpc,indent=2));save()
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    try:
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs']);env.env.reset()
        if env.robot.controllers['base'].motor_type!='position':raise ValueError('Expected native local-displacement base controller')
        if int(og.sim.dump_state(serialized=True).numel())!=len(initial):raise ValueError('Serialized reset layout mismatch')
        og.sim.load_state(torch.tensor(initial,dtype=torch.float32),serialized=True)
        if args.grasping_mode=='physical':activate_physical_after_reset(env.robot)
        result['grasping_mode']=env.robot.grasping_mode
        audit=ObjectPoseAudit(env.env.task.object_scope)
        if series:
            result.update(mode='reset_observation_series_audit_only',reset_snapshots=[],
                simulator_control_calls=0,inference_calls=0,task_success=False)
            for count in series:
                while result['reset_physics_steps']<count:
                    og.sim.step()
                    result['reset_physics_steps']+=1
                    assisted=any(value is not None for value in env.robot._ag_obj_constraints.values())
                    result['assisted_constraint_seen']|=assisted
                    if args.grasping_mode=='physical' and (assisted or audit.events):
                        raise ValueError('Assistance or object pose writes during reset series')
                for _ in range(3):og.sim.render()
                reset_obs,_=env.get_obs_IL();reset_cameras=camera_keys(reset_obs)
                reset_inputs={key:rgb_image(reset_obs[name]) for key,name in reset_cameras.items()}
                reset_inputs['state']=as_numpy(reset_obs['prop_state']).copy()
                snapshot=args.output/f'observation-physics-{count}.npz'
                temporary=snapshot.with_suffix('.tmp')
                with temporary.open('wb') as handle:np.savez_compressed(handle,**reset_inputs)
                temporary.replace(snapshot)
                root_position,root_quaternion=env.robot.get_position_orientation()
                assisted=any(value is not None for value in env.robot._ag_obj_constraints.values())
                result['assisted_constraint_seen']|=assisted
                if args.grasping_mode=='physical' and (assisted or audit.events):
                    raise ValueError('Assistance or object pose writes during snapshot capture')
                result['reset_snapshots'].append({'physics_steps':count,'path':str(snapshot),
                    'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                    'robot_root_pose':np.concatenate([as_numpy(root_position),as_numpy(root_quaternion)]).tolist(),
                    'camera_keys':reset_cameras,'object_pose_write_count':len(audit.events)})
                result['object_pose_write_count']=len(audit.events);save()
                print(json.dumps({'reset_snapshot_physics_steps':count,'snapshot':str(snapshot)}),flush=True)
            result['snapshot_series_completed']=True;save();return
        # Opt-in candidate only: let physics update restored articulation/Fabric
        # before reading cameras. No env.step, motor command, pose setter, teacher
        # action or planner. Do not adopt as a validated fix without a new audit.
        for _ in range(args.reset_settle_steps):
            og.sim.step()
            result['reset_physics_steps']+=1
            assisted=any(value is not None for value in env.robot._ag_obj_constraints.values())
            result['assisted_constraint_seen']|=assisted
            if args.grasping_mode=='physical' and (assisted or audit.events):
                raise ValueError('Assistance or object pose writes during reset settling')
        for _ in range(3):og.sim.render()
        predicate=env.env.task._termination_conditions['predicate'];predicate.reset(env.env.task,env.env)
        if env.is_success()['task']:raise ValueError('Initial state already satisfies task')
        cup=env.env.task.object_scope['teacup.n.02_1'].unwrapped
        initial_cup_position=as_numpy(cup.get_position_orientation()[0]).copy()
        initial_cup_z=float(initial_cup_position[2])
        max_cup_lift=0.0;cup_grasped=False
        obs,_=env.get_obs_IL();cameras=camera_keys(obs)
        # Exact reset policy modalities for offline data/live domain audits.
        # No object pose, serialized state, action label or clock is in this file.
        initial_inputs={key:rgb_image(obs[name]) for key,name in cameras.items()}
        initial_inputs['state']=as_numpy(obs['prop_state']).copy()
        snapshot=args.output/'initial-observation.npz'
        snapshot_tmp=args.output/'initial-observation.tmp'
        with snapshot_tmp.open('wb') as handle:np.savez_compressed(handle,**initial_inputs)
        snapshot_tmp.replace(snapshot)
        result.update(initial_observation_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
            initial_observation_camera_keys=cameras)
        if args.snapshot_only:
            result.update(simulator_control_calls=0,inference_calls=0,task_success=False)
            save();return
        writer=imageio.get_writer(str(args.output/(args.demo+'.mp4')),fps=15,codec='libx264',quality=7,macro_block_size=2)
        try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
        except OSError:font=ImageFont.load_default()
        for step in range(args.max_steps):
            audit.step=step
            if not queued:
                inputs={key:rgb_image(obs[name]) for key,name in cameras.items()}
                inputs.update(state=as_numpy(obs['prop_state']).copy(),prompt=PROMPT)
                prediction,info=infer_local(inputs,token,args.port)
                platform=info.get('inference_platform')
                if platform not in ('cpu','gpu'):raise ValueError('Inference service did not report its platform')
                result['inference_platform']=platform
                result['denoising_steps']=info.get('denoising_steps')
                result['sampling_noise']=info.get('sampling_noise')
                if info.get('action_representation','native')!=args.action_representation:raise ValueError('Wrong action representation service')
                queued.extend(prediction[:args.execute_steps]);rpc.append({'step':step,'inference_seconds':info['inference_seconds'],'inference_platform':platform})
            proposed=np.asarray(queued.popleft(),np.float32);proposals.append(proposed.copy())
            target=np.clip(proposed,lower,upper)
            clipped+=int(np.any(target!=proposed));targets.append(target.copy())
            position,quaternion=env.robot.get_position_orientation()
            pose=np.concatenate([as_numpy(position),as_numpy(quaternion)])
            if args.action_representation=='absolute_base':
                # Current ROBOT odometry is used only by this ordinary motor
                # target conversion, not as a VLA input or navigation planner.
                action=absolute_base_to_native(target,pose)
                capped=np.clip(action[:3],base_lower,base_upper)
                base_clipped+=int(np.any(capped!=action[:3]));action[:3]=capped
            else:action=target
            root_poses.append(pose.copy())
            states.append(as_numpy(obs['prop_state']).copy());actions.append(action.copy());path.append(as_numpy(env.robot.get_position_orientation()[0]).copy())
            raw,reward,done,truncated,info=env.env.step(torch.tensor(action))
            obs,_=env.get_obs_IL(di=(raw,{}))
            assisted=any(value is not None for value in env.robot._ag_obj_constraints.values())
            result['assisted_constraint_seen']|=assisted
            if args.grasping_mode=='physical' and assisted:raise ValueError('Assisted attachment appeared in physical eval')
            if args.grasping_mode=='physical' and audit.events:raise ValueError('Task-object explicit pose write during closed-loop eval')
            for i,name in enumerate(('grasp_left','grasp_right')):grasp[i]|=int(as_numpy(obs[name]).reshape(-1)[0])==1
            cup_grasp_now=int(env.robot.is_grasping(arm='left',candidate_obj=cup))==1
            cup_grasped|=cup_grasp_now
            # Evaluation-only pose/contact audit. These arrays NEVER enter the
            # RPC request or influence model actions / stopping conditions.
            cup_position=as_numpy(cup.get_position_orientation()[0]).copy()
            cup_positions.append(cup_position);cup_grasp_flags.append(cup_grasp_now)
            post_base_positions.append(as_numpy(obs['base_link_pose'])[:3].copy())
            max_cup_lift=max(max_cup_lift,float(cup_position[2])-initial_cup_z)
            success=bool(env.is_success()['task']);stable=stable+1 if success else 0
            if step%2==0:
                primary=rgb_image(env.env._external_sensors['external_sensor2'].get_obs()[0]['rgb'])
                head=rgb_image(obs[cameras['head_image']])
                frame=Image.new('RGB',(960,508),'#161616');frame.paste(Image.fromarray(primary).resize((480,480)),(0,28));frame.paste(Image.fromarray(head).resize((480,480)),(480,28))
                ImageDraw.Draw(frame).text((5,5),f'PI0 LoRA32 / {result["inference_platform"].upper()} / {env.robot.grasping_mode.upper()} / step {step} / success {success}',font=font,fill='white')
                writer.append_data(np.asarray(frame))
                if step%200==0:frame.save(args.output/'progress.jpg')
            if step%100==0:print(json.dumps({'step':step,'success':success,'grasped_ever':grasp,'robot_root_pose':pose.tolist()}),flush=True)
            if stable>=30 or truncated or (done and not success):break
        writer.close();writer=None
        finalize_episode()
    except BaseException as error:
        failed=True;result.update(error=repr(error),task_success=False)
        if audit:result['object_pose_write_count']=len(audit.events)
        result['failure_context']={'attempted_step':step,'completed_steps':len(cup_positions),
            'robot_root_pose':None if pose is None else pose.tolist()}
        traceback.print_exc()
        try:finalize_episode(aborted=True)
        except Exception as trace_error:
            result['trace_export_error']=repr(trace_error);traceback.print_exc()
        save()
    finally:
        if audit:audit.close()
        if writer:writer.close()
        try:og.shutdown()
        except SystemExit:pass
    if failed:raise SystemExit(7)


if __name__=='__main__':main()
