#!/usr/bin/env python3
"""SERVER bounded source635 INITIALIZED local grasp: replay gate -> learned pi0.

NOT full Tidy Table, navigation, heldout, or physical grasp acceptance. Original
configured assisted mode. One source state initialization per independent case;
no per-step restores or teacher actions in the learned case. No training.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/mnt/yuanyue/double-arm')
FRAME=635
STEPS=120


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def replay_gate(episode):
    return (episode.get('mode')=='source_action_replay' and episode.get('steps')==STEPS
        and episode.get('aborted') is False and episode.get('local_cup_grasp_report_seen') is True
        and episode.get('object_pose_write_count')==0 and episode.get('initial_state_load_count')==1)


def review_label(episode):
    if episode['mode']=='source_action_replay':
        return 'SOURCE ACTION REPLAY - INIT635 ASSISTED - NOT LEARNED OR FULL TASK'
    outcome='ASSISTED LOCAL GRASP OBSERVED' if episode.get('local_cup_grasp_report_seen') else 'FAILED LOCAL GRASP'
    if episode.get('aborted'):outcome='ABORTED LOCAL DIAGNOSTIC'
    return outcome+' - LEARNED PI0 INIT635 - NOT FULL TASK OR PHYSICAL SUCCESS'


def simulate(c,parent):
    if os.getppid()!=parent or read(c/'status.json')['controller_pid']!=parent:raise ValueError('Owning parent required')
    from resource_allocation import resolve_gpu
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=resolve_gpu(ROOT,'1'):raise ValueError('Explicit GPU1 UUID only')
    flags={}
    for line in (ROOT/'code/config/licenses.local.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            key,value=line.split('=',1);flags[key.strip()]=value.strip().strip('"\x27')
    if any(os.environ.get(k,flags.get(k))!='YES' for k in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS')):raise ValueError('Existing confirmations required; no new terms acceptance')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    import h5py
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from collections import deque
    from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
    from object_pose_audit import ObjectPoseAudit
    from pi0_base_targets import root_rotation_xy_and_yaw
    from pi0_rpc import infer_local
    from grasp_telemetry import read_grasp_telemetry
    from motor_contract import snapshot_motor_contract
    from probe_pi0_consistent_feedback import nominal_gate
    from evaluation_split_contract import verify_evaluation_episode
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    plan=read(c/'plan.json');source=Path(plan['source']['source']);demo=plan['source']['source_demo']
    verify_evaluation_episode(read(ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'),source,demo,'train')
    with h5py.File(source) as file:meta=json.loads(file['data'].attrs['env_args'])
    meta['env_kwargs']['init_curobo']=False
    bounds=read(Path(plan['action_bounds']));lower=np.asarray(bounds['min'],np.float32);upper=np.asarray(bounds['max'],np.float32)
    token=(c/'rpc-token.local').read_text().strip()
    result={'stage':'initializing_simulator','mode':'PRIVILEGED_SOURCE635_INITIALIZATION_local_diagnostic_NOT_full_task',
        'checkpoint':plan['checkpoint'],'evaluation_split':'train','grasping_mode':'assisted','policy_training':'not_run',
        'task_success':False,'physical_task_acceptance':False,'initialization_source_frame':FRAME,
        'source_replay_and_learned_results_separate':True,'episodes':[]}
    out=c/'local';out.mkdir()
    def save(**changes):result.update(changes);write(out/'result.json',result)
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    save()
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    audit=None;writer=None
    try:
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs'])
        for mode in ('source_action_replay','learned_pi0_local_grasp'):
            # Independent initialization, never a mid-rollout teacher restore.
            env.env.reset()
            with h5py.File(source) as file:
                group=file['data/'+demo];state=group['states'][FRAME]
                reference_state=group['obs/prop_state'][FRAME];keys=camera_keys(group['obs'])
                reference_rgb={k:rgb_image(group['obs'][v][FRAME]) for k,v in keys.items()}
                replay_actions=group['actions'][FRAME:FRAME+STEPS].copy() if mode=='source_action_replay' else None
            if mode=='learned_pi0_local_grasp' and replay_actions is not None:raise ValueError('No teacher actions in learned case')
            if int(og.sim.dump_state(serialized=True).numel())!=len(state):raise ValueError('State layout mismatch')
            og.sim.load_state(torch.tensor(state,dtype=torch.float32),serialized=True)
            if env.robot.controllers['base'].motor_type!='position':raise ValueError('Native base POSITION required')
            if env.robot.grasping_mode!='assisted':raise ValueError('Preserve author configured mode')
            # Holding actual INITIAL q including fingers prevents previous replay
            # closure targets from leaking into the independent learned reset.
            names=('trunk','arm_left','arm_right','gripper_left','gripper_right')
            for name in names:
                control=env.robot.controllers[name]
                if getattr(control,'motor_type',getattr(control,'_motor_type',None))!='position':raise ValueError('Expected position holding initialization')
            body=torch.unique(torch.cat([torch.as_tensor(env.robot.controllers[name].dof_idx,dtype=torch.int64) for name in names]))
            held=env.robot.get_joint_positions()[body]
            env.robot.set_joint_positions(held,indices=body,drive=True)
            env.robot.set_joint_velocities(torch.zeros_like(held),indices=body,drive=False)
            before=array(env.robot.get_position_orientation()[0]).copy();audit=ObjectPoseAudit(env.env.task.object_scope)
            ag=False
            for _ in range(20):
                og.sim.step();ag|=any(v is not None for v in env.robot._ag_obj_constraints.values())
            for _ in range(3):og.sim.render()
            obs,_=env.get_obs_IL();cameras=camera_keys(obs);initial_inputs={k:rgb_image(obs[v]) for k,v in cameras.items()};initial_inputs['state']=array(obs['prop_state']).copy()
            pose=np.r_[array(env.robot.get_position_orientation()[0]),array(env.robot.get_position_orientation()[1])];root_rotation_xy_and_yaw(pose)
            alignment={'source_rgb_mae_0_255':{k:float(np.abs(initial_inputs[k].astype(float)-reference_rgb[k]).mean()) for k in keys},
                'source_body_q_max_abs_rad':float(np.max(np.abs(initial_inputs['state'][3:19]-reference_state[3:19]))),
                'root_translation_during_refresh_m':float(np.linalg.norm(pose[:3]-before)),
                'object_pose_write_count':len(audit.events),'assisted_constraint_seen':ag}
            # Finger-width reset leakage is checked independently as well.
            width_error=float(np.max(np.abs(initial_inputs['state'][19:21]-reference_state[19:21])))
            if not nominal_gate(alignment) or width_error>.001:raise ValueError('Local reset not source-aligned/independent; no action execution')
            np.savez_compressed(out/(mode+'-initial.npz'),**initial_inputs)
            cup=env.env.task.object_scope['teacup.n.02_1'].unwrapped;cup0=array(cup.get_position_orientation()[0]).copy()
            episode={'mode':mode,'initial_state_load_count':1,'privileged_initialization':True,'initialization_source_frame':FRAME,
                'initial_body_and_finger_drive_target_writes':1,'initial_velocity_writes':1,'initial_robot_joint_pose_writes':0,
                'initial_alignment':alignment,'initial_gripper_width_max_error_m':width_error,
                'teacher_action_reads':STEPS if mode=='source_action_replay' else 0,
                'privileged_policy_inputs':False,'planner_enabled':False,
                'policy_profile':plan['policy_profile'] if mode=='learned_pi0_local_grasp' else None,
                'action_source':'official_source_action_replay' if mode=='source_action_replay' else 'original_pi0_real_RGB_proprio_inference',
                'runtime_motor_contract':snapshot_motor_contract(env.robot),'local_cup_grasp_report_seen':False,
                'assisted_constraint_seen':False,'max_cup_lift_m':0.,'object_pose_write_count':0,
                'task_success':False,'physical_task_acceptance':False,'steps':0,'aborted':False,
                'inference_calls':0,'clipped_action_steps':0,'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            result['episodes'].append(episode);save(stage=mode+'_in_progress')
            queue=deque();observations=[];actions=[];proposals=[];contacts=[];eef=[];cups=[];widths=[];chunks=[];roots=[]
            writer=imageio.get_writer(str(out/(mode+'.mp4')),fps=15,codec='libx264',quality=7,macro_block_size=2)
            error=None
            try:
                for step in range(STEPS):
                    audit.step=step
                    pose=np.r_[array(env.robot.get_position_orientation()[0]),array(env.robot.get_position_orientation()[1])];root_rotation_xy_and_yaw(pose)
                    if mode=='source_action_replay':proposed=replay_actions[step].copy()
                    else:
                        if not queue:
                            inputs={k:rgb_image(obs[v]) for k,v in cameras.items()};inputs.update(state=array(obs['prop_state']).copy(),prompt=PROMPT)
                            chunk,info=infer_local(inputs,token,port=18732,timeout=600)
                            if info.get('policy_profile')!=plan['policy_profile'] or info.get('action_representation')!='native':raise ValueError('No profile/encoding fallback')
                            chunks.append(chunk.copy());queue.extend(chunk[:10]);episode['inference_calls']+=1
                            episode['serving_source_sha256']=info['serving_source_sha256']
                        proposed=queue.popleft().copy()
                    action=np.clip(proposed,lower,upper);episode['clipped_action_steps']+=int(np.any(action!=proposed))
                    observations.append(array(obs['prop_state']).copy());proposals.append(proposed);actions.append(action);roots.append(pose)
                    raw,*_=env.env.step(torch.tensor(action));obs,_=env.get_obs_IL(di=(raw,{}))
                    positions,counts,errors=read_grasp_telemetry(env.robot,cup)
                    position=array(cup.get_position_orientation()[0]).copy()
                    eef.append(positions);contacts.append(counts);cups.append(position);widths.append(array(obs['prop_state'])[19:21].copy())
                    if errors:episode['telemetry_errors']=sorted(set(episode.get('telemetry_errors',[])+errors))
                    episode['local_cup_grasp_report_seen']|=int(env.robot.is_grasping(arm='left',candidate_obj=cup))==1
                    episode['assisted_constraint_seen']|=any(v is not None for v in env.robot._ag_obj_constraints.values())
                    episode['max_cup_lift_m']=max(episode['max_cup_lift_m'],float(position[2]-cup0[2]))
                    episode['steps']=len(cups);episode['object_pose_write_count']=len(audit.events)
                    if audit.events:raise ValueError('Explicit task-object pose setter during local control')
                    if step%2==0:
                        primary=rgb_image(env.env._external_sensors['external_sensor2'].get_obs()[0]['rgb'])
                        frame=Image.new('RGB',(960,536),'black');frame.paste(Image.fromarray(primary).resize((480,480)),(0,28));frame.paste(Image.fromarray(rgb_image(obs[cameras['head_image']])).resize((480,480)),(480,28))
                        draw=ImageDraw.Draw(frame);draw.text((5,5),f'{mode.upper()} / ASSISTED / local step {step} / cup grasp {episode["local_cup_grasp_report_seen"]}',fill='white')
                        draw.text((5,515),'SOURCE635 PRIVILEGED INITIALIZATION - NOT FULL TASK OR PHYSICAL SUCCESS',fill='yellow');writer.append_data(np.asarray(frame))
                    if step%10==0:save()
            except BaseException as exc:
                error=exc;episode.update(aborted=True,error=repr(exc))
            finally:
                writer.close();writer=None;n=len(cups)
                if n:
                    np.savez_compressed(out/(mode+'-trace.npz'),observations=np.stack(observations[:n]),actions=np.stack(actions[:n]),
                        model_proposed_targets=np.stack(proposals[:n]),root_poses=np.stack(roots[:n]),
                        post_step_eef_world_poses=np.stack(eef),post_step_cup_contact_finger_counts=np.stack(contacts),
                        post_step_cup_positions=np.stack(cups),post_step_gripper_widths=np.stack(widths),
                        predicted_chunks=np.stack(chunks) if chunks else np.empty((0,50,21),np.float32))
                episode['object_pose_write_count']=len(audit.events);audit.close();audit=None;save()
            del replay_actions
            if error:raise error
            if mode=='source_action_replay' and not replay_gate(episode):
                save(stage='source_local_replay_gate_failed_NO_learned_case');return
        save(stage='completed_local_grasp_diagnostic_NOT_full_task_success')
    except BaseException as exc:
        save(stage='local_initialization_or_control_aborted_NOT_full_task',error=repr(exc));raise
    finally:
        if audit:audit.close()
        if writer:writer.close()
        try:og.shutdown()
        except SystemExit:pass


def export(c):
    result=read(c/'local/result.json');out=c/'review';out.mkdir();records=[]
    from pi0_review_contract import fingerprint_checkpoint
    fingerprint=fingerprint_checkpoint(Path(read(c/'plan.json')['checkpoint']))
    write(out/'checkpoint-fingerprint.json',fingerprint)
    for episode in result['episodes']:
        mode=episode['mode'];source=c/'local'/(mode+'.mp4')
        if episode.get('steps',0)<2 or not source.exists():continue
        destination=out/(mode+'.mp4');label=review_label(episode)
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(source),'-vf',"drawbox=x=0:y=ih-30:w=iw:h=30:color=black:t=fill,drawtext=text='"+label+"':x=5:y=h-22:fontsize=12:fontcolor=yellow",'-c:v','libx264','-preset','medium','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(destination)],check=True,timeout=120)
        subprocess.run(['ffmpeg','-v','error','-i',str(destination),'-f','null','-'],check=True,timeout=120)
        records.append({'label':label,'video':str(destination),'video_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'episode':episode})
    write(out/'review.json',{'scope':'PRIVILEGED INIT635 LOCAL DIAGNOSTIC ONLY; NOT full Tidy Table/heldout/physical success',
        'task_success':False,'physical_task_acceptance':False,'formal_training_completed':False,'training_is_diagnostic':True,
        'checkpoint_fingerprint':fingerprint['sha256'],'checkpoint':fingerprint['checkpoint'],'checkpoint_optimizer_updates':20000,
        'optimizer_updates_this_job':0,'source_replay_is_not_learned_policy':True,'records':records})


def controller(c):
    import fcntl
    import signal
    import secrets
    import time
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import process_alive
    from pi0_relative_full_profile import load_contract,PROFILE
    if c.exists() or not c.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh remote campaign')
    gpu=resolve_gpu(ROOT,'1');lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(ROOT/'runs/latest-gpu1-job.json');prior=read(old['status_file'])
    if prior.get('stage')!='completed_consistent_sensor_feedback_NOT_policy_task_success' or prior.get('active_pid')!=0 or process_alive(old.get('controller_pid')):raise ValueError('Prior complete sensor probe must exit')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(row.split(',')[0].strip()==gpu for row in apps.splitlines()):raise ValueError('GPU1 occupied, no fallback')
    if (ROOT/'runs/USER_PAUSED.json').exists():raise ValueError('User paused')
    training_dir=Path((ROOT/'runs/latest-pi0-relative-full-demo-workflow.txt').read_text().strip());trained=read(training_dir/'training-report.json');cp=Path(trained['checkpoint']);contract=load_contract(cp)
    c.mkdir();script=Path(__file__).resolve();(c/script.name).write_bytes(script.read_bytes())
    plan={'checkpoint':str(cp),'policy_profile':PROFILE,'source':trained['source'],'initialization_source_frame':FRAME,'max_steps_each_case':STEPS,
        'action_bounds':str(training_dir/'train-author-diagnostic/action-bounds.json'),'checkpoint_full_parameter_sha256':contract['full_parameter_sha256'],
        'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'sampling':'unchanged Gaussian10 / execute10; independent replay has no model calls',
        'scope':'replay validity gate -> actual learned local case, both privileged-source initialization; NOT full task'}
    write(c/'plan.json',plan);write(c/'previous-owner.json',old)
    token=c/'rpc-token.local';token.write_text(secrets.token_urlsafe(48));token.chmod(0o600)
    status={'stage':'starting_local_grasp_diagnostic','controller_pid':os.getpid(),'gpu_uuid':gpu,'active_pid':0,
            'policy_training':'not_run','task_success':False,'paper_reproduction_complete':False,'privileged_initialization':True}
    def save(**updates):status.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(c/'status.json',status)
    save();pointer={'campaign':str(c),'status_file':str(c/'status.json'),'controller_pid':os.getpid(),'gpu_uuid':gpu,'max_gpu_count':1,'role':'SOURCE635_LOCAL_replay_gate_then_learned_original_pi0_ASSISTED_NOT_full_task'}
    write(ROOT/'runs/latest-gpu1-job.json',pointer);mapping=read(ROOT/'runs/latest-multi-gpu-jobs.json');mapping['completed_consistent_feedback_probe']=old;mapping['GPU1']=pointer;mapping['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',mapping)
    (ROOT/'runs/latest-pi0-local-grasp-probe.txt').write_text(str(c)+'\n')
    env={**os.environ,'PYTHONPATH':str(ROOT/'code/scripts')+':'+str(ROOT/'src/openpi-215abfb/src'),'OPENPI_DATA_HOME':str(ROOT/'cache/openpi'),
        'HF_HOME':str(ROOT/'cache/huggingface'),'HF_LEROBOT_HOME':str(ROOT/'data/pi0/lerobot'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    env.pop('PI0_INITIAL_PARAMS',None);server=None;child=None
    def interrupted(*_):raise TimeoutError('Bounded local diagnostic watchdog')
    signal.signal(signal.SIGTERM,interrupted)
    def stop(process):
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        with (c/'inference.log').open('x') as log:
            server=subprocess.Popen([str(ROOT/'envs/openpi/bin/python'),'-u',str(ROOT/'code/scripts/serve_pi0_relative_full_r1.py'),
                '--checkpoint',str(cp),'--ready-file',str(c/'inference-ready.json'),'--token-file',str(token),
                '--port','18732','--hours','1','--inference-platform','cpu'],env={**env,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(stage='loading_original_pi0_CPU',server_pid=server.pid)
        deadline=time.monotonic()+1200
        while not (c/'inference-ready.json').exists():
            if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('CPU model did not become ready')
            time.sleep(5)
        ready=read(c/'inference-ready.json')
        if ready.get('policy_profile')!=PROFILE or ready.get('norm_stats_sha256')!=contract['norm_stats_sha256']:raise ValueError('Wrong model profile/norm')
        # Compile only; this is a previously RECAPTURED actual sensor snapshot,
        # never a replacement for current cameras during learned actions.
        import numpy as np
        from momagen_pi0_schema import PROMPT
        from pi0_rpc import infer_local
        snapshot=Path(old['campaign'])/'snapshots/frame635_nominal.npz'
        with np.load(snapshot) as file:warm={k:file[k].copy() for k in file.files}
        warm['prompt']=PROMPT;infer_local(warm,token.read_text().strip(),port=18732,timeout=600)
        save(warmup_scope='prior_actual_snapshot_NO_actions_NOT_live_input',stage='source_replay_gate_then_local_learned_case')
        with (c/'simulation.log').open('x') as log:
            child=subprocess.Popen(['bash','-c','source "$1"; shift; exec "$@"','local-env',str(ROOT/'code/scripts/server_env.sh'),
                str(ROOT/'envs/momagen/bin/python'),'-u',str(script),'--campaign',str(c),'--simulate','--parent-pid',str(os.getpid())],
                env={**env,'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cpu'},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(active_pid=child.pid);code=child.wait(timeout=2400)
        save(active_pid=0,simulation_exit_status=code);stop(server)
        if (c/'local/result.json').exists():export(c)
        result=read(c/'local/result.json')
        if code or result.get('stage')!='completed_local_grasp_diagnostic_NOT_full_task_success':
            save(stage='local_diagnostic_stopped_CHECK_replay_or_setup',result_stage=result.get('stage'));return
        learned=next(x for x in result['episodes'] if x['mode']=='learned_pi0_local_grasp')
        save(stage='completed_local_grasp_DIAGNOSTIC_NOT_full_task',local_learned_cup_grasp_report_seen=learned['local_cup_grasp_report_seen'],
            local_learned_assisted_constraint_seen=learned['assisted_constraint_seen'],local_learned_max_cup_lift_m=learned['max_cup_lift_m'],review_directory=str(c/'review'))
    except BaseException as error:
        save(stage='local_grasp_probe_setup_or_control_failed_NOT_full_task',error=repr(error));raise
    finally:stop(child);stop(server)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--simulate',action='store_true');p.add_argument('--parent-pid',type=int);a=p.parse_args()
    if a.simulate:simulate(a.campaign,a.parent_pid)
    else:controller(a.campaign)
