#!/usr/bin/env python3
"""SERVER: bounded consistent reset-observation probe, NOT a policy rollout.

Six independent single-state initializations, optional INITIAL robot-joint
perturbation/holding targets, then real physics/refreshed RGB+proprio. No teacher
or learned actions are executed. CPU inference occurs only after Isaac exits.
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
NAMES=('nominal','minus','plus')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def perturb_q(q,lower,upper,sign):
    import numpy as np
    q=np.asarray(q,dtype=np.float32);lower=np.asarray(lower);upper=np.asarray(upper)
    if q.shape!=(6,) or lower.shape!=(6,) or upper.shape!=(6,) or sign not in (-1,0,1):raise ValueError('Six actual joint coordinates and bounded sign required')
    if not all(np.isfinite(x).all() for x in (q,lower,upper)) or np.any(lower>=upper):raise ValueError('Finite unchanged native joint bounds required')
    # Perturb only INITIAL robot q; never relax native bounds or modify labels.
    return np.clip(q+sign*.02,lower,upper).astype(np.float32) if sign else q.copy()


def nominal_gate(row):
    import math
    values=list(row['source_rgb_mae_0_255'].values())
    return (len(values)==3 and all(math.isfinite(x) and x<=20 for x in values)
        and math.isfinite(row['source_body_q_max_abs_rad']) and row['source_body_q_max_abs_rad']<=.01
        and math.isfinite(row['root_translation_during_refresh_m']) and row['root_translation_during_refresh_m']<=.02
        and row['object_pose_write_count']==0 and row['assisted_constraint_seen'] is False)


def snapshots(c,parent):
    if os.getppid()!=parent or read(c/'status.json')['controller_pid']!=parent:raise ValueError('Owning parent required')
    from resource_allocation import resolve_gpu
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=resolve_gpu(ROOT,'1'):raise ValueError('Explicit UUID1 only')
    flags={}
    for line in (ROOT/'code/config/licenses.local.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);flags[k.strip()]=v.strip().strip('"\x27')
    if any(os.environ.get(k,flags.get(k))!='YES' for k in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS')):
        raise ValueError('Existing explicit license confirmations required; no new acceptance')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    import h5py
    import numpy as np
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    from momagen_pi0_schema import camera_keys,rgb_image
    from object_pose_audit import ObjectPoseAudit
    from pi0_base_targets import root_rotation_xy_and_yaw
    from grasp_telemetry import read_grasp_telemetry
    from evaluation_split_contract import verify_evaluation_episode
    plan=read(c/'plan.json');entry=plan['source'];source=Path(entry['source'])
    provenance=read(ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json')
    verify_evaluation_episode(provenance,source,entry['source_demo'],'train')
    with h5py.File(source) as file:meta=json.loads(file['data'].attrs['env_args'])
    meta['env_kwargs']['init_curobo']=False
    out=c/'snapshots';out.mkdir()
    result={'stage':'initializing_simulator','mode':'controlled_initialization_real_sensor_snapshots_ONLY',
        'policy_or_teacher_actions_executed':0,'inference_calls':0,'task_success':False,
        'initial_scene_state_loads':0,'initial_robot_joint_pose_writes':0,'initial_body_drive_target_writes':0,
        'initial_body_velocity_writes':0,'frames':[0,635],'physics_refresh_steps_per_condition':20,
        'condition_state_loads':1,'conditions':[],'source_grasping_mode':'assisted',
        'scope':'Snapshot INITIALIZATION intervention only; NOT full-task, physical-grasp validation, or closed-loop recovery'}
    def save(**changes):
        result.update(changes);write(out/'result.json',result)
    save()
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    audit=None
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    try:
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs']);env.env.reset()
        if env.robot.controllers['base'].motor_type!='position':raise ValueError('Native base POSITION contract required')
        indices=env.robot.arm_control_idx['left']
        if len(indices)!=6:raise ValueError('Expected R1 left six DOFs')
        body=torch.unique(torch.cat([torch.as_tensor(env.robot.controllers[name].dof_idx,dtype=torch.int64) for name in ('trunk','arm_left','arm_right')]))
        bounds=read(Path(plan['action_bounds']));lo=np.asarray(bounds['min'])[7:13];hi=np.asarray(bounds['max'])[7:13]
        with h5py.File(source) as file:
            group=file['data/'+entry['source_demo']];keys=camera_keys(group['obs'])
            for frame in (0,635):
                nominal=None
                for name,sign in zip(NAMES,(0,-1,1)):
                    state=group['states'][frame]
                    if int(og.sim.dump_state(serialized=True).numel())!=len(state):raise ValueError('Serialized layout mismatch')
                    og.sim.load_state(torch.tensor(state,dtype=torch.float32),serialized=True)
                    result['initial_scene_state_loads']+=1
                    q=array(env.robot.get_joint_positions()).copy();initial_left=q[array(indices).astype(int)].copy()
                    target=perturb_q(initial_left,lo,hi,sign)
                    if sign:
                        env.robot.set_joint_positions(torch.tensor(target),indices=indices,drive=False)
                        result['initial_robot_joint_pose_writes']+=1
                    # Common initial hold/zero-velocity setup for nominal AND
                    # perturbed conditions, explicitly counted, not policy acts.
                    held=env.robot.get_joint_positions()[body]
                    env.robot.set_joint_positions(held,indices=body,drive=True)
                    env.robot.set_joint_velocities(torch.zeros_like(held),indices=body,drive=False)
                    result['initial_body_drive_target_writes']+=1;result['initial_body_velocity_writes']+=1
                    before=np.r_[array(env.robot.get_position_orientation()[0]),array(env.robot.get_position_orientation()[1])]
                    audit=ObjectPoseAudit(env.env.task.object_scope)
                    for _ in range(20):og.sim.step()
                    for _ in range(3):og.sim.render()
                    obs,_=env.get_obs_IL();cameras=camera_keys(obs)
                    inputs={k:rgb_image(obs[v]) for k,v in cameras.items()};inputs['state']=array(obs['prop_state']).copy()
                    if inputs['state'].shape!=(21,) or not np.isfinite(inputs['state']).all():raise ValueError('Actual finite original21D state required')
                    after=np.r_[array(env.robot.get_position_orientation()[0]),array(env.robot.get_position_orientation()[1])]
                    root_rotation_xy_and_yaw(after)  # unchanged .98 planarity
                    cup=env.env.task.object_scope['teacup.n.02_1'].unwrapped
                    eef,contact,errors=read_grasp_telemetry(env.robot,cup)
                    path=out/f'frame{frame}_{name}.npz';np.savez_compressed(path,**inputs)
                    row={'frame':frame,'condition':name,'snapshot':str(path),'snapshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'left_q_before_intervention':initial_left.tolist(),'left_q_initial_target':target.tolist(),
                        'left_q_after_refresh':inputs['state'][7:13].tolist(),
                        'source_body_q_max_abs_rad':float(np.max(np.abs(inputs['state'][3:19]-group['obs/prop_state'][frame][3:19]))),
                        'source_rgb_mae_0_255':{k:float(np.abs(inputs[k].astype(float)-rgb_image(group['obs'][v][frame])).mean()) for k,v in keys.items()},
                        'root_pose':after.tolist(),'root_translation_during_refresh_m':float(np.linalg.norm(after[:3]-before[:3])),
                        'object_pose_write_count':len(audit.events),
                        'assisted_constraint_seen':any(v is not None for v in env.robot._ag_obj_constraints.values()),
                        'read_only_finger_contacts':contact.tolist(),'read_only_telemetry_errors':errors,
                        'read_only_left_eef_origin':eef[0,:3].tolist() if np.isfinite(eef[0,:3]).all() else None}
                    if name=='nominal':
                        nominal=inputs
                        row['nominal_initialization_gate']=nominal_gate(row)
                        if not row['nominal_initialization_gate']:
                            result['conditions'].append(row);save(stage='nominal_snapshot_alignment_gate_failed_NOT_policy_result')
                            raise ValueError('Nominal intermediate reset not aligned/stable; no scoring or bypass')
                    else:
                        delta=inputs['state'][7:13]-nominal['state'][7:13]
                        row['actual_left_q_delta_from_nominal_rad']=delta.tolist()
                        row['other_state_max_abs_change']=float(np.max(np.abs(np.delete(inputs['state']-nominal['state'],np.s_[7:13]))))
                        row['actual_rgb_mae_from_nominal']={k:float(np.abs(inputs[k].astype(float)-nominal[k]).mean()) for k in keys}
                        if (np.linalg.norm(delta)<.01 or np.max(np.abs(inputs['state'][7:13]-target))>.01
                            or row['object_pose_write_count'] or row['assisted_constraint_seen']):
                            result['conditions'].append(row);save(stage='perturbation_not_established_NOT_policy_result')
                            raise ValueError('Actual perturbation did not persist safely; no fake-input fallback')
                    audit.close();audit=None
                    result['conditions'].append(row);save(stage='collecting_real_sensor_initialization_conditions')
        save(stage='completed_consistent_sensor_snapshots_NOT_policy_control')
    except BaseException as error:
        save(error=repr(error));raise
    finally:
        if audit:audit.close()
        try:og.shutdown()
        except SystemExit:pass


def score(c,parent):
    if os.getppid()!=parent or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('Owning CPU-only parent required')
    import numpy as np
    from pi0_relative_full_profile import make_policy
    from momagen_pi0_schema import PROMPT
    from probe_pi0_relative_feedback import feedback_gain
    plan=read(c/'plan.json');shots=read(c/'snapshots/result.json')
    if shots['stage']!='completed_consistent_sensor_snapshots_NOT_policy_control':raise ValueError('No valid real-sensor set')
    policy,contract=make_policy(Path(plan['checkpoint']))
    predictions={};rows=[]
    for frame in (0,635):
        baseline={}
        for name in NAMES:
            row=next(x for x in shots['conditions'] if x['frame']==frame and x['condition']==name)
            path=Path(row['snapshot'])
            if not path.resolve().is_relative_to((c/'snapshots').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['snapshot_sha256']:raise ValueError('Actual sensor snapshot changed')
            with np.load(path) as file:inputs={key:file[key].copy() for key in file.files}
            inputs['prompt']=PROMPT
            for seed in (123,124,125):
                noise=np.random.default_rng(seed).standard_normal((50,32)).astype(np.float32)
                pred=np.asarray(policy.infer(inputs,noise=noise)['actions'])
                if pred.shape!=(50,21) or not np.isfinite(pred).all():raise ValueError('Invalid native output')
                predictions[f'frame{frame}_{name}_{seed}']=pred
                if name=='nominal':baseline[seed]=pred
                else:
                    rows.append({'frame':frame,'condition':name,'seed':seed,
                        **feedback_gain(baseline[seed],pred,np.asarray(row['actual_left_q_delta_from_nominal_rad'])),
                        'scope':'same noise, ACTUAL cameras/proprio after independent INITIAL perturbation; multiple modalities change; no unique corrective target',
                        'input_scope':'ACTUAL recaptured RGB and actual proprio after a robot INITIALIZATION perturbation; NOT fake q with source RGB',
                        'other_state_max_abs_change':row['other_state_max_abs_change'],
                        'actual_rgb_mae_from_nominal':row['actual_rgb_mae_from_nominal']})
    np.savez_compressed(c/'predictions.npz',**predictions)
    write(c/'score.json',{'stage':'completed_real_sensor_offline_feedback_scoring_NOT_task_success','samples':18,
        'checkpoint_full_parameter_sha256':contract['full_parameter_sha256'],'policy_or_teacher_actions_executed':0,
        'task_success':False,'contrasts':rows,
        'caveat':'Initialization snapshots, not a natural closed-loop perturbation or unique corrective target. Gain near1 is descriptive; no task/physical grasp acceptance or sole-cause proof.'})


def controller(c):
    import fcntl
    import signal
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import process_alive
    if c.exists() or not c.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh remote campaign required')
    gpu=resolve_gpu(ROOT,'1');lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(ROOT/'runs/latest-gpu1-job.json');dependency=read(old['status_file'])
    finalizer=read(Path((ROOT/'runs/latest-raw-pool-finalize.txt').read_text().strip())/'status.json')
    if (dependency.get('stage')!='raw_official_pool_target_complete_alignment_and_split_pending'
        or dependency.get('verified_unique_demos')!=5 or dependency.get('active_pid')!=0
        or finalizer.get('stage')!='completed_raw_pool_registry_CPU_QA' or finalizer.get('dependency')!=old['campaign']
        or any(process_alive(x) for x in (old.get('controller_pid'),finalizer.get('controller_pid')))):
        raise ValueError('Prior full raw5 + CPU QA have not exited; no preemption')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(row.split(',')[0].strip()==gpu for row in apps.splitlines()):raise ValueError('GPU1 occupied; no fallback')
    if (ROOT/'runs/USER_PAUSED.json').exists():raise ValueError('User paused')
    trained=Path((ROOT/'runs/latest-pi0-relative-full-demo-workflow.txt').read_text().strip())
    training=read(trained/'training-report.json')
    if training.get('training')!='completed' or training.get('actual_optimizer_updates')!=20000:raise ValueError('Require completed relative19999 diagnostic')
    c.mkdir();script=Path(__file__).resolve();(c/script.name).write_bytes(script.read_bytes())
    plan={'checkpoint':training['checkpoint'],'source':training['source'],
        'action_bounds':str(trained/'train-author-diagnostic/action-bounds.json'),'parent_pid':os.getpid(),
        'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'source_frames':[0,635],
        'diagnostic_initialization_only':True,'teacher_actions_executed':0,'policy_actions_executed':0}
    write(c/'plan.json',plan);write(c/'previous-owner.json',old)
    state={'stage':'starting_consistent_sensor_feedback_probe','controller_pid':os.getpid(),'gpu_uuid':gpu,
           'policy_training':'not_run','task_success':False,'paper_reproduction_complete':False,
           'policy_or_teacher_actions_executed':0,'active_pid':0}
    def save(**updates):state.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(c/'status.json',state)
    save()
    pointer={'campaign':str(c),'status_file':str(c/'status.json'),'controller_pid':os.getpid(),'gpu_uuid':gpu,'max_gpu_count':1,
             'role':'controlled_INITIAL_robot_perturbation_actual_sensor_snapshots_then_CPU_OFFLINE_NOT_policy_rollout'}
    write(ROOT/'runs/latest-gpu1-job.json',pointer)
    mapping=read(ROOT/'runs/latest-multi-gpu-jobs.json');mapping['completed_raw5_GPU1']=old;mapping['GPU1']=pointer
    mapping['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',mapping)
    (ROOT/'runs/latest-pi0-consistent-feedback-probe.txt').write_text(str(c)+'\n')
    child=None
    def abort(*_):raise TimeoutError('Bounded probe watchdog')
    signal.signal(signal.SIGTERM,abort)
    env={**os.environ,'PYTHONPATH':str(ROOT/'code/scripts')+':'+str(ROOT/'src/openpi-215abfb/src'),
         'OPENPI_DATA_HOME':str(ROOT/'cache/openpi'),'HF_HOME':str(ROOT/'cache/huggingface'),
         'HF_LEROBOT_HOME':str(ROOT/'data/pi0/lerobot'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
         'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    env.pop('PI0_INITIAL_PARAMS',None)
    try:
        commands=[('snapshots',['bash','-c','source "$1"; shift; exec "$@"','snapshot-env',str(ROOT/'code/scripts/server_env.sh'),
            str(ROOT/'envs/momagen/bin/python'),'-u',str(script),'--campaign',str(c),'--phase','snapshots','--parent-pid',str(os.getpid())],gpu),
            ('score',[str(ROOT/'envs/openpi/bin/python'),'-u',str(script),'--campaign',str(c),'--phase','score','--parent-pid',str(os.getpid())],'')]
        for phase,command,visible in commands:
            with (c/(phase+'.log')).open('x') as log:
                child=subprocess.Popen(command,env={**env,'CUDA_VISIBLE_DEVICES':visible,'JAX_PLATFORMS':'cpu'},
                    stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
                save(stage=phase+'_in_progress',active_pid=child.pid)
                code=child.wait(timeout=1200)
            save(active_pid=0)
            if code:raise RuntimeError(f'{phase} exited{code}; evidence retained, no automatic bypass')
        save(stage='completed_consistent_sensor_feedback_NOT_policy_task_success')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save(stage='consistent_sensor_probe_failed_NOT_learned_policy_result',active_pid=0,error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--phase',choices=('snapshots','score'));p.add_argument('--parent-pid',type=int);a=p.parse_args()
    if a.phase=='snapshots':snapshots(a.campaign,a.parent_pid)
    elif a.phase=='score':score(a.campaign,a.parent_pid)
    else:controller(a.campaign)
