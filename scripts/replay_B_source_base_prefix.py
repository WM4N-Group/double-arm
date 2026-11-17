#!/usr/bin/env python3
"""SERVER bounded official OFFLINE HYBRID replay: source base prefix + recorded B body/arms/grippers.

Tests one base-prefix intervention after exact B tape drop reproduction; no model calls.
Single original frame0 initialization; no per-step restore, no policy inference,
no planner, no physics tuning. Both-USD-arm assistance checks at physics refresh
and action boundaries; not every internal physics substep.
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


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def completed_hybrid(code,result):
    return (code==0 and result.get('stage')=='hybrid_replay_completed_NOT_live_policy'
        and result.get('aborted') is False and not result.get('error')
        and result.get('hybrid_actions_executed')==1200
        and result.get('no_assistance_all_checked_samples_confirmed') is True)


def simulate(c,parent):
    if os.getppid()!=parent or read(c/'status.json')['controller_pid']!=parent:raise ValueError('Owning parent required')
    from resource_allocation import resolve_gpu
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=resolve_gpu(ROOT,'1'):raise ValueError('Explicit GPU1 only')
    flags={}
    for line in (ROOT/'code/config/licenses.local.env').read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);flags[k.strip()]=v.strip().strip('"\x27')
    if any(os.environ.get(k,flags.get(k))!='YES' for k in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS')):raise ValueError('Existing explicit license confirmations required')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    import h5py
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from grasping_validation import activate_physical_after_reset
    from physical_grasp_observation import snapshot_no_assistance,no_assistance_confirmed,FIELDS,ARMS
    from momagen_pi0_schema import camera_keys,rgb_image
    from object_pose_audit import ObjectPoseAudit
    from pi0_base_targets import root_rotation_xy_and_yaw
    from grasp_telemetry import read_grasp_telemetry
    from motor_contract import snapshot_motor_contract
    from evaluation_split_contract import verify_evaluation_episode
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    plan=read(c/'plan.json');entry=plan['source'];source=Path(entry['source'])
    if hashlib.sha256((ROOT/'code/scripts/diagnostic_base_prefix.py').read_bytes()).hexdigest()!=plan['intervention_source_sha256']:raise ValueError('In-flight intervention helper changed')
    verify_evaluation_episode(read(ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'),source,entry['source_demo'],'train')
    with h5py.File(source) as file:
        g=file['data/'+entry['source_demo']];actions=g['actions'][:].astype(np.float32);state=g['states'][0]
        reference_state=g['obs/prop_state'][0];keys=camera_keys(g['obs']);source_rgb={k:rgb_image(g['obs'][v][0]) for k,v in keys.items()}
        meta=json.loads(file['data'].attrs['env_args'])
    if hashlib.sha256(actions.tobytes()+reference_state.tobytes()).hexdigest()!=entry['fingerprint'] or len(actions)!=1714:raise ValueError('Immutable source changed')
    # SOURCE BASE coordinates are an explicit OFFLINE control intervention.
    # No live model runs and no expert arm/gripper commands are executed.
    source_actions=actions.copy()
    tape=Path(plan['recorded_B_trace'])
    if not tape.resolve().is_relative_to(ROOT/'runs') or hashlib.sha256(tape.read_bytes()).hexdigest()!=plan['recorded_B_trace_sha256']:
        raise ValueError('Recorded failed B trace changed')
    with np.load(tape) as recorded:actions=recorded['actions'].copy()
    if actions.shape!=(3000,21) or actions.dtype!=np.float32:raise ValueError('Actual 3000 native B commands required')
    from diagnostic_base_prefix import replace_initial_base_only
    actions=replace_initial_base_only(actions,source_actions)
    meta['env_kwargs']['init_curobo']=False
    bounds=read(Path(plan['action_bounds']));lo=np.asarray(bounds['min']);hi=np.asarray(bounds['max'])
    if not np.isfinite(actions).all() or (actions<lo-1e-6).any() or (actions>hi+1e-6).any():raise ValueError('Source outside unchanged native limits; no action modification')
    result={'mode':'OFFLINE_HYBRID_SOURCE_BASE_PREFIX_B_OTHER_AXES_NOT_LIVE_POLICY','source':entry,'policy_inference_calls':0,
        'hybrid_actions_executed':0,'expected_hybrid_actions':1200,'initial_state_load_count':1,
        'initial_source_frame':0,'per_step_state_restores':0,'planner_enabled':False,'grasping_mode':'physical',
        'source_grasping_mode':'assisted','action_source':'OFFLINE_source_base_actions_steps0to681_plus_B_body_arms_grippers_all1200',
        'source_expert_base_commands_executed':0,'source_expert_arm_gripper_commands_executed':0,
        'base_prefix_replaced_steps':682,'hybrid_actions_sha256':hashlib.sha256(actions.tobytes()).hexdigest(),
        'recorded_B_trace_sha256':plan['recorded_B_trace_sha256'],'learned_policy_task_success':False,'physical_parameters_calibrated_to_real_robot':False,
        'upstream_finger_friction':4.0,'object_pose_write_count':0,'reset_physics_steps':0,
        'stage':'initializing_simulator','aborted':False,'replay_inside_goal_sustained':False,
        'task_success_claim':False,'no_assistance_observation_scope':'before/after every control action and after each of20refresh steps; NOT every internal physics substep',
        'no_assistance_checked_samples':0,'no_assistance_all_checked_samples_confirmed':False}
    out=c/'replay';out.mkdir()
    def save(**updates):result.update(updates);write(out/'result.json',result)
    save();writer=None;audit=None;observations=[];cups=[];eef=[];contacts=[];grasp=[];roots=[];ag_samples=[];goals=[]
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    def check_assistance(robot,phase,step):
        snapshot=snapshot_no_assistance(robot,lambda path:og.sim.stage.GetPrimAtPath(path).IsValid(),
            inactive_freeze_none_source=ROOT/'src/MoMaGen/BEHAVIOR-1K/OmniGibson/omnigibson/robots/manipulation_robot.py')
        result['no_assistance_checked_samples']+=1
        if result['no_assistance_checked_samples']==1:write(out/'initial-no-assistance-evidence.json',snapshot)
        if not no_assistance_confirmed(snapshot):
            write(out/('assistance-rejected-'+str(result['no_assistance_checked_samples'])+'.json'),snapshot)
            raise ValueError('Physical/AG Python/USD absence not confirmed at '+phase+' '+str(step))
        ag_samples.append([snapshot[f][a] for f in FIELDS for a in ARMS])
    error=None
    try:
        env=EnvOmniGibson(env_name=meta['env_name'],**meta['env_kwargs']);env.env.reset()
        if env.robot.controllers['base'].motor_type!='position':raise ValueError('Native base POSITION required')
        if int(og.sim.dump_state(serialized=True).numel())!=len(state):raise ValueError('Serialized layout mismatch')
        og.sim.load_state(torch.tensor(state,dtype=torch.float32),serialized=True)
        # ONLY original empty-handed source0 reset, never a held/mid-demo state.
        activate_physical_after_reset(env.robot);check_assistance(env.robot,'initial',0)
        result['runtime_motor_contract']=snapshot_motor_contract(env.robot)
        audit=ObjectPoseAudit(env.env.task.object_scope)
        for i in range(20):
            og.sim.step();result['reset_physics_steps']+=1;check_assistance(env.robot,'refresh',i)
            if audit.events:raise ValueError('Object pose setter during reset refresh')
        for _ in range(3):og.sim.render()
        env.env.task._termination_conditions['predicate'].reset(env.env.task,env.env)
        if env.is_success()['task']:raise ValueError('Goal already satisfied at source0')
        obs,_=env.get_obs_IL();cameras=camera_keys(obs)
        inputs={k:rgb_image(obs[v]) for k,v in cameras.items()};inputs['state']=array(obs['prop_state']).copy();np.savez_compressed(out/'initial-observation.npz',**inputs)
        alignment={k:float(np.abs(inputs[k].astype(float)-source_rgb[k]).mean()) for k in keys}
        qerr=float(np.max(np.abs(inputs['state'][3:19]-reference_state[3:19])))
        if any(v>20 for v in alignment.values()) or qerr>.01:raise ValueError('Source0 physical reset sensor misalignment')
        result.update(initial_rgb_mae_0_255=alignment,initial_body_q_max_abs_rad=qerr)
        cup=env.env.task.object_scope['teacup.n.02_1'].unwrapped;cup0=array(cup.get_position_orientation()[0]).copy()
        writer=imageio.get_writer(str(out/'hybrid-replay.mp4'),fps=15,codec='libx264',quality=7,macro_block_size=2)
        stable=0;max_stable=0;max_lift=0.;grasp_seen=False
        for step,action in enumerate(actions):
            if (ROOT/'runs/USER_PAUSED.json').exists():raise RuntimeError('User paused')
            audit.step=step;check_assistance(env.robot,'pre_action',step)
            root=np.r_[array(env.robot.get_position_orientation()[0]),array(env.robot.get_position_orientation()[1])];root_rotation_xy_and_yaw(root)
            observation=array(obs['prop_state']).copy()
            raw,*_=env.env.step(torch.tensor(action));obs,_=env.get_obs_IL(di=(raw,{}))
            check_assistance(env.robot,'post_action',step)
            ep,cc,errors=read_grasp_telemetry(env.robot,cup);position=array(cup.get_position_orientation()[0]).copy()
            gflag=int(env.robot.is_grasping(arm='left',candidate_obj=cup))==1;goal=bool(env.is_success()['task'])
            observations.append(observation);roots.append(root);cups.append(position);eef.append(ep);contacts.append(cc);grasp.append(gflag);goals.append(goal)
            if errors:result['telemetry_errors']=sorted(set(result.get('telemetry_errors',[])+errors))
            stable=stable+1 if goal else 0;max_stable=max(max_stable,stable);max_lift=max(max_lift,float(position[2]-cup0[2]));grasp_seen|=gflag
            result.update(hybrid_actions_executed=len(cups),source_expert_base_commands_executed=min(len(cups),682),max_cup_lift_m=max_lift,cup_grasp_report_seen=grasp_seen,
                max_consecutive_inside_goal_frames=max_stable,replay_inside_goal_sustained=max_stable>=30,
                object_pose_write_count=len(audit.events),final_goal_status=str(env.env.task._termination_conditions['predicate'].goal_status))
            if audit.events:raise ValueError('Explicit task-object pose setter in physical replay')
            if step%2==0:
                frame=Image.new('RGB',(960,536),'black');primary=rgb_image(env.env._external_sensors['external_sensor2'].get_obs()[0]['rgb'])
                frame.paste(Image.fromarray(primary).resize((480,480)),(0,28));frame.paste(Image.fromarray(rgb_image(obs[cameras['head_image']])).resize((480,480)),(480,28))
                draw=ImageDraw.Draw(frame);draw.text((5,5),f'OFFLINE HYBRID SOURCE-BASE PREFIX + B / PHYSICAL / step {step} / inside {goal}',fill='white')
                draw.text((5,515),'OFFLINE HYBRID REPLAY - NO LIVE PI0 / NOT TASK SUCCESS',fill='yellow');writer.append_data(np.asarray(frame))
            if step%100==0:save(stage='hybrid_replay_in_progress');print(json.dumps({'step':step,'cup_grasp':gflag,'lift_m':max_lift,'inside_goal':goal}),flush=True)
        result['no_assistance_all_checked_samples_confirmed']=True;save(stage='hybrid_replay_completed_NOT_live_policy')
    except BaseException as exc:
        error=exc;save(stage='hybrid_replay_aborted_NOT_live_policy',aborted=True,error=repr(exc))
    finally:
        if writer:writer.close()
        n=len(cups)
        if n:
            np.savez_compressed(out/'trace.npz',observations=np.stack(observations),actions=actions[:n],root_poses=np.stack(roots),
                post_step_cup_positions=np.stack(cups),post_step_eef_world_poses=np.stack(eef),post_step_cup_contact_finger_counts=np.stack(contacts),
                post_step_cup_grasp_flags=np.asarray(grasp),inside_goal_flags=np.asarray(goals),initial_cup_position=cup0,
                no_assistance_observations=np.asarray(ag_samples,dtype=np.int8))
        save()
        if audit:audit.close()
        try:og.shutdown()
        except SystemExit:pass
    if error:raise error


def export(c):
    r=read(c/'replay/result.json');p=c/'replay/hybrid-replay.mp4'
    if not p.exists() or r['hybrid_actions_executed']<2:return
    out=c/'review';out.mkdir();label=('ABORTED' if r['aborted'] else 'HYBRID REPLAY INSIDE GOAL OBSERVED' if r['replay_inside_goal_sustained'] else 'HYBRID LOCAL REPLAY')+' - PHYSICAL - NO LIVE POLICY OR TASK CLAIM'
    target=out/'demo.mp4'
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(p),'-vf',"drawbox=x=0:y=ih-30:w=iw:h=30:color=black:t=fill,drawtext=text='"+label+"':x=6:y=h-22:fontsize=14:fontcolor=yellow",'-c:v','libx264','-preset','medium','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(target)],check=True,timeout=300)
    subprocess.run(['ffmpeg','-v','error','-i',str(target),'-f','null','-'],check=True,timeout=120)
    write(out/'review.json',{'label':label,'mode':r['mode'],'learned_policy_task_success':False,'result':r,
        'video_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'trace_sha256':hashlib.sha256((c/'replay/trace.npz').read_bytes()).hexdigest()})


def controller(c):
    import fcntl
    import signal
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import process_alive
    if c.exists() or not c.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh project campaign')
    gpu=resolve_gpu(ROOT,'1');lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(ROOT/'runs/latest-gpu1-job.json');state=read(old['status_file'])
    previous=Path(old['campaign'])
    if (state.get('stage')!='completed_recorded_B_physical_replay_NOT_live_policy' or state.get('replay_completed') is not True
        or state.get('active_pid')!=0 or process_alive(old.get('controller_pid'))):raise ValueError('Whole recorded-B tape workflow must complete/exit')
    audit=read(previous/'replay-equivalence.json');prior=read(previous/'replay/result.json')
    if (audit.get('all_six_arrays_exact_equal') is not True or prior.get('recorded_B_actions_executed')!=3000
        or prior.get('no_assistance_all_checked_samples_confirmed') is not True):raise ValueError('Exact B drop replay and no-assistance evidence required')
    baseline=Path(read(previous/'plan.json')['paired_learned_B_campaign'])
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(row.split(',')[0].strip()==gpu for row in apps.splitlines()):raise ValueError('GPU1 occupied, no fallback')
    if (ROOT/'runs/USER_PAUSED.json').exists():raise ValueError('User paused')
    c.mkdir();script=Path(__file__).resolve();(c/script.name).write_bytes(script.read_bytes())
    training=read(baseline/'training-report.json')
    write(c/'plan.json',{'source':training['source'],'action_bounds':str(baseline/'train-physical-diagnostic/action-bounds.json'),
        'paired_learned_B_campaign':str(baseline),'recorded_B_equivalence_campaign':str(previous),
        'intervention':'EXPLICIT OFFLINE ONLY replace base axes0:3 in steps0:682 from original source; retain all B body/arm/gripper targets, stop1200',
        'intervention_source_sha256':hashlib.sha256((ROOT/'code/scripts/diagnostic_base_prefix.py').read_bytes()).hexdigest(),
        'recorded_B_trace':str(baseline/'train-physical-diagnostic/physical-rollout/demo_0-trace.npz'),
        'recorded_B_trace_sha256':'170050b3e0d0b6fcc59f01c0fc1eb490c66b9117346cc49dade996d5dcd09f3c','source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'max_actions':1200,'policy_training':'not_run'})
    write(c/'previous-owner.json',old)
    status={'stage':'starting_hybrid_base_prefix_replay','controller_pid':os.getpid(),'active_pid':0,'gpu_uuid':gpu,
            'policy_inference_calls':0,'policy_training':'not_run','learned_policy_task_success':False}
    def save(**updates):status.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(c/'status.json',status)
    save();pointer={'campaign':str(c),'status_file':str(c/'status.json'),'controller_pid':os.getpid(),'gpu_uuid':gpu,'max_gpu_count':1,'role':'OFFLINE_HYBRID_BASE_PREFIX_INTERVENTION_NO_MODEL'}
    write(ROOT/'runs/latest-gpu1-job.json',pointer);m=read(ROOT/'runs/latest-multi-gpu-jobs.json');m['completed_recorded_B_exact_replay']=old;m['GPU1']=pointer;m['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',m)
    (ROOT/'runs/latest-B-base-prefix-diagnostic.txt').write_text(str(c)+'\n')
    child=None
    def abort(*_):raise TimeoutError('Bounded physical source replay watchdog')
    signal.signal(signal.SIGTERM,abort)
    try:
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cpu','PYTHONPATH':str(ROOT/'code/scripts'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        with (c/'simulation.log').open('x') as log:
            child=subprocess.Popen(['bash','-c','source "$1"; shift; exec "$@"','replay-env',str(ROOT/'code/scripts/server_env.sh'),str(ROOT/'envs/momagen/bin/python'),'-u',str(script),'--campaign',str(c),'--simulate','--parent-pid',str(os.getpid())],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(stage='hybrid_base_prefix_replay_in_progress',active_pid=child.pid);code=child.wait(timeout=2400)
        save(active_pid=0,simulation_exit_status=code)
        if (c/'replay/result.json').exists():export(c)
        result=read(c/'replay/result.json')
        completed=completed_hybrid(code,result)
        save(stage='completed_hybrid_base_prefix_replay_NOT_live_policy' if completed else 'hybrid_base_prefix_replay_failed_or_aborted_NOT_live_policy',
             actual_hybrid_commands=result.get('hybrid_actions_executed',0),replay_completed=completed,
             replay_inside_goal_sustained=result.get('replay_inside_goal_sustained',False),
             result_stage=result.get('stage'),result_error=result.get('error'))
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save(stage='hybrid_base_prefix_replay_orchestration_failed',active_pid=0,error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--simulate',action='store_true');p.add_argument('--parent-pid',type=int);a=p.parse_args()
    if a.simulate:simulate(a.campaign,a.parent_pid)
    else:controller(a.campaign)
