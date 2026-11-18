#!/usr/bin/env python3
"""Run the frozen B action prefix in two fresh processes; never a live policy.

Server only. GPU1 must be released by the official auxiliary generation
workflow. This does not wait, pause, preempt or migrate any GPU task.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path('/mnt/yuanyue/double-arm')
RUNS = ROOT / 'runs'
SCRIPTS = ROOT / 'code/scripts'
PLAN_SHA = 'dcdba8950dddffb3c44d1b7c436ed5756080e4c20b4884740e2ea263f3c5805e'
PLAN = RUNS / 'B-assisted-tape-pair-preflight-20261118-112520/plan.json'
DONE_RAW = 'raw_official_pool_target_complete_alignment_and_split_pending'
MODES = ('assisted', 'physical')


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def run_path(path):
    p = Path(path).resolve()
    if not p.is_relative_to(RUNS):
        raise ValueError('Expected an in-project run path')
    return p


def release_blockers(owner, status, gpu, live, apps):
    return [reason for condition, reason in (
        (owner.get('role') != 'official_assisted_paper_data_generation_not_policy_training', 'unexpected GPU1 owner'),
        (owner.get('gpu_uuid') != gpu or owner.get('max_gpu_count') != 1, 'GPU1 allocation mismatch'),
        (status.get('stage') != DONE_RAW or status.get('error') or status.get('active_pid') != 0,
         'auxiliary raw workflow has not successfully finished in full'),
        (status.get('source_grasping_mode') != 'assisted' or status.get('policy_training') != 'not_run'
         or status.get('target_unique_demos') != 100 or type(status.get('verified_unique_demos')) is not int
         or status['verified_unique_demos'] < 100, 'auxiliary-100 acceptance incomplete'),
        (type(status.get('controller_pid')) is not int or status['controller_pid'] <= 0
         or owner.get('controller_pid') != status['controller_pid'], 'controller identity mismatch'),
        (any(pid in live for pid in (owner.get('controller_pid'), status.get('controller_pid'), status.get('active_pid'))),
         'prior controller or child alive'),
        (gpu in apps, 'GPU1 compute process still present'),
    ) if condition]


def compute_gpus():
    return set(subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).split())


def check_release(gpu):
    from preflight_pi0_accumulation_gpu import process_alive
    if (RUNS / 'USER_PAUSED.json').exists():
        raise ValueError('User paused project')
    owner = read(RUNS / 'latest-gpu1-job.json')
    campaign = run_path(owner['campaign'])
    if run_path(owner['status_file']) != campaign / 'status.json':
        raise ValueError('Prior GPU1 pointer inconsistent')
    s = read(campaign / 'status.json')
    pids = {p for p in (owner.get('controller_pid'), s.get('controller_pid'), s.get('active_pid')) if process_alive(p)}
    return owner, release_blockers(owner, s, gpu, pids, compute_gpus())


def inputs(plan):
    import h5py
    import numpy as np
    from compare_paired_grasp_replay import verify_tape
    from momagen_pi0_schema import camera_keys, native_actions, rgb_image
    tape = run_path(plan['tape'])
    if digest(tape.read_bytes()) != plan['tape_sha256']:
        raise ValueError('Original executed action trace changed')
    with np.load(tape, allow_pickle=False) as z:
        actions = verify_tape(plan, z['actions']).copy()
    bounds = run_path(plan['bounds']).read_bytes()
    if digest(bounds) != plan['bounds_sha256']:
        raise ValueError('B native limits changed')
    limits = json.loads(bounds)
    lo, hi = np.asarray(limits['min']), np.asarray(limits['max'])
    if (lo.shape != (21,) or hi.shape != (21,) or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or (actions < lo - 1e-6).any() or (actions > hi + 1e-6).any()):
        raise ValueError('Action prefix outside unchanged limits; do not clip in replay')
    entry = plan['source']
    if entry['split'] != 'train' or entry['source_demo'] != 'demo_0':
        raise ValueError('Original TRAIN/demo0 only')
    with h5py.File(run_path(entry['source']), 'r') as f:
        g = f['data'][entry['source_demo']]
        init = np.asarray(g['states'][0], np.float32)
        proprio = np.asarray(g['obs/prop_state'][0], np.float32)
        reference = {k: rgb_image(g['obs'][v][0]) for k,v in camera_keys(g['obs']).items()}
        expert = native_actions(g['actions'][()])  # only fingerprint check; never control
        meta = json.loads(f['data'].attrs['env_args'])
    if (init.ndim != 1 or not np.isfinite(init).all() or proprio.shape != (21,) or not np.isfinite(proprio).all()
            or digest(init.tobytes()) != plan['initial_state_float32_sha256']
            or len(expert) != entry['frames'] or digest(expert.tobytes()+proprio.tobytes()) != entry['fingerprint']
            or meta['env_name'] != 'r1_tidy_table_D0' or meta['env_kwargs']['robots'][0]['grasping_mode'] != 'assisted'
            or plan['unchanged_physics_friction'] != 4.0):
        raise ValueError('Frame0 initialization provenance changed')
    meta['env_kwargs']['init_curobo'] = False
    return actions, init, proprio, reference, meta


def simulate(c, mode, parent, gpu, *, load_expert=None):
    if os.getppid() != parent or read(c/'status.json')['controller_pid'] != parent or os.environ.get('CUDA_VISIBLE_DEVICES') != gpu:
        raise ValueError('Need authorized parent plus single pinned GPU1 UUID')
    if any(os.environ.get(k) != 'YES' for k in ('ACCEPT_NVIDIA_EULA','ACCEPT_BEHAVIOR_DATASET_TERMS')):
        raise ValueError('Require existing user-provided license confirmations')
    os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from compare_paired_grasp_replay import BOUNDARY_FIELDS, RESET
    from grasp_telemetry import read_grasp_telemetry
    from grasping_validation import activate_physical_after_reset, require_empty_assisted_grasps
    from momagen_pi0_schema import camera_keys, rgb_image
    from motor_contract import snapshot_motor_contract
    from object_pose_audit import ObjectPoseAudit
    from physical_grasp_observation import ARMS, snapshot_no_assistance
    from pi0_base_targets import root_rotation_xy_and_yaw

    if load_expert is not None:
        if mode != 'physical':
            raise ValueError('Expert pilot accepts physical mode only')
        actions, init, source_proprio, ref_images, meta = load_expert()
        label = 'EXPERT TAPE'
    else:
        frozen = (c/'plan.json').read_bytes()
        if digest(frozen) != PLAN_SHA:
            raise ValueError('Frozen plan hash mismatch')
        actions, init, source_proprio, ref_images, meta = inputs(json.loads(frozen))
        label = 'SAME B ACTION TAPE'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    out = c/mode
    out.mkdir()
    r = {'grasping_mode': None, 'requested_mode': mode, 'completed_actions': 0, 'initial_state_load_count': 0,
         'reset_physics_steps': 0, 'explicit_object_pose_writes': 0, 'policy_calls': 0, 'planner_enabled': False,
         'recorded_boundary_fields': list(BOUNDARY_FIELDS), 'aborted': True, 'stage': 'initializing',
         'kind': 'expert_tape_physical_not_live_policy' if load_expert else 'recorded_same_B_commands_not_live_policy',
         'learned_policy_task_success': False}
    def save(**updates):
        r.update(updates)
        write(out/'result.json', r)
    save()
    import omnigibson as og
    from robomimic.envs.env_omnigibson import EnvOmniGibson

    source_py = ROOT/'src/MoMaGen/BEHAVIOR-1K/OmniGibson/omnigibson/robots/manipulation_robot.py'
    boundaries, executed, cups, eefs, contacts = [], [], [], [], []
    grasps, inside, widths = [], [], []
    first = {}
    audit = writer = None
    error = None

    def array(x):
        return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)

    def pose(obj):
        p,q = obj.get_position_orientation()
        return np.r_[array(p),array(q)].copy()

    def observe(robot, cup, phase, step):
        snapshot = snapshot_no_assistance(robot, lambda p: og.sim.stage.GetPrimAtPath(p).IsValid(),
                                          inactive_freeze_none_source=source_py)
        valid = snapshot['grasping_mode'] == mode and not snapshot['errors']
        row = [snapshot[k][a] for k in ('constraint_active','object_in_hand','gripper_freeze','usd_ag_constraint_present') for a in ARMS]
        if not boundaries:
            write(out/'initial-assistance-observation.json', snapshot)
        # A non-task attachment in the assisted arm is not a comparable cup event.
        for arm, flag in zip(ARMS,row[2:4]):
            valid &= (flag == 0 or (flag == 1 and robot._ag_obj_in_hand[arm] is cup))
        if (not valid or not all(v in (0,1) for v in row) or row[:2] != row[6:]
                or ((mode == 'physical' or phase in ('initial','refresh')) and any(row))):
            write(out/'rejected-assistance-observation.json', {'phase':phase,'step':step,'snapshot':snapshot})
            raise ValueError('Invalid or unexpected Python/USD assistance observation')
        boundaries.append(row)

    try:
        env = EnvOmniGibson(env_name=meta['env_name'], **meta['env_kwargs'])
        env.env.reset()
        robot = env.robot
        if robot.grasping_mode != 'assisted' or robot.controllers['base'].motor_type != 'position':
            raise ValueError('Unexpected initial mode or native base controller')
        if int(og.sim.dump_state(serialized=True).numel()) != len(init):
            raise ValueError('Serialized initialization layout mismatch')
        og.sim.load_state(torch.tensor(init, dtype=torch.float32), serialized=True)
        r['initial_state_load_count'] = 1
        require_empty_assisted_grasps(robot)
        if mode == 'physical':
            activate_physical_after_reset(robot)
        r['grasping_mode'] = robot.grasping_mode
        cup = env.env.task.object_scope['teacup.n.02_1'].unwrapped
        audit = ObjectPoseAudit(env.env.task.object_scope)
        observe(robot,cup,'initial',0)
        r['runtime_motor_contract'] = snapshot_motor_contract(robot)
        for i in range(RESET):
            og.sim.step()
            r['reset_physics_steps'] += 1
            observe(robot,cup,'refresh',i)
        for _ in range(3):
            og.sim.render()
        env.env.task._termination_conditions['predicate'].reset(env.env.task,env.env)
        if audit.events or env.is_success()['task']:
            raise ValueError('Unexpected object write or already-satisfied goal at init')
        obs,_ = env.get_obs_IL()
        cameras = camera_keys(obs)
        first = dict(initial_proprio=array(obs['prop_state']).copy(), initial_root_pose=pose(robot),
                     initial_cup_pose=pose(cup), initial_left_eef_pose=pose(robot.eef_links['left']))
        image_mae = {k: float(np.abs(rgb_image(obs[v]).astype(float)-ref_images[k]).mean()) for k,v in cameras.items()}
        qerr = float(np.max(np.abs(first['initial_proprio'][3:19]-source_proprio[3:19])))
        if any(x>20 for x in image_mae.values()) or qerr>.01:
            raise ValueError('Post-refresh observations misaligned with source0')
        save(stage='replay_in_progress', initial_rgb_mae_0_255=image_mae, initial_body_q_max_abs_rad=qerr)
        writer = imageio.get_writer(str(out/'view.mp4'),fps=15,codec='libx264',quality=7,macro_block_size=2)
        for step,action in enumerate(actions):
            if (RUNS/'USER_PAUSED.json').exists():
                raise RuntimeError('User paused project')
            root_rotation_xy_and_yaw(pose(robot))
            audit.step = step
            observe(robot,cup,'pre',step)
            raw,*_ = env.env.step(torch.tensor(action,dtype=torch.float32))
            executed.append(action.copy())
            r['completed_actions'] = len(executed)
            obs,_ = env.get_obs_IL(di=(raw,{}))
            observe(robot,cup,'post',step)
            _, counts, errors = read_grasp_telemetry(robot,cup)
            if errors or audit.events:
                raise ValueError(f'Replay telemetry error or explicit object pose write: {errors}')
            cups.append(pose(cup)); eefs.append(pose(robot.eef_links['left'])); contacts.append(int(counts[0]))
            if load_expert is not None:
                grasps.append(int(robot.is_grasping(arm='left', candidate_obj=cup)) == 1)
                inside.append(bool(env.is_success()['task']))
                widths.append(float(array(obs['prop_state'])[19]))
            if step % 2 == 0:
                frame = Image.new('RGB',(960,536),'black')
                external = rgb_image(env.env._external_sensors['external_sensor2'].get_obs()[0]['rgb'])
                frame.paste(Image.fromarray(external).resize((480,480)),(0,28))
                frame.paste(Image.fromarray(rgb_image(obs[cameras['head_image']])).resize((480,480)),(480,28))
                text = ImageDraw.Draw(frame)
                text.text((5,5), f'{label} | {mode.upper()} | step {step}', fill='white')
                text.text((5,515),'OPEN LOOP COMMAND REPLAY - NOT LIVE PI0 OR TASK ACCEPTANCE',fill='yellow')
                writer.append_data(np.asarray(frame))
            if step % 100 == 0:
                save()
                print(mode,step,flush=True)
        writer.close()
        writer = None
        save(stage='replay_complete_NOT_live_policy',aborted=False)
    except BaseException as e:
        error = e
        save(stage='replay_aborted_NOT_live_policy',error=repr(e),aborted=True)
    finally:
        if writer:
            writer.close()
        if audit:
            save(explicit_object_pose_writes=len(audit.events))
            audit.close()
        np.savez_compressed(out/'trace.npz', **first, actions=np.asarray(executed,np.float32).reshape(-1,21),
                           post_step_cup_poses=np.asarray(cups).reshape(-1,7),
                           post_step_left_eef_poses=np.asarray(eefs).reshape(-1,7),
                           post_step_left_cup_contacts=np.asarray(contacts,np.int32),
                           grasp_boundaries=np.asarray(boundaries,np.int8).reshape(-1,8),
                           post_step_cup_grasp_flags=np.asarray(grasps,bool), inside_goal_flags=np.asarray(inside,bool),
                           post_step_left_gripper_width=np.asarray(widths,np.float32))
        og.shutdown()
    if error:
        raise error


def controller(c, gpu, plan):
    c = run_path(c)
    with (RUNS/f'gpu-{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old, reasons = check_release(gpu)
        if reasons:
            raise ValueError('; '.join(reasons))
        c.mkdir()
        (c/'plan.json').write_bytes(plan)
        write(c/'previous-owner.json',old)
        script = c/Path(__file__).name
        script.write_bytes(Path(__file__).read_bytes())
        owner = {'campaign':str(c), 'status_file':str(c/'status.json'), 'controller_pid':os.getpid(),
                 'gpu_uuid':gpu, 'max_gpu_count':1, 'role':'same_tape_attachment_replay_not_learned_policy'}
        s = {**owner,'stage':'starting_same_tape_replay','active_pid':0,'learned_policy_task_success':False}
        def save(**updates):
            s.update(updates,updated_at=datetime.now(timezone.utc).isoformat())
            write(c/'status.json',s)
        save()
        write(RUNS/'latest-gpu1-job.json',owner)
        m = read(RUNS/'latest-multi-gpu-jobs.json'); m['GPU1']=owner; m['last_completed_GPU1_raw_generation']=old
        m['updated_at']=datetime.now(timezone.utc).isoformat(); write(RUNS/'latest-multi-gpu-jobs.json',m)
        child = None
        def on_term(*_):
            raise TimeoutError('Replay controller interrupted')
        signal.signal(signal.SIGTERM,on_term)
        try:
            gpu_env = {**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cpu', 'PYTHONPATH':str(SCRIPTS),
                       'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
            cpu_env = {**gpu_env,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1'}
            for mode in MODES:
                if (RUNS/'USER_PAUSED.json').exists():
                    raise RuntimeError('User paused project')
                if gpu in compute_gpus():
                    raise RuntimeError('Unexpected live GPU1 process before starting a replay branch')
                with (c/f'{mode}.log').open('x') as log:
                    child = subprocess.Popen([
                        'taskset','-c','88-103','bash','-c','source "$1"; set -a; source "$2"; set +a; shift 2; exec "$@"',
                        'replay-env',str(SCRIPTS/'server_env.sh'),str(ROOT/'code/config/licenses.local.env'),
                        str(ROOT/'envs/momagen/bin/python'),'-u',str(script),'--campaign',str(c), '--simulate',mode,
                        '--parent',str(os.getpid())],env=gpu_env,stdin=subprocess.DEVNULL,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                    save(stage=f'{mode}_replay_in_progress',active_pid=child.pid)
                    if child.wait(timeout=2400) != 0:
                        raise RuntimeError(f'{mode} replay process failed')
                save(active_pid=0)
                result = read(c/mode/'result.json')
                if (result['stage'] != 'replay_complete_NOT_live_policy' or result['aborted'] is not False
                        or result['completed_actions'] != json.loads(plan)['planned_prefix_steps']):
                    raise ValueError(f'{mode} did not complete actual commands')
                child = None
            if read(c/'assisted/result.json')['runtime_motor_contract'] != read(c/'physical/result.json')['runtime_motor_contract']:
                raise ValueError('Runtime motor contracts differ between branches')
            subprocess.run([str(ROOT/'envs/openpi/bin/python'),str(SCRIPTS/'compare_paired_grasp_replay.py'),
                            '--plan',str(c/'plan.json'),'--case-root',str(c),'--output',str(c/'comparison.json')],
                           env=cpu_env,check=True,timeout=120)
            save(stage='completed_same_tape_replay_NOT_live_policy', comparison=str(c/'comparison.json'))
        except BaseException as e:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:
                    child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL)
                    child.wait()
            save(stage='same_tape_replay_failed_or_aborted',active_pid=0,error=repr(e))
            raise


def main():
    from resource_allocation import resolve_gpu
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--campaign',type=Path)
    p.add_argument('--simulate',choices=MODES)
    p.add_argument('--parent',type=int)
    a=p.parse_args()
    gpu=resolve_gpu(ROOT,'1')
    if a.simulate:
        simulate(run_path(a.campaign),a.simulate,a.parent,gpu)
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('Launch or preflight controller with CUDA_VISIBLE_DEVICES empty')
    plan=PLAN.read_bytes()
    if digest(plan)!=PLAN_SHA:
        raise ValueError('Original frozen plan changed')
    inputs(json.loads(plan))
    if a.check_only:
        if a.campaign or a.parent:
            raise ValueError('Check-only does not accept a launch campaign')
        _, reasons=check_release(gpu)
        print(json.dumps({'inputs_checked':True,'gpu_launch_performed':False,'gpu1_release_blockers':reasons}))
        return
    if a.campaign is None or a.parent:
        raise ValueError('Explicit fresh campaign required for an actual launch')
    controller(a.campaign,gpu,plan)


if __name__=='__main__':
    main()
