#!/usr/bin/env python3
"""Queue the frozen eight expert commands on GPU1, strictly physical-only.

Waits on CPU for the entire auxiliary raw workflow to exit; never signals it.
Each case gets a fresh simulator, no learned policy or imitation update.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from resource_allocation import resolve_gpu
from run_B_paired_attachment_replay import (ROOT, RUNS, SCRIPTS, check_release, compute_gpus,
                                            digest, read, run_path, simulate, write)
from verify_expert_physical_pilot_inputs import PANEL_SHA, verify_bound_source

PANEL = RUNS/'expert-physical-pilot-20261118-122448/plan.json'
INPUTS = RUNS/'expert-physical-pilot-inputs-20261118-152505/report.json'
INPUTS_SHA = '26758174a7dea626ca74dcba4df29f3c3b3d80db1395e53454d32e4a27825a2d'


def frozen():
    panel_raw = PANEL.read_bytes()
    report_raw = INPUTS.read_bytes()
    if digest(panel_raw) != PANEL_SHA or digest(report_raw) != INPUTS_SHA:
        raise ValueError('Frozen pilot identities changed')
    panel, report = json.loads(panel_raw), json.loads(report_raw)
    audit = run_path(panel['audit']).read_bytes()
    if digest(audit) != panel['audit_sha256'] or report['panel_sha256'] != PANEL_SHA or len(panel['episodes']) != 8 or len(report['episodes']) != 8:
        raise ValueError('Frozen pilot/audit contract mismatch')
    return panel['episodes'], report['episodes'], json.loads(audit)['shards']


def load_case(index):
    eps, rows, shards = frozen()
    if not 1 <= index <= len(eps):
        raise ValueError('Episode index outside frozen panel')
    e, row = eps[index-1], rows[index-1]
    values = verify_bound_source(e, shards[e['shard']])
    a, init, _, images, meta = values
    if (e['fingerprint'] != row['fingerprint'] or digest(a.tobytes()) != row['original_actions_float32_sha256']
            or digest(init.tobytes()) != row['frame0_state_float32_sha256']
            or {k:digest(v.tobytes()) for k,v in images.items()} != row['frame0_rgb_sha256']
            or digest(json.dumps(meta,sort_keys=True).encode()) != row['env_config_canonical_sha256']):
        raise ValueError('Expert input bytes differ from CPU preflight')
    meta['env_kwargs']['init_curobo'] = False
    return values


def score(case, expected):
    """Strict completed-trace gate and separate, conservative release observation."""
    import numpy as np
    from compare_paired_grasp_replay import validate_case
    from motor_contract import runtime_binary_threshold
    d = case/'physical'
    r = read(d/'result.json')
    if r.get('kind') != 'expert_tape_physical_not_live_policy' or r.get('stage') != 'replay_complete_NOT_live_policy':
        raise ValueError('Not a completed physical-only expert case')
    with np.load(d/'trace.npz', allow_pickle=False) as t:
        initial, cup, _, contact, _ = validate_case(expected, 'physical', r, t)
        flags, inside, widths = t['post_step_cup_grasp_flags'], t['inside_goal_flags'], t['post_step_left_gripper_width']
        n = len(expected)
        if (any(q.shape != (n,) or q.dtype != np.bool_ for q in (flags,inside)) or widths.shape != (n,) or not np.isfinite(widths).all()):
            raise ValueError('Incomplete release evidence')
        threshold = runtime_binary_threshold(r['runtime_motor_contract'])
        if threshold is None:
            raise ValueError('Unknown runtime opening/closing convention')
        lift = cup[:,2]-initial['initial_cup_pose'][2]
        held = np.flatnonzero(flags & (contact > 0) & (lift > .03))
        opened = expected[:,13] >= threshold
        released = inside & (contact == 0) & ~flags & opened
        # Release requires measured opening relative to a preceding held sample,
        # then 30 frames of inside/no contact/open; inside alone is insufficient.
        released_from = None
        if len(held):
            for i in range(int(held[0])+1, n-29):
                previous = held[held < i]
                if (released[i:i+30].all() and len(previous)
                        and float(widths[i:i+30].max()-widths[previous[-1]]) >= .005):
                    released_from = i
                    break
        status = ('completed_release_stable_inside' if released_from is not None else
                  'no_contact' if not contact.any() else 'no_held_lift' if not len(held) else
                  'held_lift_no_verified_release_inside')
        return {'outcome':status,'actions':n,'first_held_lift_step':int(held[0]) if len(held) else None,
                'max_lift_m':float(lift.max()),'contact_steps':int(np.count_nonzero(contact)),
                'no_assistance_boundaries':int(len(t['grasp_boundaries'])),
                'cup_inside_ever':bool(inside.any()),'release_stable_inside_start_step':released_from,
                'final_contact_links':int(contact[-1]),'final_backend_grasp_report':bool(flags[-1]),
                'scope':'Fixed physical expert actions; release requires 30 observed frames and measured jaw opening; not a live policy or a hardware claim'}


def run(c, gpu, expected_owner):
    e, _, _ = frozen()
    status = {'campaign':str(c),'status_file':str(c/'status.json'),'controller_pid':os.getpid(),
              'gpu_uuid':gpu,'max_gpu_count':1,'role':'expert_physical_only_replay_not_learned_policy',
              'grasping_mode':'physical_only','auxiliary_grasp_allowed':False,'active_pid':0,
              'cases':[], 'policy_training':'not_run','policy_calls':0,'gpu_replay_launched':False}
    def save(**kw):
        status.update(kw,updated_at=datetime.now(timezone.utc).isoformat())
        write(c/'status.json', status)
    child = None
    def term(*_):
        raise InterruptedError('Physical replay queue interrupted')
    signal.signal(signal.SIGTERM,term)
    save(stage='waiting_for_existing_GPU1_generation_to_finish')
    try:
        deadline = time.monotonic()+32*3600
        while True:
            owner, blockers = check_release(gpu)
            if owner != expected_owner:
                raise ValueError('GPU1 owner changed; will not take over another workflow')
            old_status = read(run_path(owner['status_file']))
            if old_status.get('error') or old_status.get('stage') == 'generation_failed_or_interrupted':
                raise ValueError('Existing generation failed or was interrupted; manual review, no automatic takeover')
            if not blockers:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('GPU1 did not cleanly release by queue watchdog')
            save(release_blockers=blockers)
            time.sleep(30)
        with (RUNS/f'gpu-{gpu}.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            owner, blockers = check_release(gpu)
            if owner != expected_owner or blockers:
                raise ValueError('GPU1 release changed during lock acquisition')
            pointer = {k:status[k] for k in ('campaign','status_file','controller_pid','gpu_uuid','max_gpu_count','role')}
            write(c/'previous-owner.json',owner)
            save(stage='starting_physical_only_replay',release_blockers=[])
            write(RUNS/'latest-gpu1-job.json',pointer)
            m = read(RUNS/'latest-multi-gpu-jobs.json'); m['GPU1']=pointer; m['last_completed_GPU1_raw_generation']=owner
            m['updated_at']=datetime.now(timezone.utc).isoformat(); write(RUNS/'latest-multi-gpu-jobs.json',m)
            env = {**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cpu','PYTHONPATH':str(SCRIPTS),
                   'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
            for i,entry in enumerate(e,1):
                if (RUNS/'USER_PAUSED.json').exists() or gpu in compute_gpus():
                    raise RuntimeError('User pause or occupied GPU1 before physical-only case')
                case = c/f'case-{i:02d}'; case.mkdir()
                write(case/'status.json',{'controller_pid':os.getpid(),'plan_index':i})
                with (case/'simulation.log').open('x') as log:
                    child = subprocess.Popen(['taskset','-c','88-103','bash','-c',
                        'source "$1"; set -a; source "$2"; set +a; shift 2; exec "$@"',
                        'physical-expert',str(SCRIPTS/'server_env.sh'),str(ROOT/'code/config/licenses.local.env'),
                        str(ROOT/'envs/momagen/bin/python'),'-u',str(c/Path(__file__).name),
                        '--simulate',str(i),'--parent',str(os.getpid()),'--campaign',str(c)],
                        env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    save(stage='physical_only_case_running',active_pid=child.pid,case_index=i,gpu_replay_launched=True)
                    code = child.wait(timeout=2400)
                child = None
                save(active_pid=0)
                row = {'index':i,'seed':entry['seed'],'demo':entry['source_demo'],'sim_exit_status':code}
                try:
                    if code != 0:
                        raise ValueError('Physical case process failed')
                    row.update(score(case,load_case(i)[0]))
                    subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(case/'physical/view.mp4'),
                                    '-f','null','-'],check=True,timeout=120)
                    row['video_server_decode_verified']=True
                    row['video_sha256']=digest((case/'physical/view.mp4').read_bytes())
                except (ValueError, OSError, KeyError, subprocess.SubprocessError) as e:
                    artifact = case/'physical/result.json'
                    raw = read(artifact) if artifact.is_file() else {}
                    row.update(outcome='setup_failure' if not raw.get('completed_actions') else 'aborted_or_unverified',
                               error=repr(e),simulator_error=raw.get('error'),executed_actions=raw.get('completed_actions',0))
                status['cases'].append(row)
                write(case/'evidence.json',row)
                save(stage='physical_only_case_recorded')
            from collections import Counter
            save(stage='physical_expert_panel_finished_not_policy_success',counts=dict(Counter(x['outcome'] for x in status['cases'])))
    except BaseException as e:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:
                child.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
        save(stage='physical_expert_panel_orchestration_stopped',active_pid=0,error=repr(e))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--simulate',type=int)
    p.add_argument('--parent',type=int)
    a=p.parse_args()
    gpu=resolve_gpu(ROOT,'1')
    if a.simulate is not None:
        c=run_path(a.campaign)
        if os.getppid()!=a.parent or read(c/'status.json')['controller_pid']!=a.parent:
            raise ValueError('Owning queue controller required')
        case = c/f'case-{a.simulate:02d}'
        simulate(case,'physical',a.parent,gpu,load_expert=lambda:load_case(a.simulate))
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.parent:
        raise ValueError('Controller starts CPU-only')
    episodes,_,_=frozen()
    owner,blockers=check_release(gpu)
    if a.check_only:
        for i in range(1,len(episodes)+1):
            load_case(i)
        print(json.dumps({'physical_only':True,'inputs_checked':len(episodes),'gpu_launched':False,'blockers':blockers}))
        return
    c=run_path(a.campaign)
    with (RUNS/'expert-physical-pilot-queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        c.mkdir()
        names = (Path(__file__).name,'run_B_paired_attachment_replay.py','verify_expert_physical_pilot_inputs.py',
                 'compare_paired_grasp_replay.py','plan_expert_physical_pilot.py','resource_allocation.py',
                 'grasping_validation.py','grasp_telemetry.py','object_pose_audit.py','physical_grasp_observation.py',
                 'motor_contract.py','momagen_pi0_schema.py','pi0_base_targets.py','state_policy_features.py')
        hashes = {}
        for name in names:
            blob = (SCRIPTS/name).read_bytes()
            (c/name).write_bytes(blob)
            hashes[name] = digest(blob)
        write(c/'runtime-source-sha256.json',hashes)
        (RUNS/'latest-expert-physical-pilot-queue.txt').write_text(str(c)+'\n')
        run(c,gpu,owner)


if __name__=='__main__':
    main()
