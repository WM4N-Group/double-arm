#!/usr/bin/env python3
"""SERVER ONLY: bounded continuation until a genuine learned mobile task video.

GPU1 keeps generating useful successful data; GPU0 evaluates trained policies or
collects more data and refits them. A task is never marked successful from an
exit code or low supervised loss. Job scheduling is not robot task control.
"""
import argparse
import fcntl
import re
import traceback
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def next_round_directory(campaign):
    """Allocate a new round without overwriting previous interrupted work."""
    existing = [int(match.group(1)) for child in campaign.iterdir()
                if (match := re.fullmatch(r"continued-round-(\d+)", child.name))]
    index = max(existing, default=0) + 1
    while True:
        path = campaign / f"continued-round-{index}"
        try:
            path.mkdir(exist_ok=False)
            return index, path
        except FileExistsError:
            index += 1


def next_generation_seed(root, prior, state):
    seeds = [200, int(prior.get('gpu1_seed', 0)), int(state.get('gpu1_seed', 0))]
    for path in (root/'runs').glob('tidy-learning-campaign-*/continued-gpu1-seed-*.log'):
        match = re.fullmatch(r'continued-gpu1-seed-(\d+)\.log', path.name)
        if match:
            seeds.append(int(match.group(1)))
    for path in (root/'runs').glob('tidy_table-data-*/generation-request.json'):
        seeds.append(int(read(path, {}).get('seed', 0)))
    return max(seeds) + 100


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True,type=Path);p.add_argument('--hours',type=float,default=8)
    p.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    a=p.parse_args();c=a.campaign;root=a.root;script=Path(__file__).resolve().parent
    if (root/'runs/USER_PAUSED.json').exists():
        print('Project paused by user; no server workloads will start',flush=True)
        return 0
    if not str(c).startswith(str(root/'runs')+'/'):raise ValueError('Remote campaign path required')
    lock=(c/'continuation.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=read(c/'continuation-status.json',{})
    original_start=prior.get('started_at',datetime.now(timezone.utc).isoformat())
    elapsed=max(0,(datetime.now(timezone.utc)-datetime.fromisoformat(original_start)).total_seconds())
    deadline=time.monotonic()+max(0,a.hours*3600-elapsed)
    state=read(c/'status.json',{});gpu1_run=Path(prior.get('gpu1_run') or state['gpu1_run'])
    gpu1_process=None;gpu1_log=None;seed=next_generation_seed(root,prior,state)
    status={'stage':'continuing','controller_pid':os.getpid(),'started_at':original_start,'deadline_hours':a.hours,
            'goal':'learned closed-loop navigation + grasp + placement video','goal_achieved':False}
    def save(**kwargs):
        status.update(kwargs,updated_at=datetime.now(timezone.utc).isoformat())
        f=c/'continuation-status.json';tmp=f.with_suffix('.json.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(f)
    def spawn(command,log,env=None):
        with log.open('a',buffering=1) as handle:
            return subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                    env=env or os.environ.copy(),start_new_session=True)
    def maintain_gpu1():
        nonlocal gpu1_run,gpu1_process,gpu1_log,seed
        if time.monotonic()>=deadline or (c/'goal-demo-ready.json').exists():return
        if gpu1_process is not None:
            if gpu1_log:
                for line in gpu1_log.read_text(errors='replace').splitlines():
                    if line.startswith('Run directory: '):gpu1_run=Path(line.split(': ',1)[1]);break
            if gpu1_process.poll() is None:return
            gpu1_process=None
        if (gpu1_run/'process-result.json').exists():
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','TASK':'tidy_table','DATA_ATTEMPTS':'12',
               'DATA_GUARANTEE':'1','DATA_MIN_SUCCESSES':'4','DATA_SEED':str(seed),
               'DATA_TIMEOUT_SECONDS':'14400','STATE_CPU_THREADS':'4','STATE_CPUSET':'88-103'}
            gpu1_log=c/f'continued-gpu1-seed-{seed}.log'
            gpu1_process=spawn(['bash',str(script/'run_long_task_data.sh')],gpu1_log,env)
            save(gpu1_worker_pid=gpu1_process.pid,gpu1_seed=seed);seed+=100
    def tick(seconds=30):
        maintain_gpu1()
        save(gpu1_run=str(gpu1_run))
        time.sleep(seconds)
    def run_wait(command,log,env):
        process=spawn(command,log,env);save(active_pid=process.pid,active_log=str(log))
        while process.poll() is None:
            if time.monotonic()>=deadline:
                # Jobs have their own finite limits; do not label timeout success.
                save(stage='deadline_reached',goal_achieved=False);return 124
            tick()
        return process.returncode
    def publish_if_success(policy):
        result=read(policy/'evaluation/result.json',{})
        if result.get('mode')!='closed_loop_neural_policy' or result.get('planner_enabled') is not False or result.get('teacher_action_reads')!=0:return False
        if result.get('grasping_mode')!='physical' or result.get('assisted_constraint_seen') is not False:return False
        candidates=[e for e in result.get('episodes',[]) if e.get('sustained_success')
             and any(e.get('grasped_ever',[])) and e.get('base_path_length_m',0)>0.5
             and e.get('initial_state_load_count')==1]
        if not candidates:return False
        candidates.sort(key=lambda e:e.get('split')=='validation',reverse=True);episode=candidates[0]
        output=policy/'goal-review'
        if not output.exists():
            subprocess.run([sys.executable,str(script/'export_policy_review.py'),'--policy-run',str(policy),
                 '--output',str(output),'--demo',episode['demo'],'--require-success'],check=True)
        review=read(output/'review.json',{})
        if review.get('task_success') is not True:raise ValueError('Review failed task success validation')
        ready={'policy_run':str(policy),'review_dir':str(output),'video':str(output/'demo.mp4'),
              'episode':episode,'training_method':review['method'],'privileged_state':True,
              'task':'r1_tidy_table_D0','caveat':'Single active arm; distinguish training-reset from heldout result',
              'created_at':datetime.now(timezone.utc).isoformat()}
        (c/'goal-demo-ready.json').write_text(json.dumps(ready,indent=2));save(stage='goal_video_ready',goal_achieved=True,**ready)
        print('GOAL_VIDEO_READY',json.dumps(ready),flush=True);return True
    def evaluate(policy,checkpoint='best.pt',diagnostic_only=False):
        save(stage='evaluating_candidate',active_policy=str(policy),checkpoint=checkpoint)
        process=None
        existing=read(policy/'evaluation-process-result.json',{})
        if not (policy/'evaluation').exists() and existing.get('stage')!='evaluating':
            manifest=read(policy/'manifest.json',{});names=manifest.get('train_demos',[])
            count=min(1,len(names));indices=[round(i*(len(names)-1)/max(1,count-1)) for i in range(count)]
            selected=list(dict.fromkeys([names[i] for i in indices]+manifest.get('val_demos',[])))
            process=spawn(['bash',str(script/'run_policy_evaluation.sh'),str(policy)],policy/'evaluation-controller.log',
                    {**os.environ,'EVAL_GPU':'0','EVAL_MAX_STEPS':'3000','CHECKPOINT_NAME':checkpoint,'EVAL_DEMOS':' '.join(selected)})
        while time.monotonic()<deadline:
            if not diagnostic_only and publish_if_success(policy):return True
            report=read(policy/'evaluation-process-result.json',{})
            if report and report.get('stage')!='evaluating':break
            if process is not None and process.poll() is not None:break
            tick(20)
        return not diagnostic_only and publish_if_success(policy)
    save()
    # A bounded diagnostic may temporarily own GPU0. Its lock, not a stale JSON
    # stage, determines whether another simulator can be started safely.
    probe_lock=(c/'memory-probe.lock').open('a')
    while time.monotonic()<deadline:
        try:
            fcntl.flock(probe_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            fcntl.flock(probe_lock,fcntl.LOCK_UN)
            break
        except BlockingIOError:
            save(stage='waiting_for_external_gpu0_probe')
            tick(20)
    probe_lock.close()
    for job in read(c/'pending-policy-evaluations.json',[]):
        policy=Path(job['policy'])
        if not str(policy.resolve()).startswith(str(c.resolve())+'/'):
            raise ValueError('Pending policy must belong to this campaign')
        while time.monotonic()<deadline and not (policy/'training-result.json').exists():
            pid=job.get('training_pid')
            if not pid or not Path(f'/proc/{int(pid)}').exists():break
            save(stage='waiting_for_pending_training',active_policy=str(policy))
            tick(20)
        if time.monotonic()>=deadline:break
        if (policy/'training-result.json').exists() and evaluate(policy,job.get('checkpoint','best.pt'),job.get('diagnostic_only',False)):
            return 0
    # Reuse the running one-step evaluation; never open another simulator on GPU0.
    initial=[(c/'state-rbf-world-goal-seven','best.pt'),(c/'state-bc-one-step','best.pt'),(c/'state-rbf-absolute-seven','best.pt'),(c/'state-gru-seven','last.pt'),(c/'state-rbf-no-velocity-seven','best.pt'),(c/'state-bc-absolute-seven','last.pt')]
    for policy,checkpoint in initial:
        if time.monotonic()>=deadline:break
        if (policy/'training-result.json').exists() and evaluate(policy,checkpoint):return 0
    first_round=True
    while time.monotonic()<deadline:
        round_index,round_dir=next_round_directory(c)
        save(stage='collecting_more_data',round=round_index)
        code=0
        # Consume existing successful data first; do not wait for another simulator batch.
        if not first_round:
            code=run_wait(['bash',str(script/'run_long_task_data.sh')],round_dir/'gpu0-generation.log',
              {**os.environ,'CUDA_VISIBLE_DEVICES':'0','TASK':'tidy_table','DATA_ATTEMPTS':'6','DATA_GUARANTEE':'1',
               'DATA_MIN_SUCCESSES':'4','DATA_SEED':str(500+100*round_index),'DATA_TIMEOUT_SECONDS':'7200',
               'STATE_CPU_THREADS':'4','STATE_CPUSET':'0-15'})
        if code==124 or time.monotonic()>=deadline:break
        first_round=False
        dataset=round_dir/'successful-snapshot.hdf5'
        command=[sys.executable,str(script/'collect_success_demos.py'),'--run-dir',state['official_run']]
        for run in sorted((root/'runs').glob('tidy_table-data-*')):command+=['--run-dir',str(run)]
        command+=['--output',str(dataset),'--min-demos','8']
        snapshot_exit=run_wait(command,round_dir/'snapshot.log',os.environ.copy())
        if snapshot_exit==124:break
        if snapshot_exit!=0:save(stage='insufficient_new_data',snapshot_exit=snapshot_exit);continue
        save(stage='training_more_data',dataset=str(dataset))
        models=[('neural-world-goal',[sys.executable,str(script/'train_goal_policy.py'),'--steps','40000']),
                ('rbf-world-goal',[sys.executable,str(script/'train_rbf_policy.py'),'--action-mode','world_base_goal','--ignore-base-velocity','--normalize-basis','--orientation-noise-floor','0.15','--ridge','1e-5']),
                ('mlp',[sys.executable,str(script/'train_state_policy.py'),'--steps','40000','--horizon','1','--hidden','768','--val-demo','demo_0']),
                ('rbf',[sys.executable,str(script/'train_rbf_policy.py'),'--action-mode','absolute']),
                ('gru',[sys.executable,str(script/'train_recurrent_policy.py'),'--updates','1500'])]
        for name,base in models:
            policy=round_dir/name
            code=run_wait(base+['--dataset',str(dataset),'--output',str(policy)],round_dir/f'{name}-training.log',
                     {**os.environ,'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
            if code==124:break
            if code!=0:save(training_error=name,training_exit=code);continue
            if evaluate(policy,'last.pt' if name in ('gru','neural-world-goal') else 'best.pt'):return 0
    save(stage='deadline_reached_without_task_success',goal_achieved=False)
    return 124


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except BlockingIOError:
        print('Another controller owns this campaign',file=sys.stderr)
        raise SystemExit(2)
    except Exception as error:
        # A dead controller must not leave a misleading 'evaluating' status.
        if '--campaign' in sys.argv:
            campaign=Path(sys.argv[sys.argv.index('--campaign')+1])
            path=campaign/'continuation-status.json'
            status=read(path,{})
            status.update(stage='controller_failed',goal_achieved=False,
                          error=f'{type(error).__name__}: {error}',
                          updated_at=datetime.now(timezone.utc).isoformat())
            tmp=path.with_suffix('.error.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(path)
        traceback.print_exc()
        raise SystemExit(1)
