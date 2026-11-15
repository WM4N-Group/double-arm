#!/usr/bin/env python3
"""Bounded one-GPU feedback-codec experiment, serial simulation and honest gates."""
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


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def process_alive(pid, expected=None):
    proc=Path(f'/proc/{pid}')
    try:
        state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0]
        cmd=(proc/'cmdline').read_bytes()
    except FileNotFoundError:return False
    return state!='Z' and (expected is None or expected.encode() in cmd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    parser.add_argument('--hours',type=float,default=3)
    parser.add_argument('--wait-pid',type=int,required=True)
    parser.add_argument('--train-pid',type=int,required=True)
    args=parser.parse_args();c=args.campaign;root=args.root
    if not str(c).startswith(str(root/'runs')+'/'):raise ValueError('Remote persistent campaign required')
    gpu=resolve_gpu(root);deadline=time.monotonic()+args.hours*3600
    lock=(root/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=Path(__file__).resolve().parent
    state={'controller_pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),'max_hours':args.hours,
           'gpu_uuid':gpu,'reserved_gpu_index':0,'goal_achieved':False,
           'change':'Learn measured-state residual motor goals; noise targets corrected for perturbed anchor',
           'old_campaign':read(c/'status.json',{}),'candidates':[]}
    def save(**kw):
        state.update(kw,updated_at=datetime.now(timezone.utc).isoformat());tmp=c/'feedback-status.tmp'
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(c/'feedback-status.json')
    def paused():return (root/'runs/USER_PAUSED.json').exists()
    def wait(pid,expected=None):
        while process_alive(pid,expected):
            if paused() or time.monotonic()>deadline:return False
            save(stage='waiting_for_existing_job',waiting_pid=pid);time.sleep(20)
        return True
    if not wait(args.wait_pid) or not wait(args.train_pid):save(stage='paused_or_deadline');return 75
    for offset in (0,4):
        policy=c/f'feedback-neural-offset{offset}'
        training=read(policy/'training-result.json',{})
        if training.get('status')!='training_completed':
            state['candidates'].append({'policy':str(policy),'status':'training_failed_or_missing'});continue
        if paused() or time.monotonic()>deadline:save(stage='paused_or_deadline');return 75
        # Independently verify no unrelated task occupies our allocated card.
        reply=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        free={line.split(',')[0].strip():int(line.split(',')[1].strip()) for line in reply.splitlines()}
        if free.get(gpu,0)<20000:save(stage='allocated_gpu_unavailable');return 76
        env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'EVAL_GPU':gpu,'EVAL_DEMOS':'demo_4 demo_0',
             'EVAL_MAX_STEPS':'2200','EVAL_TIMEOUT_SECONDS':str(min(3600,int(deadline-time.monotonic()))),
             'CHECKPOINT_NAME':'last.pt','EVAL_GRASPING_MODE':'physical','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
        with (policy/'evaluation-controller.log').open('a') as log:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=90s',str(max(1,int(deadline-time.monotonic()))),
                'taskset','-c','88-103','bash',str(script/'run_policy_evaluation.sh'),str(policy)],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
        save(stage='evaluating_residual_neural_policy',active_policy=str(policy),active_pid=child.pid)
        while child.poll() is None:
            if paused() or time.monotonic()>deadline:
                os.killpg(child.pid,signal.SIGTERM);save(stage='paused_or_deadline');return 75
            time.sleep(20)
        result=read(policy/'evaluation/result.json',{})
        state['candidates'].append({'policy':str(policy),'exit_status':child.returncode,'episodes':result.get('episodes',[])})
        eligible=[e for e in result.get('episodes',[]) if e.get('sustained_success') and any(e.get('grasped_ever',[]))
                  and e.get('base_path_length_m',0)>.5 and e.get('initial_state_load_count')==1 and e.get('object_pose_write_count')==0]
        if result.get('mode')!='closed_loop_neural_policy' or result.get('planner_enabled') is not False or result.get('teacher_action_reads')!=0:
            eligible=[]
        if result.get('grasping_mode')!='physical' or result.get('assisted_constraint_seen') is not False or result.get('object_pose_write_count')!=0:
            eligible=[]
        if eligible:
            eligible.sort(key=lambda e:e['split']=='validation',reverse=True);episode=eligible[0]
            subprocess.run([sys.executable,str(script/'export_policy_review.py'),'--policy-run',str(policy),'--output',str(policy/'physical-goal-review'),
                            '--demo',episode['demo'],'--require-success','--require-physical'],check=True)
            save(stage='physical_neural_video_ready',goal_achieved=True,episode=episode,review_dir=str(policy/'physical-goal-review'));return 0
        save(stage='candidate_closed_loop_failed',goal_achieved=False)
    save(stage='all_feedback_candidates_failed',goal_achieved=False);return 6


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        if '--campaign' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--campaign')+1])/'feedback-status.json';state=read(p,{})
            state.update(stage='controller_failed',goal_achieved=False,error=repr(error));p.write_text(json.dumps(state,indent=2))
        import traceback
        traceback.print_exc();raise SystemExit(1)
