#!/usr/bin/env python3
"""SERVER ONLY: one allocated GPU, physical data -> neural BC -> real rollout.

Never launches work on a reserved GPU, never revives the old dual-card runner.
Physical demonstrations are collected from a fitted memory policy, not replayed
teacher actions; neural acceptance remains independent and teacher-free.
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
from resource_allocation import allocation, resolve_gpu


def read(path, default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--source-policy',type=Path,required=True)
    parser.add_argument('--wait-pid',type=int)
    parser.add_argument('--audit-dir',type=Path)
    parser.add_argument('--hours',type=float,default=4)
    parser.add_argument('--target-successes',type=int,default=6)
    parser.add_argument('--prior-physical-data',type=Path)
    args=parser.parse_args();root=args.root;c=args.campaign
    if (root/'runs/USER_PAUSED.json').exists():return 75
    config=allocation(root)
    if config.get('max_gpu_count')!=1:raise ValueError('Explicit single-GPU resource allocation required')
    gpu=resolve_gpu(root,config['allowed_gpu_uuid'])
    if not str(c).startswith(str(root/'runs')+'/'):raise ValueError('Remote persistent campaign required')
    c.mkdir(exist_ok=False)
    lock=(root/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    script=Path(__file__).resolve().parent;start=time.time();deadline=time.monotonic()+args.hours*3600
    state={'stage':'starting','controller_pid':os.getpid(),'started_at':datetime.now(timezone.utc).isoformat(),
           'max_hours':args.hours,'allowed_gpu_uuid':gpu,'reserved_gpu_index':config['reserved_gpu_index'],
           'max_gpu_count':1,'goal_achieved':False,'acceptance':'neural policy; physical mode; zero assisted constraints and object pose writes; sustained goal; actual video',
           'source_policy':str(args.source_policy)}
    def save(**kwargs):
        state.update(kwargs,updated_at=datetime.now(timezone.utc).isoformat())
        tmp=c/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(c/'status.json')
    save();(root/'runs/latest-single-gpu-campaign.txt').write_text(str(c)+'\n')
    (root/'runs/latest-single-gpu-job.json').write_text(json.dumps({'controller_pid':os.getpid(),'campaign':str(c),'gpu_uuid':gpu,'max_gpu_count':1,'max_hours':args.hours},indent=2))
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'EVAL_GPU':gpu,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'OPENBLAS_NUM_THREADS':'4','OMP_WAIT_POLICY':'PASSIVE'}
    def remaining():return max(0,deadline-time.monotonic())
    def stop_requested():return (root/'runs/USER_PAUSED.json').exists()
    def wait_for_free_gpu():
        while remaining()>0 and not stop_requested():
            reply=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            free={row.split(',')[0].strip():int(row.split(',')[1].strip()) for row in reply.splitlines()}
            if free.get(gpu,0)>=20000:return True
            save(stage='waiting_for_allocated_gpu');time.sleep(20)
        return False
    def run(command,log,stage,extra_env=None):
        if remaining()<=0 or stop_requested():return 75
        if not wait_for_free_gpu():return 75
        save(stage=stage,active_log=str(log))
        timeout=min(int(remaining()),7200)
        with log.open('a') as handle:
            process=subprocess.Popen(['timeout','--signal=TERM','--kill-after=60s',str(timeout),'taskset','-c','88-103']+command,
                stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**env,**(extra_env or {})})
        save(active_pid=process.pid)
        while process.poll() is None:
            if stop_requested() or remaining()<=0:
                os.killpg(process.pid,signal.SIGTERM);save(stage='paused_or_deadline',goal_achieved=False);return 75
            time.sleep(15)
        save(last_exit_status=process.returncode)
        return process.returncode
    if args.wait_pid:
        save(stage='waiting_for_existing_grasp_audit',prior_pid=args.wait_pid,audit_dir=str(args.audit_dir))
        while remaining()>0 and not stop_requested():
            proc=Path(f'/proc/{args.wait_pid}')
            try:
                fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
                cmd=(proc/'cmdline').read_bytes()
            except FileNotFoundError:break
            if fields[0]=='Z' or b'audit_grasp_physics.py' not in cmd:break
            time.sleep(15)
        if stop_requested() or remaining()<=0:save(stage='paused_or_deadline');return 75
        if args.audit_dir:
            audit=read(args.audit_dir/'audit-result.json',{})
            save(prior_grasp_audit=audit)
            if audit.get('error') or len(audit.get('cases',[]))!=2:
                save(stage='grasp_audit_failed',error='Inspect preceding audit; no policy success claimed');return 2
    source=read(args.source_policy/'manifest.json',{})
    dataset=source['dataset'];recorded=c/'physical-successes.hdf5';rollout=c/'physical-data-collection'
    # Whole-episode holdout demo_0 stays held out for the neural learner.
    import h5py
    with h5py.File(dataset) as file:
        names=sorted((name for name in file['data'] if name.startswith('demo_')),key=lambda name:int(name.split('_')[-1]))
    names=['demo_0']+[name for name in names if name not in ('demo_0','demo_1')]+(['demo_1'] if 'demo_1' in names else [])
    command=[sys.executable,'-u',str(script/'evaluate_state_policy.py'),'--dataset',dataset,
        '--checkpoint',str(args.source_policy/'best.pt'),'--output',str(rollout),'--grasping-mode','physical',
        '--save-success-data',str(recorded),'--target-successes',str(args.target_successes),'--max-steps','2200','--no-video']
    for name in names:command+=['--demo',name]
    result=run(command,c/'physical-collection.log','collecting_verified_physical_data')
    if result!=0:save(stage='physical_collection_failed',exit_status=result);return 2
    if not recorded.exists():
        save(stage='insufficient_physical_demonstrations',physical_demonstrations=0);return 3
    if args.prior_physical_data and args.prior_physical_data.exists():
        with h5py.File(args.prior_physical_data,'r') as old,h5py.File(recorded,'a') as new:
            for name in old['data']:
                if not name.startswith('demo_') or name in new['data']:continue
                proof=json.loads(old['data'][name].attrs.get('provenance','{}'))
                if proof.get('grasping_mode')!='physical' or proof.get('assisted_constraint_seen') is not False or proof.get('object_pose_write_count')!=0:
                    raise ValueError('Prior physical data lacks valid zero-assistance / zero-pose-write provenance')
                old.copy(old['data'][name],new['data'],name=name)
            new['data'].attrs['total']=sum(len(new['data'][name]['actions']) for name in new['data'] if name.startswith('demo_'))
    with h5py.File(recorded) as file:
        demos=sorted(name for name in file['data'] if name.startswith('demo_'))
    save(physical_demonstrations=len(demos),physical_dataset=str(recorded))
    if 'demo_0' not in demos or len(demos)<3:
        save(stage='insufficient_physical_demonstrations',error='Need >=2 physical training successes and a separate physical holdout');return 3
    policy=c/'physical-neural-goal-bc'
    result=run([sys.executable,'-u',str(script/'train_goal_policy.py'),'--dataset',str(recorded),'--output',str(policy),
                '--steps','40000','--target-offset','4','--val-demo','demo_0'],c/'neural-training.log','training_neural_policy_on_physical_data')
    if result!=0:save(stage='neural_training_failed',exit_status=result);return 4
    fit=read(policy/'manifest.json',{});selected=fit['train_demos'][:2]+['demo_0']
    result=run(['bash',str(script/'run_policy_evaluation.sh'),str(policy)],c/'neural-evaluation.log','evaluating_neural_policy',
                {'EVAL_DEMOS':' '.join(selected),'CHECKPOINT_NAME':'last.pt','EVAL_GRASPING_MODE':'physical','EVAL_MAX_STEPS':'3500'})
    if result!=0:save(stage='neural_evaluation_failed',exit_status=result);return 5
    evaluation=read(policy/'evaluation/result.json',{})
    eligible=[episode for episode in evaluation.get('episodes',[]) if episode.get('sustained_success')
        and any(episode.get('grasped_ever',[])) and episode.get('base_path_length_m',0)>.5
        and episode.get('initial_state_load_count')==1 and episode.get('object_pose_write_count')==0]
    if evaluation.get('mode')!='closed_loop_neural_policy' or evaluation.get('planner_enabled') is not False or evaluation.get('teacher_action_reads')!=0:eligible=[]
    if evaluation.get('grasping_mode')!='physical' or evaluation.get('assisted_constraint_seen') is not False or evaluation.get('object_pose_write_count')!=0:
        eligible=[]
    if not eligible:save(stage='neural_closed_loop_failed',goal_achieved=False,episodes=evaluation.get('episodes',[]));return 6
    eligible.sort(key=lambda episode:episode['split']=='validation',reverse=True);episode=eligible[0]
    subprocess.run([sys.executable,str(script/'export_policy_review.py'),'--policy-run',str(policy),'--output',str(policy/'goal-review'),
        '--demo',episode['demo'],'--require-success','--require-physical'],check=True,env=env)
    save(stage='physical_neural_goal_video_ready',goal_achieved=True,episode=episode,review_dir=str(policy/'goal-review'),
         caveat='R1 single active arm; upstream friction=4, not real-hardware calibration; distinguish training reset from heldout result')
    print(json.dumps(state,indent=2),flush=True);return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        if '--campaign' in sys.argv:
            campaign=Path(sys.argv[sys.argv.index('--campaign')+1]);path=campaign/'status.json'
            if campaign.exists():
                status=read(path,{})
                status.update(stage='controller_failed',goal_achieved=False,error=f'{type(error).__name__}: {error}',
                              updated_at=datetime.now(timezone.utc).isoformat())
                path.write_text(json.dumps(status,indent=2))
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
