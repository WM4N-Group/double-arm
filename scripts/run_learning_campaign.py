#!/usr/bin/env python3
"""SERVER ONLY: bounded two-GPU generation -> state BC -> teacher-free video.

Reuses the running native control and GPU1 worker. Launches a bounded GPU0
worker only after the native process exits. Training uses spare VRAM while data
is generated; evaluation waits for GPU0's generator to exit. No success is
inferred from GPU usage, package imports, or network validation loss.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now(): return datetime.now(timezone.utc).isoformat()


class AttachedWorker:
    """Observe an existing generator without claiming ownership of its process."""
    def __init__(self,pid,run): self.pid,self.run=pid,run
    def poll(self):
        result=self.run/'process-result.json'
        if result.exists():return int(json.loads(result.read_text())['generator_exit_status'])
        try:
            state=Path(f'/proc/{self.pid}/stat').read_text().split()[2]
            return 1 if state=='Z' else None
        except FileNotFoundError:return 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--official-run',required=True,type=Path)
    p.add_argument('--gpu1-run',required=True,type=Path)
    p.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    p.add_argument('--max-hours',type=float,default=7)
    p.add_argument('--gpu0-run',type=Path)
    p.add_argument('--gpu0-pid',type=int)
    p.add_argument('--extra-run-dir',action='append',default=[],type=Path)
    a=p.parse_args()
    if bool(a.gpu0_run)!=bool(a.gpu0_pid):p.error('--gpu0-run and --gpu0-pid must be supplied together')
    if not str(a.root).startswith('/mnt/'): raise ValueError('Server persistent root required')
    script=Path(__file__).resolve().parent
    campaign=a.root/'runs'/f'tidy-learning-campaign-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    campaign.mkdir();(a.root/'runs/latest-learning-campaign.txt').write_text(str(campaign)+'\n')
    deadline=time.monotonic()+a.max_hours*3600
    status={'created_at':now(),'objective':'successful data -> learned closed-loop long-range mobile manipulation video',
       'task':'r1_tidy_table_D0','robot':'R1','caveat':'Single active arm baseline; not a bimanual completion claim',
       'official_run':str(a.official_run),'gpu1_run':str(a.gpu1_run),'stage':'waiting_for_data',
       'training':'not_started','evaluation':'not_started','campaign':str(campaign),'sustained_success':False}
    processes=[];gpu0=AttachedWorker(a.gpu0_pid,a.gpu0_run) if a.gpu0_run else None
    gpu0_run=a.gpu0_run;train=None;evaluation=None;train_dir=None;snapshot=None
    if gpu0 is not None:status.update(gpu0_worker_pid=gpu0.pid,gpu0_run=str(gpu0_run),attached_existing_gpu0=True)
    status['extra_run_dirs']=[str(p) for p in a.extra_run_dir]
    def save(**updates):
        status.update(updates,updated_at=now());path=campaign/'status.json'
        tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(path)
    def start(command,log,env=None):
        handle=log.open('a',buffering=1)
        proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                              env=env or os.environ.copy(),start_new_session=True)
        handle.close();processes.append(proc);return proc
    save()
    try:
        while time.monotonic()<deadline:
            if gpu0 is None and (a.official_run/'process-result.json').exists():
                env={**os.environ,'TASK':'tidy_table','DATA_ATTEMPTS':'6','DATA_GUARANTEE':'1',
                    'DATA_MIN_SUCCESSES':'4','DATA_SEED':'200','DATA_TIMEOUT_SECONDS':'14400','CUDA_VISIBLE_DEVICES':'0',
                    'STATE_CPU_THREADS':'4','STATE_CPUSET':'0-15'}
                gpu0=start(['bash',str(script/'run_long_task_data.sh')],campaign/'gpu0-generation.log',env)
                save(gpu0_worker_pid=gpu0.pid)
            if gpu0 is not None and gpu0_run is None:
                text=(campaign/'gpu0-generation.log').read_text(errors='replace')
                for line in text.splitlines():
                    if line.startswith('Run directory: '):
                        gpu0_run=Path(line.split(': ',1)[1]);save(gpu0_run=str(gpu0_run));break
            if train is None:
                candidate=campaign/'training-snapshot.hdf5'
                commands=[sys.executable,str(script/'collect_success_demos.py'),'--run-dir',str(a.official_run),
                          '--run-dir',str(a.gpu1_run)]
                if gpu0_run: commands+=['--run-dir',str(gpu0_run)]
                for extra in a.extra_run_dir:commands+=['--run-dir',str(extra)]
                commands+=['--output',str(candidate),'--min-demos','4']
                result=subprocess.run(commands,capture_output=True,text=True,timeout=120)
                (campaign/'snapshot-latest.log').write_text(result.stdout+result.stderr)
                if result.returncode==0:
                    snapshot=candidate;train_dir=campaign/'state-bc-initial'
                    report=json.loads(snapshot.with_suffix('.json').read_text())
                    # Small state MLP normally needs <1 GiB. Do not start it if
                    # cuRobo's current allocation leaves too little headroom.
                    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
                    available=[(int(line.split(',')[1]),int(line.split(',')[0])) for line in memory.splitlines()]
                    free,gpu=max(available)
                    if free<1500:
                        # Snapshot is immutable; retry GPU allocation next iteration.
                        snapshot.rename(campaign/'snapshot-pending.hdf5')
                        snapshot.with_suffix('.json').rename(campaign/'snapshot-pending.json')
                        save(stage='waiting_for_training_vram',free_vram_mb=free)
                    else:
                        train=start([sys.executable,str(script/'train_state_policy.py'),'--dataset',str(snapshot),
                            '--output',str(train_dir),'--val-demo','demo_0','--steps','15000'],campaign/'training.log',
                            {**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)})
                        save(stage='training_with_parallel_generation',training='running',training_pid=train.pid,
                             training_gpu=gpu,training_snapshot=str(snapshot),training_demos=report['num_success_demos'],
                             policy_run=str(train_dir))
                elif result.returncode!=3:
                    save(snapshot_error=result.stderr[-2000:])
            if train is not None and train.poll() is not None:
                if train.returncode!=0:
                    save(stage='failed',training='failed',error='See training.log',training_exit_status=train.returncode);return 2
                if evaluation is None:save(training='completed',stage='waiting_for_evaluation_gpu')
                if evaluation is None and gpu0 is not None and gpu0.poll() is not None:
                    assert snapshot and train_dir
                    evaluation=start(['timeout','--signal=TERM','--kill-after=90s','7200',sys.executable,
                         str(script/'evaluate_state_policy.py'),'--dataset',str(snapshot),
                         '--checkpoint',str(train_dir/'best.pt'),'--output',str(train_dir/'evaluation'),
                         '--demo','demo_1','--demo','demo_0','--max-steps','4000'],campaign/'evaluation.log',
                         {**os.environ,'CUDA_VISIBLE_DEVICES':'0'})
                    save(stage='evaluating',evaluation='running',evaluation_pid=evaluation.pid)
            if evaluation is not None and evaluation.poll() is not None:
                result_file=train_dir/'evaluation/result.json'
                result=json.loads(result_file.read_text()) if result_file.exists() else {}
                episodes=result.get('episodes',[])
                heldout=[e for e in episodes if e.get('split')=='validation']
                success=any(e.get('sustained_success',False) for e in heldout)
                save(stage='evaluated' if evaluation.returncode==0 else 'failed',evaluation='completed' if evaluation.returncode==0 else 'failed',
                     evaluation_exit_status=evaluation.returncode,sustained_success=success,episodes=episodes,
                     evaluation_result=str(result_file),error=result.get('error'))
                return 0 if success else 4
            # Small telemetry only; never generate fake workload to heat the GPU.
            telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
            with (campaign/'gpu-usage.jsonl').open('a') as log:log.write(json.dumps({'at':now(),'gpu':telemetry.strip()})+'\n')
            save(gpu_usage=telemetry.strip())
            time.sleep(60)
        save(stage='timed_out',error='Campaign deadline reached; preserve data and inspect finite worker logs');return 124
    except BaseException as error:
        save(stage='failed',error=repr(error));raise


if __name__=='__main__':raise SystemExit(main())
