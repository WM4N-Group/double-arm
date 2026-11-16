#!/usr/bin/env python3
"""SERVER controller: audit reset RGB, then same trained pi0 physical rollout.

User renewed GPU1 time budget explicitly. No inherited twelve-hour deadline is
reset, no new training is started, and per-job watchdogs still bound stuck jobs.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from resource_allocation import allocation,resolve_gpu


def read(path):return json.loads(Path(path).read_text())


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workflow',type=Path,required=True);args=p.parse_args()
    root=Path('/mnt/yuanyue/double-arm');w=args.workflow;script=Path(__file__).resolve().parent
    if not w.resolve().is_relative_to(root/'runs'):raise ValueError('Remote workflow required')
    lock=(w/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gpu=resolve_gpu(root);authorized=allocation(root)
    if authorized.get('compute_time_budget')!='open_ended_explicit_user_authorization' or authorized.get('requires_per_window_approval') is not False:
        raise ValueError('Explicit renewed user time authorization required')
    state=read(w/'status.json');audit_pid=state['active_pid'];child=None
    def interrupted(signum,frame):raise InterruptedError(f'Controller signal{signum}')
    signal.signal(signal.SIGTERM,interrupted)
    def save(**updates):
        state.update(updates,updated_at=datetime.now(timezone.utc).isoformat());tmp=w/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(w/'status.json')
    def check():
        if (root/'runs/USER_PAUSED.json').exists():raise RuntimeError('User paused workflow')
        if resolve_gpu(root)!=gpu:raise RuntimeError('Allocation changed')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    def run(command,log,seconds):
        nonlocal child
        check()
        with (w/log).open('x') as stream:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=60s',str(seconds),'taskset','-c','88-103',*command],
                stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        save(active_pid=child.pid,active_log=str(w/log))
        while child.poll() is None:check();time.sleep(5)
        return child.returncode
    try:
        save(controller_pid=os.getpid(),stage='waiting_reset_sync_series')
        end=time.monotonic()+2400
        while alive(audit_pid):
            check()
            if time.monotonic()>end:raise TimeoutError('Reset audit watchdog')
            time.sleep(5)
        source=read(root/'data/pi0/lerobot/momagen/r1_tidy_table_holdout_v1/momagen-provenance.json')['episodes'][0]['source']
        py=str(root/'envs/openpi/bin/python')
        save(stage='scoring_true_reset_rgb')
        code=run([py,str(script/'audit_reset_sync_series.py'),'--series',str(w/'reset-sync-series'),
            '--source-dataset',source,'--output',str(w/'reset-sync-image-audit')],'reset-sync-image-audit.log',600)
        if code:raise RuntimeError(f'Reset image audit exited{code}')
        report=read(w/'reset-sync-image-audit/report.json');selected=report['selected_reset_settle_steps']
        if selected is None:save(stage='reset_sync_not_validated',task_success=False,active_pid=0);return 6
        save(stage='reset_rgb_gate_passed_physical_evaluation_required',selected_reset_settle_steps=selected,
             reset_image_audit=str(w/'reset-sync-image-audit/report.json'),task_success=False)
        # Let our completed simulator release driver allocations; never kill
        # another task, use GPU0, or run the model on an unapproved device.
        for attempt in range(60):
            check();query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            if not any(line.split(',')[0].strip()==gpu for line in query.splitlines()):break
            time.sleep(5)
        else:raise RuntimeError('Allowed GPU still occupied')
        save(stage='evaluating_synced_real_visual_pi0_physical')
        evaluation=w/'pi0-synced-physical-evaluation'
        code=run([py,str(script/'run_pi0_evaluation.py'),'--checkpoint',state['checkpoint'],'--source-dataset',source,
            '--training-provenance',str(root/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'),
            '--output',str(evaluation),'--hours','2','--action-representation','absolute_base',
            '--reset-settle-steps',str(selected),'--grasping-mode','physical'],'pi0-synced-physical-evaluation.log',7500)
        assessed=read(evaluation/'status.json')
        save(stage='synced_physical_success_manual_review_required' if assessed.get('task_success') is True else 'synced_physical_evaluation_failed',
             evaluation_exit_status=code,evaluation_result=assessed,task_success=assessed.get('task_success') is True,active_pid=0)
        return code
    except BaseException as error:
        save(stage='reset_sync_workflow_failed',error=repr(error),task_success=False)
        if child is not None and child.poll() is None:
            # SIGINT reaches Python's finally blocks, unlike a default SIGTERM;
            # allow the evaluator to stop its own authenticated CPU service and
            # simulator. Watchdogs still bound any unresponsive descendants.
            child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        elif child is None and alive(audit_pid):
            command=Path(f'/proc/{audit_pid}/cmdline').read_bytes()
            if str(w).encode() in command:
                try:os.killpg(audit_pid,signal.SIGINT)
                except ProcessLookupError:pass
        raise


if __name__=='__main__':raise SystemExit(main())
