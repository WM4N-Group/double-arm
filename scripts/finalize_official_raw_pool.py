#!/usr/bin/env python3
"""Finite CPU-only: wait for existing raw pool exit, register provenance, audit.

Does not start/stop GPU jobs, train/select a split, clean records or change norms.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/mnt/yuanyue/double-arm')
COMPLETE='raw_official_pool_target_complete_alignment_and_split_pending'


def outcome(status):
    if status.get('error') or status.get('stage')=='generation_failed_or_interrupted':return 'failed'
    if status.get('stage')!=COMPLETE:return 'pending'
    if (status.get('active_pid')!=0 or status.get('verified_unique_demos',0)<status.get('target_unique_demos',1)
        or status.get('source_grasping_mode')!='assisted' or status.get('policy_training')!='not_run'):
        return 'failed'
    return 'ready'


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dependency',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU only')
    if (a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs')
        or not a.dependency.resolve().is_relative_to(ROOT/'runs')):raise ValueError('New project run and existing raw pool required')
    from preflight_pi0_accumulation_gpu import process_alive
    lock=(ROOT/'runs/raw-pool-finalize.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    a.output.mkdir();(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    state={'stage':'waiting_for_raw_pool_exit','controller_pid':os.getpid(),'dependency':str(a.dependency),
           'cpu_only':True,'gpu_jobs_started':0,'policy_training':'not_run','split_created':False,
           'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def save(**updates):
        state.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(a.output/'status.json',state)
    save();(ROOT/'runs/latest-raw-pool-finalize.txt').write_text(str(a.output)+'\n')
    try:
        deadline=time.monotonic()+7200
        while True:
            if time.monotonic()>deadline:raise TimeoutError('Finite CPU dependency wait exhausted; existing GPU job untouched')
            if (ROOT/'runs/USER_PAUSED.json').exists():raise RuntimeError('User paused')
            dependency=read(a.dependency/'status.json');result=outcome(dependency)
            if result=='failed':raise RuntimeError('Raw dependency failed; no audit/completion bypass')
            if result=='ready' and not any(process_alive(dependency.get(k)) for k in ('controller_pid','active_pid')):break
            time.sleep(15)
        write(a.output/'dependency-status.json',dependency)
        provenance=a.dependency/'raw-episode-provenance.json'
        if not provenance.is_file():raise ValueError('No finalized provenance')
        registry=ROOT/'runs/official-raw-pool-registry.json'
        with (ROOT/'runs/official-raw-pool-registry.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            paths=sorted(set(read(registry)['provenance_paths']+[str(provenance)]))
            if any(not Path(x).resolve().is_relative_to(ROOT/'runs') or Path(x).name!='raw-episode-provenance.json' or not Path(x).is_file() for x in paths):
                raise ValueError('Registry must preserve only finalized project raw sources')
            write(registry,{'scope':'UNSPLIT official author-assisted raw sources; no per-worker count summation',
                'updated_at':datetime.now(timezone.utc).isoformat(),'provenance_paths':paths})
            write(a.output/'registry-snapshot.json',read(registry))
        save(stage='cross_pool_CPU_QA')
        script=Path(__file__).resolve().parent/'audit_official_pool_snapshot.py'
        (a.output/script.name).write_bytes(script.read_bytes())
        command=[str(ROOT/'envs/momagen/bin/python'),str(script),'--output',str(a.output/'audit')]
        for path in paths:command+=['--raw-provenance',path]
        for name in ('r1_tidy_table_train_v1','r1_tidy_table_holdout_v1'):
            command+=['--exclude-provenance',str(ROOT/'data/pi0/lerobot/momagen'/name/'momagen-provenance.json')]
        with (a.output/'audit.log').open('x') as log:
            subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        report=read(a.output/'audit/report.json')
        (ROOT/'runs/latest-official-raw-pool-audit.txt').write_text(str(a.output/'audit')+'\n')
        save(stage='completed_raw_pool_registry_CPU_QA',eligible_unique_episodes=report['eligible_unique_episodes'],
             report_sha256=hashlib.sha256((a.output/'audit/report.json').read_bytes()).hexdigest())
    except BaseException as error:
        save(stage='raw_pool_registry_QA_failed',error=repr(error));raise


if __name__=='__main__':main()
