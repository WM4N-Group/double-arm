#!/usr/bin/env python3
"""One bounded TRAIN physical LIVE original-pi0 diagnostic: execute5 vs prior10.

Same completed B checkpoint/norm/inputs/safety. No new training, teacher, planner,
base substitution, privileged mid-state, or heldout tuning. Extra Python/USD
no-assistance evidence is read-only; unknown evidence aborts, not a workaround.
"""
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT=Path('/mnt/yuanyue/double-arm')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    tmp=Path(path).with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def actual_complete(result):
    episodes=result.get('episodes',[])
    return (len(episodes)==1 and episodes[0].get('steps',0)>0 and episodes[0].get('aborted') is False
        and not result.get('error') and result.get('execute_steps')==5
        and result.get('grasping_mode')=='physical' and result.get('evaluation_split')=='train'
        and result.get('teacher_action_reads')==0 and result.get('initial_state_load_count')==1
        and result.get('no_assistance_all_checked_samples_confirmed') is True
        and result.get('no_assistance_checked_samples')==21+2*episodes[0]['steps'])


def main(c):
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import process_alive
    from pi0_relative_full_profile import load_contract,PROFILE
    if c.exists() or not c.resolve().is_relative_to(ROOT/'runs'):raise ValueError('Fresh project campaign')
    gpu=resolve_gpu(ROOT,'1');lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(ROOT/'runs/latest-gpu1-job.json');prior=read(old['status_file'])
    if (prior.get('stage')!='completed_hybrid_base_prefix_replay_NOT_live_policy' or prior.get('actual_hybrid_commands')!=1200
        or prior.get('replay_completed') is not True or prior.get('active_pid')!=0 or process_alive(old.get('controller_pid'))):
        raise ValueError('Whole base-prefix diagnostic must actually terminate/release')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    if any(row.split(',')[0].strip()==gpu for row in apps.splitlines()):raise ValueError('GPU1 occupied; no preemption/fallback')
    if (ROOT/'runs/USER_PAUSED.json').exists():raise ValueError('User paused')
    baseline=Path((ROOT/'runs/latest-pi0-relative-uniform-workflow.txt').read_text().strip());trained=read(baseline/'training-report.json')
    if read(baseline/'status.json')['stage']!='train-physical-diagnostic_FAILED':raise ValueError('Preserved actual B execute10 failed physical required')
    cp=Path(trained['checkpoint']);contract=load_contract(cp)
    if contract.get('training_variant')!='uniform_full_trajectory_same_base_v1':raise ValueError('Not actual B checkpoint')
    c.mkdir();script=Path(__file__).resolve();(c/script.name).write_bytes(script.read_bytes());write(c/'previous-owner.json',old)
    names=('run_pi0_evaluation.py','evaluate_pi0_r1.py','physical_grasp_observation.py','serve_pi0_relative_full_r1.py','pi0_relative_full_profile.py','pi0_relative_joint_diagnostic.py','momagen_pi0_schema.py','pi0_r1_config.py')
    context=c/'source-context';context.mkdir();hashes={}
    for name in names:
        raw=(ROOT/'code/scripts'/name).read_bytes();(context/name).write_bytes(raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    write(c/'plan.json',{'checkpoint':str(cp),'checkpoint_full_parameter_sha256':contract['full_parameter_sha256'],
        'norm_stats_sha256':contract['norm_stats_sha256'],'policy_profile':PROFILE,'source':trained['source'],
        'baseline_execute10_campaign':str(baseline),'execute_steps':5,'denoising_steps':10,'sampling_noise':'gaussian',
        'scope':'LIVE original pi0 TRAIN/source0 physical diagnostic, same weights; NOT heldout/paper/real-time claim',
        'no_new_training':True,'optimizer_updates_this_job':0,'runtime_source_sha256':hashes,
        'comparison_caveat':'Changing receding horizon also changes query/noise count and observed history; one TRAIN episode, not independent parameter-effect proof'})
    status={'stage':'starting_B_chunk5_TRAIN_physical','controller_pid':os.getpid(),'active_pid':0,'gpu_uuid':gpu,
        'checkpoint':str(cp),'training':'completed','training_is_diagnostic':True,'formal_training_completed':False,
        'paper_reproduction_complete':False,'policy_profile':PROFILE,'train_demonstrations':1,'train_frames':1714,
        'execute_steps':5,'optimizer_updates_this_job':0,'task_success':False,'evaluation_split':'train'}
    def save(**updates):status.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(c/'status.json',status)
    save();pointer={'campaign':str(c),'status_file':str(c/'status.json'),'controller_pid':os.getpid(),'gpu_uuid':gpu,'max_gpu_count':1,'role':'LIVE_ORIGINAL_PI0_B_chunk5_TRAINphysical_diagnostic_NO_new_training'}
    write(ROOT/'runs/latest-gpu1-job.json',pointer);mapping=read(ROOT/'runs/latest-multi-gpu-jobs.json');mapping['completed_base_prefix_diagnostic']=old;mapping['GPU1']=pointer;mapping['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',mapping)
    (ROOT/'runs/latest-B-chunk5-physical.txt').write_text(str(c)+'\n')
    child=None
    def abort(*_):raise TimeoutError('Bounded chunk5 evaluation watchdog')
    signal.signal(signal.SIGTERM,abort)
    try:
        e=c/'train-physical-chunk5';entry=trained['source'];env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cpu',
            'PYTHONPATH':str(ROOT/'code/scripts')+':'+str(ROOT/'src/openpi-215abfb/src'),
            'OPENPI_DATA_HOME':str(ROOT/'cache/openpi'),'HF_HOME':str(ROOT/'cache/huggingface'),
            'HF_LEROBOT_HOME':str(ROOT/'data/pi0/lerobot'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
        env.pop('PI0_INITIAL_PARAMS',None)
        command=[str(ROOT/'envs/openpi/bin/python'),'-u',str(ROOT/'code/scripts/run_pi0_evaluation.py'),
            '--checkpoint',str(cp),'--source-dataset',entry['source'],'--demo',entry['source_demo'],'--evaluation-split','train',
            '--training-provenance',str(ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'),
            '--output',str(e),'--hours','3','--inference-platform','cpu','--execute-steps','5','--denoising-steps','10',
            '--sampling-noise','gaussian','--action-representation','native','--grasping-mode','physical',
            '--reset-settle-steps','20','--policy-profile',PROFILE,'--audit-no-assistance']
        with (c/'evaluation.log').open('x') as log:
            child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(stage='B_chunk5_TRAIN_physical_in_progress',active_pid=child.pid,evaluation=str(e));code=child.wait(timeout=11100)
        save(active_pid=0,evaluation_exit_status=code)
        assessed=read(e/'status.json');result=read(e/'physical-rollout/result.json')
        if code!=0 or not actual_complete(result):
            save(stage='B_chunk5_setup_or_rollout_ABORTED_check_evidence',result_error=result.get('error'),review_directory=assessed.get('review_directory'));return
        passed=assessed.get('training_split_physical_task_success') is True
        save(stage='completed_B_chunk5_TRAIN_physical_PASS_manual_review_required' if passed else 'completed_B_chunk5_TRAIN_physical_FAILED',
            training_split_physical_task_success=passed,review_directory=assessed.get('review_directory'),
            heldout_evaluation='not_run_THIS_is_single_TRAIN_diagnostic')
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save(stage='B_chunk5_orchestration_failed',active_pid=0,error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();main(a.campaign)
