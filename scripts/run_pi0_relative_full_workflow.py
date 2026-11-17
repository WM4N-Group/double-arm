#!/usr/bin/env python3
"""One GPU1 owner: full-demo fit -> real checkpoint reload -> honest auto evals.

GPU0 official generation stays untouched. No paper/physical success from train
loss or source-assisted result. Existing controls/safety/task criteria unchanged.
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
from pi0_relative_full_profile import ROOT,PROFILE,RUNTIME_FILES,full_schedule,sha
from pi0_single_demo_fit import sequence_sha256


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    p=Path(path);t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));t.replace(p)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args();c=a.campaign
    if not c.resolve().is_relative_to(ROOT/'runs') or c.exists() or c.is_symlink():raise ValueError('Fresh remote campaign required')
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import inspect_release,process_alive
    gpu=resolve_gpu(ROOT,'1');lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(ROOT/'runs/latest-gpu1-job.json');previous=read(old['status_file'])
    if previous.get('stage')!='completed_relative_joint_fit_NOT_task_success' or process_alive(old.get('controller_pid')):
        raise ValueError('Relative small-fit owner not completed/released or GPU1 pointer changed')
    if not (Path(old['campaign'])/'encoding-comparison.json').exists():raise ValueError('Completed encoding comparison required')
    inspect_release(Path((ROOT/'runs/latest-pi0-coverage-campaign.txt').read_text().strip()))
    preflight=Path((ROOT/'runs/latest-pi0-relative-preflight.txt').read_text().strip());probe=read(preflight/'report.json')
    if probe['stage']!='completed_relative_joint_cpu_preflight' or probe['windows']!=96 or probe['roundtrip_max_abs']>1e-6:
        raise ValueError('Relative input/inverse preflight not passed')
    if sha(preflight/'relative-norm-stats.json')!=probe['normalization_sha256']:raise ValueError('Preflight norm changed')
    ppath=ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json';provenance=read(ppath)
    if provenance['split']!='train' or provenance['episodes'][0]['frames']!=1714:raise ValueError('Wrong source')
    before=read(Path(old['campaign'])/'original_vision_trainable/report.json')
    plan={'policy_profile':PROFILE,'optimizer_updates':20000,'batch_size':1,'seed':42,'rank':32,
        'controller_pid':os.getpid(),'relative_preflight':str(preflight),'relative_norm_source_sha256':probe['normalization_sha256'],
        'norm_anchor':probe['checkpoint'],'training_provenance_sha256':sha(ppath),
        'frame_schedule_sha256':sequence_sha256(full_schedule(1714,636,20000,42)),
        'initial_full_parameter_sha256':before['initial_full_parameter_sha256'],
        'training_scope':'ONE complete TRAIN/demo0 +50%9 transition windows, not paper recipe',
        'parameter_initialization':'official pi0_base, fresh AdamW; not small-fit/v3 warmstart',
        'lr_schedule':{'warmup_steps':1000,'peak_lr':2.5e-5,'decay_steps':30000,'decay_lr':2.5e-6},
        'save_every_updates':5000,'save_all_new_checkpoints':True,'inference_platform':'cpu',
        'automatic_evals':'TRAIN/configured -> ifsuccess TRAIN/physical -> ifsuccess heldout/physical; no bypass',
        'runtime_source_sha256':{name:sha(ROOT/'code/scripts'/name) for name in RUNTIME_FILES},
        'controller_source_sha256':sha(Path(__file__)),'trainer_source_sha256':sha(ROOT/'code/scripts/train_pi0_relative_full_demo.py')}
    c.mkdir();write(c/'plan.json',plan)
    status={'stage':'starting_relative_full_demo_fit','controller_pid':os.getpid(),'gpu_uuid':gpu,
        'policy_profile':PROFILE,'training':'in_progress','training_is_diagnostic':True,'formal_training_completed':False,
        'train_demonstrations':1,'train_frames':1714,'paper_reproduction_complete':False,'task_success':False,
        'started_at':datetime.now(timezone.utc).isoformat(),'previous_small_fit':old['campaign']}
    def save(**changes):
        status.update(changes,updated_at=datetime.now(timezone.utc).isoformat());write(c/'status.json',status)
    save()
    write(ROOT/'runs/latest-gpu1-job.json',{'controller_pid':os.getpid(),'campaign':str(c),'status_file':str(c/'status.json'),
        'gpu_uuid':gpu,'max_gpu_count':1,'role':'relative_FULL_DEMO_diagnostic_training_reload_auto_evals_NOT_paper'})
    m=read(ROOT/'runs/latest-multi-gpu-jobs.json');m['completed_relative_joint_small_fit']=old
    m['GPU1']={'campaign':str(c),'job_pointer':str(ROOT/'runs/latest-gpu1-job.json'),'gpu_uuid':gpu,
        'role':'relative_FULL_DEMO_diagnostic_training_reload_auto_evals_NOT_paper'}
    m['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',m)
    (ROOT/'runs/latest-pi0-relative-full-demo-workflow.txt').write_text(str(c)+'\n')
    script=Path(__file__).resolve().parent;python=ROOT/'envs/openpi/bin/python'
    env={**os.environ,'DOUBLE_ARM_ROOT':str(ROOT),'OPENPI_ROOT':str(ROOT/'src/openpi-215abfb'),
        'PYTHONPATH':str(script)+':'+str(ROOT/'src/openpi-215abfb/src'),
        'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cuda','XLA_PYTHON_CLIENT_PREALLOCATE':'true','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.90',
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'OPENPI_DATA_HOME':str(ROOT/'cache/openpi'),'HF_HOME':str(ROOT/'cache/huggingface'),
        'HF_LEROBOT_HOME':str(ROOT/'data/pi0/lerobot'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    env.pop('PI0_INITIAL_PARAMS',None)
    child=None
    def abort(*_):raise TimeoutError('Bounded controller watchdog')
    signal.signal(signal.SIGTERM,abort)
    def run(command,name,seconds,extra=None):
        nonlocal child
        if (ROOT/'runs/USER_PAUSED.json').exists():raise RuntimeError('User paused')
        with (c/(name+'.log')).open('x') as log:
            child=subprocess.Popen(command,env={**env,**(extra or {})},stdout=log,stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,start_new_session=True)
            save(active_pid=child.pid,active_log=str(c/(name+'.log')))
            code=child.wait(timeout=seconds)
        save(active_pid=0)
        return code
    try:
        save(stage='relative_full_demo_training')
        code=run([str(python),'-u',str(script/'train_pi0_relative_full_demo.py'),'--campaign',str(c)],'training',10800)
        trained=read(c/'training-report.json')
        if code!=0 or trained.get('training')!='completed' or trained.get('actual_optimizer_updates')!=20000:
            raise RuntimeError('Full-demo training/checkpoint did not finish; no evaluation of partial weights')
        from pi0_relative_full_profile import load_contract
        checkpoint=Path(trained['checkpoint']);load_contract(checkpoint)
        save(stage='relative_full_demo_checkpoint_CPU_reload',training='completed',checkpoint=str(checkpoint),training_exit_status=code)
        code=run([str(python),'-u',str(script/'verify_pi0_relative_full_checkpoint.py'),'--campaign',str(c),
            '--output',str(c/'checkpoint-reload.json')],'checkpoint-reload',2400,{'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'})
        if code!=0 or read(c/'checkpoint-reload.json').get('stage')!='cpu_reload_parameters_exact_actions_finite_NOT_task_success':
            raise RuntimeError('New-profile checkpoint reload failed; do not start simulator')
        steps=[('train-author-diagnostic','train','configured'),('train-physical-diagnostic','train','physical'),
               ('heldout-physical-evaluation','validation','physical')]
        for name,split,grasp in steps:
            source_entry=provenance['episodes'][0] if split=='train' else read(ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_holdout_v1/momagen-provenance.json')['episodes'][0]
            evaluation=c/name;save(stage=name,evaluation=str(evaluation))
            code=run([str(python),'-u',str(script/'run_pi0_evaluation.py'),'--checkpoint',str(checkpoint),
                '--source-dataset',source_entry['source'],'--demo',source_entry['source_demo'],'--evaluation-split',split,
                '--training-provenance',str(ppath),'--output',str(evaluation),'--hours','3','--inference-platform','cpu',
                '--execute-steps','10','--denoising-steps','10','--sampling-noise','gaussian','--action-representation','native',
                '--grasping-mode',grasp,'--reset-settle-steps','20','--policy-profile',PROFILE],name,11100,
                {'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cpu'})
            assessed=read(evaluation/'status.json')
            field='training_split_author_config_task_success' if grasp=='configured' else 'training_split_physical_task_success' if split=='train' else 'task_success'
            if code!=0 or assessed.get(field) is not True:
                save(stage=name+'_FAILED',last_evaluation_exit_status=code,task_success=False,
                    review_directory=assessed.get('review_directory'));return 0
            save(last_completed_evaluation=name,last_review_directory=assessed.get('review_directory'))
        save(stage='relative_full_demo_HELDOUT_physical_success_MANUAL_REVIEW_REQUIRED',task_success=True)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=90)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save(stage='relative_full_demo_workflow_failed',active_pid=0,error=repr(error)[:3000],task_success=False);raise


if __name__=='__main__':main()
