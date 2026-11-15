#!/usr/bin/env python3
"""SERVER ONLY: bounded official-data -> pi0 LoRA preparation/training pipeline.

Only the allocated GPU is used. Installation/downloads are CPU/network jobs;
JAX GPU work waits for official MoMaGen simulation to finish. No task success is
inferred from package import, checkpoint load, or decreasing training loss.
"""
import argparse
from datetime import datetime,timezone
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


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--generation-pid',type=int,required=True)
    p.add_argument('--install-pid',type=int,required=True)
    p.add_argument('--download-pid',type=int,required=True)
    p.add_argument('--hours',type=float,default=12)
    a=p.parse_args();root=a.root;c=a.campaign;gpu=resolve_gpu(root)
    if not str(c).startswith(str(root/'runs')+'/'):raise ValueError('Remote persistent campaign required')
    initial=read(c/'status.json',{})
    elapsed=max(0,time.time()-datetime.fromisoformat(initial.get('started_at',datetime.now(timezone.utc).isoformat())).timestamp())
    deadline=time.monotonic()+max(0,a.hours*3600-elapsed);script=Path(__file__).resolve().parent
    state=read(c/'status.json',{});state.update(controller_pid=os.getpid(),controller_started_at=datetime.now(timezone.utc).isoformat(),max_hours=a.hours,
        official_policy_framework='openpi pi0 JAX/Flax LoRA rank32; frozen pretrained base; no state-BC substitution',gpu_uuid=gpu)
    def save(**kw):
        state.update(kw,updated_at=datetime.now(timezone.utc).isoformat());tmp=c/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(c/'status.json')
    def stop():return (root/'runs/USER_PAUSED.json').exists() or time.monotonic()>deadline
    def wait(pid,stage):
        while alive(pid):
            if stop():save(stage='paused_or_deadline',task_success=False);return False
            save(stage=stage,waiting_pid=pid);time.sleep(30)
        return True
    env={**os.environ,'DOUBLE_ARM_ROOT':str(root),'OPENPI_ROOT':str(root/'src/openpi-215abfb'),
         'PYTHONPATH':str(script)+':'+str(root/'src/openpi-215abfb/src')+':'+str(root/'src/openpi-215abfb/scripts'),
         'CUDA_VISIBLE_DEVICES':gpu,'XLA_PYTHON_CLIENT_MEM_FRACTION':'0.90',
         'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
         'HF_HOME':str(root/'cache/huggingface'),'HF_LEROBOT_HOME':str(root/'data/pi0/lerobot'),
         'PI0_EXP_NAME':c.name,'PI0_BATCH_SIZE':'1','WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'}
    py=str(root/'envs/openpi/bin/python')
    def run(command,name,gpu_job=False,extra=None):
        if stop():return 75
        local={**env,**(extra or {})}
        if not gpu_job:local.update(JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='')
        else:
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            if any(line.split(',')[0].strip()==gpu for line in processes.splitlines()):
                save(stage='allocated_gpu_busy',error='Will not steal another task or start concurrent JAX/simulator');return 76
        with (c/(name+'.log')).open('a') as log:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=90s',str(max(1,int(deadline-time.monotonic()))),
                'taskset','-c','88-103']+command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=local)
        save(stage=name,active_pid=child.pid,active_log=str(c/(name+'.log')))
        while child.poll() is None:
            if stop():os.killpg(child.pid,signal.SIGTERM);save(stage='paused_or_deadline');return 75
            time.sleep(20)
        save(last_exit_status=child.returncode)
        return child.returncode
    save()
    if not wait(a.install_pid,'waiting_for_isolated_openpi_environment'):return 75
    if not (root/'envs/openpi/requirements-frozen.txt').exists():save(stage='environment_install_failed',error='Inspect environment-install.log');return 2
    code=run([py,str(script/'run_pi0_entrypoint.py'),'config'],'pi0_adapter_import_check')
    if code:save(stage='pi0_adapter_failed',exit_status=code);return 3
    if not wait(a.generation_pid,'waiting_for_official_rgb_demonstrations'):return 75
    generation=None
    for line in (c/'official-generation.log').read_text(errors='replace').splitlines():
        if line.startswith('RUN_DIR='):generation=Path(line.split('=',1)[1]);break
    if generation is None:save(stage='generation_run_missing');return 4
    report=read(generation/'generation-result.json',{})
    # Preserve and use only fully completed success episodes, including batches
    # that stopped at a finite bound before their requested count was reached.
    source=root/'runs/tidy_table-official-20261114-232245-5ufWgQ'
    code=run([py,str(script/'convert_momagen_pi0.py'),'--run-dir',str(source),'--run-dir',str(generation),'--min-train-episodes','4'],
             'convert_official_momagen_to_lerobot')
    if code:save(stage='visual_data_conversion_failed',generation_report=report,exit_status=code);return 5
    train=root/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1';provenance=read(train/'momagen-provenance.json',{})
    save(train_demonstrations=provenance.get('num_episodes'),train_frames=provenance.get('total_frames'),visual_dataset=str(train),generation_run=str(generation))
    code=run([py,str(script/'run_pi0_entrypoint.py'),'norms'],'compute_pi0_r1_normalization')
    if code:save(stage='normalization_failed',exit_status=code);return 6
    if not wait(a.download_pid,'waiting_for_verified_pi0_base'):return 75
    checkpoint=root/'data/pi0/checkpoints/pi0_base';verified=read(checkpoint/'download-manifest.json',{})
    if verified.get('status')!='download_verified':save(stage='pretrained_download_incomplete',error='Resume verified download, never train random weights as pi0 fine-tuning');return 7
    code=run([py,str(script/'run_pi0_entrypoint.py'),'train'],'pi0_lora_memory_smoke',True,{'PI0_TRAIN_STEPS':'20','PI0_EXP_NAME':c.name+'-smoke'})
    if code:save(stage='pi0_lora_smoke_failed',exit_status=code,error='Check memory/API/adapter; never expand to reserved GPU0');return 8
    save(pi0_gradient_smoke='completed_20_updates',training='smoke_completed_not_task_success')
    code=run([py,str(script/'run_pi0_entrypoint.py'),'train'],'pi0_lora_finetuning',True,{'PI0_TRAIN_STEPS':'30000'})
    if code:save(stage='pi0_finetuning_failed_or_timed_out',exit_status=code,task_success=False);return 9
    save(stage='pi0_finetuning_completed_evaluation_required',training='completed',task_success=False,
         next_step='Separate native R1 visual closed-loop rollout; no planner/teacher, physical mode, no object pose writes; video required')
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        if '--campaign' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--campaign')+1])/'status.json';state=read(path,{})
            state.update(stage='pi0_pipeline_failed',error=repr(error),task_success=False);path.write_text(json.dumps(state,indent=2))
        import traceback
        traceback.print_exc();raise SystemExit(1)
