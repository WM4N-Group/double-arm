#!/usr/bin/env python3
"""SERVER: executable-source gate -> native pi0 warm-start -> honest evaluations.

No fresh twelve-hour user window: GPU1 time was explicitly renewed. Individual
jobs remain bounded. Teacher action replay is a diagnostic prerequisite, NEVER
reported as learned-policy success. Old checkpoints/optimizer state stay intact.
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
from pi0_initialization_contract import training_initialization
from pi0_review_contract import fingerprint_checkpoint


def read(path):return json.loads(Path(path).read_text())


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False


def source_replay_passed(report):
    n=report.get('num_recorded_actions',0)
    return (report.get('mode')=='recorded_action_replay_audit' and report.get('learned_policy') is False
        and report.get('initial_state_load_count')==1 and type(n) is int and n>=30
        and report.get('completed_actions')==n and report.get('teacher_action_reads')==n
        and report.get('model_inference_calls')==0 and report.get('goal_success') is True
        and any(report.get('grasped_ever',[])) and not report.get('error'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--audit-pid',type=int,required=True)
    p.add_argument('--audit-result',type=Path,required=True);p.add_argument('--initial-params',type=Path,required=True)
    p.add_argument('--steps',type=int,default=90000);args=p.parse_args()
    root=Path('/mnt/yuanyue/double-arm');c=args.campaign;script=Path(__file__).resolve().parent
    if not c.resolve().is_relative_to(root/'runs') or not 1000<=args.steps<=200000:raise ValueError('Invalid bounded experiment')
    lock=(c/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(root/'src/openpi-215abfb/src'))
    gpu=resolve_gpu(root)
    if allocation(root).get('compute_time_budget')!='open_ended_explicit_user_authorization':raise ValueError('Explicit renewed budget required')
    initial,initialization=training_initialization(root,str(args.initial_params),'native')
    state=read(c/'status.json');child=None
    def save(**updates):
        state.update(updates,updated_at=datetime.now(timezone.utc).isoformat());tmp=c/'status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(c/'status.json')
    def check():
        if (root/'runs/USER_PAUSED.json').exists():raise RuntimeError('User pause')
        if resolve_gpu(root)!=gpu:raise RuntimeError('Allocation changed')
    def interrupted(signum,frame):raise InterruptedError(f'Controller signal{signum}')
    signal.signal(signal.SIGTERM,interrupted)
    env={**os.environ,'DOUBLE_ARM_ROOT':str(root),'OPENPI_ROOT':str(root/'src/openpi-215abfb'),
        'PYTHONPATH':str(script)+':'+str(root/'src/openpi-215abfb/src')+':'+str(root/'src/openpi-215abfb/scripts'),
        'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cuda','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.90',
        'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
        'HF_HOME':str(root/'cache/huggingface'),'HF_LEROBOT_HOME':str(root/'data/pi0/lerobot'),
        'OPENPI_DATA_HOME':str(root/'cache/openpi'),'XDG_CACHE_HOME':str(root/'cache/xdg'),'TMPDIR':str(root/'cache/tmp'),
        'PI0_INITIAL_PARAMS':str(initial),'PI0_ACTION_REPRESENTATION':'native','PI0_BATCH_SIZE':'1',
        'PI0_LR_DECAY_STEPS':str(args.steps),'PI0_EXP_NAME':c.name,'WANDB_MODE':'disabled','TOKENIZERS_PARALLELISM':'false'}
    py=str(root/'envs/openpi/bin/python')
    def run(command,name,seconds,extra=None):
        nonlocal child
        check()
        for _ in range(60):
            query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
            if not any(line.split(',')[0].strip()==gpu for line in query.splitlines()):break
            check();time.sleep(5)
        else:raise RuntimeError('Allowed GPU occupied; will not steal it or use GPU0')
        with (c/(name+'.log')).open('x') as log:
            child=subprocess.Popen(['timeout','--signal=TERM','--kill-after=90s',str(seconds),'taskset','-c','88-103',*command],
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env={**env,**(extra or {})})
        save(stage=name,active_pid=child.pid,active_log=str(c/(name+'.log')))
        while child.poll() is None:check();time.sleep(10)
        save(last_exit_status=child.returncode)
        return child.returncode
    try:
        save(stage='waiting_exact_training_source_action_audit',controller_pid=os.getpid(),initialization=initialization)
        deadline=time.monotonic()+3600
        while alive(args.audit_pid):
            check()
            if time.monotonic()>deadline:raise TimeoutError('Source audit watchdog')
            time.sleep(5)
        audit=read(args.audit_result);save(source_action_audit=audit)
        if not source_replay_passed(audit):save(stage='source_actions_not_reproducible_no_blind_training',active_pid=0);return 6
        save(stage='verifying_warm_start_inference_tree')
        fingerprint=fingerprint_checkpoint(initial.parent);(c/'initial-checkpoint-fingerprint.json').write_text(json.dumps(fingerprint,indent=2))
        # Same native dataset/statistics, not a new normalization or label variant.
        from pi0_r1_config import register_config
        os.environ.update({key:env[key] for key in ('DOUBLE_ARM_ROOT','PI0_INITIAL_PARAMS','PI0_LR_DECAY_STEPS','PI0_EXP_NAME','PI0_ACTION_REPRESENTATION')})
        config=register_config();data=config.data.create(config.assets_dirs,config.model)
        import numpy as np
        source_stats=read(initial.parent/'assets'/data.asset_id/'norm_stats.json')['norm_stats']
        current_stats=read(Path(config.data.assets.assets_dir or config.assets_dirs)/data.asset_id/'norm_stats.json')['norm_stats']
        if source_stats.keys()!=current_stats.keys():raise ValueError('Normalization feature keys changed')
        for feature,values in source_stats.items():
            if values.keys()!=current_stats[feature].keys():raise ValueError('Normalization statistic keys changed')
            for name,value in values.items():
                if value is None or current_stats[feature][name] is None:
                    if value is not current_stats[feature][name]:raise ValueError('Missing normalization statistic')
                elif not np.allclose(value,current_stats[feature][name],rtol=1e-6,atol=1e-7):raise ValueError('Warm-start normalization mismatch')
        save(initial_checkpoint_fingerprint=fingerprint['sha256'],normalization_verified_unchanged=True)
        code=run([py,str(script/'run_pi0_entrypoint.py'),'train'],'coverage_memory_smoke',2400,
            {'PI0_TRAIN_STEPS':'20','PI0_EXP_NAME':c.name+'-smoke'})
        if code:save(stage='coverage_smoke_failed_no_formal_training',training='not_started',task_success=False,active_pid=0);return code
        save(training='in_progress',real_smoke_gradient_updates=20)
        code=run([py,str(script/'run_pi0_entrypoint.py'),'train'],'native_coverage_finetuning',43200,{'PI0_TRAIN_STEPS':str(args.steps)})
        if code:save(stage='coverage_training_failed',training='failed',task_success=False,active_pid=0);return code
        checkpoint=root/'runs/pi0-checkpoints/pi0_r1_tidy_table_lora32'/c.name/str(args.steps-1)
        if not (checkpoint/'assets/r1-adapter-contract.json').is_file():raise ValueError('Final checkpoint contract missing')
        save(training='completed',checkpoint=str(checkpoint),completed_additional_updates=args.steps,task_success=False)
        train_provenance=root/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'
        train=read(train_provenance)['episodes'][0];holdout=read(root/'data/pi0/lerobot/momagen/r1_tidy_table_holdout_v1/momagen-provenance.json')['episodes'][0]
        def evaluate(entry,split,mode,name):
            output=c/name
            code=run([py,str(script/'run_pi0_evaluation.py'),'--checkpoint',str(checkpoint),'--source-dataset',entry['source'],
                '--demo',entry['source_demo'],'--evaluation-split',split,'--training-provenance',str(train_provenance),
                '--output',str(output),'--hours','2','--action-representation','native','--reset-settle-steps','20',
                '--grasping-mode',mode],name,7500,{'PI0_INITIAL_PARAMS':'','PI0_LR_DECAY_STEPS':'30000'})
            return code,read(output/'status.json')
        code,assessed=evaluate(train,'train','configured','coverage-training-author-diagnostic')
        save(training_condition_result=assessed)
        if code or assessed.get('training_split_author_config_task_success') is not True:
            save(stage='coverage_trained_training_condition_still_failed',task_success=False,active_pid=0);return 7
        code,assessed=evaluate(holdout,'validation','physical','coverage-heldout-physical-evaluation')
        save(stage='coverage_heldout_physical_success_manual_review_required' if assessed.get('task_success') is True else 'coverage_heldout_physical_failed',
             heldout_result=assessed,task_success=assessed.get('task_success') is True,active_pid=0)
        return code
    except BaseException as error:
        save(stage='coverage_pipeline_failed',error=repr(error),task_success=False)
        if child is not None and child.poll() is None:
            try:os.killpg(child.pid,signal.SIGINT)
            except ProcessLookupError:pass
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
        raise


if __name__=='__main__':raise SystemExit(main())
