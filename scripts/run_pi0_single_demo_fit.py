#!/usr/bin/env python3
"""SERVER ONLY: bounded single-demo fit diagnostic, sequential same-base freeze A/B.

Real gradient updates, NOT paper training or task rollouts. Keeps common storage
casts despite different gradient filters. No checkpoint save or live control.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
from pi0_single_demo_fit import VARIANTS,transition_indices,frame_schedule,sequence_sha256
ROOT=Path('/mnt/yuanyue/double-arm')


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def worker(campaign,variant,parent_pid):
    if os.getppid()!=parent_pid or read(campaign/'status.json').get('controller_pid')!=parent_pid:
        raise ValueError('Worker must belong to owning live controller')
    from resource_allocation import resolve_gpu
    gpu=resolve_gpu(ROOT,'1')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or os.environ.get('JAX_PLATFORMS')!='cuda':
        raise ValueError('Explicit GPU1 only, no fallback')
    plan=read(campaign/'plan.json');out=campaign/variant;out.mkdir()
    report={'mode':'bounded_single_demo_grasp_transition_fit','variant':variant,'gpu_uuid':gpu,
        'actual_optimizer_updates':0,'checkpoint_saved':False,'simulator_control_calls':0,'policy_task_success':False,
        'not_paper_training':True,'started_at':datetime.now(timezone.utc).isoformat(),'samples':[],'training_log':[]}
    def save(**updates):
        report.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(out/'report.json',report)
    save(stage='loading')
    try:
        import copy
        import dataclasses
        import functools
        import gc
        import importlib.util
        import h5py
        import numpy as np
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi import transforms
        from openpi.models import model as model_module
        from openpi.training import checkpoints,data_loader,optimizer,sharding
        from pi0_r1_config import register_config
        from pi0_single_demo_fit import initialize_state,variant_freeze_filter,vision_filter
        from preflight_pi0_accumulation_gpu import pinned_upstream_revision,trainable_fingerprint
        from probe_pi0_failure_mechanisms import digest_inputs,motion_metrics
        from momagen_pi0_schema import PROMPT,camera_keys,rgb_image
        from probe_pi0_grasp_onset import onset_frames
        if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('Wrong backend')
        source=ROOT/'src/openpi-215abfb';revision=pinned_upstream_revision(source)
        spec=importlib.util.spec_from_file_location('_single_demo_upstream_train',source/'scripts/train.py')
        upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
        base=register_config('native')
        if base.policy_metadata['initialization']['kind']!='official_pretrained_pi0_base':
            raise ValueError('Both branches must start official base, not old LoRA warm start')
        cfg=dataclasses.replace(base,freeze_filter=variant_freeze_filter(base.freeze_filter,variant),
            num_workers=0,batch_size=1,num_train_steps=plan['optimizer_updates'],ema_decay=None,
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=100,peak_lr=2.5e-5,decay_steps=2000,decay_lr=2.5e-6))
        anchor=Path(plan['norm_anchor']);contract=read(anchor/'assets/r1-adapter-contract.json')
        data=cfg.data.create(cfg.assets_dirs,cfg.model);norm_path=anchor/'assets'/data.asset_id/'norm_stats.json'
        if data.use_quantile_norm or contract.get('action_representation')!='native' or contract.get('quantile_normalization') is not False or hashlib.sha256(norm_path.read_bytes()).hexdigest()!=contract['norm_stats_sha256']:
            raise ValueError('Native mean/std diagnostic only')
        if any(hashlib.sha256((ROOT/'code/scripts'/name).read_bytes()).hexdigest()!=sha for name,sha in contract['training_source_sha256'].items()):
            raise ValueError('Actual old training adapter source changed')
        norms=checkpoints.load_norm_stats(anchor/'assets',data.asset_id)
        for key in norms:
            for field in ('mean','std'):
                np.testing.assert_array_equal(getattr(norms[key],field),getattr(data.norm_stats[key],field))
        provenance=read(ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json')
        if provenance.get('split')!='train' or len(provenance['episodes'])!=16:raise ValueError('Original TRAIN provenance required')
        entry=provenance['episodes'][0]
        if entry['source_demo']!='demo_0':raise ValueError('Unexpected source ordering')
        raw_ds=data_loader.create_torch_dataset(data,cfg.model.action_horizon,cfg.model)
        transformed_ds=data_loader.transform_dataset(raw_ds,data)
        cache={};references={};states={};inputs_hashes={};batch_digest=hashlib.sha256()
        mesh=sharding.make_mesh(1);replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        batch_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        with h5py.File(entry['source']) as f:
            g=f['data/'+entry['source_demo']];actions=g['actions'][:];source_states=g['obs/prop_state'][:]
            if hashlib.sha256(actions.tobytes()+source_states[0].tobytes()).hexdigest()!=entry['fingerprint']:
                raise ValueError('Source actions/initial observation provenance changed')
            onset=onset_frames(actions[:,13])[0];indices=transition_indices(len(actions),onset);keys=camera_keys(g['obs'])
            for index in indices:
                raw=raw_ds[index];np.testing.assert_array_equal(raw['state'],source_states[index])
                if int(np.asarray(raw['episode_index']).item())!=0 or raw['prompt']!=PROMPT:raise ValueError('Wrong episode/prompt')
                reference=actions[index:index+50];np.testing.assert_array_equal(raw['actions'],reference)
                inputs={'state':source_states[index].copy(),'prompt':PROMPT}
                for name,key in keys.items():
                    inputs[name]=rgb_image(g['obs'][key][index]);np.testing.assert_array_equal(rgb_image(raw[name]),inputs[name])
                inputs_hashes[str(index)]=digest_inputs(inputs)
                item=transformed_ds[index]
                if set(item)-{'state','actions','image','image_mask','tokenized_prompt','tokenized_prompt_mask'}:
                    raise ValueError('Unexpected model fields / index or privileged input leakage')
                # Match actual loader batching/type conversion; no scalar-mask fixture.
                batch=jax.tree.map(jnp.asarray,data_loader._collate_fn([copy.deepcopy(item)]))
                observation=model_module.Observation.from_dict(batch)
                for path,leaf in jax.tree_util.tree_flatten_with_path(batch)[0]:
                    value=np.asarray(leaf);batch_digest.update(str(path).encode());batch_digest.update(value.tobytes())
                cache[index]=jax.device_put((observation,batch['actions']),batch_shard)
                references[index]=reference.copy();states[index]=source_states[index].copy()
        del raw_ds,transformed_ds,raw,item,batch
        order=frame_schedule(indices,plan['optimizer_updates'],plan['seed'])
        save(stage='initializing_from_official_base',indices=indices,first_close=onset,source=entry,
            input_hashes=inputs_hashes,batch_sha256=batch_digest.hexdigest(),frame_schedule_sha256=sequence_sha256(order),
            lr_schedule=dataclasses.asdict(cfg.lr_schedule),seed=plan['seed'],upstream_revision=revision,
            optimizer='official clip then AdamW; fresh state; batch1',storage_policy='both variants use ORIGINAL freeze filter for storage cast',
            normalization_sha256=hashlib.sha256(norm_path.read_bytes()).hexdigest(),
            actual_training_source_matches=True)
        rng,init_rng=jax.random.split(jax.random.key(plan['seed']))
        state,state_placement=initialize_state(cfg,base.freeze_filter,init_rng,mesh,upstream)
        jax.block_until_ready(state)
        initial_hash,total_count=trainable_fingerprint(state.params)
        if plan.get('reference_initial_full_parameter_sha256') and initial_hash!=plan['reference_initial_full_parameter_sha256']:
            raise ValueError('Allocator repair changed initial full model parameters/dtypes')
        train_hash,train_count=trainable_fingerprint(state.params.filter(cfg.trainable_filter))
        img_hash,img_count=trainable_fingerprint(state.params.filter(vision_filter()))
        expected=495908624 if variant==VARIANTS[0] else 81104928
        if train_count!=expected or img_count!=414803696:raise ValueError('Unexpected real NNX gradient partition')
        dtypes={}
        for leaf in jax.tree.leaves(state.params):dtypes[str(leaf.dtype)]=dtypes.get(str(leaf.dtype),0)+int(leaf.size)
        save(initial_full_parameter_sha256=initial_hash,initial_trainable_sha256=train_hash,
            initial_vision_sha256=img_hash,total_parameter_count=total_count,trainable_parameter_count=train_count,
            vision_parameter_count=img_count,initial_storage_dtype_counts=dtypes)
        if variant==VARIANTS[1]:
            first=read(campaign/VARIANTS[0]/'report.json')
            for key in ('initial_full_parameter_sha256','initial_storage_dtype_counts','batch_sha256',
                        'frame_schedule_sha256','normalization_sha256','input_hashes','lr_schedule','seed'):
                if report[key]!=first[key]:raise ValueError(f'Invalid freeze A/B: mismatched {key}')
        std=np.asarray(read(norm_path)['norm_stats']['actions']['std'])+1e-6
        def sample(params,observation,noise):
            model=nnx.merge(state.model_def,params);model.eval()
            return model.sample_actions(jax.random.key(123),observation,num_steps=10,noise=noise[None])
        sample_jit=jax.jit(sample)
        predictions={}
        def evaluate(stage):
            save(stage='offline_'+stage+'_sampling')
            for index in indices:
                obs=cache[index][0]
                for seed in (123,124,125):
                    noise=np.random.default_rng(seed).standard_normal((50,32)).astype(np.float32)
                    with sharding.set_mesh(mesh):normalized=sample_jit(state.params,obs,noise)
                    normalized=np.asarray(normalized[0]);predicted=transforms.Unnormalize(norms,use_quantiles=False)(
                        {'actions':normalized,'state':np.asarray(obs.state[0])})['actions'][:,:21]
                    row={'stage':stage,'frame':index,'seed':seed,'input_sha256':inputs_hashes[str(index)],
                        'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                        'first1':motion_metrics(predicted,references[index],states[index],std,1),
                        'first10':motion_metrics(predicted,references[index],states[index],std,10)}
                    report['samples'].append(row);predictions[f'{stage}_{index}_{seed}']=predicted
            save()
        evaluate('initial')
        jax.clear_caches();gc.collect()
        train_jit=jax.jit(functools.partial(upstream.train_step,cfg),
            in_shardings=(replicated,state_placement,batch_shard),out_shardings=(state_placement,replicated),donate_argnums=(1,))
        save(stage='compiling_and_fitting',actual_optimizer_updates=0)
        started=time.monotonic()
        for update,index in enumerate(order):
            if (ROOT/'runs/USER_PAUSED.json').exists() or resolve_gpu(ROOT,'1')!=gpu:raise RuntimeError('Paused or allocation changed')
            with sharding.set_mesh(mesh):state,info=train_jit(rng,state,cache[index])
            jax.block_until_ready((state,info))
            values={k:float(v) for k,v in jax.device_get(info).items()}
            if not all(np.isfinite(v) for v in values.values()):raise ValueError('Nonfinite official train step')
            if int(state.step)!=update+1:raise ValueError('Wrong actual optimizer count')
            if update==0 or (update+1)%25==0:
                report['training_log'].append({'update':update+1,'frame':index,**values})
                save(stage='fitting',actual_optimizer_updates=update+1,
                    updates_per_second=(update+1)/(time.monotonic()-started),last_info=values)
                print(json.dumps({'variant':variant,'update':update+1,**values}),flush=True)
        counts={jax.tree_util.keystr(path):int(leaf) for path,leaf in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]
                if path and getattr(path[-1],'name',None)=='count'}
        if not counts or any(v!=plan['optimizer_updates'] for v in counts.values()):raise ValueError('AdamW/LR counter mismatch')
        final_hash,_=trainable_fingerprint(state.params.filter(cfg.trainable_filter))
        final_img_hash,_=trainable_fingerprint(state.params.filter(vision_filter()))
        if final_hash==train_hash:raise ValueError('Trainables never changed')
        if variant==VARIANTS[1] and final_img_hash!=img_hash:raise ValueError('Frozen vision unexpectedly changed')
        save(stage='final_parameter_checks',actual_optimizer_updates=int(state.step),optimizer_and_lr_counts=counts,
             final_trainable_sha256=final_hash,final_vision_sha256=final_img_hash,
             vision_parameters_changed=final_img_hash!=img_hash,device_memory_stats=jax.devices()[0].memory_stats())
        # Free optimizer buffers and training executable before final sampling.
        del train_jit
        state=dataclasses.replace(state,opt_state=None)
        jax.clear_caches();gc.collect()
        evaluate('final')
        np.savez_compressed(out/'offline-predictions.npz',**predictions)
        save(stage='completed_single_demo_fit_NOT_task_success',actual_optimizer_updates=int(state.step))
    except BaseException as error:
        save(stage='single_demo_fit_failed',error=repr(error));raise


def controller(campaign):
    import fcntl
    import signal
    import subprocess
    import sys
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import inspect_release,process_alive
    if not campaign.resolve().is_relative_to(ROOT/'runs') or campaign.exists() or campaign.is_symlink():raise ValueError('Fresh remote campaign required')
    gpu=resolve_gpu(ROOT,'1');lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(ROOT/'runs/latest-gpu1-job.json');prior=read(old['status_file'])
    repaired_from=None;repaired_initial_hash=None
    if process_alive(old.get('controller_pid')):raise ValueError('Current GPU1 owner still alive')
    if prior.get('stage')=='controlled_single_demo_fit_failed':
        previous=Path(old['campaign'])
        reports=list(previous.glob('*/report.json'))
        if prior.get('completed_variants') or prior.get('active_pid')!=0 or not reports or any(read(p).get('actual_optimizer_updates')!=0 for p in reports):
            raise ValueError('Do not auto-retry partial training or change a failed recipe')
        repaired_from=str(previous)
        repaired_initial_hash=read(reports[0]).get('initial_full_parameter_sha256')
        precision=Path((ROOT/'runs/latest-pi0-restore-precision-probe.txt').read_text().strip())
        prior=read(precision/'status.json')
    else:precision=Path(old['campaign'])
    if prior.get('stage')!='completed_TRAIN_restore_precision_probe_NOT_task_success':
        raise ValueError('Precision probe not completed/verified')
    if hashlib.sha256((precision/'report.json').read_bytes()).hexdigest()!=prior['report_sha256']:
        raise ValueError('Precision probe immutable report hash mismatch')
    inspect_release(Path((ROOT/'runs/latest-pi0-coverage-campaign.txt').read_text().strip()))
    campaign.mkdir()
    plan={'mode':'single_demo_fit_freeze_AB_v1','optimizer_updates':2000,'seed':42,'variants':list(VARIANTS),
          'norm_anchor':prior['checkpoint'],'initialization':'official pi0_base BOTH branches, fresh optimizer',
          'action_representation':'native','normalization':'original TRAIN mean/std','batch':1,'rank':32,
          'storage_cast':'original freeze filter shared in BOTH conditions','pregradient_attempt_repaired_from':repaired_from,
          'reference_initial_full_parameter_sha256':repaired_initial_hash,
          'gpu_allocator_fraction':0.90,'gpu_preallocate':True,'changed_only':'additional image gradient freezing',
          'checkpoint_saved':False,'simulator_control_calls':0,'not_paper_training':True,
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'helper_sha256':hashlib.sha256(Path(__file__).with_name('pi0_single_demo_fit.py').read_bytes()).hexdigest()}
    write(campaign/'plan.json',plan)
    status={'stage':'starting_controlled_fit','controller_pid':os.getpid(),'gpu_uuid':gpu,
        'started_at':datetime.now(timezone.utc).isoformat(),'policy_training':'bounded_single_demo_diagnostic_NOT_paper',
        'learned_policy_task_success':False,'simulator_control_calls':0,'checkpoint_saved':False,'completed_variants':[]}
    def save(**updates):
        status.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(campaign/'status.json',status)
    save()
    write(ROOT/'runs/latest-gpu1-job.json',{'campaign':str(campaign),'status_file':str(campaign/'status.json'),
        'controller_pid':os.getpid(),'gpu_uuid':gpu,'max_gpu_count':1,'role':'bounded_single_demo_fit_freeze_AB_NOT_paper_training'})
    m=read(ROOT/'runs/latest-multi-gpu-jobs.json');m['completed_pi0_restore_precision']=old
    m['GPU1']={'campaign':str(campaign),'job_pointer':str(ROOT/'runs/latest-gpu1-job.json'),'gpu_uuid':gpu,'role':'bounded_single_demo_fit_freeze_AB_NOT_paper_training'}
    m['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',m)
    (ROOT/'runs/latest-pi0-single-demo-fit.txt').write_text(str(campaign)+'\n')
    child=None
    def terminate(*_):raise TimeoutError('Bounded controller watchdog requested termination')
    signal.signal(signal.SIGTERM,terminate)
    try:
        for variant in VARIANTS:
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'JAX_PLATFORMS':'cuda','OPENPI_DATA_HOME':str(ROOT/'cache/openpi'),
                 'DOUBLE_ARM_ROOT':str(ROOT),'HF_HOME':str(ROOT/'cache/huggingface'),
                 'HF_LEROBOT_HOME':str(ROOT/'data/pi0/lerobot'),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                 'XLA_PYTHON_CLIENT_PREALLOCATE':'true','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.90','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
            env.pop('PI0_INITIAL_PARAMS',None)
            with (campaign/(variant+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--campaign',str(campaign),
                    '--worker',variant,'--parent-pid',str(os.getpid())],env=env,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(stage='single_demo_fit_'+variant,active_pid=child.pid,active_variant=variant)
                code=child.wait(timeout=3600)
            if code!=0:raise RuntimeError(f'{variant} worker exit{code}; no automatic recipe change/fallback')
            result=read(campaign/variant/'report.json')
            if result.get('stage')!='completed_single_demo_fit_NOT_task_success' or result.get('actual_optimizer_updates')!=2000:
                raise ValueError('Worker did not complete actual requested diagnostic updates')
            status['completed_variants'].append(variant);save(active_pid=0)
        save(stage='completed_controlled_single_demo_fit_AB_NOT_task_success',active_pid=0)
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        save(stage='controlled_single_demo_fit_failed',active_pid=0,error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--worker',choices=VARIANTS);p.add_argument('--parent-pid',type=int);a=p.parse_args()
    if a.worker:worker(a.campaign,a.worker,a.parent_pid)
    else:controller(a.campaign)
