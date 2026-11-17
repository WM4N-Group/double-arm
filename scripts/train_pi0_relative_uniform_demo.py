#!/usr/bin/env python3
"""SERVER GPU1: NEW uniform whole-demo sampling-only B diagnostic, save real params.

20k real original-pi0 updates, official base/fresh optimizer. Host RGB cache,
ONE device batch, native-base/relative-joint labels. Not the paper recipe.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time
from pi0_relative_full_profile import ROOT,PROFILE,ASSET_ID,MODEL_ENCODING,RUNTIME_FILES,sha
from pi0_uniform_sampling_diagnostic import uniform_schedule as full_schedule,VARIANT,RECIPE


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)


def run(c):
    from resource_allocation import resolve_gpu
    gpu=resolve_gpu(ROOT,'1')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or os.environ.get('JAX_PLATFORMS')!='cuda':raise ValueError('Only authorized GPU1')
    plan=json.loads((c/'plan.json').read_text())
    if plan.get('controller_pid')!=os.getppid():raise ValueError('Worker must belong to owning full-workflow controller')
    if sha(Path(__file__))!=plan['trainer_source_sha256']:raise ValueError('Actual B trainer source changed')
    if any(sha(ROOT/'code/scripts'/name)!=value for name,value in plan['runtime_source_sha256'].items()):raise ValueError('Profile runtime sources changed')
    if (plan.get('training_variant')!=VARIANT or sha(ROOT/'code/scripts/pi0_uniform_sampling_diagnostic.py')!=plan['sampler_source_sha256']):raise ValueError('Uniform variant/sampler changed')
    if plan['policy_profile']!=PROFILE or plan['optimizer_updates']!=20000 or plan['batch_size']!=1:raise ValueError('Wrong explicit plan')
    report={'stage':'loading','policy_profile':PROFILE,'training':'in_progress','training_is_diagnostic':True,
        'training_variant':VARIANT,'sampling_recipe':RECIPE,'actual_optimizer_updates':0,'checkpoint_saved':False,'simulator_control_calls':0,'task_success':False,
        'paper_reproduction_complete':False,'started_at':datetime.now(timezone.utc).isoformat(),'training_log':[],'samples':[]}
    def save(**updates):
        report.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(c/'training-report.json',report)
    save()
    try:
        import copy
        import dataclasses
        import functools
        import gc
        import importlib.util
        import types
        import h5py
        import numpy as np
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi import transforms
        from openpi.models import model as model_module
        from openpi.training import checkpoints,data_loader,optimizer,sharding
        from pi0_r1_config import register_config
        from pi0_single_demo_fit import initialize_state,inference_payload,sequence_sha256,transition_indices
        from pi0_relative_full_profile import data_profile
        from preflight_pi0_accumulation_gpu import pinned_upstream_revision,trainable_fingerprint
        from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
        from probe_pi0_failure_mechanisms import digest_inputs,motion_metrics
        source=ROOT/'src/openpi-215abfb';revision=pinned_upstream_revision(source)
        if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('Wrong backend')
        spec=importlib.util.spec_from_file_location('_full_demo_upstream_train',source/'scripts/train.py')
        upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
        base=register_config('native')
        if base.policy_metadata['initialization']['kind']!='official_pretrained_pi0_base':raise ValueError('No v3 warm start')
        cfg=dataclasses.replace(base,name=PROFILE,batch_size=1,num_workers=0,num_train_steps=20000,ema_decay=None,
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1000,peak_lr=2.5e-5,decay_steps=30000,decay_lr=2.5e-6))
        original_data=cfg.data.create(cfg.assets_dirs,cfg.model)
        preflight=Path(plan['relative_preflight']);norm_json=json.loads((preflight/'relative-norm-stats.json').read_text())
        if sha(preflight/'relative-norm-stats.json')!=plan['relative_norm_source_sha256']:raise ValueError('Preflight norm changed')
        from openpi.shared.normalize import NormStats
        norms={key:NormStats(**{k:np.asarray(v) if v is not None else None for k,v in values.items()}) for key,values in norm_json['norm_stats'].items()}
        data=data_profile(original_data,norms)
        transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(norms,use_quantiles=False),*data.model_transforms.inputs])
        output_transform=transforms.compose([*data.model_transforms.outputs,transforms.Unnormalize(norms,use_quantiles=False),*data.data_transforms.outputs])
        provenance_path=ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'
        if sha(provenance_path)!=plan['training_provenance_sha256']:raise ValueError('TRAIN provenance changed')
        provenance=json.loads(provenance_path.read_text());entry=provenance['episodes'][0]
        if provenance['split']!='train' or entry['source_demo']!='demo_0' or entry['frames']!=1714:raise ValueError('Wrong source scope')
        host_cache=[];input_hashes={};references={};raw_states={};cache_hash=hashlib.sha256();maximum=0
        save(stage='caching_full_source_on_HOST_not_GPU',train_demonstrations=1,train_frames=1714,
            source=entry,source_grasping='author_assisted',label_horizon=50,cache_location='server_HOST_RAM',lr_schedule=dataclasses.asdict(cfg.lr_schedule))
        with h5py.File(entry['source']) as f:
            g=f['data/'+entry['source_demo']];actions=g['actions'][:];states=g['obs/prop_state'][:];keys=camera_keys(g['obs'])
            if hashlib.sha256(actions.tobytes()+states[0].tobytes()).hexdigest()!=entry['fingerprint']:raise ValueError('Source fingerprint changed')
            onset=int(np.flatnonzero(actions[:,13]<.5)[0]);eval_indices=sorted(set([0,len(actions)//3,2*len(actions)//3,len(actions)-1]+transition_indices(len(actions),onset)))
            for i in range(len(actions)):
                inputs={name:rgb_image(g['obs'][key][i]) for name,key in keys.items()}
                inputs.update(state=states[i].copy(),prompt=PROMPT)
                ref=actions[np.minimum(np.arange(i,i+50),len(actions)-1)]
                item=transform({**inputs,'actions':ref})
                if set(item)-{'state','actions','image','image_mask','tokenized_prompt','tokenized_prompt_mask'}:
                    raise ValueError('Index/phase/privileged input leakage')
                recovered=output_transform({'state':item['state'].copy(),'actions':item['actions'].copy()})['actions']
                delta=float(np.max(np.abs(recovered-ref)));maximum=max(maximum,delta)
                if delta>1e-6:raise ValueError('Full-demo native inverse roundtrip failed')
                np.testing.assert_array_equal(item['state'][21:],0);np.testing.assert_array_equal(item['actions'][:,21:],0)
                input_hashes[str(i)]=digest_inputs(inputs);cache_hash.update(input_hashes[str(i)].encode());host_cache.append(item)
                if i in eval_indices:references[i]=ref.copy();raw_states[i]=states[i].copy()
                if i%100==0:save(cached_frames=i+1)
        if cache_hash.hexdigest()!=plan['baseline_source_input_sha256']:raise ValueError('B source observations differ from paired A')
        order=full_schedule(len(host_cache),onset,20000,42)
        order_hash=sequence_sha256(order)
        if order_hash!=plan['frame_schedule_sha256']:raise ValueError('Training schedule changed')
        save(stage='initializing_official_base',cached_frames=len(host_cache),first_close=onset,eval_indices=eval_indices,
            frame_schedule_sha256=order_hash,source_input_sha256=cache_hash.hexdigest(),all1714_roundtrip_max_abs=maximum,
            runtime_source_sha256={name:sha(ROOT/'code/scripts'/name) for name in RUNTIME_FILES})
        mesh=sharding.make_mesh(1);replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
        batch_shard=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rng,init_rng=jax.random.split(jax.random.key(42));state,placement=initialize_state(cfg,base.freeze_filter,init_rng,mesh,upstream)
        jax.block_until_ready(state)
        initial,count=trainable_fingerprint(state.params)
        if initial!=plan['initial_full_parameter_sha256']:raise ValueError('Official base/LoRA initialization differs from diagnosed variant')
        train_initial,train_count=trainable_fingerprint(state.params.filter(cfg.trainable_filter))
        if train_count!=495908624:raise ValueError('Wrong trainable partition')
        save(initial_full_parameter_sha256=initial,trainable_parameters=train_count,total_parameters=count)
        experiment_id=c.parent.name if c.name=='campaign' else c.name
        checkpoint_root=ROOT/'runs/pi0-checkpoints'/PROFILE/experiment_id
        manager,resuming=checkpoints.initialize_checkpoint_dir(checkpoint_root,keep_period=1,overwrite=False,resume=False)
        if resuming:raise ValueError('Unexpected optimizer resume')
        loader_assets=types.SimpleNamespace(data_config=lambda:data)
        def optimizer_counts(s):
            return {jax.tree_util.keystr(path):int(leaf) for path,leaf in jax.tree_util.tree_flatten_with_path(s.opt_state)[0]
                    if path and getattr(path[-1],'name',None)=='count'}
        def save_checkpoint(s,updates):
            if sha(Path(__file__))!=plan['trainer_source_sha256']:raise ValueError('In-flight B trainer source changed')
            if sha(ROOT/'code/scripts/pi0_uniform_sampling_diagnostic.py')!=plan['sampler_source_sha256']:raise ValueError('In-flight sampler changed')
            if any(sha(ROOT/'code/scripts'/name)!=value for name,value in plan['runtime_source_sha256'].items()):raise ValueError('In-flight profile source changed')
            counts=optimizer_counts(s)
            if int(s.step)!=updates or not counts or any(v!=updates for v in counts.values()):raise ValueError('Actual AdamW/LR count mismatch')
            save(stage='saving_real_checkpoint',actual_optimizer_updates=updates)
            checkpoints.save_state(manager,s,loader_assets,updates-1);manager.wait_until_finished()
            cp=checkpoint_root/str(updates-1)
            if not json.loads((cp/'_CHECKPOINT_METADATA').read_text()).get('commit_timestamp_nsecs'):raise ValueError('Uncommitted Orbax checkpoint')
            saved=checkpoints.load_norm_stats(cp/'assets',ASSET_ID)
            for key in norms:
                for field in ('mean','std'):np.testing.assert_array_equal(getattr(saved[key],field),getattr(norms[key],field))
            dtype_counts={}
            for leaf in jax.tree.leaves(s.params):dtype_counts[str(leaf.dtype)]=dtype_counts.get(str(leaf.dtype),0)+int(leaf.size)
            full_hash,_=trainable_fingerprint(s.params)
            profile_contract={'schema_version':1,'policy_profile':PROFILE,'model':'pi0','lora_rank':32,
                'model_action_encoding':MODEL_ENCODING,'motor_action_encoding':'native_21_after_inverse',
                'normalization_mode':'mean_std','norm_stats_asset_id':ASSET_ID,'original_proprio_dim':21,
                'model_action_dim':32,'action_horizon':50,'parameter_restore_dtype':'preserve_saved',
                'training_is_diagnostic':True,'paper_reproduction_complete':False,'training_completed':updates==20000,
                'train_demonstrations':1,'train_frames':1714,'training_source_demo':'demo_0','optimizer_updates':updates,
                'optimizer_and_lr_counts':counts,'openpi_revision':revision,'teacher_enabled':False,'planner_enabled':False,
                'object_pose_inputs':False,'norm_stats_sha256':sha(cp/'assets'/ASSET_ID/'norm_stats.json'),
                'full_parameter_sha256':full_hash,'frame_schedule_sha256':order_hash,'parameter_dtype_counts':dtype_counts,
                'runtime_source_sha256':{name:sha(ROOT/'code/scripts'/name) for name in RUNTIME_FILES},
                'source_preflight_norm_sha256':plan['relative_norm_source_sha256'],'source_provenance':entry,
                'training_source_sha256':{'train_pi0_relative_uniform_demo.py':sha(Path(__file__)),
                    'pi0_uniform_sampling_diagnostic.py':sha(ROOT/'code/scripts/pi0_uniform_sampling_diagnostic.py'),
                    'pi0_single_demo_fit.py':sha(ROOT/'code/scripts/pi0_single_demo_fit.py')},
                'sampling_recipe':RECIPE,'training_variant':VARIANT,'paired_baseline_campaign':plan['paired_baseline_campaign'],
                'paired_common_fields_exact':plan['paired_common_fields_exact'],'baseline_source_input_sha256':plan['baseline_source_input_sha256'],
                'lr_schedule':dataclasses.asdict(cfg.lr_schedule),'initialization':'official_pi0_base_fresh_optimizer'}
            if updates==20000:
                from pi0_relative_full_profile import validate_fields
                validate_fields(profile_contract)
            write(cp/'assets/r1-relative-profile-contract.json',profile_contract)
            save(checkpoint_saved=True,checkpoint=str(cp),full_parameter_sha256=full_hash,
                parameter_dtype_counts=dtype_counts,optimizer_and_lr_counts=counts)
            return cp
        train_step=jax.jit(functools.partial(upstream.train_step,cfg),
            in_shardings=(replicated,placement,batch_shard),out_shardings=(placement,replicated),donate_argnums=(1,))
        save(stage='compiling_full_demo_fit');started=time.monotonic()
        for update,index in enumerate(order):
            if (ROOT/'runs/USER_PAUSED.json').exists() or resolve_gpu(ROOT,'1')!=gpu:raise RuntimeError('Paused/allocation changed')
            batch=jax.tree.map(jnp.asarray,data_loader._collate_fn([host_cache[index]]))
            observation=model_module.Observation.from_dict(batch)
            device_batch=jax.device_put((observation,batch['actions']),batch_shard)
            with sharding.set_mesh(mesh):state,info=train_step(rng,state,device_batch)
            jax.block_until_ready((state,info));values={k:float(v) for k,v in jax.device_get(info).items()}
            if not all(np.isfinite(v) for v in values.values()) or int(state.step)!=update+1:raise ValueError('Nonfinite/count failure')
            del batch,observation,device_batch
            if update==0 or (update+1)%25==0:
                report['training_log'].append({'update':update+1,'source_frame':index,**values})
                save(stage='full_demo_fitting',actual_optimizer_updates=update+1,last_info=values,
                    updates_per_second=(update+1)/(time.monotonic()-started))
                print(json.dumps({'update':update+1,'frame':index,**values}),flush=True)
            if (update+1)%5000==0:save_checkpoint(state,update+1)
        manager.wait_until_finished();manager.close()
        final_train,_=trainable_fingerprint(state.params.filter(cfg.trainable_filter))
        if final_train==train_initial:raise ValueError('No parameter updates')
        save(training='completed',stage='training_completed_final_sampling_pending',actual_optimizer_updates=20000,
            final_trainable_sha256=final_train,formal_training_completed=False)
        params,model_def,completed_step=inference_payload(state);del state,train_step
        jax.clear_caches();gc.collect()
        def sample(p,obs,noise):
            m=nnx.merge(model_def,p);m.eval();return m.sample_actions(jax.random.key(123),obs,num_steps=10,noise=noise[None])
        sampler=jax.jit(sample);predictions={}
        native_stats=json.loads((Path(plan['norm_anchor'])/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json').read_text())['norm_stats']['actions']
        native_std=np.asarray(native_stats['std'])+1e-6
        try:
            for frame in eval_indices:
                batch=jax.tree.map(jnp.asarray,data_loader._collate_fn([host_cache[frame]]));obs=model_module.Observation.from_dict(batch)
                noise=np.random.default_rng(123).standard_normal((50,32)).astype(np.float32)
                with sharding.set_mesh(mesh):normalized=sampler(params,obs,noise)
                native=output_transform({'state':np.asarray(obs.state[0]),'actions':np.asarray(normalized[0])})['actions']
                predictions[f'frame_{frame}']=native
                report['samples'].append({'frame':frame,'input_sha256':input_hashes[str(frame)],
                    'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                    'first10':motion_metrics(native,references[frame],raw_states[frame],native_std,10)})
            np.savez_compressed(c/'in-memory-final-predictions.npz',**predictions)
            save(stage='completed_full_demo_training_restore_required',training='completed',actual_optimizer_updates=completed_step)
        except Exception as error:
            # Trained, committed weights survive any noncritical sampling issue.
            # Parent MUST perform independent CPU checkpoint restore before eval.
            save(stage='completed_training_sampling_failed_restore_required',training='completed',
                in_memory_sampling_error=repr(error)[:2000],actual_optimizer_updates=completed_step)
    except BaseException as error:
        import traceback
        (c/'training-exception.txt').write_text(traceback.format_exc())
        save(stage='full_demo_training_failed',error=repr(error)[:2000]);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);a=p.parse_args()
    run(a.campaign)
