#!/usr/bin/env python3
"""SERVER ONLY: same-process BF16 versus FP32 PARAMETER-restore diagnostic.

Pinned official serving casts checkpoint params to BF16. Test preservation of
saved FP32 trainables WITHOUT changing configured model compute dtype, labels,
normalization, sampler, or controller. NOT full-FP32 compute or task evaluation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from probe_pi0_failure_mechanisms import ROOT, digest_inputs, motion_metrics
from probe_pi0_grasp_onset import onset_frames


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import inspect_release, process_alive, pinned_upstream_revision
    gpu=resolve_gpu(ROOT,'1')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or os.environ.get('JAX_PLATFORMS')!='cuda':
        raise ValueError('Only explicitly authorized GPU1 CUDA; no fallback')
    if not a.output.resolve().is_relative_to(ROOT/'runs') or a.output.exists() or a.output.is_symlink():
        raise ValueError('Fresh remote evidence directory required')
    lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pointer=json.loads((ROOT/'runs/latest-gpu1-job.json').read_text());prior=json.loads(Path(pointer['status_file']).read_text())
    if prior.get('stage')!='completed_TRAIN_gripper_width_counterfactual_NOT_task_success' or process_alive(pointer.get('controller_pid')):
        raise ValueError('GPU1 predecessor not fully released or pointer changed')
    inspect_release(Path((ROOT/'runs/latest-pi0-coverage-campaign.txt').read_text().strip()))
    source=ROOT/'src/openpi-215abfb';revision=pinned_upstream_revision(source)
    checkpoint=Path(prior['checkpoint'])
    a.output.mkdir()
    def write(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
    status={'stage':'loading','controller_pid':os.getpid(),'checkpoint':str(checkpoint),'gpu_uuid':gpu,
            'started_at':datetime.now(timezone.utc).isoformat(),'policy_training':'not_run',
            'simulator_control_calls':0,'learned_policy_task_success':False}
    def save(**updates):
        status.update(updates,updated_at=datetime.now(timezone.utc).isoformat());write(a.output/'status.json',status)
    save()
    write(ROOT/'runs/latest-gpu1-job.json',{'campaign':str(a.output),'status_file':str(a.output/'status.json'),
        'controller_pid':os.getpid(),'gpu_uuid':gpu,'max_gpu_count':1,'role':'TRAIN_offline_parameter_restore_precision_NOT_training'})
    mapping=json.loads((ROOT/'runs/latest-multi-gpu-jobs.json').read_text());mapping['completed_pi0_width_counterfactual']=pointer
    mapping['GPU1']={'campaign':str(a.output),'job_pointer':str(ROOT/'runs/latest-gpu1-job.json'),'gpu_uuid':gpu,
                    'role':'TRAIN_offline_parameter_restore_precision_NOT_training'}
    mapping['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',mapping)
    (ROOT/'runs/latest-pi0-restore-precision-probe.txt').write_text(str(a.output)+'\n')
    try:
        import h5py
        import numpy as np
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi import transforms
        from openpi.models import model as model_module
        from openpi.policies.policy import Policy
        from openpi.training.checkpoints import load_norm_stats
        from pi0_r1_config import register_config
        from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
        if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('Wrong backend')
        config=register_config('native');data=config.data.create(config.assets_dirs,config.model)
        contract=json.loads((checkpoint/'assets/r1-adapter-contract.json').read_text())
        norm_path=checkpoint/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json'
        if (contract.get('action_representation')!='native' or contract.get('quantile_normalization') is not False
            or data.use_quantile_norm or hashlib.sha256(norm_path.read_bytes()).hexdigest()!=contract['norm_stats_sha256']):
            raise ValueError('Native mean/std contract mismatch')
        hashes={name:hashlib.sha256((ROOT/'code/scripts'/name).read_bytes()).hexdigest()==sha for name,sha in contract['training_source_sha256'].items()}
        if not all(hashes.values()):raise ValueError('Actual training adapter source changed')
        norm=load_norm_stats(checkpoint/'assets',data.asset_id)
        std=np.asarray(json.loads(norm_path.read_text())['norm_stats']['actions']['std'])+1e-6
        provenance_path=ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'
        provenance=json.loads(provenance_path.read_text())
        if provenance.get('split')!='train' or len(provenance['episodes'])!=16:raise ValueError('Original16 TRAIN only')
        bounds_path=ROOT/'runs/pi0-native-coverage-v3-20261116-172526/coverage-training-author-diagnostic/action-bounds.json'
        bounds=json.loads(bounds_path.read_text());lo=np.asarray(bounds['min']);hi=np.asarray(bounds['max'])
        if lo.shape!=(21,) or hi.shape!=(21,) or np.any(lo>hi):raise ValueError('Bad native bounds')
        # Same real model observations for both restore conditions; stored only
        # in server RAM. This is not source-RGB substitution in a live policy.
        examples=[]
        for entry in provenance['episodes']:
            with h5py.File(entry['source']) as file:
                g=file['data/'+entry['source_demo']];obs=g['obs'];actions=g['actions'][:];keys=camera_keys(obs)
                for phase,index in zip(('close_onset','onset_plus15'),onset_frames(actions[:,13])):
                    inputs={name:rgb_image(obs[key][index]) for name,key in keys.items()}
                    inputs.update(state=obs['prop_state'][index],prompt=PROMPT)
                    examples.append((entry['source_demo'],phase,index,inputs,actions[np.minimum(np.arange(index,index+50),len(actions)-1)]))
        noises={seed:np.random.default_rng(seed).standard_normal((50,32)).astype(np.float32) for seed in (123,124,125)}
        report={'mode':'TRAIN16_same_process_parameter_restore_precision','checkpoint':str(checkpoint),
            'scope':'32 actual TRAIN observations x3 noises x2 restore conditions; no heldout/rollout/training',
            'changed_only':'checkpoint restore target parameter dtype; configured model compute dtype unchanged',
            'not_full_float32_compute':True,'model_compute_dtype':str(config.model.dtype),
            'bf16_saved_parameters_cannot_regain_precision_by_upcast':True,
            'simulator_control_calls':0,'gradient_updates':0,'checkpoint_saved':False,'not_task_evaluation':True,
            'openpi_revision':revision,'training_source_matches':hashes,'seeds':[123,124,125],'denoising_steps':10,
            'norm_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),
            'provenance_sha256':hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'pinned_policy_factory_sha256':hashlib.sha256((source/'src/openpi/policies/policy_config.py').read_bytes()).hexdigest(),
            'conditions':{},'samples':[]}
        predictions={}
        for condition,dtype in (('official_bfloat16_restore',jnp.bfloat16),('preserve_float32_params',jnp.float32)):
            save(stage='loading_restore_condition',condition=condition,completed_samples=len(report['samples']))
            model=config.model.load(model_module.restore_params(checkpoint/'params',dtype=dtype))
            dtype_counts={}
            for leaf in jax.tree.leaves(nnx.state(model)):
                if hasattr(leaf,'dtype'):
                    name=str(leaf.dtype);dtype_counts[name]=dtype_counts.get(name,0)+int(leaf.size)
            report['conditions'][condition]={'restored_parameter_dtype_counts':dtype_counts,
                'requested_restore_dtype':str(dtype),'model_compute_dtype':str(config.model.dtype)}
            # Mirror the pinned create_trained_policy transforms exactly; only
            # restore_params(dtype=...) differs. No monkeypatch / source edit.
            policy=Policy(model,transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,
                    transforms.Normalize(norm,use_quantiles=False),*data.model_transforms.inputs],
                output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(norm,use_quantiles=False),
                                   *data.data_transforms.outputs],sample_kwargs={'num_steps':10},metadata=config.policy_metadata)
            for demo,phase,frame,inputs,reference in examples:
                for seed,noise in noises.items():
                    save(stage='offline_precision_inference',condition=condition,source_demo=demo,phase=phase,
                         seed=seed,completed_samples=len(report['samples']))
                    started=time.monotonic();predicted=np.asarray(policy.infer(inputs,noise=noise)['actions']);clipped=np.clip(predicted,lo,hi)
                    row={'condition':condition,'source_demo':demo,'phase':phase,'frame':frame,'seed':seed,
                        'input_sha256':digest_inputs(inputs),'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                        'seconds':time.monotonic()-started,'clipped_first1':motion_metrics(clipped,reference,inputs['state'],std,1),
                        'clipped_first10':motion_metrics(clipped,reference,inputs['state'],std,10)}
                    report['samples'].append(row);predictions[f'{condition}_{demo}_{phase}_{seed}']=predicted
                    write(a.output/'report.json',report)
                    print(json.dumps({'condition':condition,'demo':demo,'phase':phase,'seed':seed,
                        'gripper':row['clipped_first10']['left_gripper'],
                        'left':row['clipped_first10']['left_arm']}),flush=True)
            # Explicitly release each model before loading the next condition.
            del policy,model
            jax.clear_caches();gc.collect()
        np.savez_compressed(a.output/'offline-predictions.npz',**predictions)
        report['completed']=True;write(a.output/'report.json',report)
        save(stage='completed_TRAIN_restore_precision_probe_NOT_task_success',completed_samples=len(report['samples']),
            report_sha256=hashlib.sha256((a.output/'report.json').read_bytes()).hexdigest())
    except BaseException as error:
        save(stage='restore_precision_probe_failed',error=repr(error));raise


if __name__=='__main__':main()
