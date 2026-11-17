#!/usr/bin/env python3
"""SERVER ONLY CPU: compare official pi0 sampling on TRAIN-only calibration.

Keep one trained model and RGB/proprio/prompt inputs; score native action labels
only after inference. No simulator or held-out episode is used to select settings.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time


def sampling_variants(default_only):
    return ((10,'fixed_gaussian'),) if default_only else ((10,'fixed_gaussian'),(10,'zero'),(50,'fixed_gaussian'),(50,'zero'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--training-provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pretrained-control',action='store_true',help='Official base weights + R1 adapter + explicit training norms; NOT robot zero-shot evaluation')
    parser.add_argument('--normalization-checkpoint',type=Path)
    parser.add_argument('--default-sampling-only',action='store_true')
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU-only benchmark')
    if args.output.exists():raise FileExistsError(args.output)
    if not str(args.output.resolve()).startswith('/mnt/yuanyue/double-arm/runs/'):raise ValueError('Remote report required')
    root=Path('/mnt/yuanyue/double-arm')
    if args.pretrained_control and (args.normalization_checkpoint is None or args.checkpoint.resolve()!=(root/'data/pi0/checkpoints/pi0_base').resolve()):raise ValueError('Explicit official base path and R1 normalization anchor required')
    if not args.pretrained_control and args.normalization_checkpoint is not None:raise ValueError('Do not silently override fine-tuned normalization')
    import h5py
    import numpy as np
    from openpi.policies.policy_config import create_trained_policy
    from pi0_r1_config import register_config
    from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
    config=register_config()
    anchor=args.normalization_checkpoint if args.pretrained_control else args.checkpoint
    norm_path=anchor/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json'
    if args.pretrained_control:
        from openpi.models.pi0_config import Pi0Config
        from openpi.training.checkpoints import load_norm_stats
        contract=json.loads((anchor/'assets/r1-adapter-contract.json').read_text())
        if contract.get('model')!='pi0' or contract.get('action_representation')!='native' or contract.get('quantile_normalization') is not False or hashlib.sha256(norm_path.read_bytes()).hexdigest()!=contract.get('norm_stats_sha256'):raise ValueError('Wrong R1 norm anchor')
        config=dataclasses.replace(config,model=Pi0Config(paligemma_variant='gemma_2b',action_expert_variant='gemma_300m',action_horizon=50),policy_metadata={**config.policy_metadata,'lora_rank_vlm':0,'lora_rank_action':0,'initialization':'official_pretrained_no_gradient_updates_R1_adapter_control'})
        policy=create_trained_policy(config,args.checkpoint,norm_stats=load_norm_stats(anchor/'assets','momagen/r1_tidy_table_train_v1'))
    else:policy=create_trained_policy(config,args.checkpoint)
    norm=json.loads(norm_path.read_text())['norm_stats']
    std=np.asarray(norm['actions']['std'])+1e-6
    provenance=json.loads(args.training_provenance.read_text())
    if provenance.get('split')!='train':raise ValueError('Calibration must use only the training split')
    entry=provenance['episodes'][0]
    report={'checkpoint':str(args.checkpoint),'mode':'train_only_offline_sampling_calibration',
        'no_heldout_tuning':True,'teacher_labels_for_metrics_only':True,'simulator_control_calls':0,
        'not_task_evaluation':True,'gpu_initialized':False,'samples':[],
        'base_rmse_units':'Mixed XY metres / yaw radians; not a pure metric distance',
        'weight_condition':'official_pretrained_weights_R1_adapter_TRAIN_norms' if args.pretrained_control else 'fine_tuned_weights_R1_adapter_TRAIN_norms',
        'not_embodiment_zero_shot_task_test':True,'gradient_updates':0,'checkpoint_saved':False,
        'norm_anchor':str(anchor),'norm_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        'provenance_sha256':hashlib.sha256(args.training_provenance.read_bytes()).hexdigest(),
        'benchmark_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    variants=sampling_variants(args.default_sampling_only)
    def save():args.output.write_text(json.dumps(report,indent=2))
    rng=np.random.default_rng(123)
    with h5py.File(entry['source']) as file:
        group=file['data'][entry['source_demo']];obs=group['obs'];keys=camera_keys(obs);length=len(obs['prop_state'])
        actions=group['actions'][()]
        for index in (0,length//3,2*length//3):
            inputs={key:rgb_image(obs[name][index]) for key,name in keys.items()}
            inputs.update(state=obs['prop_state'][index],prompt=PROMPT)
            reference=actions[np.minimum(np.arange(index,index+50),length-1)]
            fixed_noise=rng.standard_normal((50,32)).astype(np.float32)
            input_digest=hashlib.sha256()
            for key in sorted(inputs):
                input_digest.update(key.encode());value=inputs[key];input_digest.update(value.encode() if isinstance(value,str) else np.asarray(value).tobytes())
            for steps,noise_name in variants:
                # Official Policy.infer exposes noise publicly; its construction
                # sample kwargs are switched here only to reuse the same weights.
                # Production service uses create_trained_policy(sample_kwargs=...).
                policy._sample_kwargs={'num_steps':steps}
                noise=fixed_noise if noise_name=='fixed_gaussian' else np.zeros((50,32),np.float32)
                started=time.monotonic();prediction=policy.infer(inputs,noise=noise)
                predicted=np.asarray(prediction['actions']);difference=predicted-reference
                if predicted.shape!=(50,21) or not np.isfinite(predicted).all():raise ValueError('Invalid actions')
                sample={'source_demo':entry['source_demo'],'frame':index,'steps':steps,'noise':noise_name,
                    'input_sha256':input_digest.hexdigest(),'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                    'base_xy_coordinate_rmse_first_10_m':float(np.sqrt(np.mean(difference[:10,:2]**2))),
                    'base_yaw_delta_rmse_first_10_rad':float(np.sqrt(np.mean(difference[:10,2]**2))),
                    'predicted_first_action':predicted[0].tolist(),'reference_first_action':reference[0].tolist(),
                    'group_normalized_mse_first_10':{name:float(np.mean((difference[:10,axes]/std[axes])**2)) for name,axes in [('base',slice(0,3)),('trunk',slice(3,7)),('left_arm',slice(7,13)),('left_gripper',slice(13,14)),('right_arm',slice(14,20)),('right_gripper',slice(20,21))]},
                    'seconds':time.monotonic()-started,'normalized_mse_first_10':float(np.mean((difference[:10]/std)**2)),
                    'base_rmse_first_10':float(np.sqrt(np.mean(difference[:10,:3]**2))),
                    'arm_rmse_first_10':float(np.sqrt(np.mean(difference[:10,7:13]**2))),
                    'predicted_first_base':predicted[0,:3].tolist(),'reference_first_base':reference[0,:3].tolist()}
                report['samples'].append(sample);save();print(json.dumps(sample),flush=True)
    aggregate=[]
    for steps,noise_name in variants:
        subset=[s for s in report['samples'] if s['steps']==steps and s['noise']==noise_name]
        aggregate.append({'steps':steps,'noise':noise_name,'normalized_mse_first_10':float(np.mean([s['normalized_mse_first_10'] for s in subset])),
                          'base_rmse_first_10':float(np.mean([s['base_rmse_first_10'] for s in subset]))})
    report.update(status='completed_train_only_calibration',aggregate=aggregate,
                  best_train_only=min(aggregate,key=lambda x:x['normalized_mse_first_10']))
    save();print(json.dumps(report['aggregate'],indent=2),flush=True)


if __name__=='__main__':main()
