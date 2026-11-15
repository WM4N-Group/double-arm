#!/usr/bin/env python3
"""SERVER ONLY CPU: compare official pi0 sampling on TRAIN-only calibration.

Keep one trained model and RGB/proprio/prompt inputs; score native action labels
only after inference. No simulator or held-out episode is used to select settings.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--training-provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU-only benchmark')
    if args.output.exists():raise FileExistsError(args.output)
    if not str(args.output.resolve()).startswith('/mnt/yuanyue/double-arm/runs/'):raise ValueError('Remote report required')
    import h5py
    import numpy as np
    from openpi.policies.policy_config import create_trained_policy
    from pi0_r1_config import register_config
    from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
    config=register_config();policy=create_trained_policy(config,args.checkpoint)
    norm=json.loads((args.checkpoint/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json').read_text())['norm_stats']
    std=np.asarray(norm['actions']['std'])+1e-6
    provenance=json.loads(args.training_provenance.read_text())
    if provenance.get('split')!='train':raise ValueError('Calibration must use only the training split')
    entry=provenance['episodes'][0]
    report={'checkpoint':str(args.checkpoint),'mode':'train_only_offline_sampling_calibration',
        'no_heldout_tuning':True,'teacher_labels_for_metrics_only':True,'simulator_control_calls':0,
        'not_task_evaluation':True,'gpu_initialized':False,'samples':[],
        'base_rmse_units':'Mixed XY metres / yaw radians; not a pure metric distance'}
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
            for steps,noise_name in ((10,'fixed_gaussian'),(10,'zero'),(50,'fixed_gaussian'),(50,'zero')):
                # Official Policy.infer exposes noise publicly; its construction
                # sample kwargs are switched here only to reuse the same weights.
                # Production service uses create_trained_policy(sample_kwargs=...).
                policy._sample_kwargs={'num_steps':steps}
                noise=fixed_noise if noise_name=='fixed_gaussian' else np.zeros((50,32),np.float32)
                started=time.monotonic();prediction=policy.infer(inputs,noise=noise)
                predicted=np.asarray(prediction['actions']);difference=predicted-reference
                if predicted.shape!=(50,21) or not np.isfinite(predicted).all():raise ValueError('Invalid actions')
                sample={'source_demo':entry['source_demo'],'frame':index,'steps':steps,'noise':noise_name,
                    'seconds':time.monotonic()-started,'normalized_mse_first_10':float(np.mean((difference[:10]/std)**2)),
                    'base_rmse_first_10':float(np.sqrt(np.mean(difference[:10,:3]**2))),
                    'arm_rmse_first_10':float(np.sqrt(np.mean(difference[:10,7:13]**2))),
                    'predicted_first_base':predicted[0,:3].tolist(),'reference_first_base':reference[0,:3].tolist()}
                report['samples'].append(sample);save();print(json.dumps(sample),flush=True)
    aggregate=[]
    for steps,noise_name in ((10,'fixed_gaussian'),(10,'zero'),(50,'fixed_gaussian'),(50,'zero')):
        subset=[s for s in report['samples'] if s['steps']==steps and s['noise']==noise_name]
        aggregate.append({'steps':steps,'noise':noise_name,'normalized_mse_first_10':float(np.mean([s['normalized_mse_first_10'] for s in subset])),
                          'base_rmse_first_10':float(np.mean([s['base_rmse_first_10'] for s in subset]))})
    report.update(status='completed_train_only_calibration',aggregate=aggregate,
                  best_train_only=min(aggregate,key=lambda x:x['normalized_mse_first_10']))
    save();print(json.dumps(report['aggregate'],indent=2),flush=True)


if __name__=='__main__':main()
