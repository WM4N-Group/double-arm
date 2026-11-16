#!/usr/bin/env python3
"""SERVER CPU ONLY: paired source/live RGB audit, never simulator control.

Use identical Gaussian action noise and proprioception to isolate each camera's
influence. No teacher motor commands, object poses, episode clock or phase are
read or passed to pi0. This diagnostic cannot establish closed-loop success.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source-dataset',type=Path,required=True)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wait-seconds',type=int,default=240)
    args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':
        raise ValueError('Explicit CPU-only runtime required')
    root=Path(os.environ.get('DOUBLE_ARM_ROOT','/mnt/yuanyue/double-arm'))
    for path in (args.checkpoint,args.snapshot,args.output):
        if not path.resolve().is_relative_to((root/'runs').resolve()):raise ValueError('Remote run paths required')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir()
    report={'mode':'paired_rgb_input_diagnostic_only','task_success_claim':False,'simulator_control_calls':0,
            'teacher_action_reads':0,'gpu_initialized':False,'checkpoint':str(args.checkpoint),
            'same_noise_within_each_trial':True,'same_proprioception_for_camera_ablations':True,
            'source_dataset':str(args.source_dataset),'snapshot':str(args.snapshot),'samples':[]}
    def save(**updates):
        report.update(updates);tmp=args.output/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(args.output/'report.json')
    save(stage='loading_cpu_policy')
    try:
        import h5py
        import jax
        import numpy as np
        from PIL import Image,ImageDraw
        from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
        from pi0_r1_config import register_config
        from openpi.policies.policy_config import create_trained_policy
        if any(d.platform!='cpu' for d in jax.devices()):raise ValueError('Non-CPU device')
        contract=json.loads((args.checkpoint/'assets/r1-adapter-contract.json').read_text())
        config=register_config(action_representation='absolute_base')
        if (contract.get('model')!='pi0' or contract.get('config_name')!=config.name
            or contract.get('action_representation')!='absolute_base' or contract.get('quantile_normalization') is not True):
            raise ValueError('Wrong checkpoint adapter contract')
        asset_id=config.data.assets.asset_id or config.data.repo_id
        norm=args.checkpoint/'assets'/asset_id/'norm_stats.json'
        if contract.get('norm_stats_asset_id')!=asset_id or hashlib.sha256(norm.read_bytes()).hexdigest()!=contract.get('norm_stats_sha256'):
            raise ValueError('Checkpoint normalization contract mismatch')
        policy=create_trained_policy(config,args.checkpoint,sample_kwargs={'num_steps':10})
        save(stage='waiting_for_live_snapshot')
        deadline=time.monotonic()+max(0,args.wait_seconds)
        while not args.snapshot.is_file():
            if time.monotonic()>deadline:raise TimeoutError('No live reset snapshot')
            if (root/'runs/USER_PAUSED.json').exists():raise RuntimeError('User pause')
            time.sleep(2)
        with h5py.File(args.source_dataset) as file:
            obs=file['data/demo_0/obs']
            source={key:rgb_image(obs[name][0]) for key,name in camera_keys(obs).items()}
            source.update(state=obs['prop_state'][0],prompt=PROMPT)
        with np.load(args.snapshot,allow_pickle=False) as loaded:
            live={key:loaded[key].copy() for key in ('head_image','left_wrist_image','right_wrist_image','state')}
        live['prompt']=PROMPT
        delta=live['state']-source['state']
        pixel={}
        sheet=Image.new('RGB',(256*3,280*2),'#161616');draw=ImageDraw.Draw(sheet)
        for column,key in enumerate(('head_image','left_wrist_image','right_wrist_image')):
            left=rgb_image(source[key]);right=rgb_image(live[key])
            if left.shape!=right.shape or left.shape!=(256,256,3):raise ValueError('Unexpected camera shape')
            difference=right.astype(float)-left.astype(float)
            pixel[key]={'mae_0_255':float(np.abs(difference).mean()),'identical':bool(np.array_equal(left,right)),
                'source_sha256':hashlib.sha256(left.tobytes()).hexdigest(),'live_sha256':hashlib.sha256(right.tobytes()).hexdigest()}
            for row,image in enumerate((left,right)):
                sheet.paste(Image.fromarray(image),(column*256,row*280+24));draw.text((column*256+5,row*280+5),('SOURCE ' if row==0 else 'LIVE ')+key,fill='yellow')
        sheet.save(args.output/'source-live-cameras.jpg',quality=93)
        save(stage='paired_camera_inference',state_max_abs_difference=float(np.abs(delta).max()),
             state_exact_match=bool(np.array_equal(live['state'],source['state'])),pixel_comparison=pixel)
        variants={'source':source,'all_live':live}
        for key in pixel:variants['live_'+key]={**source,key:live[key]}
        for seed in (42,43):
            noise=np.random.default_rng(seed).standard_normal((50,config.model.action_dim)).astype(np.float32)
            for name,inputs in variants.items():
                if (root/'runs/USER_PAUSED.json').exists():raise RuntimeError('User pause')
                started=time.monotonic();actions=np.asarray(policy.infer(inputs,noise=noise)['actions'])
                if actions.shape!=(50,21) or not np.isfinite(actions).all():raise ValueError('Invalid model output')
                np.save(args.output/f'actions-{seed}-{name}.npy',actions)
                report['samples'].append({'seed':seed,'variant':name,'first_action':actions[0].tolist(),
                    'inference_seconds':time.monotonic()-started});save()
        save(stage='completed_paired_diagnostic_not_task_success')
    except BaseException as error:
        save(stage='paired_diagnostic_failed',error=repr(error));raise


if __name__=='__main__':main()
