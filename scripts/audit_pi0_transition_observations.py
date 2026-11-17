#!/usr/bin/env python3
"""SERVER CPU: source close-boundary observation separation and tiny state stem.

Only33792 state-projection elements partially restored; no full VLA, GPU,
gradient or simulator. CPU state-token mirror is NOT actual GPU activation
capture or proof of full-policy indistinguishability / a task failure cause.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
ROOT=Path('/mnt/yuanyue/double-arm')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU only')
    if not a.output.resolve().is_relative_to(ROOT/'runs') or a.output.exists():raise ValueError('Fresh remote evidence only')
    import h5py
    import numpy as np
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi import transforms
    from openpi.training.checkpoints import load_norm_stats
    from pi0_r1_config import register_config
    from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
    from probe_pi0_grasp_onset import onset_frames
    cfg=register_config('native');data=cfg.data.create(cfg.assets_dirs,cfg.model)
    checkpoint=ROOT/'runs/pi0-checkpoints/pi0_r1_tidy_table_lora32/pi0-native-coverage-v3-20261116-172526/89999'
    contract=json.loads((checkpoint/'assets/r1-adapter-contract.json').read_text())
    norm_path=checkpoint/'assets'/data.asset_id/'norm_stats.json'
    if contract['action_representation']!='native' or contract['quantile_normalization'] is not False or hashlib.sha256(norm_path.read_bytes()).hexdigest()!=contract['norm_stats_sha256']:
        raise ValueError('Wrong normalization')
    stats=load_norm_stats(checkpoint/'assets',data.asset_id)
    transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(stats,use_quantiles=False),*data.model_transforms.inputs])
    item={'kernel':np.empty((32,1024),np.float32),'bias':np.empty(1024,np.float32)}
    mapping={key:ocp.Transform(original_key=f'params/state_proj/{key}/value') for key in item}
    restore={key:ocp.ArrayRestoreArgs(restore_type=np.ndarray,dtype=np.float32) for key in item}
    with ocp.PyTreeCheckpointer() as c:
        values=c.restore(checkpoint/'params',ocp.args.PyTreeRestore(item=item,restore_args=restore,transforms=mapping,transforms_default_to_original=False))
    # Default serving restores ALL params BF16. nnx.Linear dtype=None then
    # promotes these BF16 params with FP32 normalized state to FP32 arithmetic;
    # Gemma subsequently casts the state token to configured embed dtypeBF16.
    kernel=jnp.asarray(values['kernel'],dtype=jnp.bfloat16).astype(jnp.float32)
    bias=jnp.asarray(values['bias'],dtype=jnp.bfloat16).astype(jnp.float32)
    provenance=json.loads((ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json').read_text())
    if provenance['split']!='train' or len(provenance['episodes'])!=16:raise ValueError('Original16 TRAIN only')
    report={'mode':'CPU_source_close_boundary_separation_partial_state_stem','checkpoint':str(checkpoint),
        'partial_parameter_elements':33792,'full_model_loaded':False,'gpu_initialized':False,
        'simulator_control_calls':0,'gradient_updates':0,'checkpoint_saved':False,'not_task_evaluation':True,
        'caveat':'CPU mirror of state_proj then Gemma BF16 cast, NOT actual GPU activations; no vision embeddings or full-policy identifiability proof',
        'normalization_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'episodes':[]}
    for entry in provenance['episodes']:
        with h5py.File(entry['source']) as file:
            g=file['data/'+entry['source_demo']];obs=g['obs'];actions=g['actions'][:];onset=onset_frames(actions[:,13])[0];keys=camera_keys(obs)
            processed=[]
            for frame in (onset-1,onset):
                inputs={name:rgb_image(obs[key][frame]) for name,key in keys.items()};inputs.update(state=obs['prop_state'][frame],prompt=PROMPT)
                processed.append(transform(inputs))
            before,after=processed
            old=jnp.asarray(before['state'],jnp.float32);new=jnp.asarray(after['state'],jnp.float32)
            old_token=(old@kernel+bias).astype(jnp.bfloat16);new_token=(new@kernel+bias).astype(jnp.bfloat16)
            different=np.asarray(old_token!=new_token)
            image_metrics={}
            for key in before['image']:
                x=np.asarray(before['image'][key]);y=np.asarray(after['image'][key]);delta=y.astype(np.float32)-x.astype(np.float32)
                image_metrics[key]={'dtype':str(x.dtype),'shape':list(x.shape),'mean_abs_pixel_delta':float(np.abs(delta).mean()),
                    'different_scalar_fraction':float(np.mean(delta!=0)),'max_abs_pixel_delta':float(np.abs(delta).max())}
            row={'source_demo':entry['source_demo'],'frames':[onset-1,onset],'source_gripper_labels':actions[onset-1:onset+1,13].tolist(),
                'normalized_state_max_abs_delta':float(np.max(np.abs(np.asarray(new-old)))),
                'normalized_state_different_coordinates':int(np.sum(np.asarray(old!=new))),
                'cpu_mirrored_bf16_state_token_different_elements':int(different.sum()),'state_token_elements':1024,
                'processed_images':image_metrics}
            report['episodes'].append(row)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'episodes':len(report['episodes']),'cpu_state_tokens_identical_pairs':sum(e['cpu_mirrored_bf16_state_token_different_elements']==0 for e in report['episodes']),
        'all_processed_images_identical_pairs':sum(all(v['different_scalar_fraction']==0 for v in e['processed_images'].values()) for e in report['episodes'])}))


if __name__=='__main__':main()
