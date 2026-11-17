#!/usr/bin/env python3
"""SERVER CPU: partial state-stem and actual sampling-exposure audit.

No full VLA, GPU, gradient, action execution or data writes. A CPU state-token
mirror is NOT actual full-model GPU attention or a causal task-success test.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path('/mnt/yuanyue/double-arm')


def exposure_counts(order, length, critical):
    counts=Counter(order)
    if set(counts)!=set(range(length)) or not set(critical).issubset(counts):
        raise ValueError('All frames and explicit transition subset required')
    ordinary=[count for frame,count in counts.items() if frame not in critical]
    return {'optimizer_updates':len(order),'unique_frames':len(counts),
            'ordinary_frame_visit_min_max':[min(ordinary),max(ordinary)],
            'transition_frame_visit_min_max':[min(counts[f] for f in critical),max(counts[f] for f in critical)],
            'visits_by_frame':{str(f):counts[f] for f in range(length)},
            'scope':'sampling exposures per source frame, not epochs of perturbed recovery or effective batch size'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':
        raise ValueError('Explicit CPU-only environment required')
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Fresh remote report required')
    import h5py
    import numpy as np
    import jax
    import jax.numpy as jnp
    import orbax.checkpoint as ocp
    from openpi import transforms
    from openpi.training.checkpoints import load_norm_stats
    from pi0_r1_config import register_config
    from pi0_relative_full_profile import load_contract,ASSET_ID,data_profile,full_schedule
    from pi0_single_demo_fit import sequence_sha256,transition_indices
    from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
    if any(d.platform!='cpu' for d in jax.devices()):raise ValueError('No GPU')
    report=json.loads((a.campaign/'training-report.json').read_text());plan=json.loads((a.campaign/'plan.json').read_text())
    cp=Path(report['checkpoint']);contract=load_contract(cp)
    order=full_schedule(1714,636,20000,42)
    if sequence_sha256(order)!=report['frame_schedule_sha256'] or sequence_sha256(order)!=plan['frame_schedule_sha256']:
        raise ValueError('Actual saved sampling plan changed')
    exposure=exposure_counts(order,1714,transition_indices(1714,636))
    item={'kernel':np.empty((32,1024),np.float32),'bias':np.empty(1024,np.float32)}
    mapping={key:ocp.Transform(original_key=f'params/state_proj/{key}/value') for key in item}
    restore={key:ocp.ArrayRestoreArgs(restore_type=np.ndarray) for key in item}
    with ocp.PyTreeCheckpointer() as checkpointer:
        values=checkpointer.restore(cp/'params',ocp.args.PyTreeRestore(item=item,restore_args=restore,
            transforms=mapping,transforms_default_to_original=False))
    if any(v.dtype!=np.float32 for v in values.values()):raise ValueError('Preserve saved state-stem FP32, no implicit BF16 restore')
    cfg=register_config('native');norms=load_norm_stats(cp/'assets',ASSET_ID)
    data=data_profile(cfg.data.create(cfg.assets_dirs,cfg.model),norms)
    transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(norms,use_quantiles=False),*data.model_transforms.inputs])
    rows=[]
    with h5py.File(report['source']['source']) as file:
        obs=file['data/'+report['source']['source_demo']+'/obs'];keys=camera_keys(obs)
        for frame in (0,636):
            inputs={name:rgb_image(obs[key][frame]) for name,key in keys.items()}
            inputs.update(state=obs['prop_state'][frame],prompt=PROMPT)
            original=np.asarray(jnp.asarray(transform(inputs)['state']))
            if original.dtype!=np.float32:raise ValueError('Actual pinned model input conversion must be FP32')
            linear=original@values['kernel']+values['bias']
            token=np.asarray(jnp.asarray(linear).astype(jnp.bfloat16).astype(jnp.float32))
            for sign in (-1,1):
                changed=dict(inputs);changed['state']=inputs['state'].copy();changed['state'][7:13]+=sign*.02
                normalized=np.asarray(jnp.asarray(transform(changed)['state']));delta=normalized-original
                expected=(changed['state'][7:13]-inputs['state'][7:13])/(norms['state'].std[7:13]+1e-6)
                np.testing.assert_allclose(delta[7:13],expected,rtol=1e-4,atol=1e-6)
                np.testing.assert_array_equal(delta[np.r_[0:7,13:32]],0)
                altered=normalized@values['kernel']+values['bias']
                altered_token=np.asarray(jnp.asarray(altered).astype(jnp.bfloat16).astype(jnp.float32))
                rows.append({'frame':frame,'q_delta_rad':sign*.02,
                    'normalized_state_changed_coordinates':np.flatnonzero(delta).tolist(),
                    'normalized_state_delta':delta[7:13].tolist(),
                    'cpu_mirrored_bf16_token_changed_elements':int(np.count_nonzero(altered_token!=token)),
                    'token_elements':1024,'cpu_mirrored_token_l2_change':float(np.linalg.norm(altered_token-token))})
    result={'stage':'completed_CPU_relative_state_stem_and_exposure_audit','created_at':datetime.now(timezone.utc).isoformat(),
        'checkpoint':str(cp),'checkpoint_full_parameter_sha256':contract['full_parameter_sha256'],
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'partial_parameter_elements':33792,'saved_state_stem_dtype':'float32','model_input_dtype':'float32','full_model_loaded':False,
        'gpu_initialized':False,'gradient_updates':0,'model_or_simulator_control_calls':0,'checkpoint_saved':False,
        'source_split':'train','source_demo':'demo_0','frame_schedule_sha256':sequence_sha256(order),
        'sampling_exposures':exposure,'state_perturbations':rows,
        'caveat':'CPU state-stem mirror only; token difference does not prove downstream attention/output use. Inconsistent q/RGB is OFFLINE only. Sampling imbalance is not isolated causal proof or paper recipe.'}
    a.output.mkdir();(a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (ROOT/'runs/latest-pi0-relative-state-stem-audit.txt').write_text(str(a.output)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='sampling_exposures'},indent=2))
    print(json.dumps({k:v for k,v in exposure.items() if k!='visits_by_frame'}))


if __name__=='__main__':main()
