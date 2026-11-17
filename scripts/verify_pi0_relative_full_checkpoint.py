#!/usr/bin/env python3
"""SERVER CPU-only: genuine new-profile checkpoint restore, not task evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from pi0_relative_full_profile import make_policy,ROOT,PROFILE


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU only')
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):raise ValueError('New remote report required')
    report=json.loads((a.campaign/'training-report.json').read_text());checkpoint=Path(report['checkpoint'])
    if report.get('training')!='completed' or report.get('actual_optimizer_updates')!=20000:raise ValueError('Training incomplete')
    import h5py
    import jax
    import numpy as np
    from flax import nnx
    from preflight_pi0_accumulation_gpu import trainable_fingerprint
    from probe_pi0_failure_mechanisms import digest_inputs
    from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
    policy,contract=make_policy(checkpoint)
    digest,count=trainable_fingerprint(nnx.state(policy._model))
    if digest!=contract['full_parameter_sha256'] or digest!=report['full_parameter_sha256'] or count!=3315905296:
        raise ValueError('Reloaded model parameters are NOT bit-identical to trained saved state')
    checked=[];entry=report['source'];path=a.campaign/'in-memory-final-predictions.npz'
    in_memory=np.load(path) if path.exists() else None
    with h5py.File(entry['source']) as file:
        g=file['data/'+entry['source_demo']];obs=g['obs'];keys=camera_keys(obs)
        for frame in sorted(set((0,report['first_close'],len(obs['prop_state'])//2,len(obs['prop_state'])-1))):
            inputs={name:rgb_image(obs[key][frame]) for name,key in keys.items()};inputs.update(state=obs['prop_state'][frame],prompt=PROMPT)
            noise=np.random.default_rng(123).standard_normal((50,32)).astype(np.float32)
            native=np.asarray(policy.infer(inputs,noise=noise)['actions'])
            if native.shape!=(50,21) or not np.isfinite(native).all():raise ValueError('Invalid reloaded native actions')
            row={'frame':frame,'input_sha256':digest_inputs(inputs),'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                 'native_shape':list(native.shape),'finite':True}
            if in_memory is not None and f'frame_{frame}' in in_memory:
                row['max_abs_cpu_vs_gpu_native_difference']=float(np.max(np.abs(native-in_memory[f'frame_{frame}'])))
            checked.append(row)
    result={'stage':'cpu_reload_parameters_exact_actions_finite_NOT_task_success','policy_profile':PROFILE,
        'checkpoint':str(checkpoint),'full_parameter_sha256':digest,'actual_parameter_count':count,
        'norm_stats_sha256':contract['norm_stats_sha256'],'simulator_control_calls':0,'gpu_initialized':False,
        'task_success':False,'samples':checked,
        'numeric_caveat':'CPU/GPU sampling need not be bit-exact; parameter bits/dtypes and savednorm are exact. Not full real-time equivalence or task acceptance.'}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
