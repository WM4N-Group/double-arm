#!/usr/bin/env python3
"""SERVER CPU-only: real TRAIN relative-label/source/preprocessing roundtrip."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
ROOT=Path('/mnt/yuanyue/double-arm')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('CPU only')
    if not a.output.resolve().is_relative_to(ROOT/'runs') or a.output.exists():raise ValueError('Fresh remote report directory required')
    import h5py
    import numpy as np
    from openpi import transforms
    from openpi.training.checkpoints import load_norm_stats
    from pi0_r1_config import register_config
    from pi0_relative_joint_diagnostic import RelativeJointInputs,RelativeJointOutputs,relative_joint_stats
    from audit_pi0_training_strategy import probe_indices
    from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
    checkpoint=ROOT/'runs/pi0-checkpoints/pi0_r1_tidy_table_lora32/pi0-native-coverage-v3-20261116-172526/89999'
    cfg=register_config('native');data=cfg.data.create(cfg.assets_dirs,cfg.model)
    anchor=checkpoint/'assets'/data.asset_id/'norm_stats.json';original=load_norm_stats(checkpoint/'assets',data.asset_id)
    contract=json.loads((checkpoint/'assets/r1-adapter-contract.json').read_text())
    if contract['action_representation']!='native' or contract['quantile_normalization'] is not False or hashlib.sha256(anchor.read_bytes()).hexdigest()!=contract['norm_stats_sha256']:
        raise ValueError('Wrong anchor')
    pp=ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json';provenance=json.loads(pp.read_text())
    action_stats,source_meta=relative_joint_stats(provenance,json.loads(anchor.read_text())['norm_stats']['actions'])
    effective={**original,'actions':dataclasses.replace(original['actions'],mean=np.asarray(action_stats['mean']),std=np.asarray(action_stats['std']),q01=None,q99=None)}
    old_transform=transforms.compose([*data.data_transforms.inputs,transforms.Normalize(original,use_quantiles=False),*data.model_transforms.inputs])
    new_transform=transforms.compose([RelativeJointInputs(),transforms.Normalize(effective,use_quantiles=False),*data.model_transforms.inputs])
    inverse=transforms.compose([*data.model_transforms.outputs,transforms.Unnormalize(effective,use_quantiles=False),RelativeJointOutputs()])
    windows=0;maximum=0
    for entry in provenance['episodes']:
        with h5py.File(entry['source']) as f:
            g=f['data/'+entry['source_demo']];obs=g['obs'];actions=g['actions'][:];keys=camera_keys(obs)
            close=int(np.flatnonzero(actions[:,13]<.5)[0])
            for frame in probe_indices(len(actions),close):
                reference=actions[np.minimum(np.arange(frame,frame+50),len(actions)-1)]
                inputs={name:rgb_image(obs[key][frame]) for name,key in keys.items()}
                inputs.update(state=obs['prop_state'][frame],prompt=PROMPT,actions=reference)
                before=old_transform(copy.deepcopy(inputs));after=new_transform(copy.deepcopy(inputs))
                for key in ('state','tokenized_prompt','tokenized_prompt_mask'):
                    np.testing.assert_array_equal(before[key],after[key])
                for key in before['image']:
                    np.testing.assert_array_equal(before['image'][key],after['image'][key])
                    np.testing.assert_array_equal(before['image_mask'][key],after['image_mask'][key])
                for axis in (0,1,2,13,20):np.testing.assert_array_equal(before['actions'][:,axis],after['actions'][:,axis])
                recovered=inverse({'state':after['state'].copy(),'actions':after['actions'].copy()})['actions']
                np.testing.assert_allclose(recovered,reference,atol=1e-6,rtol=1e-6)
                maximum=max(maximum,float(np.max(np.abs(recovered-reference))));windows+=1
    if windows!=96:raise ValueError('Incomplete sampled preflight')
    a.output.mkdir()
    nj=copy.deepcopy(json.loads(anchor.read_text()));nj['norm_stats']['actions']=action_stats
    norm_path=a.output/'relative-norm-stats.json';norm_path.write_text(json.dumps(nj,indent=2,allow_nan=False))
    report={'stage':'completed_relative_joint_cpu_preflight','checkpoint':str(checkpoint),'windows':windows,
        'roundtrip_max_abs':maximum,'model_observations_identical':True,'base_gripper_labels_and_norm_unchanged':True,
        'original21_proprio_unchanged':True,'gpu_initialized':False,'full_model_loaded':False,
        'simulator_control_calls':0,'gradient_updates':0,'dataset_modified':False,'not_task_evaluation':True,
        'source_provenance_sha256':hashlib.sha256(pp.read_bytes()).hexdigest(),
        'reference_native_normalization_sha256':hashlib.sha256(anchor.read_bytes()).hexdigest(),
        'normalization_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),'action_norm_provenance':source_meta,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    (ROOT/'runs/latest-pi0-relative-preflight.txt').write_text(str(a.output)+'\n')
    print(json.dumps({k:report[k] for k in ('stage','windows','roundtrip_max_abs','normalization_sha256')}))


if __name__=='__main__':main()
