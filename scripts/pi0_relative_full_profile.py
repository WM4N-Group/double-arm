"""Versioned original-pi0 FULL-demo diagnostic profile, not a paper recipe.

Old native/absolute-base contracts are not accepted here. Saved model params
retain their original mixed dtypes; inverse joint transform uses only CURRENT
21D input proprio. This module never opens source demonstrations in serving.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/yuanyue/double-arm')
PROFILE='pi0_r1_relative_full_demo_v1'
ASSET_ID='momagen/r1_relative_full_demo_v1'
MODEL_ENCODING='native_base_observation_relative_joints_v1'
RUNTIME_FILES=('pi0_relative_full_profile.py','pi0_relative_joint_diagnostic.py','momagen_pi0_schema.py','pi0_r1_config.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_schedule(length,onset,updates=20000,seed=42):
    """50% complete-demo uniform cycles +50% transition cycles, OFFLINE only."""
    import random
    from pi0_single_demo_fit import transition_indices
    if type(updates) is not int or updates<2*length or updates>100000:
        raise ValueError('Bounded schedule must cover every full-demo frame')
    critical=transition_indices(length,onset);rng=random.Random(seed);uniform=[];transition=[];result=[]
    for i in range(updates):
        pool=uniform if i%2==0 else transition
        if not pool:
            pool.extend(range(length) if i%2==0 else critical);rng.shuffle(pool)
        result.append(pool.pop())
    if set(result)!=set(range(length)):raise ValueError('Incomplete full-demo coverage')
    return result


def data_profile(base_data,norms):
    import dataclasses
    from openpi import transforms
    from pi0_relative_joint_diagnostic import RelativeJointInputs,RelativeJointOutputs
    return dataclasses.replace(base_data,asset_id=ASSET_ID,norm_stats=norms,use_quantile_norm=False,
        data_transforms=transforms.Group(inputs=[RelativeJointInputs()],outputs=[RelativeJointOutputs()]))


def validate_fields(contract):
    fixed={'schema_version':1,'policy_profile':PROFILE,'model':'pi0','lora_rank':32,
        'model_action_encoding':MODEL_ENCODING,'motor_action_encoding':'native_21_after_inverse',
        'normalization_mode':'mean_std','norm_stats_asset_id':ASSET_ID,'original_proprio_dim':21,
        'model_action_dim':32,'action_horizon':50,'parameter_restore_dtype':'preserve_saved',
        'training_is_diagnostic':True,'paper_reproduction_complete':False,'training_completed':True,
        'train_demonstrations':1,'optimizer_updates':20000,
        'openpi_revision':'215abfb217dbac7d5f1273282331b9b1866c0479',
        'teacher_enabled':False,'planner_enabled':False,'object_pose_inputs':False}
    for key,value in fixed.items():
        if type(contract.get(key)) is not type(value) or contract[key]!=value:
            raise ValueError(f'Wrong full-demo profile contract field {key}')
    if (contract.get('training_source_demo')!='demo_0' or contract.get('train_frames')!=1714
        or not contract.get('optimizer_and_lr_counts')
        or any(type(x) is not int or x!=20000 for x in contract['optimizer_and_lr_counts'].values())):
        raise ValueError('Full-demo diagnostic completion/counts not verified')
    if contract.get('parameter_dtype_counts')!={'float32':495908624,'bfloat16':2819996672}:
        raise ValueError('Wrong original-pi0 mixed parameter dtype partition')
    for key in ('norm_stats_sha256','full_parameter_sha256','frame_schedule_sha256'):
        value=contract.get(key,'')
        if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Missing hash '+key)
    if set(contract.get('runtime_source_sha256',{}))!=set(RUNTIME_FILES):raise ValueError('Missing runtime source provenance')
    return contract


def load_contract(checkpoint,root=ROOT,verify_sources=True):
    checkpoint=Path(checkpoint);root=Path(root)
    if checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(root/'runs') or checkpoint.name!='19999':
        raise ValueError('Only committed final diagnostic19999 under project runs/')
    meta=json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())
    if not meta.get('commit_timestamp_nsecs') or not (checkpoint/'params').is_dir():raise ValueError('Uncommitted checkpoint')
    contract=validate_fields(json.loads((checkpoint/'assets/r1-relative-profile-contract.json').read_text()))
    norm=checkpoint/'assets'/ASSET_ID/'norm_stats.json'
    if sha(norm)!=contract['norm_stats_sha256']:raise ValueError('Saved normalization changed')
    if verify_sources:
        for name,digest in contract['runtime_source_sha256'].items():
            if sha(root/'code/scripts'/name)!=digest:raise ValueError('Runtime transform source changed: '+name)
    return contract


def make_policy(checkpoint):
    """Official openpi Policy, original pi0 LoRA, checkpoint norms and dtypes."""
    import jax
    import numpy as np
    from flax import nnx
    from openpi import transforms
    from openpi.models import model as model_module
    from openpi.policies.policy import Policy
    from openpi.training.checkpoints import load_norm_stats
    from pi0_r1_config import register_config
    contract=load_contract(checkpoint)
    from preflight_pi0_accumulation_gpu import pinned_upstream_revision
    pinned_upstream_revision(ROOT/'src/openpi-215abfb')
    cfg=register_config('native');norms=load_norm_stats(Path(checkpoint)/'assets',ASSET_ID)
    for key in ('state','actions'):
        if norms[key].mean.shape!=(21,) or norms[key].std.shape!=(21,) or not np.isfinite(norms[key].mean).all() or not np.isfinite(norms[key].std).all():
            raise ValueError('Invalid saved21D stats')
    data=data_profile(cfg.data.create(cfg.assets_dirs,cfg.model),norms)
    model=cfg.model.load(model_module.restore_params(Path(checkpoint)/'params',dtype=None))
    counts={}
    for leaf in jax.tree.leaves(nnx.state(model)):counts[str(leaf.dtype)]=counts.get(str(leaf.dtype),0)+int(leaf.size)
    if counts!=contract['parameter_dtype_counts']:raise ValueError('Restored parameter dtype/count mismatch')
    policy=Policy(model,transforms=[transforms.InjectDefaultPrompt(None),*data.data_transforms.inputs,
            transforms.Normalize(norms,use_quantiles=False),*data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,transforms.Unnormalize(norms,use_quantiles=False),*data.data_transforms.outputs],
        sample_kwargs={'num_steps':10},metadata={'policy_profile':PROFILE,'model_action_encoding':MODEL_ENCODING})
    return policy,contract
