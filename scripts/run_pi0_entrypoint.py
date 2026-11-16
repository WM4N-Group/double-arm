#!/usr/bin/env python3
"""SERVER ONLY: register the R1 data adapter, then call official openpi routines."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def capture_adapter_contract(config,root):
    representation=config.policy_metadata.get('action_representation','native')
    data=config.data.create(config.assets_dirs,config.model)
    quantile=data.use_quantile_norm
    if quantile!=(representation=='absolute_base'):raise ValueError('Normalization/action contract mismatch')
    norm=Path(config.data.assets.assets_dir or config.assets_dirs)/data.asset_id/'norm_stats.json'
    scripts=Path(__file__).resolve().parent
    report={'model':'pi0','config_name':config.name,'action_representation':representation,
            'quantile_normalization':quantile,'model_action_dim':config.model.action_dim,
            'motor_action_dim':21,'action_horizon':config.model.action_horizon,
            'policy_inputs':'real head + both wrist RGB, original 21D proprioception, instruction',
            'norm_stats_asset_id':data.asset_id,'source_norm_stats_sha256':hashlib.sha256(norm.read_bytes()).hexdigest(),
            'training_source_sha256':{name:hashlib.sha256((scripts/name).read_bytes()).hexdigest()
                for name in ('run_pi0_entrypoint.py','pi0_r1_config.py','momagen_pi0_schema.py')},
            'base_odometry_scope':'low-level motor conversion only, not VLA input',
            'initialization':config.policy_metadata.get('initialization'),
            'training_steps':config.num_train_steps,'batch_size':config.batch_size,
            'lr_schedule':{'warmup_steps':config.lr_schedule.warmup_steps,'peak_lr':config.lr_schedule.peak_lr,
                'decay_steps':config.lr_schedule.decay_steps,'decay_lr':config.lr_schedule.decay_lr}}
    if representation=='absolute_base':
        manifest=root/'data/pi0/absolute-base-v2/manifest.json'
        report['derived_label_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
    return report


def save_adapter_contracts(checkpoint_directory,contract):
    if not contract.get('norm_stats_asset_id'):raise ValueError('Normalization asset id required')
    saved=[]
    for checkpoint in sorted(Path(checkpoint_directory).iterdir()):
        if not checkpoint.name.isdigit() or not (checkpoint/'params').is_dir():continue
        metadata=json.loads((checkpoint/'_CHECKPOINT_METADATA').read_text())
        if not metadata.get('commit_timestamp_nsecs'):raise ValueError('Incomplete checkpoint after trainer exit')
        path=checkpoint/'assets/r1-adapter-contract.json'
        if not path.parent.is_dir():raise ValueError('Missing checkpoint assets')
        # Fingerprint the ACTUAL saved stats: checkpoint serialization may not
        # preserve byte formatting of the global training asset JSON.
        saved_contract=dict(contract)
        norm=path.parent/contract['norm_stats_asset_id']/'norm_stats.json'
        saved_contract['norm_stats_sha256']=hashlib.sha256(norm.read_bytes()).hexdigest()
        if path.exists() and json.loads(path.read_text())!=saved_contract:raise ValueError('Conflicting immutable adapter contract')
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(saved_contract,indent=2));temporary.replace(path)
        saved.append(str(path))
    if not saved:raise ValueError('No committed checkpoints to annotate')
    return saved


def run_official_training(source,config_name,remaining,root):
    # The pinned upstream main explicitly selects ~/.cache/jax. Redirect only
    # that filesystem setting for this process; preserve the upstream source,
    # training loop, optimizer and model, and do not move shared user caches.
    import jax
    original_update=jax.config.update
    def project_update(name,val):
        if name=='jax_compilation_cache_dir':val=str(root/'cache/jax')
        return original_update(name,val)
    jax.config.update=project_update
    try:
        sys.argv=[str(source/'scripts/train.py'),config_name,*remaining]
        runpy.run_path(str(source/'scripts/train.py'),run_name='__main__')
    finally:jax.config.update=original_update


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['config','norms','train'])
    args,remaining=parser.parse_known_args()
    root=Path(os.environ.get('DOUBLE_ARM_ROOT','/mnt/yuanyue/double-arm'))
    os.environ.setdefault('OPENPI_DATA_HOME',str(root/'cache/openpi'))
    os.environ.setdefault('XDG_CACHE_HOME',str(root/'cache/xdg'))
    from pi0_r1_config import register_config
    config=register_config()
    if args.mode=='config':print(config);return
    source=Path(os.environ['OPENPI_ROOT']);sys.path.insert(0,str(source/'scripts'))
    if args.mode=='norms':
        import compute_norm_stats
        limit=os.environ.get('PI0_NORM_MAX_FRAMES')
        compute_norm_stats.main(config.name,max_frames=int(limit) if limit else None)
    else:
        contract=capture_adapter_contract(config,root)
        run_official_training(source,config.name,remaining,root)
        print('ADAPTER_CONTRACTS_SAVED',json.dumps(save_adapter_contracts(config.checkpoint_dir,contract)),flush=True)


if __name__=='__main__':main()
