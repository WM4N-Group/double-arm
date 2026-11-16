"""Explicit warm-start provenance for original pi0; never overwrite the source."""
import hashlib
import json
from pathlib import Path


def training_initialization(root,requested,representation):
    root=Path(root)
    if not requested:
        return root/'data/pi0/checkpoints/pi0_base/params',{'kind':'official_pretrained_pi0_base','optimizer_state_restored':False}
    path=Path(requested)
    if representation!='native':raise ValueError('Coverage warm start currently supports native pi0 only')
    if path.is_symlink() or not path.resolve().is_relative_to((root/'runs').resolve()) or path.name!='params' or not path.is_dir():
        raise ValueError('Warm start requires committed project pi0 params')
    checkpoint=path.parent
    if not checkpoint.name.isdigit() or checkpoint.parent.parent.name!='pi0_r1_tidy_table_lora32':
        raise ValueError('Warm start must use the native pi0 configuration, not absolute-base/state models')
    metadata=checkpoint/'_CHECKPOINT_METADATA'
    if not json.loads(metadata.read_text()).get('commit_timestamp_nsecs'):raise ValueError('Uncommitted initialization checkpoint')
    norm=checkpoint/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json'
    if not norm.is_file():raise ValueError('Missing source normalization asset')
    contract_path=checkpoint/'assets/r1-adapter-contract.json'
    if contract_path.is_file():
        contract=json.loads(contract_path.read_text())
        if contract.get('model')!='pi0' or contract.get('action_representation')!='native' or contract.get('quantile_normalization') is not False:
            raise ValueError('Wrong initialization adapter contract')
        if contract.get('norm_stats_sha256')!=hashlib.sha256(norm.read_bytes()).hexdigest():raise ValueError('Source normalization changed')
    return path,{'kind':'warm_start_existing_native_pi0','source_checkpoint':str(checkpoint),
        'source_step_index':int(checkpoint.name),'optimizer_state_restored':False,
        'source_completion_sha256':hashlib.sha256(metadata.read_bytes()).hexdigest(),
        'source_norm_stats_sha256':hashlib.sha256(norm.read_bytes()).hexdigest()}
