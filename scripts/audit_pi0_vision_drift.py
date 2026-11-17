#!/usr/bin/env python3
"""SERVER CPU: partial restore TWO tiny SigLip tensors; NOT full model drift.

Records aggregate differences only, never writes weights or runs inference.
A changed parameter is NOT proof of semantic forgetting or task failure cause.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/yuanyue/double-arm')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':
        raise ValueError('Explicit CPU only')
    if not args.output.resolve().is_relative_to(ROOT/'runs') or args.output.exists():
        raise ValueError('New remote report required')
    import numpy as np
    import orbax.checkpoint as ocp
    campaign=Path((ROOT/'runs/latest-pi0-coverage-campaign.txt').read_text().strip())
    current=Path(json.loads((campaign/'status.json').read_text())['checkpoint'])/'params'
    cases={'pretrained':ROOT/'data/pi0/checkpoints/pi0_base/params',
           'native30k':ROOT/'runs/pi0-checkpoints/pi0_r1_tidy_table_lora32/pi0-momagen-20261115-233529/29999/params',
           'coverage_v3':current}
    records={};base=None
    for name,path in cases.items():
        suffix='' if name=='pretrained' else '/value'
        item={key:np.empty(1152,np.float32) for key in ('bias','scale')}
        mapping={key:ocp.Transform(original_key='params/PaliGemma/img/Transformer/encoder_norm/'+key+suffix) for key in item}
        restore={key:ocp.ArrayRestoreArgs(restore_type=np.ndarray,dtype=np.float32) for key in item}
        with ocp.PyTreeCheckpointer() as checkpointer:
            values=checkpointer.restore(path,ocp.args.PyTreeRestore(item=item,restore_args=restore,
                transforms=mapping,transforms_default_to_original=False))
        if base is None:base=values
        records[name]={'params_path':str(path),'tensors':{key:{'elements':int(value.size),
            'changed_vs_base':int(np.sum(value!=base[key])),
            'relative_l2_delta':float(np.linalg.norm(value-base[key])/max(1e-12,np.linalg.norm(base[key]))),
            'max_absolute_delta':float(np.abs(value-base[key]).max())} for key,value in values.items()}}
    report={'mode':'partial_SigLip_parameter_update_audit','gpu_initialized':False,
        'gradient_updates':0,'weights_saved':False,
        'scope':'Only encoder_norm scale/bias,2304 elements total per checkpoint, NOT whole414M encoder or semantic forgetting proof',
        'records':records,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
