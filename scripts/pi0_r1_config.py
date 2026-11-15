"""Official openpi pi0 LoRA adaptation for MoMaGen R1, not a new policy network."""
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from momagen_pi0_schema import make_pi0_inputs,unpad_pi0_actions,PROMPT


@dataclasses.dataclass(frozen=True)
class R1Inputs:
    def __call__(self,data):return make_pi0_inputs(data)


@functools.lru_cache(maxsize=4)
def load_absolute_overlay(directory,split):
    directory=Path(directory)
    labels=np.load(directory/(split+'-actions.npy'),mmap_mode='r')
    ends=np.load(directory/(split+'-episode-ends.npy'),mmap_mode='r')
    metadata=json.loads((directory/'manifest.json').read_text())['splits'][split]
    source=Path(metadata['source_provenance'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=metadata['source_provenance_sha256']:
        raise ValueError('Source RGB/proprio provenance changed after label derivation')
    if labels.shape!=(metadata['frames'],21) or ends.shape!=(len(labels),):raise ValueError('Invalid absolute label shape')
    if hashlib.sha256(labels.tobytes()).hexdigest()!=metadata['label_sha256']:raise ValueError('Absolute label hash mismatch')
    if np.any(ends<np.arange(len(labels))) or np.any(ends>=len(labels)):raise ValueError('Invalid episode boundaries')
    return labels,ends


@dataclasses.dataclass(frozen=True)
class R1AbsoluteBaseInputs:
    directory:str
    split:str='train'
    def __call__(self,data):
        result=make_pi0_inputs(data)
        if 'actions' in data:
            # Dataset index is ONLY for offline supervised label lookup; it is
            # stripped here and never becomes a VLA observation. Inference has
            # no action labels or index and does not open the overlay at all.
            labels,ends=load_absolute_overlay(self.directory,self.split)
            index=int(np.asarray(data['index']).item())
            horizon=result['actions'].shape[0]
            if not 0<=index<len(labels):raise ValueError('Invalid label-overlay index')
            indices=np.minimum(np.arange(index,index+horizon),ends[index])
            result['actions']=np.asarray(labels[indices],np.float32).copy()
        return result


@dataclasses.dataclass(frozen=True)
class R1Outputs:
    def __call__(self,data):return {'actions':unpad_pi0_actions(data['actions'])}


def register_config(action_representation=None):
    from openpi import transforms
    from openpi.models import gemma
    from openpi.models import pi0_config
    from openpi.training import config,weight_loaders
    root=Path(os.environ.get('DOUBLE_ARM_ROOT','/mnt/yuanyue/double-arm'))
    representation=action_representation or os.environ.get('PI0_ACTION_REPRESENTATION','native')
    if representation not in ('native','absolute_base'):raise ValueError('Unknown pi0 action representation')
    absolute=representation=='absolute_base'
    # The pinned official default is rank16 for the VLM, rank32 for the action
    # expert. Match MoMaGen's reported rank32 on both without editing upstream.
    if not getattr(gemma.get_config,'_r1_rank32',False):
        original=gemma.get_config
        def rank32(variant):
            result=original(variant)
            if variant in ('gemma_2b_lora','gemma_300m_lora'):
                result=dataclasses.replace(result,lora_configs={name:dataclasses.replace(value,rank=32,alpha=32.0) for name,value in result.lora_configs.items()})
            return result
        rank32._r1_rank32=True;gemma.get_config=rank32
    @dataclasses.dataclass(frozen=True)
    class R1DataConfig(config.DataConfigFactory):
        def create(self,assets_dirs,model_config):
            return dataclasses.replace(self.create_base_config(assets_dirs,model_config),
                use_quantile_norm=absolute,  # Override pinned factory's PI0=False default AFTER create_base_config.
                data_transforms=transforms.Group(inputs=[R1AbsoluteBaseInputs(str(root/'data/pi0/absolute-base-v2'),
                    'validation' if 'holdout' in (self.repo_id or '') else 'train') if absolute else R1Inputs()],outputs=[R1Outputs()]),
                model_transforms=config.ModelTransformFactory(default_prompt=PROMPT)(model_config))
    model=pi0_config.Pi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50)
    name='pi0_r1_tidy_table_absolute_base_lora32' if absolute else 'pi0_r1_tidy_table_lora32'
    train=config.TrainConfig(name=name,project_name='double-arm-pi0',exp_name=os.environ.get('PI0_EXP_NAME','pilot'),
        model=model,data=R1DataConfig(repo_id=os.environ.get('PI0_DATASET_REPO','momagen/r1_tidy_table_train_v1'),base_config=config.DataConfig(prompt_from_task=True,use_quantile_norm=absolute)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(root/'data/pi0/checkpoints/pi0_base/params')),
        freeze_filter=model.get_freeze_filter(),ema_decay=None,
        batch_size=int(os.environ.get('PI0_BATCH_SIZE','1')),num_workers=2,
        num_train_steps=int(os.environ.get('PI0_TRAIN_STEPS','30000')),
        save_interval=1000,log_interval=10,keep_period=5000,
        assets_base_dir=str(root/'data/pi0/assets'),checkpoint_base_dir=str(root/'runs/pi0-checkpoints'),
        wandb_enabled=False,fsdp_devices=1,
        policy_metadata={'robot':'MoMaGen R1','model':'pi0','action_dim_native':21,'action_horizon':50,'lora_rank_vlm':32,'lora_rank_action':32,
                         'action_representation':representation,'action_encoding':
                         'absolute world-base targets + absolute joint targets + native grippers; robot odometry used only in low-level controller' if absolute else
                         'native local-delta base + absolute joint targets + native grippers; no ALOHA conversion or extra delta transform'})
    config._CONFIGS_DICT[name]=train
    return train
