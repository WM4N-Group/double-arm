"""Official openpi pi0 LoRA adaptation for MoMaGen R1, not a new policy network."""
import dataclasses
import os
from pathlib import Path
import numpy as np
from momagen_pi0_schema import make_pi0_inputs,unpad_pi0_actions,PROMPT


@dataclasses.dataclass(frozen=True)
class R1Inputs:
    def __call__(self,data):return make_pi0_inputs(data)


@dataclasses.dataclass(frozen=True)
class R1Outputs:
    def __call__(self,data):return {'actions':unpad_pi0_actions(data['actions'])}


def register_config():
    from openpi import transforms
    from openpi.models import gemma
    from openpi.models import pi0_config
    from openpi.training import config,weight_loaders
    root=Path(os.environ.get('DOUBLE_ARM_ROOT','/mnt/yuanyue/double-arm'))
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
                data_transforms=transforms.Group(inputs=[R1Inputs()],outputs=[R1Outputs()]),
                model_transforms=config.ModelTransformFactory(default_prompt=PROMPT)(model_config))
    model=pi0_config.Pi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora',action_horizon=50)
    name='pi0_r1_tidy_table_lora32'
    train=config.TrainConfig(name=name,project_name='double-arm-pi0',exp_name=os.environ.get('PI0_EXP_NAME','pilot'),
        model=model,data=R1DataConfig(repo_id=os.environ.get('PI0_DATASET_REPO','momagen/r1_tidy_table_train_v1'),base_config=config.DataConfig(prompt_from_task=True)),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(root/'data/pi0/checkpoints/pi0_base/params')),
        freeze_filter=model.get_freeze_filter(),ema_decay=None,
        batch_size=int(os.environ.get('PI0_BATCH_SIZE','1')),num_workers=2,
        num_train_steps=int(os.environ.get('PI0_TRAIN_STEPS','30000')),
        save_interval=1000,log_interval=10,keep_period=5000,
        assets_base_dir=str(root/'data/pi0/assets'),checkpoint_base_dir=str(root/'runs/pi0-checkpoints'),
        wandb_enabled=False,fsdp_devices=1,
        policy_metadata={'robot':'MoMaGen R1','model':'pi0','action_dim_native':21,'action_horizon':50,'lora_rank_vlm':32,'lora_rank_action':32,
                         'action_encoding':'native local-delta base + absolute joint targets + native grippers; no ALOHA conversion or extra delta transform'})
    config._CONFIGS_DICT[name]=train
    return train
