#!/usr/bin/env python3
"""SERVER ONLY: CPU shape-trace the real pi0 topology and freeze partition.

No full model tensors, gradients or GPU context are created. Report official
freeze-filter behavior rather than assuming every non-LoRA leaf is frozen.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only audit: explicitly set JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES=""')

    import flax.nnx as nnx
    import jax
    import numpy as np
    from openpi.models import gemma
    from pi0_r1_config import register_config

    config = register_config()
    start = time.monotonic()
    state = jax.eval_shape(lambda rng: nnx.state(config.model.create(rng)), jax.random.key(123))

    def count(tree):
        return sum(int(np.prod(leaf.shape)) for leaf in jax.tree.leaves(tree.to_pure_dict()))

    train = state.filter(config.trainable_filter)
    frozen = state.filter(config.freeze_filter)
    groups = {}
    non_lora = []
    # Pinned Flax 0.10.2 exposes State.flat_state(), not nnx.to_flat_state().
    for path, leaf in train.flat_state().items():
        name = '/'.join(map(str, path))
        elements = int(np.prod(leaf.value.shape))
        prefix = '/'.join(name.split('/')[:2])
        groups[prefix] = groups.get(prefix, 0) + elements
        if 'lora' not in name:
            non_lora.append({'path': name, 'shape': list(leaf.value.shape), 'elements': elements})
    ranks = {
        name: {kind: value.rank for kind, value in gemma.get_config(name).lora_configs.items()}
        for name in ('gemma_2b_lora', 'gemma_300m_lora')
    }
    if not all(values and all(rank == 32 for rank in values.values()) for values in ranks.values()):
        raise ValueError(f'Unexpected LoRA ranks: {ranks}')
    if count(train) + count(frozen) != count(state):
        raise ValueError('Freeze and trainable filters do not partition the full model state')
    report = {
        'mode': 'cpu_shape_and_freeze_contract_only',
        'gpu_initialized': False,
        'gradient_updates': 0,
        'model': 'pi0',
        'action_dim_padded': config.model.action_dim,
        'native_action_dim': 21,
        'action_horizon': config.model.action_horizon,
        'lora_ranks': ranks,
        'total_parameters': count(state),
        'trainable_parameters': count(train),
        'frozen_parameters': count(frozen),
        'trainable_parameter_groups': groups,
        'non_lora_trainable_leaves': non_lora,
        'not_every_non_lora_leaf_frozen': bool(non_lora),
        'meaning': 'Official openpi freeze partition; can include non-LoRA vision/projection leaves, not assumed pure LoRA-only',
        'elapsed_s': time.monotonic() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
