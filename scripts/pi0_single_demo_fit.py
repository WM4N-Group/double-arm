"""Isolated single-demo fitting helpers; no old adapter/recipe mutation."""
import hashlib
import json
import random

OFFSETS=(-30,-15,-5,-1,0,1,5,15,30)
VARIANTS=('original_vision_trainable','extra_vision_frozen')


def transition_indices(length, onset):
    if type(length) is not int or type(onset) is not int or length<1:
        raise ValueError('Integer source length/onset required')
    indices=[onset+x for x in OFFSETS]
    if min(indices)<0 or max(indices)+49>=length:
        raise ValueError('Need nine real transition observations with full within-episode horizons')
    return indices


def frame_schedule(indices, updates, seed=42):
    if type(updates) is not int or not 1<=updates<=2000 or len(set(indices))!=9:
        raise ValueError('Bounded9-window diagnostic required')
    rng=random.Random(seed);order=[]
    while len(order)<updates:
        chunk=list(indices);rng.shuffle(chunk);order.extend(chunk)
    return order[:updates]


def sequence_sha256(values):
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


def vision_filter():
    from openpi.shared.nnx_utils import PathRegex
    return PathRegex(r'.*PaliGemma/img/.*')


def variant_freeze_filter(base_filter, variant):
    from flax import nnx
    if variant not in VARIANTS:raise ValueError('Unknown explicitly versioned diagnostic variant')
    return base_filter if variant==VARIANTS[0] else nnx.Any(base_filter,vision_filter())


def storage_cast(params, common_filter):
    """Same cast filter for BOTH conditions, not the variable gradient filter."""
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    return nnx_utils.state_map(params,common_filter,lambda p:p.replace(p.value.astype(jnp.bfloat16)))


def initialize_state(config, common_storage_filter, init_rng, mesh, upstream):
    """Pinned init algorithm, except storage cast is explicitly decoupled.

    Same official weight loader / optimizer / shape validation / sharding. This
    is a NEW diagnostic init, not a patch or silent change to upstream train.py.
    """
    import jax
    from flax import nnx
    from openpi.training import optimizer,sharding,utils
    tx=optimizer.create_optimizer(config.optimizer,config.lr_schedule,weight_decay_mask=None)
    def init(rng, partial_params=None):
        _,model_rng=jax.random.split(rng)
        model=config.model.create(model_rng)
        if partial_params is not None:
            graph,params=nnx.split(model);params.replace_by_pure_dict(partial_params);model=nnx.merge(graph,params)
        params=storage_cast(nnx.state(model),common_storage_filter)
        return utils.TrainState(step=0,params=params,model_def=nnx.graphdef(model),tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),ema_decay=None,ema_params=None)
    shape=jax.eval_shape(init,init_rng)
    placement=sharding.fsdp_sharding(shape,mesh,log=False)
    partial=upstream._load_weights_and_validate(config.weight_loader,shape.params.to_pure_dict())
    replicated=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    state=jax.jit(init,donate_argnums=(1,),in_shardings=replicated,out_shardings=placement)(init_rng,partial)
    return state,placement
