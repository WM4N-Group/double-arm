#!/usr/bin/env python3
"""SERVER CPU ONLY: synthetic optimizer algebra, NOT pi0 training or GPU fit."""
import argparse
import json
import os
from pathlib import Path
from pi0_accumulation_preflight import accumulation_plan,accumulating_optimizer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('Explicit CPU only')
    if not args.output.resolve().is_relative_to(Path('/mnt/yuanyue/double-arm/runs')) or args.output.exists():raise ValueError('New remote report required')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    if any(d.platform!='cpu' for d in jax.devices()):raise ValueError('Non-CPU device')
    k=64;rng=np.random.default_rng(42)
    schedule=lambda count:.01/(count+1)
    inner=optax.chain(optax.clip_by_global_norm(.7),optax.adamw(schedule,b1=.9,b2=.95,weight_decay=.001))
    multi=accumulating_optimizer(inner,k)
    # Match the pinned upstream TrainState.tx runtime type, not duck typing.
    acc=optax.GradientTransformation(multi.init,multi.update)
    assert isinstance(acc,optax.GradientTransformation)
    reference=jnp.zeros(3);micro=reference;ref_state=inner.init(reference);state=acc.init(micro)
    def loss(w,x,y):return jnp.mean((x@w-y)**2)
    @jax.jit
    def micro_step(w,s,x,y):
        gradient=jax.grad(lambda p:(x@p-y)**2)(w);update,s=acc.update(gradient,s,w)
        return optax.apply_updates(w,update),s
    errors=[]
    for update_index in range(3):
        x=jnp.asarray(rng.normal(size=(k,3)),jnp.float32);y=jnp.asarray(rng.normal(size=k)*5,jnp.float32)
        gradients=jax.grad(loss)(reference,x,y);delta,ref_state=inner.update(gradients,ref_state,reference)
        next_reference=optax.apply_updates(reference,delta)
        for i in range(k):
            before=np.asarray(micro).copy();micro,state=micro_step(micro,state,x[i],y[i])
            if i<k-1:np.testing.assert_array_equal(micro,before)
        np.testing.assert_allclose(micro,next_reference,rtol=2e-5,atol=2e-6)
        for actual,expected in zip(jax.tree.leaves(state.inner_opt_state),jax.tree.leaves(ref_state),strict=True):
            np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
        assert int(state.gradient_step)==update_index+1 and int(state.mini_step)==0
        errors.append(float(np.max(np.abs(micro-next_reference))));reference=next_reference
    # Counterexample: per-micro clipping is not global-batch clipping.
    gradients=np.array([100.,-99.]+[0.]*62,np.float32)
    correct_mean=float(gradients.mean());incorrect_mean=float(np.clip(gradients,-1,1).mean())
    assert abs(correct_mean-incorrect_mean)>1e-3
    clipped=accumulating_optimizer(optax.chain(optax.clip_by_global_norm(1.),optax.sgd(1.)),k)
    value=jnp.zeros(1);cs=clipped.init(value)
    for gradient in gradients:
        delta,cs=clipped.update(jnp.array([gradient]),cs,value);value=optax.apply_updates(value,delta)
    np.testing.assert_allclose(value,-correct_mean,atol=1e-6)
    report={'mode':'synthetic_cpu_optimizer_preflight','robot_policy_trained':False,'gpu_initialized':False,
        'actual_pi0_gpu_memory_test_performed':False,'active_training_untouched':True,
        'plan':accumulation_plan(1,64,50000),'three_effective_adamw_updates_match_batch64':True,
        'pinned_trainstate_tx_gradient_transformation_type_verified':True,
        'parameters_unchanged_before_each_window_boundary':True,'optimizer_moments_and_schedule_counts_match':True,
        'clip_after_averaging_verified':True,'max_parameter_absolute_differences':errors,
        'per_micro_clipping_counterexample':{'correct_average':correct_mean,'incorrect_average':incorrect_mean},
        'limitations':'Small deterministic linear loss only; not proof of actual pi0 stochastic-gradient equivalence, throughput or 24GiB memory feasibility'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
