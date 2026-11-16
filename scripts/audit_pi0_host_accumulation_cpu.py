#!/usr/bin/env python3
"""SERVER CPU ONLY: host raw-gradient accumulation arithmetic, NOT real pi0."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('Explicit CPU only')
    if not args.output.resolve().is_relative_to(Path('/mnt/yuanyue/double-arm/runs')) or args.output.exists():
        raise ValueError('New remote report required')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from pi0_accumulation_preflight import HostGradientAccumulator
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('CPU devices only')
    # NNX State gradient roundtrip as used by the real model, NOT a large model.
    class Tiny(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.zeros(3))
    model = Tiny()
    graph, reference = nnx.split(model)
    current = reference
    tx = optax.chain(optax.clip_by_global_norm(.7), optax.adamw(lambda n: .01/(n+1), b2=.95))
    rs, cs = tx.init(reference), tx.init(current)
    @jax.jit
    def gradient(params, x, y):
        model = nnx.merge(graph, params)
        return nnx.grad(lambda m: jnp.mean((x @ m.w.value-y)**2))(model)
    rng = np.random.default_rng(42)
    differences = []
    for effective_update in range(3):
        x = jnp.asarray(rng.normal(size=(64, 3)), jnp.float32)
        y = jnp.asarray(rng.normal(size=64) * 5, jnp.float32)
        rg = gradient(reference, x, y)
        delta, rs = tx.update(rg, rs, reference)
        reference = optax.apply_updates(reference, delta)
        acc = HostGradientAccumulator(64)
        before = np.asarray(current['w'].value).copy()
        for i in range(64):
            raw = jax.device_get(gradient(current, x[i:i+1], y[i:i+1]))
            leaves, structure = jax.tree.flatten(raw)
            acc.add(leaves)
            np.testing.assert_array_equal(np.asarray(current['w'].value), before)
        mean = jax.tree.unflatten(structure, acc.mean())
        delta, cs = tx.update(mean, cs, current)
        current = optax.apply_updates(current, delta)
        for actual, expected in zip(jax.tree.leaves(current), jax.tree.leaves(reference), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        for actual, expected in zip(jax.tree.leaves(cs), jax.tree.leaves(rs), strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        counts = [int(leaf) for path, leaf in jax.tree_util.tree_flatten_with_path(cs)[0]
                  if path and getattr(path[-1], 'name', None) == 'count']
        assert counts and all(n == effective_update + 1 for n in counts)
        differences.append(float(np.max(np.abs(current['w'].value-reference['w'].value))))
    report = {'mode': 'synthetic_host_accumulator_nnx_cpu_check', 'gpu_initialized': False,
              'actual_pi0_test': False, 'checkpoint_saved': False,
              'three_effective_adamw_updates_match_batch64': True,
              'nnx_gradient_tree_host_roundtrip_verified': True,
              'parameters_fixed_for_all64_micros': True,
              'optimizer_moments_and_lr_update_counts_match': True,
              'clip_after_host_raw_gradient_mean': True,
              'max_parameter_absolute_differences': differences,
              'scope': 'Small deterministic NNX linear loss only, not actual pi0 GPU fit or direct batch64 equivalence'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
