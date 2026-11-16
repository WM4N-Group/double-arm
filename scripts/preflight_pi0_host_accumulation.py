"""Isolated GPU1 preflight variant: GPU gradients, host FP32 accumulation.

Invoked ONLY behind preflight_pi0_accumulation_gpu.py's release/allocation gates.
Same model, native labels, trainable filter, fresh official optimizer, mean/std.
Separates gradient and optimizer executables; never changes active experiments.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def run_host_probe(campaign, initial_params, output):
    import dataclasses
    import importlib.util
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax
    from flax import nnx
    from openpi.training import data_loader, sharding
    from pi0_accumulation_preflight import HostGradientAccumulator, accumulation_plan
    from pi0_r1_config import register_config
    from preflight_pi0_accumulation_gpu import ROOT, pinned_upstream_revision, trainable_fingerprint
    from resource_allocation import ALLOWED_GPU_UUID, resolve_gpu

    source = ROOT / 'src/openpi-215abfb'
    report = {
        'mode': 'isolated_real_pi0_host_accumulation_preflight',
        'variant': 'host_fp32_raw_gradient_sum_separate_gradient_and_optimizer_jit',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'campaign_guard': str(campaign), 'gpu_uuid': ALLOWED_GPU_UUID,
        'active_training_modified': False, 'checkpoint_saved': False,
        'policy_task_success': False, 'paper_reproduction': False,
        'actual_microsteps': 0, 'actual_optimizer_updates': 0,
        'actual_pi0_gpu_memory_fit_verified': False,
        'real_pi0_accumulation_smoke_passed': False,
        'direct_real_pi0_batch64_numerical_equivalence_verified': False,
        'accumulator_placement': 'host_RAM_numpy_float32_NOT_GPU',
        'scope': 'Native/mean-std memory/count variant, not paper-aligned training. '
                 'Gradient computation uses unchanged weights for all64 micros; '
                 'official clipping/AdamW only once after raw gradient mean. '
                 'Not direct real batch64 numerical equivalence or task success.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'initial_params': str(initial_params),
    }
    def save(**updates):
        report.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(output)
    save(stage='initializing_real_pi0_host_accumulation_variant')
    try:
        revision = pinned_upstream_revision(source)
        train_file = source / 'scripts/train.py'
        spec = importlib.util.spec_from_file_location('_host_probe_upstream_train', train_file)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform not in ('gpu', 'cuda'):
            raise ValueError('Exactly one CUDA device required')
        jax.config.update('jax_compilation_cache_dir', str(ROOT / 'cache/jax-host-accumulation-preflight'))
        config = dataclasses.replace(register_config('native'), num_workers=0, batch_size=1, ema_decay=None)
        if config.policy_metadata['initialization']['optimizer_state_restored']:
            raise ValueError('Fresh optimizer required')
        if config.data.create(config.assets_dirs, config.model).use_quantile_norm:
            raise ValueError('Do not change native mean/std in a memory-only variant')
        mesh = sharding.make_mesh(1)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        rng, init_rng = jax.random.split(jax.random.key(config.seed))
        # NO MultiSteps on GPU: official fresh optimizer/state unchanged.
        state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=False)
        jax.block_until_ready(state)
        initial_hash, count = trainable_fingerprint(state.params.filter(config.trainable_filter))
        def optimizer_counts(s):
            counts = {jax.tree_util.keystr(path): int(leaf)
                      for path, leaf in jax.tree_util.tree_flatten_with_path(s.opt_state)[0]
                      if path and getattr(path[-1], 'name', None) == 'count'}
            if not counts:
                raise ValueError('Missing official AdamW/schedule counters')
            return counts
        if int(state.step) != 0 or any(optimizer_counts(state).values()):
            raise ValueError('Optimizer must start at zero')
        model_def = state.model_def
        def compute_gradients(key, params, batch):
            # This is the pinned train_step's loss/diff-filter operation, split
            # from its optimizer for placement only. No per-micro clipping.
            model = nnx.merge(model_def, params)
            model.train()
            def loss_fn(model, key, observation, actions):
                return jnp.mean(model.compute_loss(key, observation, actions, train=True))
            loss, grads = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, config.trainable_filter))(
                model, key, *batch)
            return grads, {'loss': loss, 'raw_grad_norm': optax.global_norm(grads)}
        gradient_step = jax.jit(compute_gradients,
            in_shardings=(replicated, state_sharding.params, data_sharding),
            out_shardings=(replicated, replicated))  # params intentionally NOT donated/updated
        def apply_mean(s, gradient):
            params = s.params.filter(config.trainable_filter)
            updates, new_opt_state = s.tx.update(gradient, s.opt_state, params)
            model = nnx.merge(s.model_def, s.params)
            nnx.update(model, optax.apply_updates(params, updates))
            return dataclasses.replace(s, step=s.step + 1, params=nnx.state(model), opt_state=new_opt_state)
        optimizer_step = jax.jit(apply_mean, in_shardings=(state_sharding, replicated),
                                 out_shardings=state_sharding, donate_argnums=(0, 1))
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        accumulator = HostGradientAccumulator(64)
        observations = []
        structure = None
        save(stage='compiling_gradient_only_then_host_accumulating', trainable_parameter_count=count,
             upstream_revision=revision, upstream_train_sha256=hashlib.sha256(train_file.read_bytes()).hexdigest(),
             plan=accumulation_plan(1, 64, 50000, count), initial_trainable_sha256=initial_hash,
             initial_device_memory_stats=devices[0].memory_stats())
        for index in range(64):
            if resolve_gpu(ROOT) != ALLOWED_GPU_UUID or (ROOT / 'runs/USER_PAUSED.json').exists():
                raise RuntimeError('Allocation changed or user paused')
            started = time.monotonic()
            batch = next(iterator)
            with sharding.set_mesh(mesh):
                gradients, info = gradient_step(jax.random.fold_in(rng, index), state.params, batch)
            jax.block_until_ready(gradients)
            info = {key: float(value) for key, value in jax.device_get(info).items()}
            if not all(np.isfinite(value) for value in info.values()):
                raise ValueError('Non-finite real pi0 loss/raw gradient norm')
            host = jax.device_get(gradients)
            leaves, tree = jax.tree.flatten(host)
            if structure is not None and tree != structure:
                raise ValueError('Gradient tree changed')
            structure = tree
            accumulator.add(leaves)  # verifies EVERY raw gradient element finite FP32
            del gradients, host, leaves, batch
            if int(state.step) != 0 or any(optimizer_counts(state).values()):
                raise ValueError('Optimizer/LR advanced inside micro window')
            item = {'microstep': index + 1, 'optimizer_updates': 0,
                    'seconds': time.monotonic() - started, **info}
            if index + 1 in (1, 63, 64):
                digest, _ = trainable_fingerprint(state.params.filter(config.trainable_filter))
                if digest != initial_hash:
                    raise ValueError('Parameters changed before effective optimizer update')
                item['trainable_sha256'] = digest
            observations.append(item)
            save(actual_microsteps=index + 1, actual_host_accumulator_bytes=accumulator.nbytes,
                 actual_optimizer_updates=0, observations=observations,
                 latest_device_memory_stats=devices[0].memory_stats())
        mean_leaves = accumulator.mean()
        if not all(np.isfinite(value).all() for value in mean_leaves):
            raise ValueError('Non-finite averaged host gradient')
        mean_tree = jax.tree.unflatten(structure, mean_leaves)
        mean_gpu = jax.device_put(mean_tree, replicated)
        del mean_tree
        save(stage='compiling_and_applying_one_official_optimizer_update')
        with sharding.set_mesh(mesh):
            state = optimizer_step(state, mean_gpu)
        del mean_gpu
        jax.block_until_ready(state)
        if int(state.step) != 1 or any(value != 1 for value in optimizer_counts(state).values()):
            raise ValueError('Expected exactly one optimizer/AdamW/LR update')
        digest, _ = trainable_fingerprint(state.params.filter(config.trainable_filter))
        if digest == initial_hash:
            raise ValueError('No parameter change at effective update')
        # Full trainable tree finite check also covers optimizer output.
        for leaf in jax.tree.leaves(state.params.filter(config.trainable_filter)):
            if not np.isfinite(np.asarray(jax.device_get(leaf))).all():
                raise ValueError('Non-finite parameters after effective update')
        save(stage='completed', actual_optimizer_updates=1,
             real_pi0_accumulation_smoke_passed=True, actual_pi0_gpu_memory_fit_verified=True,
             inner_optimizer_and_lr_counts=optimizer_counts(state), final_trainable_sha256=digest,
             parameters_unchanged_at_checked_microsteps=[1, 63, 64],
             parameters_changed_only_after_64_microstep_mean=True,
             optimizer_clipping_scope='official global clip/AdamW after mean of64 RAW gradients',
             latest_device_memory_stats=devices[0].memory_stats(),
             total_microstep_wall_seconds=sum(item['seconds'] for item in observations))
    except BaseException as error:
        save(stage='failed', error=repr(error)[:12000],
             real_pi0_accumulation_smoke_passed=False, actual_pi0_gpu_memory_fit_verified=False)
        raise
