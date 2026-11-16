#!/usr/bin/env python3
"""SERVER ONLY: isolated real-pi0 batch1 x64 memory/count smoke, NOT training.

Never calls upstream main, saves checkpoints, changes source/data, or launches a
simulator. Refuses even to import JAX until the coverage workflow has terminated,
its controller/child are dead, and the authorized GPU is empty. --check-only is
stdlib-only and safe while the existing workflow is running.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pi0_accumulation_preflight import accumulating_optimizer, accumulation_plan
from resource_allocation import ALLOWED_GPU_UUID, allocation, resolve_gpu

ROOT = Path('/mnt/yuanyue/double-arm')
OPENPI_REV = '215abfb217dbac7d5f1273282331b9b1866c0479'
TERMINAL_STAGES = frozenset({
    'source_actions_not_reproducible_no_blind_training',
    'coverage_smoke_failed_no_formal_training',
    'coverage_training_failed',
    'coverage_trained_training_condition_still_failed',
    'coverage_heldout_physical_success_manual_review_required',
    'coverage_heldout_physical_failed',
    'coverage_pipeline_failed',
})


def process_alive(pid):
    if type(pid) is not int or pid <= 0:
        return False
    try:
        return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def release_blockers(status, pointer, campaign, live_pids, gpu_processes):
    """Pure fail-closed gate. An empty card between train/eval is NOT release."""
    reasons = []
    if status.get('stage') not in TERMINAL_STAGES:
        reasons.append('coverage workflow is not terminal (training AND auto-evaluation must finish)')
    if pointer.get('campaign') != str(campaign) or pointer.get('status_file') != str(campaign / 'status.json'):
        reasons.append('latest single-GPU job no longer points to this coverage workflow; inspect new owner')
    if pointer.get('gpu_uuid') != ALLOWED_GPU_UUID or pointer.get('max_gpu_count') != 1:
        reasons.append('single-GPU job allocation mismatch')
    if type(status.get('controller_pid')) is not int or status.get('controller_pid', 0) <= 0:
        reasons.append('missing valid coverage controller PID')
    if pointer.get('controller_pid') != status.get('controller_pid'):
        reasons.append('controller identity differs between campaign and authoritative job pointer')
    for pid in {status.get('controller_pid'), status.get('active_pid'), pointer.get('controller_pid')}:
        if pid in live_pids:
            reasons.append(f'workflow process {pid} is still alive')
    if any(uuid == ALLOWED_GPU_UUID for uuid, _pid in gpu_processes):
        reasons.append('authorized GPU1 is occupied; do not steal it or use GPU0')
    return reasons


def inspect_release(campaign):
    if not campaign.resolve().is_relative_to(ROOT / 'runs') or campaign.is_symlink():
        raise ValueError('Expected existing campaign under remote runs/')
    gpu = resolve_gpu(ROOT)
    if allocation(ROOT).get('compute_time_budget') != 'open_ended_explicit_user_authorization':
        raise ValueError('Open-ended compute authorization missing')
    if (ROOT / 'runs/USER_PAUSED.json').exists():
        raise ValueError('User paused project')
    status = json.loads((campaign / 'status.json').read_text())
    pointer = json.loads((ROOT / 'runs/latest-single-gpu-job.json').read_text())
    pids = {status.get('controller_pid'), status.get('active_pid'), pointer.get('controller_pid')}
    live = {pid for pid in pids if process_alive(pid)}
    query = subprocess.check_output([
        'nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    apps = []
    for line in query.splitlines():
        uuid, pid = line.split(',')
        apps.append((uuid.strip(), int(pid.strip())))
    reasons = release_blockers(status, pointer, campaign, live, apps)
    if reasons:
        raise RuntimeError('; '.join(reasons))
    return gpu


def pinned_upstream_revision(source):
    """Pinned archive installs use the SAME marker checked by the installer."""
    source = Path(source)
    marker = source / 'UPSTREAM_REVISION'
    if (source / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if marker.exists() and marker.read_text().strip() != revision:
            raise ValueError('Archive marker and Git revision disagree')
    else:
        revision = marker.read_text().strip()
    if revision != OPENPI_REV:
        raise ValueError('Pinned openpi revision mismatch')
    return revision


def trainable_fingerprint(params):
    """Host hash only; never writes model tensors. Called at 0/1/63/64."""
    import jax
    import numpy as np
    digest = hashlib.sha256()
    count = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        array = np.asarray(jax.device_get(leaf))
        digest.update(str(path).encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
        count += array.size
    return digest.hexdigest(), count


def run_probe(campaign, initial_params, output):
    # Heavy imports occur ONLY after the gate and a project-local exclusive lock.
    import dataclasses
    import functools
    import gc
    import importlib.util
    import jax
    import numpy as np
    import optax
    from openpi.training import data_loader, optimizer, sharding
    from pi0_r1_config import register_config

    source = ROOT / 'src/openpi-215abfb'
    revision = pinned_upstream_revision(source)
    train_file = source / 'scripts/train.py'
    spec = importlib.util.spec_from_file_location('_isolated_pi0_upstream_train', train_file)
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    report = {
        'mode': 'isolated_real_pi0_accumulation_memory_preflight',
        'started_at': datetime.now(timezone.utc).isoformat(),
        'campaign_guard': str(campaign), 'gpu_uuid': ALLOWED_GPU_UUID,
        'upstream_revision': revision, 'upstream_train_sha256': hashlib.sha256(train_file.read_bytes()).hexdigest(),
        'active_training_modified': False, 'checkpoint_saved': False,
        'policy_task_success': False, 'paper_reproduction': False,
        'direct_real_pi0_batch64_numerical_equivalence_verified': False,
        'scope': 'Current native/mean-std diagnostic recipe; not a new paper-aligned recipe. '
                 'Tests real gradients and one effective update, NOT direct batch64 equivalence or task success.',
        'plan': accumulation_plan(1, 64, 50000),
        'actual_microsteps': 0, 'actual_optimizer_updates': 0,
        'real_pi0_accumulation_smoke_passed': False,
        'actual_pi0_gpu_memory_fit_verified': False,
    }
    def save(**updates):
        report.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(output)
    save(stage='initializing_real_pi0')
    try:
        devices = jax.devices()
        if len(devices) != 1 or devices[0].platform not in ('gpu', 'cuda'):
            raise ValueError('Exactly one CUDA device required')
        jax.config.update('jax_compilation_cache_dir', str(ROOT / 'cache/jax-accumulation-preflight'))
        config = register_config('native')
        config = dataclasses.replace(config, num_workers=0, batch_size=1, ema_decay=None)
        if config.policy_metadata['initialization']['optimizer_state_restored']:
            raise ValueError('Preflight requires a fresh optimizer')
        config_data = config.data.create(config.assets_dirs, config.model)
        if config_data.use_quantile_norm:
            raise ValueError('This preflight must not silently change the current native mean/std recipe')
        mesh = sharding.make_mesh(1)
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        # An in-process optimizer factory override only; restore immediately.
        original = optimizer.create_optimizer
        def factory(*args, **kwargs):
            multi = accumulating_optimizer(original(*args, **kwargs), 64)
            # Pinned TrainState runtime-checks tx as this NamedTuple, not the
            # duck-typed MultiSteps object. Preserve identical init/update/state.
            return optax.GradientTransformation(multi.init, multi.update)
        optimizer.create_optimizer = factory
        try:
            rng, init_rng = jax.random.split(jax.random.key(config.seed))
            state, state_sharding = upstream.init_train_state(config, init_rng, mesh, resume=False)
            jax.block_until_ready(state)
        finally:
            optimizer.create_optimizer = original
        params = state.params.filter(config.trainable_filter)
        initial_hash, parameter_count = trainable_fingerprint(params)
        del params
        if int(state.opt_state.gradient_step) != 0 or int(state.opt_state.mini_step) != 0:
            raise ValueError('Accumulation counters must start at zero')
        accumulator_leaves = jax.tree.leaves(state.opt_state.acc_grads)
        accumulator_bytes = sum(value.size * value.dtype.itemsize for value in accumulator_leaves)
        accumulator_dtypes = sorted({str(value.dtype) for value in accumulator_leaves})
        if accumulator_dtypes != ['float32']:
            raise ValueError('Expected explicit FP32 gradient accumulator for this native recipe')
        save(stage='compiling_and_executing_64_real_microsteps', trainable_parameter_count=parameter_count,
             actual_gradient_accumulator_bytes=accumulator_bytes,
             actual_gradient_accumulator_dtypes=accumulator_dtypes,
             initial_trainable_sha256=initial_hash,
             plan=accumulation_plan(1, 64, 50000, parameter_count),
             initial_device_memory_stats=devices[0].memory_stats())
        loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
        iterator = iter(loader)
        step = jax.jit(functools.partial(upstream.train_step, config),
                       in_shardings=(replicated, state_sharding, data_sharding),
                       out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        def optimizer_counts(state):
            counts = {}
            for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state.inner_opt_state)[0]:
                if path and getattr(path[-1], 'name', None) == 'count':
                    counts[jax.tree_util.keystr(path)] = int(leaf)
            if not counts:
                raise ValueError('No inner AdamW/schedule counters found')
            return counts
        if any(optimizer_counts(state).values()):
            raise ValueError('Inner optimizer did not start fresh')
        observations = []
        for index in range(64):
            # Also detects a pause/allocation change before each microstep.
            if resolve_gpu(ROOT) != ALLOWED_GPU_UUID or (ROOT / 'runs/USER_PAUSED.json').exists():
                raise RuntimeError('Allocation changed or user paused')
            started = time.monotonic()
            batch = next(iterator)
            with sharding.set_mesh(mesh):
                state, info = step(rng, state, batch)
            jax.block_until_ready(state)
            info = jax.device_get(info)
            if not all(np.isfinite(np.asarray(value)).all() for value in info.values()):
                raise ValueError('Non-finite real pi0 loss/gradient/parameter norm')
            micro = index + 1
            effective = int(state.opt_state.gradient_step)
            pending = int(state.opt_state.mini_step)
            if effective != micro // 64 or pending != micro % 64 or int(state.step) != micro:
                raise ValueError('Optimizer/microstep counter mismatch')
            counts = optimizer_counts(state)
            if any(count != effective for count in counts.values()):
                raise ValueError('Inner AdamW/schedule counts differ from effective optimizer updates')
            item = {'inner_optimizer_and_lr_counts': counts, 'microstep': micro, 'optimizer_updates': effective,
                    'pending_microsteps': pending, 'seconds': time.monotonic() - started,
                    **{key: float(value) for key, value in info.items()}}
            if micro in (1, 63, 64):
                digest, count = trainable_fingerprint(state.params.filter(config.trainable_filter))
                if count != parameter_count or ((digest == initial_hash) != (micro < 64)):
                    raise ValueError('Parameters changed before boundary or did not change at update64')
                item['trainable_sha256'] = digest
            observations.append(item)
            save(actual_microsteps=micro, actual_optimizer_updates=effective,
                 observations=observations, latest_device_memory_stats=devices[0].memory_stats())
        # No inference checkpoint is produced; the test state dies with process.
        save(stage='completed', real_pi0_accumulation_smoke_passed=True,
             actual_pi0_gpu_memory_fit_verified=True,
             parameters_unchanged_at_checked_microsteps=[1, 63],
             parameters_changed_at_microstep=64,
             optimizer_clipping_scope='inner official clip/AdamW after mean of 64 raw gradients',
             numerical_scope='CPU synthetic equivalence audited separately; no direct real batch64 comparison')
        del iterator, loader, state, step
        gc.collect()
    except BaseException as error:
        save(stage='failed', error=repr(error), real_pi0_accumulation_smoke_passed=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--initial-params', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        inspect_release(args.campaign)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(json.dumps({'ready': False, 'gpu_initialized': False, 'blocker': str(error)}))
        return 3
    if args.check_only:
        print(json.dumps({'ready': True, 'gpu_initialized': False, 'note': 'Release check only, not GPU fit'}))
        return 0
    if args.initial_params is None or args.output is None:
        parser.error('--initial-params and --output required unless --check-only')
    if not args.output.resolve().is_relative_to(ROOT / 'runs') or args.output.exists():
        raise ValueError('New report under remote runs/ required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'runs/pi0-accumulation-preflight.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inspect_release(args.campaign)  # Recheck after acquiring the preflight lock.
    # Set exact device restrictions BEFORE importing anything GPU-capable.
    os.environ.update({
        'CUDA_VISIBLE_DEVICES': ALLOWED_GPU_UUID, 'JAX_PLATFORMS': 'cuda',
        'DOUBLE_ARM_ROOT': str(ROOT), 'OPENPI_ROOT': str(ROOT / 'src/openpi-215abfb'),
        'PI0_INITIAL_PARAMS': str(args.initial_params), 'PI0_ACTION_REPRESENTATION': 'native',
        'PI0_BATCH_SIZE': '1', 'PI0_TRAIN_STEPS': '64', 'PI0_LR_DECAY_STEPS': '30000',
        'PI0_DATASET_REPO': 'momagen/r1_tidy_table_train_v1',
        'PI0_EXP_NAME': 'isolated-accumulation-preflight-no-checkpoint',
        'HF_HOME': str(ROOT / 'cache/huggingface'), 'HF_LEROBOT_HOME': str(ROOT / 'data/pi0/lerobot'),
        'OPENPI_DATA_HOME': str(ROOT / 'cache/openpi'), 'XDG_CACHE_HOME': str(ROOT / 'cache/xdg'),
        'TMPDIR': str(ROOT / 'cache/tmp'), 'WANDB_MODE': 'disabled',
        'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.90',
    })
    sys.path.insert(0, str(ROOT / 'src/openpi-215abfb/src'))
    try:
        run_probe(args.campaign, args.initial_params, args.output)
    except BaseException as error:
        # Include import/initialization failures that precede the full report.
        if not args.output.exists():
            args.output.write_text(json.dumps({
                'mode': 'isolated_real_pi0_accumulation_memory_preflight',
                'stage': 'failed_before_probe_initialization', 'error': repr(error),
                'gpu_uuid': ALLOWED_GPU_UUID, 'actual_pi0_gpu_memory_fit_verified': False,
                'policy_task_success': False, 'checkpoint_saved': False,
            }, indent=2))
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
