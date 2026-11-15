#!/usr/bin/env python3
"""SERVER ONLY: real trained pi0 CPU inference contract, not task evaluation.

This can run while the single allocated GPU trains. It loads a completed
checkpoint and real held-out head/wrist images without reading teacher actions.
No simulator, GPU context, gradient updates or fallback policy is allowed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--source-dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('CPU preflight requires CUDA_VISIBLE_DEVICES="" and JAX_PLATFORMS=cpu')
    root = Path(os.environ.get('DOUBLE_ARM_ROOT', '/mnt/yuanyue/double-arm'))
    if not args.checkpoint.resolve().is_relative_to((root/'runs').resolve()):
        raise ValueError('Use a real project fine-tuned checkpoint, not an unfine-tuned base')
    if not (args.checkpoint/'params').is_dir() or 'tmp' in args.checkpoint.name:
        raise ValueError('A completed checkpoint is required')
    if not args.output.resolve().is_relative_to((root/'runs').resolve()):
        raise ValueError('Reports must stay on the server')
    if args.output.exists():
        raise FileExistsError(args.output)

    import h5py
    import jax
    import numpy as np
    from momagen_pi0_schema import camera_keys, rgb_image, PROMPT
    from pi0_r1_config import register_config
    from openpi.policies.policy_config import create_trained_policy

    devices = [str(device) for device in jax.devices()]
    if any(device.platform != 'cpu' for device in jax.devices()):
        raise ValueError('Unexpected non-CPU JAX device')
    with h5py.File(args.source_dataset) as file:
        obs = file['data/demo_0/obs']
        inputs = {key: rgb_image(obs[name][0]) for key, name in camera_keys(obs).items()}
        inputs.update(state=obs['prop_state'][0].copy(), prompt=PROMPT)
    report = {
        'stage': 'loading_real_trained_pi0_on_cpu', 'checkpoint': str(args.checkpoint),
        'source_dataset': str(args.source_dataset), 'not_task_evaluation': True,
        'gradient_updates': 0, 'gpu_initialized': False, 'jax_devices': devices,
        'input_keys': sorted(inputs), 'teacher_action_reads': 0,
        'privileged_state': False, 'task_success': False,
    }
    def save(**updates):
        report.update(updates)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(args.output)
    save()
    try:
        started = time.monotonic()
        policy = create_trained_policy(register_config(), args.checkpoint)
        save(stage='running_real_pi0_cpu_inference', load_seconds=time.monotonic()-started)
        calls = []
        for index in range(2):
            started = time.monotonic()
            prediction = policy.infer(inputs)
            actions = np.asarray(prediction['actions'])
            if actions.shape != (50, 21) or not np.isfinite(actions).all():
                raise ValueError(f'Invalid native output: {actions.shape}')
            calls.append({'call': index, 'seconds': time.monotonic()-started,
                          'action_shape': list(actions.shape), 'finite': True,
                          'actions_sha256': hashlib.sha256(actions.tobytes()).hexdigest()})
            save(inference_calls=calls)
        save(stage='real_pi0_cpu_inference_contract_passed')
        print(json.dumps(report, indent=2), flush=True)
    except BaseException as error:
        save(stage='real_pi0_cpu_inference_contract_failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
