#!/usr/bin/env python3
"""SERVER CPU ONLY: actual LeRobot/training/inference contract audit.

No model weights, training, simulator, dataset writes or hyperparameter tuning.
Samples every TRAIN episode at start, grasp-command onset, middle and boundaries.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/mnt/yuanyue/double-arm')


def probe_indices(length, first_close):
    if length < 30:
        raise ValueError('Full episode required')
    indices = {0, length//2, length-2, length-1}
    if first_close is not None:
        if not 0 <= first_close < length:
            raise ValueError('Close onset outside episode')
        indices.update((max(0, first_close-1), first_close))
    return sorted(indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise ValueError('Explicit CPU-only audit required')
    if not args.output.resolve().is_relative_to(ROOT/'runs') or args.output.exists():
        raise ValueError('New remote audit directory required')
    import copy
    import h5py
    import numpy as np
    import jax
    import jax.numpy as jnp
    from openpi import transforms
    from openpi.models import model
    from openpi.training import checkpoints, data_loader
    from pi0_r1_config import register_config
    from momagen_pi0_schema import ACTION_NAMES, STATE_NAMES, PROMPT, camera_keys, rgb_image, native_actions, unpad_pi0_actions
    if any(d.platform != 'cpu' for d in jax.devices()):
        raise ValueError('Non-CPU backend')
    config = register_config('native')
    data = config.data.create(config.assets_dirs, config.model)
    contract = json.loads((args.checkpoint/'assets/r1-adapter-contract.json').read_text())
    if contract['action_representation'] != 'native' or contract['quantile_normalization'] is not False:
        raise ValueError('This audits native mean/std recipe, not a changed adapter')
    norm_file = args.checkpoint/'assets'/data.asset_id/'norm_stats.json'
    if hashlib.sha256(norm_file.read_bytes()).hexdigest() != contract['norm_stats_sha256']:
        raise ValueError('Saved normalization changed')
    scripts = Path(__file__).resolve().parent
    source_hash_matches = {name: hashlib.sha256((scripts/name).read_bytes()).hexdigest() == digest
                           for name, digest in contract['training_source_sha256'].items()}
    if not all(source_hash_matches.values()):
        raise ValueError('Current adapter source differs from actual training source; audit historical source first')
    provenance_path = ROOT/'data/pi0/lerobot'/data.repo_id/'momagen-provenance.json'
    provenance = json.loads(provenance_path.read_text())
    if provenance['split'] != 'train' or len(provenance['episodes']) != 16:
        raise ValueError('Expected original16 TRAIN diagnostic data, not new raw pool')
    checkpoint_norm = checkpoints.load_norm_stats(args.checkpoint/'assets', data.asset_id)
    for key, stats in checkpoint_norm.items():
        for field in ('mean', 'std', 'q01', 'q99'):
            a, b = getattr(stats, field), getattr(data.norm_stats[key], field)
            if a is None or b is None:
                if a is not b:
                    raise ValueError('Normalization availability mismatch')
            else:
                np.testing.assert_array_equal(a, b)
    raw_dataset = data_loader.create_torch_dataset(data, config.model.action_horizon, config.model)
    training_dataset = data_loader.transform_dataset(raw_dataset, data)
    inference_transform = transforms.compose([
        transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
        transforms.Normalize(checkpoint_norm, use_quantiles=False), *data.model_transforms.inputs])
    report = {'mode': 'CPU_sampled_real_training_loader_vs_source_and_inference_audit',
              'created_at': datetime.now(timezone.utc).isoformat(),
              'checkpoint': str(args.checkpoint), 'gpu_initialized': False,
              'model_weights_loaded': False, 'dataset_modified': False,
              'model_or_simulator_control_calls': 0, 'heldout_used_for_tuning': False,
              'scope': 'Original16 train episodes only, sampled windows; not complete pixel audit or policy acceptance',
              'adapter_source_hashes_match_actual_training': source_hash_matches,
              'checkpoint_and_training_norm_arrays_identical': True,
              'trainable_filter_source': 'official model.get_freeze_filter; parameter contract audited separately',
              'native_actions': list(ACTION_NAMES), 'original_proprio': list(STATE_NAMES),
              'model_action_dim': config.model.action_dim, 'action_horizon': config.model.action_horizon,
              'known_recipe_differences_from_paper': ['16 vs1000 demos', 'batch1 vs64', 'mean/std vsquantile',
                  'weights warmstart/fresh optimizer +90k vs declared paper schedule',
                  'author vision freeze/action execution/joint-cleaning subset not verified'],
              'augmentation_caveat': 'Compare before model-internal intentional train=True image augmentation; not asserting augmented RGB identical to inference',
              'samples': [], 'episodes': []}
    args.output.mkdir()
    def save():
        tmp = args.output/'report.tmp'
        tmp.write_text(json.dumps(report, indent=2))
        tmp.replace(args.output/'report.json')
    save()
    offset = 0
    sources = []
    for episode_index, entry in enumerate(provenance['episodes']):
        with h5py.File(entry['source'], 'r') as file:
            group = file['data'][entry['source_demo']]
            actions = native_actions(group['actions'][()])
            states = group['obs/prop_state'][()]
            keys = camera_keys(group['obs'])
            length = len(actions)
            negative = np.flatnonzero(actions[:, 13] < 0)
            close = int(negative[0]) if len(negative) else None
            fingerprint = hashlib.sha256(actions.tobytes()+states[0].tobytes()).hexdigest()
            if fingerprint != entry['fingerprint'] or length != entry['frames']:
                raise ValueError('Original train source/provenance changed')
            sources.append(states)
            for index in probe_indices(length, close):
                raw = raw_dataset[offset+index]
                if int(np.asarray(raw['episode_index']).item()) != episode_index:
                    raise ValueError('LeRobot episode ordering differs from provenance')
                if raw['prompt'] != PROMPT:
                    raise ValueError('Training prompt mismatch')
                reference = actions[np.minimum(np.arange(index, index+50), length-1)]
                np.testing.assert_array_equal(np.asarray(raw['state']), states[index])
                np.testing.assert_array_equal(native_actions(raw['actions']), reference)
                source_inputs = {name: rgb_image(group['obs'][source][index]) for name, source in keys.items()}
                for key, image in source_inputs.items():
                    np.testing.assert_array_equal(rgb_image(raw[key]), image)
                source_inputs.update(state=states[index].copy(), prompt=PROMPT)
                trained = training_dataset[offset+index]
                inferred = inference_transform(copy.deepcopy(source_inputs))
                np.testing.assert_array_equal(trained['state'], inferred['state'])
                np.testing.assert_array_equal(trained['tokenized_prompt'], inferred['tokenized_prompt'])
                np.testing.assert_array_equal(trained['tokenized_prompt_mask'], inferred['tokenized_prompt_mask'])
                for key in trained['image']:
                    np.testing.assert_array_equal(trained['image'][key], inferred['image'][key])
                    np.testing.assert_array_equal(trained['image_mask'][key], inferred['image_mask'][key])
                if trained['actions'].shape != (50, 32) or trained['state'].shape != (32,):
                    raise ValueError('Unexpected model padding shape')
                np.testing.assert_array_equal(trained['actions'][:, 21:], 0)
                np.testing.assert_array_equal(trained['state'][21:], 0)
                restored = transforms.Unnormalize(checkpoint_norm, use_quantiles=False)(
                    {'state': trained['state'].copy(), 'actions': trained['actions'].copy()})
                roundtrip = unpad_pi0_actions(restored['actions'])
                np.testing.assert_allclose(roundtrip, reference, atol=1e-6, rtol=1e-6)
                # Match ACTUAL loader/policy batching and JAX dtype conversion,
                # not an invalid unbatched bool-scalar Observation fixture.
                train_batch = jax.tree.map(jnp.asarray, data_loader._collate_fn([copy.deepcopy(trained)]))
                infer_batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], copy.deepcopy(inferred))
                obs_train = model.Observation.from_dict(train_batch)
                obs_infer = model.Observation.from_dict(infer_batch)
                for key in obs_train.images:
                    np.testing.assert_array_equal(obs_train.images[key], obs_infer.images[key])
                allowed = {'state', 'image', 'image_mask', 'tokenized_prompt', 'tokenized_prompt_mask', 'actions'}
                if set(trained)-allowed:
                    raise ValueError('Unexpected model input fields / possible index or privileged input leakage')
                report['samples'].append({'episode_index': episode_index, 'source_demo': entry['source_demo'],
                    'local_frame': index, 'global_frame': offset+index,
                    'source_RGB_exact': True, '50_step_labels_exact_including_episode_clamp': True,
                    'train_vs_inference_model_inputs_exact_before_augmentation': True,
                    'normalization_roundtrip_max_abs': float(np.abs(roundtrip-reference).max())})
            report['episodes'].append({'source': entry['source'], 'source_demo': entry['source_demo'],
                'frames': length, 'first_left_close_frame': close,
                'negative_left_command_frames': int((actions[:,13]<0).sum()),
                'left_close_onsets': int(np.sum((actions[:,13]<0)&~np.r_[False, actions[:-1,13]<0])),
                'source_fingerprint': fingerprint})
        offset += length
        save()
    if offset != len(raw_dataset):
        raise ValueError('Dataset length/provenance mismatch')
    source_states = np.concatenate(sources)
    norm_mean, norm_std = checkpoint_norm['state'].mean, checkpoint_norm['state'].std
    source_z = (source_states-norm_mean)/(norm_std+1e-6)
    trace_file = Path((ROOT/'runs/latest-pi0-coverage-campaign.txt').read_text().strip())/'coverage-training-author-diagnostic/physical-rollout/demo_0-trace.npz'
    with np.load(trace_file, allow_pickle=False) as trace:
        live = trace['observations']
        live_z = (live-norm_mean)/(norm_std+1e-6)
    report.update(status='completed_sampled_contract_audit', sampled_windows=len(report['samples']),
        train_frames=offset, distinct_training_instructions=1,
        constant_instruction_caveat='One task/one prompt does not demonstrate language grounding; not a proof the language backbone is ignored',
        max_live_abs_proprio_z=float(np.abs(live_z).max()),
        live_proprio_elements_abs_z_gt10=int((np.abs(live_z)>10).sum()),
        normalized_state_caveat='Marginal scalar bounds do NOT rule out image/pose-combination distribution shift',
        state_normalization=[{'name': name, 'stored_std': float(norm_std[i]),
            'actual_source_std_float64': float(np.std(source_states[:,i].astype(np.float64))),
            'source_max_abs_z': float(np.abs(source_z[:,i]).max()),
            'live_max_abs_z': float(np.abs(live_z[:,i]).max())} for i,name in enumerate(STATE_NAMES)])
    save()
    print(json.dumps({k:report[k] for k in ('status','sampled_windows','train_frames','distinct_training_instructions',
         'max_live_abs_proprio_z','live_proprio_elements_abs_z_gt10')}, indent=2))


if __name__ == '__main__':
    main()
