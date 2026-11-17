#!/usr/bin/env python3
"""SERVER ONLY: TRAIN offline precision and input-intervention diagnostic.

No gradients, simulator, action execution, heldout tuning, or checkpoint writes.
Counterfactual image/state/prompt combinations are deliberately inconsistent;
sensitivity is NOT a standalone test of grounding or a task-success metric.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/yuanyue/double-arm')
JOINT_GROUPS = {'trunk': (slice(3, 7), slice(3, 7)),
                'left_arm': (slice(7, 13), slice(7, 13)),
                'right_arm': (slice(14, 20), slice(13, 19))}
VARIANTS = ('original', 'repeat_original', 'other_frame_rgb', 'gray_rgb',
            'gray_head_only', 'gray_wrists_only', 'other_frame_state', 'different_prompt')


def digest_inputs(inputs):
    import numpy as np
    digest = hashlib.sha256()
    for name in sorted(inputs):
        value = inputs[name]
        digest.update(name.encode())
        if isinstance(value, str):
            digest.update(value.encode())
        else:
            array = np.asarray(value)
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def intervene(inputs, other, variant):
    import numpy as np
    if variant not in VARIANTS:
        raise ValueError('Unknown intervention')
    output = {name: value.copy() if hasattr(value, 'copy') else value for name, value in inputs.items()}
    if variant == 'other_frame_rgb':
        for name in ('head_image', 'left_wrist_image', 'right_wrist_image'):
            output[name] = other[name].copy()
    elif variant.startswith('gray_'):
        names = ('head_image',) if variant == 'gray_head_only' else (
            ('left_wrist_image', 'right_wrist_image') if variant == 'gray_wrists_only' else
            ('head_image', 'left_wrist_image', 'right_wrist_image'))
        for name in names:
            output[name] = np.full_like(output[name], 127)
    elif variant == 'other_frame_state':
        output['state'] = other['state'].copy()
    elif variant == 'different_prompt':
        output['prompt'] = 'Leave the cup on the countertop. Do not pick up the cup.'
    return output


def motion_metrics(predicted, reference, state, std, steps=10):
    """Score joint target errors against constant-q hold, without gripper invention."""
    import numpy as np
    predicted, reference, state, std = map(np.asarray, (predicted, reference, state, std))
    if predicted.shape != (50, 21) or reference.shape != (50, 21) or state.shape != (21,) or std.shape != (21,):
        raise ValueError('Expected native21/state21/horizon50 contract')
    if not all(np.isfinite(x).all() for x in (predicted, reference, state, std)) or np.any(std <= 0) or not 1 <= steps <= 50:
        raise ValueError('Nonfinite values, bad scale, or invalid prefix')
    report = {}
    for name, (action_axes, state_axes) in JOINT_GROUPS.items():
        expected, actual, current = reference[:steps, action_axes], predicted[:steps, action_axes], state[state_axes]
        residual = expected - current
        error = actual - expected
        hold_mse = float(np.mean((residual / std[action_axes])**2))
        prediction_mse = float(np.mean((error / std[action_axes])**2))
        report[name] = {'prediction_rmse_rad': float(np.sqrt(np.mean(error**2))),
            'hold_current_q_rmse_rad': float(np.sqrt(np.mean(residual**2))),
            'prediction_normalized_mse': prediction_mse, 'hold_normalized_mse': hold_mse,
            'error_to_hold_rmse_ratio': float(np.sqrt(prediction_mse / hold_mse)) if hold_mse > 1e-16 else None,
            'predicted_motion_rmse_rad': float(np.sqrt(np.mean((actual-current)**2))),
            'first_target_error_rad': (actual[0]-expected[0]).tolist(),
            'first_expert_target_minus_current_q_rad': residual[0].tolist()}
    report['base_xy'] = {'prediction_rmse_m': float(np.sqrt(np.mean((predicted[:steps,:2]-reference[:steps,:2])**2))),
                         'zero_command_rmse_m': float(np.sqrt(np.mean(reference[:steps,:2]**2)))}
    report['left_gripper'] = {'predicted_first_raw': float(predicted[0,13]), 'source_first_raw': float(reference[0,13]),
        'predicted_effective_close_steps': int(np.sum(predicted[:steps,13] < .5)),
        'source_effective_close_steps': int(np.sum(reference[:steps,13] < .5)),
        'effective_command_agreement_fraction': float(np.mean((predicted[:steps,13] < .5) == (reference[:steps,13] < .5))),
        'threshold_evidence': 'pinned-source inferred binary input[0,1]/output[-1,1]; NOT runtime instance metadata'}
    return report


def sensitivity(predicted, nominal, std):
    import numpy as np
    difference = np.asarray(predicted)[:10] - np.asarray(nominal)[:10]
    return {name: {'raw_rmse_rad_or_gripper_units': float(np.sqrt(np.mean(difference[:, axes]**2))),
                   'normalized_rmse': float(np.sqrt(np.mean((difference[:,axes]/std[axes])**2)))}
            for name, axes in {'left_arm': slice(7,13), 'right_arm': slice(14,20), 'trunk': slice(3,7),
                                'left_gripper': slice(13,14)}.items()} | {
        'base_xy_rmse_m': float(np.sqrt(np.mean(difference[:,:2]**2))),
        'base_yaw_rmse_rad': float(np.sqrt(np.mean(difference[:,2]**2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import inspect_release, process_alive, pinned_upstream_revision
    gpu = resolve_gpu(ROOT, '1')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or os.environ.get('JAX_PLATFORMS') != 'cuda':
        raise ValueError('Explicit authorized GPU1 UUID / cuda required; no fallback')
    if not args.output.resolve().is_relative_to(ROOT/'runs') or args.output.exists() or args.output.is_symlink():
        raise ValueError('Fresh remote evidence directory required')
    if not args.checkpoint.resolve().is_relative_to(ROOT/'runs') or args.checkpoint.name != '89999':
        raise ValueError('Use actual completed coverage final89999')
    if not json.loads((args.checkpoint/'_CHECKPOINT_METADATA').read_text()).get('commit_timestamp_nsecs'):
        raise ValueError('Checkpoint is uncommitted')
    lock = (ROOT/'runs'/f'gpu-{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inspect_release(Path((ROOT/'runs/latest-pi0-coverage-campaign.txt').read_text().strip()))
    old = json.loads((ROOT/'runs/latest-gpu1-job.json').read_text())
    old_status = json.loads(Path(old['status_file']).read_text())
    followup = Path((ROOT/'runs/latest-paper-gpu1-followup.txt').read_text().strip())
    follow_status = json.loads((followup/'status.json').read_text())
    if (old_status.get('stage') != 'raw_official_pool_target_complete_alignment_and_split_pending'
        or old_status.get('verified_unique_demos') != 25 or old_status.get('active_pid') != 0
        or follow_status.get('stage') != 'bounded_data_followup_complete'
        or any(process_alive(p) for p in (old.get('controller_pid'), follow_status.get('controller_pid')))):
        raise ValueError('Existing GPU1 data workflow has not fully released')
    revision = pinned_upstream_revision(ROOT/'src/openpi-215abfb')
    args.output.mkdir()
    status = {'stage': 'loading', 'controller_pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
              'gpu_uuid': gpu, 'policy_training': 'not_run', 'simulator_control_calls': 0,
              'learned_policy_task_success': False, 'checkpoint': str(args.checkpoint)}
    def write_json(path, value):
        temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, allow_nan=False)); temp.replace(path)
    def save_status(**updates):
        status.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'status.json', status)
    save_status()
    write_json(ROOT/'runs/latest-gpu1-job.json', {'controller_pid': os.getpid(), 'campaign': str(args.output),
        'status_file': str(args.output/'status.json'), 'gpu_uuid': gpu, 'max_gpu_count': 1,
        'role': 'TRAIN_only_offline_input_intervention_NOT_training_NOT_simulator_control'})
    mapping = json.loads((ROOT/'runs/latest-multi-gpu-jobs.json').read_text())
    mapping['completed_gpu1_second_auxiliary_pool'] = old
    mapping['GPU1'] = {'job_pointer': str(ROOT/'runs/latest-gpu1-job.json'), 'campaign': str(args.output),
                      'gpu_uuid': gpu, 'role': 'TRAIN_only_offline_input_intervention_NOT_training_NOT_simulator_control'}
    mapping['updated_at'] = datetime.now(timezone.utc).isoformat()
    write_json(ROOT/'runs/latest-multi-gpu-jobs.json', mapping)
    (ROOT/'runs/latest-pi0-failure-mechanism-probe.txt').write_text(str(args.output)+'\n')
    try:
        import h5py
        import numpy as np
        import jax
        from openpi.policies.policy_config import create_trained_policy
        from pi0_r1_config import register_config
        from momagen_pi0_schema import camera_keys, rgb_image, PROMPT
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
            raise ValueError('Expected sole visible GPU1 CUDA backend')
        contract = json.loads((args.checkpoint/'assets/r1-adapter-contract.json').read_text())
        norm_path = args.checkpoint/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json'
        if (contract.get('action_representation') != 'native' or contract.get('quantile_normalization') is not False
            or hashlib.sha256(norm_path.read_bytes()).hexdigest() != contract['norm_stats_sha256']):
            raise ValueError('Native mean/std contract mismatch')
        sources = {name: hashlib.sha256((ROOT/'code/scripts'/name).read_bytes()).hexdigest() == sha
                   for name, sha in contract['training_source_sha256'].items()}
        if not all(sources.values()):
            raise ValueError('Actual training adapter source changed')
        std = np.asarray(json.loads(norm_path.read_text())['norm_stats']['actions']['std']) + 1e-6
        provenance_path = ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'
        provenance = json.loads(provenance_path.read_text())
        if provenance.get('split') != 'train' or len(provenance['episodes']) != 16:
            raise ValueError('Only original16 TRAIN episodes allowed')
        config = register_config('native')
        policy = create_trained_policy(config, args.checkpoint, sample_kwargs={'num_steps':10})
        report = {'mode': 'TRAIN_only_offline_precision_and_input_interventions', 'checkpoint': str(args.checkpoint),
            'simulator_control_calls': 0, 'gradient_updates': 0, 'checkpoint_saved': False, 'heldout_used': False,
            'counterfactuals_not_valid_live_states': True, 'not_task_evaluation': True,
            'counterfactual_caveat': 'Sensitivity is not semantic understanding. Gray RGB / changed prompt may be out of distribution.',
            'hold_baseline_caveat': 'Only arm/trunk absolute joint coordinates; no invented gripper-width to command conversion.',
            'checkpoint_norm_sha256': hashlib.sha256(norm_path.read_bytes()).hexdigest(),
            'provenance_sha256': hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'training_adapter_source_matches': sources, 'openpi_revision': revision,
            'sampling': {'denoising_steps':10, 'noise':'fixed Gaussian seed123 for all interventions'},
            'source_motion': [], 'samples': []}
        save_status(stage='source_precision_audit')
        for entry in provenance['episodes']:
            with h5py.File(entry['source']) as file:
                g=file['data/'+entry['source_demo']]; actions=g['actions'][:]; state=g['obs/prop_state'][:]
                summary = {'source_demo': entry['source_demo'], 'frames':len(actions), 'joint_target_minus_current_q':{}}
                for name,(aa,sa) in JOINT_GROUPS.items():
                    delta=actions[:,aa]-state[:,sa]
                    rms=np.sqrt(np.mean(delta**2,axis=1))
                    summary['joint_target_minus_current_q'][name] = {'rms_rad_p50_p95_p99':np.quantile(rms,[.5,.95,.99]).tolist(),
                        'fraction_frames_rms_over_0_01rad':float(np.mean(rms>.01))}
                report['source_motion'].append(summary)
        write_json(args.output/'report.json', report)
        entry=provenance['episodes'][0]
        predictions={}
        with h5py.File(entry['source']) as file:
            g=file['data/'+entry['source_demo']]; obs=g['obs']; actions=g['actions'][:]; keys=camera_keys(obs)
            onset=np.flatnonzero(actions[:,13]<.5)
            if not len(onset): raise ValueError('Missing source close onset')
            close=int(onset[0]); indices=list(dict.fromkeys((0,len(actions)//3,max(0,close-30),close)))
            def load_inputs(index):
                inputs={name:rgb_image(obs[key][index]) for name,key in keys.items()}
                inputs.update(state=obs['prop_state'][index],prompt=PROMPT)
                return inputs
            noise=np.random.default_rng(123).standard_normal((50,32)).astype(np.float32)
            for frame in indices:
                inputs=load_inputs(frame); other_frame=close if frame==0 else 0; other=load_inputs(other_frame)
                reference=actions[np.minimum(np.arange(frame,frame+50),len(actions)-1)]
                nominal=None
                for variant in VARIANTS:
                    save_status(stage='offline_intervention_inference', frame=frame, variant=variant, completed_samples=len(report['samples']))
                    changed=intervene(inputs,other,variant)
                    start=time.monotonic(); predicted=np.asarray(policy.infer(changed,noise=noise)['actions'])
                    if predicted.shape!=(50,21) or not np.isfinite(predicted).all(): raise ValueError('Invalid native prediction')
                    if nominal is None: nominal=predicted.copy()
                    row={'source_demo':entry['source_demo'],'frame':frame,'variant':variant,'other_frame':other_frame,
                        'input_sha256':digest_inputs(changed), 'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                        'seconds':time.monotonic()-start,'metrics_first1':motion_metrics(predicted,reference,inputs['state'],std,1),
                        'metrics_first10':motion_metrics(predicted,reference,inputs['state'],std,10),
                        'difference_from_nominal':sensitivity(predicted,nominal,std)}
                    if variant=='repeat_original': row['repeat_max_abs_difference']=float(np.max(np.abs(predicted-nominal)))
                    report['samples'].append(row)
                    predictions[f'frame{frame}_{variant}']=predicted
                    write_json(args.output/'report.json',report)
                    print(json.dumps({'frame':frame,'variant':variant,'seconds':row['seconds'],
                          'left':row['metrics_first10']['left_arm'], 'gripper':row['metrics_first10']['left_gripper'],
                          'sensitivity':row['difference_from_nominal']['left_arm']}),flush=True)
        np.savez_compressed(args.output/'offline-predictions.npz',**predictions)
        report['completed']=True; write_json(args.output/'report.json',report)
        save_status(stage='completed_offline_diagnostic_NOT_task_success',completed_samples=len(report['samples']),
                    report_sha256=hashlib.sha256((args.output/'report.json').read_bytes()).hexdigest())
    except BaseException as error:
        save_status(stage='offline_probe_failed',error=repr(error))
        raise


if __name__ == '__main__':
    main()
