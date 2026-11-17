#!/usr/bin/env python3
"""SERVER GPU1 offline source/live-start and state-feedback probe; NOT control.

Only after the entire relative-full workflow terminates. Deliberately inconsistent
state/RGB interventions are diagnostic inputs, NEVER simulator/policy deployment.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT = Path('/mnt/yuanyue/double-arm')


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def feedback_gain(nominal, altered, delta):
    """Native-target response; 1 follows changed anchor, 0 stays fixed. Not reward."""
    import numpy as np
    response = np.mean(np.asarray(altered)[:10, 7:13] - np.asarray(nominal)[:10, 7:13], axis=0)
    delta = np.asarray(delta)
    if delta.shape != (6,) or not np.isfinite(delta).all() or np.sum(delta**2) <= 0:
        raise ValueError('Finite nonzero six-joint perturbation required')
    return {'native_target_response_gain': float(response @ delta / (delta @ delta)),
            'mean_first10_target_change_rad': response.tolist(),
            'scope': 'same RGB/noise; altered proprio is inconsistent OFFLINE counterfactual, not a unique valid control target'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import process_alive
    gpu = resolve_gpu(ROOT, '1')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu or os.environ.get('JAX_PLATFORMS') != 'cuda':
        raise ValueError('Only explicit authorized GPU1/cuda; no device fallback')
    if (ROOT/'runs/USER_PAUSED.json').exists():
        raise ValueError('User paused')
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Fresh remote run required')
    lock = (ROOT/'runs'/f'gpu-{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    old = json.loads((ROOT/'runs/latest-gpu1-job.json').read_text())
    dependency = json.loads((a.campaign/'status.json').read_text())
    if (old.get('campaign') != str(a.campaign) or old.get('status_file') != str(a.campaign/'status.json')
        or old.get('controller_pid') != dependency.get('controller_pid')
        or old.get('gpu_uuid') != gpu or dependency.get('stage') != 'train-author-diagnostic_FAILED'
        or dependency.get('active_pid') != 0 or process_alive(old.get('controller_pid'))):
        raise ValueError('Complete failed full-demo workflow must actually release before probe')
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'], text=True)
    if any(row.split(',')[0].strip() == gpu for row in apps.splitlines()):
        raise ValueError('GPU1 occupied; do not preempt/fall back')
    a.output.mkdir()
    source = Path(__file__).read_bytes()
    (a.output/Path(__file__).name).write_bytes(source)
    write(a.output/'dependency-status.json', dependency)
    report = {'stage':'loading', 'controller_pid':os.getpid(), 'gpu_uuid':gpu,
              'campaign':str(a.campaign), 'source_sha256':hashlib.sha256(source).hexdigest(),
              'started_at':datetime.now(timezone.utc).isoformat(), 'policy_training':'not_run',
              'simulator_control_calls':0, 'task_success':False, 'samples':[], 'interventions':[]}
    def save(stage):
        report.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
        write(a.output/'report.json', report)
        write(a.output/'status.json', {k:v for k,v in report.items() if k not in ('samples','interventions','trace_summary')})
    save('loading')
    pointer = {'controller_pid':os.getpid(),'campaign':str(a.output),'status_file':str(a.output/'status.json'),
               'gpu_uuid':gpu,'max_gpu_count':1,'role':'relative_profile_OFFLINE_feedback_probe_NOT_training_or_control'}
    write(ROOT/'runs/latest-gpu1-job.json', pointer)
    mapping = json.loads((ROOT/'runs/latest-multi-gpu-jobs.json').read_text())
    mapping['completed_relative_full_workflow'] = old
    mapping['GPU1'] = pointer
    mapping['updated_at'] = datetime.now(timezone.utc).isoformat()
    write(ROOT/'runs/latest-multi-gpu-jobs.json', mapping)
    (ROOT/'runs/latest-pi0-relative-feedback-probe.txt').write_text(str(a.output)+'\n')
    try:
        import h5py
        import jax
        import numpy as np
        from momagen_pi0_schema import PROMPT, camera_keys, rgb_image
        from pi0_relative_full_profile import make_policy
        from probe_pi0_failure_mechanisms import motion_metrics, digest_inputs
        if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
            raise ValueError('Only one visible GPU required')
        trained = json.loads((a.campaign/'training-report.json').read_text())
        e = a.campaign/'train-author-diagnostic'
        with np.load(e/'physical-rollout/demo_0-trace.npz') as trace:
            applied = trace['actions']; proposed = trace['model_proposed_targets']
            difference = np.abs(applied-proposed)
            report['trace_summary'] = {
                'clip_counts_gt1e_6_per_coordinate':(difference>1e-6).sum(0).tolist(),
                'first_clip_step_per_coordinate':[int(np.flatnonzero(difference[:,i]>1e-6)[0]) if (difference[:,i]>1e-6).any() else None for i in range(21)],
                'max_clipping_per_coordinate':difference.max(0).tolist(),
                'native_base_any_bound_clipped_gt1e_6_steps':int(np.any(difference[:,:3]>1e-6,axis=1).sum()),
                'counter_caveat':'old result clipped_native_base_steps counts only additional absolute_base conversion; native overall clipping is separate'}
        policy, contract = make_policy(Path(trained['checkpoint']))
        report.update(checkpoint=trained['checkpoint'], checkpoint_full_parameter_sha256=contract['full_parameter_sha256'],
                      norm_sha256=contract['norm_stats_sha256'])
        # Same original native scale for offline scoring only; never model inputs.
        native_norm = json.loads((Path(json.loads((a.campaign/'plan.json').read_text())['norm_anchor'])/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json').read_text())
        std = np.asarray(native_norm['norm_stats']['actions']['std'])+1e-6
        predictions = {}
        with h5py.File(trained['source']['source']) as file:
            group = file['data/'+trained['source']['source_demo']]
            obs = group['obs']; keys = camera_keys(obs); actions = group['actions'][:]
            frames = [0,30,60,120,240,360,480,571,606,621,631,635,636,637,641,651,666,857,1142,1400,1713]
            source_inputs = {}
            save('source_condition_sampling')
            for frame in frames:
                inputs = {name:rgb_image(obs[key][frame]) for name,key in keys.items()}
                inputs.update(state=obs['prop_state'][frame],prompt=PROMPT)
                source_inputs[frame] = inputs
                reference = actions[np.minimum(np.arange(frame,frame+50),len(actions)-1)]
                for seed in (123,124,125):
                    noise = np.random.default_rng(seed).standard_normal((50,32)).astype(np.float32)
                    pred = np.asarray(policy.infer(inputs,noise=noise)['actions'])
                    if pred.shape != (50,21) or not np.isfinite(pred).all():
                        raise ValueError('Invalid predicted chunk')
                    name = f'source_{frame}_{seed}'; predictions[name] = pred
                    report['samples'].append({'kind':'source','frame':frame,'noise_seed':seed,'input_sha256':digest_inputs(inputs),
                        'first10':motion_metrics(pred,reference,inputs['state'],std,10)})
                save('source_condition_sampling')
            with np.load(e/'physical-rollout/initial-observation.npz') as initial:
                live = {name:initial[name].copy() for name in keys}
                live.update(state=initial['state'].copy(),prompt=PROMPT)
            for seed in (123,124,125):
                noise = np.random.default_rng(seed).standard_normal((50,32)).astype(np.float32)
                pred = np.asarray(policy.infer(live,noise=noise)['actions'])
                predictions[f'live_start_{seed}'] = pred
                base = predictions[f'source_0_{seed}']
                report['samples'].append({'kind':'actual_initial_observation_OFFLINE','noise_seed':seed,
                    'input_sha256':digest_inputs(live), 'first10':motion_metrics(pred,actions[:50],live['state'],std,10),
                    'source_to_live_left_first10_rms_rad':float(np.sqrt(np.mean((pred[:10,7:13]-base[:10,7:13])**2))),
                    'source_to_live_trunk_first10_rms_rad':float(np.sqrt(np.mean((pred[:10,3:7]-base[:10,3:7])**2)))})
            save('inconsistent_state_counterfactuals_OFFLINE_ONLY')
            for frame in (0,636):
                base = predictions[f'source_{frame}_123']
                for sign in (-1,1):
                    delta = np.full(6,sign*.02,np.float32)
                    altered = dict(source_inputs[frame]); altered['state'] = altered['state'].copy()
                    altered['state'][7:13] += delta
                    noise = np.random.default_rng(123).standard_normal((50,32)).astype(np.float32)
                    pred = np.asarray(policy.infer(altered,noise=noise)['actions'])
                    predictions[f'perturb_{frame}_{sign}'] = pred
                    report['interventions'].append({'frame':frame,'left_q_delta_rad':delta.tolist(),
                        'noise_seed':123,**feedback_gain(base,pred,delta)})
        np.savez_compressed(a.output/'predictions.npz',**predictions)
        report['prediction_samples'] = len(predictions)
        report['evidence_scope'] = 'TRAIN source accuracy and offline sensitivity only; source and corrupted state never simulator inputs; no causal proof or task success from these metrics'
        save('completed_relative_feedback_OFFLINE_NOT_task_success')
    except BaseException as error:
        report['error'] = repr(error)
        (a.output/'exception.txt').write_text(traceback.format_exc())
        save('relative_feedback_probe_failed')
        raise


if __name__ == '__main__':
    main()
