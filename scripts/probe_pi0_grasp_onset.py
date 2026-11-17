#!/usr/bin/env python3
"""SERVER ONLY: replicate TRAIN grasp-onset precision over16 episodes/3 noises.

Offline real source observations, not reset-to-late-state task evaluation.
No source actions enter the model; they select audit frames / score only.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from probe_pi0_failure_mechanisms import ROOT, digest_inputs, motion_metrics


def onset_frames(commands):
    """Pinned binary-close convention. Cannot turn missing onset into frame0."""
    import numpy as np
    commands=np.asarray(commands)
    if commands.ndim!=1 or not len(commands) or not np.isfinite(commands).all():
        raise ValueError('Finite nonempty gripper commands required')
    closed=np.flatnonzero(commands < .5)
    if not len(closed) or closed[0]==0:
        raise ValueError('An observable open-to-close transition is required')
    onset=int(closed[0])
    return onset, min(onset+15,len(commands)-1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prerequisite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width-intervention',action='store_true',help='Offline ONLY: swap one measured gripper width, leave all RGB/arm states fixed')
    args=p.parse_args()
    role='TRAIN_gripper_width_offline_counterfactual_NOT_live_control' if args.width_intervention else 'TRAIN_grasp_onset_offline_replication_NOT_training'
    from resource_allocation import resolve_gpu
    from preflight_pi0_accumulation_gpu import inspect_release, process_alive, pinned_upstream_revision
    gpu=resolve_gpu(ROOT,'1')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or os.environ.get('JAX_PLATFORMS')!='cuda':
        raise ValueError('Only explicit GPU1/cuda; no fallback')
    if not args.output.resolve().is_relative_to(ROOT/'runs') or args.output.exists() or args.output.is_symlink():
        raise ValueError('Fresh remote report directory required')
    if not args.prerequisite.resolve().is_relative_to(ROOT/'runs'):
        raise ValueError('Remote completed prerequisite required')
    lock=(ROOT/'runs'/f'gpu-{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    expected_stage='completed_TRAIN_grasp_onset_probe_NOT_task_success' if args.width_intervention else 'completed_offline_diagnostic_NOT_task_success'
    old=json.loads((ROOT/'runs/latest-gpu1-job.json').read_text())
    before=json.loads((args.prerequisite/'status.json').read_text())
    if (old.get('campaign')!=str(args.prerequisite) or before.get('stage')!=expected_stage
        or process_alive(old.get('controller_pid')) or process_alive(before.get('controller_pid'))
        or hashlib.sha256((args.prerequisite/'report.json').read_bytes()).hexdigest()!=before.get('report_sha256')):
        raise ValueError('GPU1 owner/prerequisite changed, active, or unverified')
    inspect_release(Path((ROOT/'runs/latest-pi0-coverage-campaign.txt').read_text().strip()))
    pinned_upstream_revision(ROOT/'src/openpi-215abfb')
    checkpoint=Path(before['checkpoint'])
    args.output.mkdir()
    def write(path,value):
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)
    status={'stage':'loading','controller_pid':os.getpid(),'checkpoint':str(checkpoint),'gpu_uuid':gpu,
            'started_at':datetime.now(timezone.utc).isoformat(),'policy_training':'not_run',
            'simulator_control_calls':0,'learned_policy_task_success':False}
    def save(**kwargs):
        status.update(kwargs,updated_at=datetime.now(timezone.utc).isoformat());write(args.output/'status.json',status)
    save()
    write(ROOT/'runs/latest-gpu1-job.json',{'campaign':str(args.output),'status_file':str(args.output/'status.json'),
        'controller_pid':os.getpid(),'gpu_uuid':gpu,'max_gpu_count':1,'role':role})
    mapping=json.loads((ROOT/'runs/latest-multi-gpu-jobs.json').read_text())
    mapping['completed_pi0_grasp_onset_replication' if args.width_intervention else 'completed_pi0_input_intervention']=old
    mapping['GPU1']={'campaign':str(args.output),'job_pointer':str(ROOT/'runs/latest-gpu1-job.json'),
                    'gpu_uuid':gpu,'role':role}
    mapping['updated_at']=datetime.now(timezone.utc).isoformat();write(ROOT/'runs/latest-multi-gpu-jobs.json',mapping)
    (ROOT/'runs'/('latest-pi0-grasp-width-probe.txt' if args.width_intervention else 'latest-pi0-grasp-onset-probe.txt')).write_text(str(args.output)+'\n')
    try:
        import h5py
        import numpy as np
        import jax
        from openpi.policies.policy_config import create_trained_policy
        from pi0_r1_config import register_config
        from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
        if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('Wrong backend')
        contract=json.loads((checkpoint/'assets/r1-adapter-contract.json').read_text())
        norm_path=checkpoint/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json'
        if contract.get('action_representation')!='native' or contract.get('quantile_normalization') is not False or hashlib.sha256(norm_path.read_bytes()).hexdigest()!=contract['norm_stats_sha256']:
            raise ValueError('Wrong saved norm/representation')
        matches={name:hashlib.sha256((ROOT/'code/scripts'/name).read_bytes()).hexdigest()==sha for name,sha in contract['training_source_sha256'].items()}
        if not all(matches.values()):raise ValueError('Actual training adapter source changed')
        provenance_path=ROOT/'data/pi0/lerobot/momagen/r1_tidy_table_train_v1/momagen-provenance.json'
        provenance=json.loads(provenance_path.read_text())
        if provenance.get('split')!='train' or len(provenance['episodes'])!=16:raise ValueError('Original16 TRAIN only')
        std=np.asarray(json.loads(norm_path.read_text())['norm_stats']['actions']['std'])+1e-6
        bounds_path=ROOT/'runs/pi0-native-coverage-v3-20261116-172526/coverage-training-author-diagnostic/action-bounds.json'
        bounds=json.loads(bounds_path.read_text());lo=np.asarray(bounds['min']);hi=np.asarray(bounds['max'])
        if lo.shape!=(21,) or hi.shape!=(21,) or np.any(lo>hi):raise ValueError('Bad native bounds')
        report={'mode':'TRAIN_gripper_width_only_offline_counterfactual' if args.width_intervention else 'TRAIN16_grasp_onset_offline_replication','checkpoint':str(checkpoint),'heldout_used':False,
            'simulator_control_calls':0,'gradient_updates':0,'checkpoint_saved':False,'not_task_evaluation':True,
            'frame_selection':'first raw<.5 (pinned binary inference) and15frames later; source labels ONLY for frame selection/scoring',
            'seeds':[123] if args.width_intervention else [123,124,125],'denoising_steps':10,'provenance_sha256':hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
            'norm_sha256':hashlib.sha256(norm_path.read_bytes()).hexdigest(),'bounds_sha256':hashlib.sha256(bounds_path.read_bytes()).hexdigest(),
            'counterfactual_caveat':'Width-swap inputs are physically inconsistent. Offline sensitivity ONLY, never fed into a live controller or claimed as a grasp success.',
            'width_intervention':args.width_intervention,
            'probe_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'training_source_matches':matches,'samples':[]}
        policy=create_trained_policy(register_config('native'),checkpoint,sample_kwargs={'num_steps':10})
        noises={seed:np.random.default_rng(seed).standard_normal((50,32)).astype(np.float32) for seed in report['seeds']}
        predictions={}
        for entry in provenance['episodes']:
            with h5py.File(entry['source']) as file:
                g=file['data/'+entry['source_demo']];obs=g['obs'];actions=g['actions'][:];keys=camera_keys(obs)
                frames=onset_frames(actions[:,13])
                for phase,index in zip(('close_onset','onset_plus15'),frames):
                    inputs={key:rgb_image(obs[name][index]) for key,name in keys.items()}
                    inputs.update(state=obs['prop_state'][index],prompt=PROMPT)
                    reference=actions[np.minimum(np.arange(index,index+50),len(actions)-1)]
                    for seed,noise in noises.items():
                        variants=('original','swap_left_width','right_width_control') if args.width_intervention else ('original',)
                        for variant in variants:
                            changed={k:v.copy() if hasattr(v,'copy') else v for k,v in inputs.items()}
                            if variant!='original':
                                other_index=frames[1] if phase=='close_onset' else frames[0]
                                other_left_width=float(obs['prop_state'][other_index][19])
                                changed['state'][19 if variant=='swap_left_width' else 20]=other_left_width
                            save(stage='TRAIN_only_grasp_onset_inference',source_demo=entry['source_demo'],frame=index,seed=seed,completed_samples=len(report['samples']))
                            start=time.monotonic();predicted=np.asarray(policy.infer(changed,noise=noise)['actions'])
                            clipped=np.clip(predicted,lo,hi)
                            row={'source_demo':entry['source_demo'],'frame':index,'phase':phase,'seed':seed,'variant':variant,
                                'original_input_sha256':digest_inputs(inputs),
                                'original_gripper_widths':inputs['state'][19:21].tolist(),
                                'intervened_gripper_widths':changed['state'][19:21].tolist(),
                                'input_sha256':digest_inputs(changed),'noise_sha256':hashlib.sha256(noise.tobytes()).hexdigest(),
                                'seconds':time.monotonic()-start,'raw_first1':motion_metrics(predicted,reference,inputs['state'],std,1),
                                'raw_first10':motion_metrics(predicted,reference,inputs['state'],std,10),
                                'clipped_first1':motion_metrics(clipped,reference,inputs['state'],std,1),
                                'clipped_first10':motion_metrics(clipped,reference,inputs['state'],std,10)}
                            report['samples'].append(row)
                            predictions[f'{entry["source_demo"]}_{index}_{seed}_{variant}']=predicted
                            write(args.output/'report.json',report)
                            print(json.dumps({'demo':entry['source_demo'],'phase':phase,'seed':seed,'variant':variant,
                                'gripper':row['clipped_first10']['left_gripper'],
                                'left':row['clipped_first10']['left_arm']}),flush=True)
        np.savez_compressed(args.output/'offline-predictions.npz',**predictions)
        report['completed']=True;write(args.output/'report.json',report)
        save(stage='completed_TRAIN_gripper_width_counterfactual_NOT_task_success' if args.width_intervention else 'completed_TRAIN_grasp_onset_probe_NOT_task_success',completed_samples=len(report['samples']),
            report_sha256=hashlib.sha256((args.output/'report.json').read_bytes()).hexdigest())
    except BaseException as error:
        save(stage='grasp_onset_probe_failed',error=repr(error));raise


if __name__=='__main__':main()
