#!/usr/bin/env python3
"""SERVER ONLY: offline teacher-referenced action diagnostics via real pi0 RPC.

Ground-truth actions are read ONLY for scoring after inference. They are never
sent to the policy or a simulator. This is not a closed-loop task success test.
"""
import argparse
import json
from pathlib import Path
from urllib.request import urlopen
import h5py
import numpy as np
from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
from pi0_rpc import infer_local
from pi0_base_targets import native_to_absolute_base

GROUPS={'base':slice(0,3),'trunk':slice(3,7),'left_arm':slice(7,13),
        'left_gripper':slice(13,14),'right_arm':slice(14,20),'right_gripper':slice(20,21)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-provenance',type=Path,required=True)
    parser.add_argument('--holdout-provenance',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--token-file',type=Path,required=True)
    parser.add_argument('--port',type=int,default=18731)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--action-representation',choices=['native','absolute_base'],default='native')
    parser.add_argument('--initial-repeats',type=int,default=1)
    args=parser.parse_args()
    if not 1<=args.initial_repeats<=5:raise ValueError('Invalid bounded initial repeats')
    if args.output.exists():raise FileExistsError(args.output)
    if not str(args.output.resolve()).startswith('/mnt/yuanyue/double-arm/runs/'):raise ValueError('Remote report required')
    with urlopen(f'http://127.0.0.1:{args.port}/health',timeout=30) as response:health=json.load(response)
    if health.get('checkpoint')!=str(args.checkpoint) or health.get('model')!='pi0':raise ValueError('Wrong checkpoint service')
    if health.get('action_representation','native')!=args.action_representation:raise ValueError('Wrong action-representation service')
    norm=json.loads((args.checkpoint/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json').read_text())
    stats=norm['norm_stats']['actions']
    std=(np.asarray(stats['q99'],np.float64)-np.asarray(stats['q01'],np.float64))/2 if args.action_representation=='absolute_base' else np.asarray(stats['std'],np.float64)
    if std.shape!=(21,):raise ValueError('Wrong normalization contract')
    token=args.token_file.read_text().strip()
    report={'checkpoint':str(args.checkpoint),'mode':'offline_action_comparison_only',
            'ground_truth_actions_used_for_metrics_only':True,'teacher_actions_sent_to_model':False,
            'simulator_control_calls':0,'task_success_claim':False,'samples':[],
            'action_representation':args.action_representation,
            'inference_platform':health.get('inference_platform'),
            'denoising_steps':health.get('denoising_steps'),'sampling_noise':health.get('sampling_noise'),
            'normalization_scale':'half 1%-99% span' if args.action_representation=='absolute_base' else 'std',
            'base_native_rmse_units':'Mixed XY metres / yaw radians; not a pure metric distance',
            'sample_scope':'First episode of each split only; small sampled diagnostic, not whole-dataset accuracy'}
    def save():args.output.write_text(json.dumps(report,indent=2))
    for split,path in (('train',args.training_provenance),('validation',args.holdout_provenance)):
        provenance=json.loads(path.read_text());entry=provenance['episodes'][0]
        with h5py.File(entry['source']) as file:
            group=file['data'][entry['source_demo']];obs=group['obs'];keys=camera_keys(obs)
            length=len(obs['prop_state'])
            for sample_number,index in enumerate([0]*args.initial_repeats+[length//3,2*length//3]):
                inputs={key:rgb_image(obs[name][index]) for key,name in keys.items()}
                inputs.update(state=obs['prop_state'][index],prompt=PROMPT)
                predicted,info=infer_local(inputs,token,args.port)
                # Teacher reference is obtained after inference and is used for
                # numerical comparisons only; no teacher fallback or env.step.
                indices=np.minimum(np.arange(index,index+50),length-1)
                reference=group['actions'][()][indices]
                if args.action_representation=='absolute_base':
                    reference=native_to_absolute_base(reference,group['obs/base_link_pose'][()][indices])
                difference=predicted-reference;normalized=difference/(std+1e-6)
                metrics={name:{'native_rmse_first_10':float(np.sqrt(np.mean(difference[:10,axes]**2))),
                               'normalized_rmse_first_10':float(np.sqrt(np.mean(normalized[:10,axes]**2))),
                               'normalized_rmse_full_50':float(np.sqrt(np.mean(normalized[:,axes]**2)))}
                         for name,axes in GROUPS.items()}
                sample={'split':split,'source_demo':entry['source_demo'],'frame':index,'sample_number':sample_number,
                        'base_xy_rmse_m_first_10':float(np.sqrt(np.mean(difference[:10,:2]**2))),
                        'base_yaw_wrapped_rmse_rad_first_10':float(np.sqrt(np.mean(((difference[:10,2]+np.pi)%(2*np.pi)-np.pi)**2))),
                        'inference_seconds':info['inference_seconds'],'metrics':metrics,
                        'predicted_first_action':predicted[0].tolist(),'reference_first_action':reference[0].tolist()}
                report['samples'].append(sample);save();print(json.dumps(sample),flush=True)
    report['status']='completed_offline_diagnostics_not_task_evaluation';save()


if __name__=='__main__':main()
