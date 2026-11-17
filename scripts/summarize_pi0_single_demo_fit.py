#!/usr/bin/env python3
"""Read-only CPU summary of completed single-demo TRAIN fitting diagnostics.

Reject incomplete final inference even when2000 optimizer steps succeeded.
This is an overfit/precision test, NOT learned-policy task/physical acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from pi0_single_demo_fit import VARIANTS


def summarize_reports(first,second):
    reports=dict(zip(VARIANTS,(first,second)))
    common=('initial_full_parameter_sha256','initial_storage_dtype_counts','batch_sha256',
            'frame_schedule_sha256','normalization_sha256','input_hashes','lr_schedule','seed','indices','first_close')
    for key in common:
        if first.get(key)!=second.get(key) or first.get(key) is None:
            raise ValueError(f'Unmatched / missing A/B recipe field: {key}')
    expected_counts=(495908624,81104928)
    pairs={}
    for (variant,report),expected in zip(reports.items(),expected_counts):
        if (report.get('variant')!=variant or report.get('stage')!='completed_single_demo_fit_NOT_task_success'
            or report.get('actual_optimizer_updates')!=2000 or report.get('trainable_parameter_count')!=expected
            or not report.get('optimizer_and_lr_counts') or any(v!=2000 for v in report['optimizer_and_lr_counts'].values())
            or len(report.get('samples',[]))!=54 or report.get('checkpoint_saved') is not False
            or report.get('simulator_control_calls')!=0):
            raise ValueError('Both variants need completed gradients AND final sampling; no task/optimizer-only substitute')
        if report.get('initial_trainable_sha256')==report.get('final_trainable_sha256') or not report.get('final_trainable_sha256'):
            raise ValueError('Missing real parameter update evidence')
        samples=report['samples']
        keys={(s['stage'],s['frame'],s['seed']) for s in samples}
        if (len(keys)!=54 or set(s['stage'] for s in samples)!={'initial','final'}
            or set(s['frame'] for s in samples)!=set(report['indices'])
            or set(s['seed'] for s in samples)!={123,124,125} or len(report['indices'])!=9):
            raise ValueError('Expected2 stages x9 frames x3 noises with no duplicates')
        pairs[variant]={(s['stage'],s['frame'],s['seed']):s for s in samples}
    if second.get('initial_vision_sha256')!=second.get('final_vision_sha256') or second.get('vision_parameters_changed') is not False:
        raise ValueError('Frozen vision actually changed / not verified')
    if first.get('initial_vision_sha256')==first.get('final_vision_sha256') or first.get('vision_parameters_changed') is not True:
        raise ValueError('Vision-trainable branch lacks actual vision-update evidence')
    for key,left in pairs[VARIANTS[0]].items():
        right=pairs[VARIANTS[1]][key]
        if any(left.get(field)!=right.get(field) for field in ('input_sha256','noise_sha256')):
            raise ValueError('A/B sampling inputs/noise do not match')
        other=pairs[VARIANTS[0]][('final' if key[0]=='initial' else 'initial',key[1],key[2])]
        if any(left.get(field)!=other.get(field) for field in ('input_sha256','noise_sha256')):
            raise ValueError('Initial/final sampling inputs/noise changed')
    result={'scope':'One TRAIN/demo_0,9 selected transition observations;3 sampling noises, NOT27 episodes',
            'not_task_evaluation':True,'policy_task_success':False,'simulator_control_calls':0,
            'matched_recipe_fields':list(common),'matched_sample_pairs':54,'variants':{}}
    for variant,report in reports.items():
        item={'actual_optimizer_updates':2000,'trainable_parameters':report['trainable_parameter_count'],
              'initial_full_parameter_sha256':report['initial_full_parameter_sha256'],
              'vision_parameters_changed':report['vision_parameters_changed'],
              'last_training_record':report['training_log'][-1],'stages':{}}
        for stage in ('initial','final'):
            samples=[s for s in report['samples'] if s['stage']==stage]
            by_frame=[]
            for frame in report['indices']:
                part=[s for s in samples if s['frame']==frame]
                by_frame.append({'frame':frame,'relative_to_first_close':frame-report['first_close'],
                    'source_first_close':part[0]['first1']['left_gripper']['source_first_raw']<.5,
                    'predicted_first_close_samples':sum(s['first1']['left_gripper']['predicted_first_raw']<.5 for s in part),
                    'first_command_agreement_samples':sum((s['first1']['left_gripper']['predicted_first_raw']<.5)==
                        (s['first1']['left_gripper']['source_first_raw']<.5) for s in part),
                    'samples':3,
                    'left_first_target_rmse_rad_median':statistics.median(s['first1']['left_arm']['prediction_rmse_rad'] for s in part),
                    'left_first10_target_rmse_rad_median':statistics.median(s['first10']['left_arm']['prediction_rmse_rad'] for s in part),
                    'left_hold_first10_rmse_rad':part[0]['first10']['left_arm']['hold_current_q_rmse_rad']})
            item['stages'][stage]={'first_command_agreement_samples':sum(x['first_command_agreement_samples'] for x in by_frame),
                'samples':27,'left_first_target_rmse_rad_median':statistics.median(s['first1']['left_arm']['prediction_rmse_rad'] for s in samples),
                'left_first10_target_rmse_rad_median':statistics.median(s['first10']['left_arm']['prediction_rmse_rad'] for s in samples),
                'left_first10_worse_than_hold_samples':sum(s['first10']['left_arm']['prediction_rmse_rad']>
                    s['first10']['left_arm']['hold_current_q_rmse_rad'] for s in samples),'per_frame':by_frame}
        result['variants'][variant]=item
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/mnt/yuanyue/double-arm/runs')
    if not a.campaign.resolve().is_relative_to(root) or not a.output.resolve().is_relative_to(root) or a.output.exists():
        raise ValueError('Read remote campaign; fresh remote summary only')
    status=json.loads((a.campaign/'status.json').read_text())
    if status.get('stage')!='completed_controlled_single_demo_fit_AB_NOT_task_success':raise ValueError('Parent not complete')
    paths=[a.campaign/v/'report.json' for v in VARIANTS]
    summary=summarize_reports(*(json.loads(p.read_text()) for p in paths))
    summary['report_sha256']={v:hashlib.sha256(p.read_bytes()).hexdigest() for v,p in zip(VARIANTS,paths)}
    import numpy as np
    arrays=[np.load(a.campaign/v/'offline-predictions.npz') for v in VARIANTS]
    initial_keys=[k for k in arrays[0] if k.startswith('initial_')]
    summary['initial_prediction_max_abs_cross_process_difference']=max(float(np.max(np.abs(arrays[0][k]-arrays[1][k]))) for k in initial_keys)
    a.output.write_text(json.dumps(summary,indent=2,allow_nan=False))
    for variant,result in summary['variants'].items():
        print(variant,json.dumps({stage:{k:v for k,v in item.items() if k!='per_frame'} for stage,item in result['stages'].items()}))
        print('transition',json.dumps([f for f in result['stages']['final']['per_frame'] if f['relative_to_first_close'] in (-1,0,15)]))


if __name__=='__main__':main()
