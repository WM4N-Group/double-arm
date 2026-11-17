#!/usr/bin/env python3
"""CPU-only paired joint-encoding/norm diagnostic; no task-success inference."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def compare_reports(absolute,relative):
    if relative.get('profile')!='relative_joints_meanstd_v1' or relative.get('prediction_coordinates')!='native_absolute_motor21_after_inverse_chunk_anchor':
        raise ValueError('Need explicit relative profile with native inverse before scoring')
    for report in (absolute,relative):
        if (report.get('stage')!='completed_single_demo_fit_NOT_task_success' or report.get('actual_optimizer_updates')!=2000
            or report.get('trainable_parameter_count')!=495908624 or len(report.get('samples',[]))!=54
            or not report.get('optimizer_and_lr_counts') or any(c!=2000 for c in report['optimizer_and_lr_counts'].values())):
            raise ValueError('Both matched vision-trainable2000-update fits and final sampling required')
    for key in ('initial_full_parameter_sha256','initial_storage_dtype_counts','frame_schedule_sha256','input_hashes',
                'indices','first_close','lr_schedule','seed'):
        if absolute.get(key)!=relative.get(key) or absolute.get(key) is None:raise ValueError(f'Unmatched {key}')
    if relative.get('reference_native_normalization_sha256')!=absolute['normalization_sha256']:
        raise ValueError('Original state/base/gripper normalization anchor changed')
    pairs=[]
    for report in (absolute,relative):
        records={(s['stage'],s['frame'],s['seed']):s for s in report['samples']}
        if len(records)!=54 or set(s['stage'] for s in report['samples'])!={'initial','final'}:
            raise ValueError('Incomplete or duplicate samples')
        pairs.append(records)
    if set(pairs[0])!=set(pairs[1]):raise ValueError('Sample keys differ')
    for key in pairs[0]:
        if any(pairs[0][key][field]!=pairs[1][key][field] for field in ('input_sha256','noise_sha256')):
            raise ValueError('Unmatched input/noise')
    result={'scope':'One TRAIN demo0,9 source observations,3 sampling noises; same initial params and2000 updates',
        'changed_factor':'joint target encoding + joint action mean/std JOINT recipe, not isolated one-factor normalization test',
        'not_task_evaluation':True,'simulator_control_calls':0,'policy_task_success':False,'conditions':{}}
    for name,report in (('absolute',absolute),('relative',relative)):
        stages={}
        for stage in ('initial','final'):
            samples=[s for s in report['samples'] if s['stage']==stage];frames=[]
            for frame in report['indices']:
                p=[s for s in samples if s['frame']==frame]
                frames.append({'frame':frame,'source_first_raw_gripper':p[0]['first1']['left_gripper']['source_first_raw'],
                    'first_gripper_agreement':sum((s['first1']['left_gripper']['source_first_raw']<.5)==
                        (s['first1']['left_gripper']['predicted_first_raw']<.5) for s in p),
                    'samples':len(p),'left_first_target_rmse_rad_median':statistics.median(s['first1']['left_arm']['prediction_rmse_rad'] for s in p),
                    'left_first10_target_rmse_rad_median':statistics.median(s['first10']['left_arm']['prediction_rmse_rad'] for s in p)})
            stages[stage]={'first_gripper_agreement':sum(f['first_gripper_agreement'] for f in frames),'samples':len(samples),
                'left_first_target_rmse_rad_median':statistics.median(s['first1']['left_arm']['prediction_rmse_rad'] for s in samples),
                'left_first10_target_rmse_rad_median':statistics.median(s['first10']['left_arm']['prediction_rmse_rad'] for s in samples),
                'first10_left_worse_than_hold':sum(s['first10']['left_arm']['prediction_rmse_rad']>
                    s['first10']['left_arm']['hold_current_q_rmse_rad'] for s in samples),'per_frame':frames}
        result['conditions'][name]=stages
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--absolute-campaign',type=Path,required=True)
    p.add_argument('--relative-campaign',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path('/mnt/yuanyue/double-arm/runs')
    if a.output.exists() or any(not x.resolve().is_relative_to(root) for x in (a.absolute_campaign,a.relative_campaign,a.output)):
        raise ValueError('Remote reports/new summary required')
    parent=json.loads((a.relative_campaign/'status.json').read_text())
    if parent.get('stage')!='completed_relative_joint_fit_NOT_task_success':raise ValueError('Relative workflow incomplete')
    paths=[c/'original_vision_trainable/report.json' for c in (a.absolute_campaign,a.relative_campaign)]
    reports=[json.loads(path.read_text()) for path in paths]
    # Canonicalize both saved prediction arrays to actual motor float32 before
    # scoring. The older absolute diagnostic scored float64 unnormalize output;
    # do NOT let that harmless output-cast difference confound this comparison.
    import h5py
    import numpy as np
    from probe_pi0_failure_mechanisms import motion_metrics
    plan=json.loads((a.absolute_campaign/'plan.json').read_text())
    norm_path=Path(plan['norm_anchor'])/'assets/momagen/r1_tidy_table_train_v1/norm_stats.json'
    std=np.asarray(json.loads(norm_path.read_text())['norm_stats']['actions']['std'])+1e-6
    cast_maximum={}
    for label,campaign,report in zip(('absolute','relative'),(a.absolute_campaign,a.relative_campaign),reports):
        arrays=np.load(campaign/'original_vision_trainable/offline-predictions.npz')
        source=report['source'];cast_maximum[label]=0.
        with h5py.File(source['source']) as file:
            group=file['data/'+source['source_demo']];actions=group['actions'][:];states=group['obs/prop_state'][:]
            for row in report['samples']:
                values=arrays[f'{row["stage"]}_{row["frame"]}_{row["seed"]}'];native=values.astype(np.float32)
                cast_maximum[label]=max(cast_maximum[label],float(np.abs(native-values).max()))
                reference=actions[np.minimum(np.arange(row['frame'],row['frame']+50),len(actions)-1)]
                row['first1']=motion_metrics(native,reference,states[row['frame']],std,1)
                row['first10']=motion_metrics(native,reference,states[row['frame']],std,10)
    summary=compare_reports(*reports)
    summary['canonical_native_scoring_dtype']='float32; derived from preserved prediction arrays, original reports NOT modified'
    summary['max_output_float32_cast_difference']=cast_maximum

    summary['report_sha256']={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in zip(('absolute','relative'),paths)}
    a.output.write_text(json.dumps(summary,indent=2,allow_nan=False))
    for name,stage in summary['conditions'].items():
        print(name,json.dumps({key:{k:v for k,v in item.items() if k!='per_frame'} for key,item in stage.items()}))
        print('transition',json.dumps([f for f in stage['final']['per_frame'] if f['frame'] in (635,636,651)]))


if __name__=='__main__':main()
