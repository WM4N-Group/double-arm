#!/usr/bin/env python3
"""CPU-only summary of completed TRAIN offline onset probes, NOT task acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(report, treatments=('raw','clipped')):
    if not report.get('completed') or report.get('mode')!='TRAIN16_grasp_onset_offline_replication':
        raise ValueError('Completed explicitly scoped TRAIN report required')
    if not treatments or any(t not in ('raw','clipped') for t in treatments):
        raise ValueError('Unknown score treatment')
    samples=report['samples']
    keys={(s['source_demo'],s['phase'],s['seed']) for s in samples}
    episodes={s['source_demo'] for s in samples}
    if len(episodes)!=16 or len(samples)!=96 or len(keys)!=96 or set(report['seeds'])!={123,124,125}:
        raise ValueError('Expected16 episodes x2 phases x3 distinct noises, no duplicate samples')
    if any(s['phase'] not in ('close_onset','onset_plus15') or s['seed'] not in report['seeds'] for s in samples):
        raise ValueError('Unexpected phase/seed')
    summary={'scope':'Original16 TRAIN,32 observations x3 Gaussian noises; NOT96 independent episodes or task rollouts',
             'checkpoint':report['checkpoint'],'learned_policy_task_success':False,'phases':{}}
    for phase in ('close_onset','onset_plus15'):
        selected=[s for s in samples if s['phase']==phase]
        detail={}
        for treatment in treatments:
            first=[s[treatment+'_first1'] for s in selected]
            ten=[s[treatment+'_first10'] for s in selected]
            valid=[m for m in first if m['left_gripper']['source_first_raw']<.5]
            erroneous=[m for m in valid if m['left_gripper']['predicted_first_raw']>=.5]
            arm=[m['left_arm'] for m in first]
            per_episode=[]
            for demo in sorted(episodes):
                subset=[s for s in selected if s['source_demo']==demo]
                wrong=sum((s[treatment+'_first1']['left_gripper']['predicted_first_raw']<.5) !=
                          (s[treatment+'_first1']['left_gripper']['source_first_raw']<.5) for s in subset)
                per_episode.append({'demo':demo,'first_command_disagreement_seeds':wrong,'seeds':3})
            detail[treatment]={'samples':len(selected),'source_first_close_samples':len(valid),
                'predicted_first_open_when_source_closed':len(erroneous),
                'first_gripper_disagreement_all3_noises_episodes':sum(d['first_command_disagreement_seeds']==3 for d in per_episode),
                'first_gripper_disagreement_at_least2_noises_episodes':sum(d['first_command_disagreement_seeds']>=2 for d in per_episode),
                'first10_predicted_close_commands':sum(m['left_gripper']['predicted_effective_close_steps'] for m in ten),
                'first10_source_close_commands':sum(m['left_gripper']['source_effective_close_steps'] for m in ten),
                'first10_total_commands':len(ten)*10,
                'left_first_target_error_rms_rad_median':statistics.median(m['prediction_rmse_rad'] for m in arm),
                'left_source_first_target_minus_current_q_rms_rad_median':statistics.median(m['hold_current_q_rmse_rad'] for m in arm),
                'left_first_target_raw_rmse_worse_than_hold_samples':sum(m['prediction_rmse_rad']>m['hold_current_q_rmse_rad'] for m in arm),
                'left_first10_target_raw_rmse_worse_than_hold_samples':sum(m['left_arm']['prediction_rmse_rad']>m['left_arm']['hold_current_q_rmse_rad'] for m in ten),
                'per_episode':per_episode}
        summary['phases'][phase]=detail
    return summary


def summarize_width(report):
    if not report.get('completed') or report.get('mode')!='TRAIN_gripper_width_only_offline_counterfactual':
        raise ValueError('Completed explicitly scoped width-counterfactual report required')
    samples=report['samples']; variants=('original','swap_left_width','right_width_control')
    episodes={s['source_demo'] for s in samples}
    keys={(s['source_demo'],s['phase'],s['variant'],s['seed']) for s in samples}
    if len(samples)!=96 or len(keys)!=96 or len(episodes)!=16 or report['seeds']!=[123]:
        raise ValueError('Expected16 TRAIN episodes x2 source frames x3 variants, seed123')
    if any(s['phase'] not in ('close_onset','onset_plus15') or s['variant'] not in variants or s['seed']!=123 for s in samples):
        raise ValueError('Unexpected variant or phase')
    result={'scope':'16 TRAIN episodes x2 source frames x3 width variants; fixed seed123, offline counterfactual NOT task rollout',
            'not_task_evaluation':True,'phases':{}}
    for phase in ('close_onset','onset_plus15'):
        details={}
        for demo in episodes:
            group=[s for s in samples if s['source_demo']==demo and s['phase']==phase]
            if len({s['original_input_sha256'] for s in group})!=1 or len({s['noise_sha256'] for s in group})!=1:
                raise ValueError('Original inputs/noise differ across a comparison triplet')
        for variant in variants:
            rows=[s for s in samples if s['phase']==phase and s['variant']==variant]
            grasps=[s['clipped_first10']['left_gripper'] for s in rows]
            flips=0
            for row in rows:
                original=next(s for s in samples if s['source_demo']==row['source_demo'] and s['phase']==phase and s['variant']=='original')
                flips+=(original['clipped_first1']['left_gripper']['predicted_first_raw']<.5)!=(row['clipped_first1']['left_gripper']['predicted_first_raw']<.5)
            details[variant]={'samples':len(rows),'first_close':sum(g['predicted_first_raw']<.5 for g in grasps),
                'first10_close':sum(g['predicted_effective_close_steps'] for g in grasps),
                'first_class_flips_vs_original':flips}
        result['phases'][phase]=details
    return result


def summarize_precision(report):
    if not report.get('completed') or report.get('mode')!='TRAIN16_same_process_parameter_restore_precision':
        raise ValueError('Completed explicitly scoped restore-precision report required')
    conditions=('official_bfloat16_restore','preserve_float32_params')
    samples=report['samples']
    if len(samples)!=192 or set(s['condition'] for s in samples)!=set(conditions):
        raise ValueError('Expected96 matched predictions per restore condition')
    result={'scope':'Same-process parameter-restore comparison, model configured compute dtype unchanged; NOT full-FP32 compute or task evaluation',
            'model_compute_dtype':report['model_compute_dtype'],'conditions':{},'matched_input_noise_pairs':0}
    per_condition={}
    for condition in conditions:
        selected=[s for s in samples if s['condition']==condition]
        adapted={'completed':True,'mode':'TRAIN16_grasp_onset_offline_replication','checkpoint':report['checkpoint'],
                 'samples':selected,'seeds':report['seeds']}
        result['conditions'][condition]=summarize(adapted,treatments=('clipped',))
        per_condition[condition]={(s['source_demo'],s['phase'],s['seed']):s for s in selected}
    for key,left in per_condition[conditions[0]].items():
        right=per_condition[conditions[1]][key]
        if left['input_sha256']!=right['input_sha256'] or left['noise_sha256']!=right['noise_sha256'] or left['frame']!=right['frame']:
            raise ValueError('Restore comparison has unmatched observations/noise')
        result['matched_input_noise_pairs']+=1
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--width-intervention',action='store_true');p.add_argument('--restore-precision',action='store_true');a=p.parse_args()
    if a.width_intervention and a.restore_precision:raise ValueError('One explicit summary mode required')
    if a.output.exists():raise FileExistsError(a.output)
    result=(summarize_precision if a.restore_precision else summarize_width if a.width_intervention else summarize)(json.loads(a.report.read_text()));result['report_sha256']=hashlib.sha256(a.report.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    if a.restore_precision:
        for condition,summary in result['conditions'].items():
            for phase,detail in summary['phases'].items():
                print(condition,phase,json.dumps({k:v for k,v in detail['clipped'].items() if k!='per_episode'}))
        return
    for phase,detail in result['phases'].items():
        print(phase,json.dumps(detail if a.width_intervention else {k:v for k,v in detail['clipped'].items() if k!='per_episode'},ensure_ascii=False))


if __name__=='__main__':main()
