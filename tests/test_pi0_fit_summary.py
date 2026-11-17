import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_pi0_single_demo_fit import summarize_reports
from pi0_single_demo_fit import VARIANTS,transition_indices


def fixtures():
    indices=transition_indices(1714,636)
    result=[]
    for i,variant in enumerate(VARIANTS):
        d={'variant':variant,'stage':'completed_single_demo_fit_NOT_task_success','actual_optimizer_updates':2000,
            'trainable_parameter_count':(495908624,81104928)[i],'optimizer_and_lr_counts':{'lr':2000,'adam':2000},
            'initial_full_parameter_sha256':'same','initial_storage_dtype_counts':{'float32':3},'batch_sha256':'same',
            'frame_schedule_sha256':'same','normalization_sha256':'same','input_hashes':{'636':'same'},'lr_schedule':{'warmup':100},
            'seed':42,'indices':indices,'first_close':636,'checkpoint_saved':False,'simulator_control_calls':0,
            'initial_trainable_sha256':'initial','final_trainable_sha256':'final',
            'initial_vision_sha256':'initial','final_vision_sha256':'initial' if i else 'changed',
            'vision_parameters_changed':i==0,'training_log':[{'update':2000,'loss':.001}],'samples':[]}
        for stage in ('initial','final'):
            for frame in indices:
                for seed in (123,124,125):
                    grip={'source_first_raw':-1 if frame>=636 else 1,'predicted_first_raw':-1 if frame>=636 else 1}
                    metric={'left_arm':{'prediction_rmse_rad':.01 if stage=='initial' else .001,'hold_current_q_rmse_rad':.004},'left_gripper':grip}
                    d['samples'].append({'stage':stage,'frame':frame,'seed':seed,'input_sha256':f'input{frame}',
                        'noise_sha256':f'noise{seed}','first1':copy.deepcopy(metric),'first10':copy.deepcopy(metric)})
        result.append(d)
    return result


class FitSummaryTests(unittest.TestCase):
    def test_requires_actual_final_sampling_not_only_optimizer_completion(self):
        a,b=fixtures();a['stage']='single_demo_fit_failed';a['samples']=a['samples'][:27]
        with self.assertRaises(ValueError):summarize_reports(a,b)

    def test_matched_completed_pair_and_subsamples_not_episodes(self):
        a,b=fixtures();summary=summarize_reports(a,b)
        self.assertEqual(summary['matched_sample_pairs'],54)
        self.assertEqual(summary['variants'][VARIANTS[0]]['stages']['final']['first_command_agreement_samples'],27)
        self.assertFalse(summary['policy_task_success'])

    def test_reject_unmatched_recipe_or_changed_frozen_vision(self):
        for key,value in [('normalization_sha256','wrong'),('final_vision_sha256','changed'),('seed',1)]:
            a,b=fixtures();b[key]=value
            with self.assertRaises(ValueError):summarize_reports(a,b)

    def test_reject_unmatched_noise_and_duplicate_samples(self):
        a,b=fixtures();b['samples'][0]['noise_sha256']='wrong'
        with self.assertRaises(ValueError):summarize_reports(a,b)
        a,b=fixtures();b['samples'][-1]=b['samples'][0]
        with self.assertRaises(ValueError):summarize_reports(a,b)


if __name__=='__main__':unittest.main()
