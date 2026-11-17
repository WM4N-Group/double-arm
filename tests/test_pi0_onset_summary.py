from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from summarize_pi0_grasp_onset import summarize, summarize_width


class OnsetSummaryTests(unittest.TestCase):
    def test_reject_incomplete_or_wrong_scope_report(self):
        for report in ({},{'completed':True,'mode':'task_rollout'},
                       {'completed':True,'mode':'TRAIN16_grasp_onset_offline_replication','samples':[],'seeds':[123,124,125]}):
            with self.assertRaises(ValueError):summarize(report)

    def test_full_cartesian_coverage_and_duplicate_rejection(self):
        import copy
        metric={'left_arm':{'prediction_rmse_rad':.03,'hold_current_q_rmse_rad':.001},
                'left_gripper':{'source_first_raw':-1.,'predicted_first_raw':1.,
                    'predicted_effective_close_steps':0,'source_effective_close_steps':10}}
        report={'completed':True,'mode':'TRAIN16_grasp_onset_offline_replication',
                'checkpoint':'fixture89999','seeds':[123,124,125],'samples':[]}
        for episode in range(16):
            for phase in ('close_onset','onset_plus15'):
                for seed in report['seeds']:
                    row={'source_demo':f'demo_{episode}','phase':phase,'seed':seed}
                    for name in ('raw_first1','raw_first10','clipped_first1','clipped_first10'):
                        row[name]=copy.deepcopy(metric)
                    report['samples'].append(row)
        summary=summarize(report)
        for phase in summary['phases'].values():
            self.assertEqual(phase['clipped']['predicted_first_open_when_source_closed'],48)
            self.assertEqual(phase['clipped']['first_gripper_disagreement_all3_noises_episodes'],16)
        report['samples'][-1]=report['samples'][0]
        with self.assertRaises(ValueError):summarize(report)

    def test_width_comparison_requires_matched_noise_and_original_input(self):
        report={'completed':True,'mode':'TRAIN_gripper_width_only_offline_counterfactual',
                'seeds':[123],'samples':[]}
        for demo in range(16):
            for phase in ('close_onset','onset_plus15'):
                for variant in ('original','swap_left_width','right_width_control'):
                    grip={'predicted_first_raw':-1 if variant=='swap_left_width' else 1,
                          'predicted_effective_close_steps':10 if variant=='swap_left_width' else 0}
                    report['samples'].append({'source_demo':f'demo_{demo}','phase':phase,'variant':variant,
                        'seed':123,'original_input_sha256':'same','noise_sha256':'same',
                        'clipped_first1':{'left_gripper':grip},'clipped_first10':{'left_gripper':grip}})
        summary=summarize_width(report)
        self.assertEqual(summary['phases']['close_onset']['swap_left_width']['first_class_flips_vs_original'],16)
        report['samples'][1]['noise_sha256']='wrong'
        with self.assertRaises(ValueError):summarize_width(report)


if __name__=='__main__':unittest.main()
