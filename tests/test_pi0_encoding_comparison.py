import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_pi0_joint_encoding_fit import compare_reports
from test_pi0_fit_summary import fixtures


class EncodingComparisonTests(unittest.TestCase):
    def pair(self):
        a=fixtures()[0];b=copy.deepcopy(a);b.update(profile='relative_joints_meanstd_v1',
            prediction_coordinates='native_absolute_motor21_after_inverse_chunk_anchor',
            reference_native_normalization_sha256=a['normalization_sha256'],normalization_sha256='new-relative')
        return a,b

    def test_relative_comparison_deliberately_allows_changed_action_norm_not_changed_inputs(self):
        a,b=self.pair();result=compare_reports(a,b)
        self.assertFalse(result['policy_task_success']);self.assertEqual(result['conditions']['relative']['final']['samples'],27)
        b['samples'][0]['input_sha256']='wrong'
        with self.assertRaises(ValueError):compare_reports(a,b)

    def test_reject_incomplete_or_uninverted_or_wrong_anchor(self):
        for key,value in [('stage','fitting'),('prediction_coordinates','model_delta_not_native'),
                          ('reference_native_normalization_sha256','wrong'),('actual_optimizer_updates',1999)]:
            a,b=self.pair();b[key]=value
            with self.assertRaises(ValueError):compare_reports(a,b)


if __name__=='__main__':unittest.main()
