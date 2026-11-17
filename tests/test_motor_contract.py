from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from motor_contract import binary_close_flags, runtime_binary_threshold, snapshot_motor_contract, choose_close_threshold


class MotorContractTests(unittest.TestCase):
    def test_binary_raw_command_threshold_not_assumed_zero(self):
        c={'gripper_left':{'mode':'binary','inverted':False,'command_input_limits':[[0.],[1.]],'command_output_limits':[[-1.],[1.]]}}
        self.assertEqual(runtime_binary_threshold(c),.5)
        self.assertEqual(binary_close_flags([-1,0,.49,.5,1],.5),[True,True,True,False,False])
        c['gripper_left']['command_input_limits']=[[-1.],[1.]]
        self.assertEqual(runtime_binary_threshold(c),0.)

    def test_unknown_or_unsupported_controller_is_not_faked(self):
        self.assertIsNone(runtime_binary_threshold({}))
        self.assertIsNone(runtime_binary_threshold({'gripper_left':{'mode':'smooth','inverted':False}}))
        with self.assertRaises(ValueError):
            binary_close_flags([float('nan')],.5)

    def test_saved_runtime_metadata_cannot_be_overridden_by_static_assumptions(self):
        self.assertEqual(choose_close_threshold(None,True)[0],.5)
        self.assertIsNone(choose_close_threshold({'gripper_left':{'mode':'smooth'}},True)[0])
        class Controller:
            motor_type='position'
            _mode='binary'
            _inverted=False
            _command_input_limits=([0.],[1.])
            _command_output_limits=([-1.],[1.])
        class Robot:
            controllers={'gripper_left':Controller()}
            _action_normalize=False
        snapshot=snapshot_motor_contract(Robot())
        self.assertEqual(choose_close_threshold(snapshot,False)[0],.5)
        self.assertEqual(len(snapshot['errors']),5)

    def test_readonly_metadata_partial_failures_are_explicit(self):
        class Robot:
            controllers={}
            _action_normalize=False
        r=snapshot_motor_contract(Robot())
        self.assertFalse(r['robot_action_normalize'])
        self.assertIsNone(r['base'])
        self.assertEqual(len(r['errors']),6)


if __name__=='__main__':
    unittest.main()
