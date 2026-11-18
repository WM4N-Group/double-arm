"""Mock-only tests of the v2 rack allowlist and single actuator contract."""
from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase, main, skipUnless
from unittest.mock import Mock, patch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
HAS_X7_RUNTIME = all(find_spec(name) is not None for name in ('numpy', 'scipy', 'PIL', 'pybullet'))
if HAS_X7_RUNTIME:
    import x7_sync_gripper_preflight as x7


@skipUnless(HAS_X7_RUNTIME, 'X7 CPU simulator dependencies not available')
class X7RackContractTests(TestCase):
    def setUp(self):
        self.world = x7.SynchronizedWorld.__new__(x7.SynchronizedWorld)
        self.world.robot = 4
        self.world.jaw = [7, 8]
        self.world.rack = 19
        self.engine = Mock(JOINT_GEAR=6, POSITION_CONTROL=2)
        self.engine.getNumConstraints.return_value = 1
        self.engine.getConstraintUniqueId.return_value = 19
        self.info = (4, 7, 4, 8, 6, (1., 0., 0.), (0.,)*3, (0.,)*3,
                     (0., 0., 0., 1.), (0., 0., 0., 1.), 60., -1., 0, 0., .55)
        self.engine.getConstraintInfo.return_value = self.info
        change = patch.object(x7, 'p', self.engine)
        change.start()
        self.addCleanup(change.stop)

    def test_accepts_exact_internal_gear(self):
        self.world.verify_rack()

    def test_rejects_extra_constraint_even_with_valid_rack(self):
        self.engine.getNumConstraints.return_value = 2
        with self.assertRaisesRegex(RuntimeError, 'Unexpected'):
            self.world.verify_rack()

    def test_rejects_substituted_rack_id(self):
        self.engine.getConstraintUniqueId.return_value = 27
        with self.assertRaisesRegex(RuntimeError, 'substituted'):
            self.world.verify_rack()

    def test_rejects_attachment_where_child_is_an_object(self):
        self.engine.getConstraintInfo.return_value = self.info[:2] + (15, -1) + self.info[4:]
        with self.assertRaisesRegex(RuntimeError, 'exact two jaw slides'):
            self.world.verify_rack()

    def test_rejects_inverted_gear_ratio(self):
        self.engine.getConstraintInfo.return_value = self.info[:11] + (1.,) + self.info[12:]
        with self.assertRaisesRegex(RuntimeError, 'rack parameters'):
            self.world.verify_rack()

    def test_target_commands_only_the_master(self):
        self.world.drive_jaws(.012)
        self.engine.setJointMotorControl2.assert_called_once_with(
            4, 7, 2, targetPosition=.012, force=x7.MASTER_FORCE_N, positionGain=.22, velocityGain=1.)
        self.engine.setJointMotorControlArray.assert_not_called()


if __name__ == '__main__':
    main()
