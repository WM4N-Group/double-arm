from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from physical_grasp_observation import physical_audit_cli_flags


class PhysicalAuditRoutingTests(unittest.TestCase):
    def test_default_legacy_paths_unchanged_and_physical_explicit_opt_in(self):
        self.assertEqual(physical_audit_cli_flags(False,'configured'),[])
        self.assertEqual(physical_audit_cli_flags(False,'physical'),[])
        self.assertEqual(physical_audit_cli_flags(True,'physical'),['--audit-no-assistance'])
    def test_configured_is_not_silently_physical_and_nonboolean_rejected(self):
        with self.assertRaises(ValueError):physical_audit_cli_flags(True,'configured')
        with self.assertRaises(ValueError):physical_audit_cli_flags('yes','physical')


if __name__=='__main__':unittest.main()
