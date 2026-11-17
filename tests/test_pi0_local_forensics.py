from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_pi0_local_grasp import close_runs


class LocalForensicsTests(unittest.TestCase):
    def test_close_steps_are_contiguous_intervals_not_attempts(self):
        self.assertEqual(close_runs([False,True,True,False,True]),[[1,2],[4,4]])
        self.assertEqual(close_runs([True]*120),[[0,119]])
        self.assertEqual(close_runs([False]*120),[])
        self.assertEqual(close_runs([]),[])


if __name__=='__main__':unittest.main()
