from collections import Counter
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pi0_uniform_sampling_diagnostic import uniform_schedule,VARIANT
from pi0_relative_full_profile import full_schedule


class UniformSamplerTests(unittest.TestCase):
    def test_uniform20k_covers_every_frame11_or12_times_without_critical_oversampling(self):
        order=uniform_schedule(1714,636);counts=Counter(order)
        self.assertEqual(len(order),20000);self.assertEqual(set(counts),set(range(1714)))
        self.assertEqual(set(counts.values()),{11,12})
        self.assertEqual(order,uniform_schedule(1714,636));self.assertEqual(order,uniform_schedule(1714,0))
        self.assertNotEqual(order,full_schedule(1714,636))
        self.assertEqual(VARIANT,'uniform_full_trajectory_same_base_v1')
    def test_no_unbounded_partial_recipe(self):
        for length,updates in ((0,20000),(1714,1000),(1714,100001)):
            with self.assertRaises(ValueError):uniform_schedule(length,636,updates)


if __name__=='__main__':unittest.main()
