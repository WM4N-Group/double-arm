"""Run numerical HDF5 fixtures only on the server; never launch a simulator."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('h5py') and importlib.util.find_spec('numpy'), 'HDF5 tests run in server environment')
class SuccessfulSnapshot(unittest.TestCase):
    def test_native_aliases_and_partial_failure_exclusion(self):
        import h5py
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'worker';tmp=run/'generated/task/tmp';tmp.mkdir(parents=True)
            for index in range(5):
                with h5py.File(tmp/f'{index}.hdf5','w') as file:
                    data=file.create_group('data');data.attrs['env_args']=json.dumps({'env_name':'r1_tidy_table_D0'})
                    demo=data.create_group('demo_0');demo.attrs['partial']=index==4
                    demo.create_dataset('actions',data=np.full((12,21),index,dtype=np.float32))
                    demo.create_dataset('states',data=np.full((12,30),index,dtype=np.float32))
                    obs=demo.create_group('obs')
                    for key,size in [('prop_eef_basepose',38),('prop_state',21),('grasp_left',1),('grasp_right',1),
                                     ('object::teacup',7),('object::sink',7),('object::countertop',7),('object::floor',7)]:
                        obs.create_dataset(key,data=np.full((12,size),index,dtype=np.float32))
            output=root/'snapshot.hdf5'
            result=subprocess.run([sys.executable,str(ROOT/'scripts/collect_success_demos.py'),'--run-dir',str(run),
                     '--output',str(output),'--min-demos','4','--stable-seconds','0'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            with h5py.File(output) as file:
                self.assertEqual(len(file['data']),4)
                self.assertIn('task::teacup.n.02_1',file['data/demo_0/obs'])
                self.assertNotIn('demo_4',file['data'])
            report=json.loads(output.with_suffix('.json').read_text())
            self.assertTrue(report['provenance'][0]['native_single_instance_aliases_expanded'])


if __name__=='__main__':unittest.main()
