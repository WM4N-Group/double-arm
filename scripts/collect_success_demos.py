#!/usr/bin/env python3
"""SERVER ONLY: take a read-only, immutable snapshot of completed successful demos.

Only upstream success tmp/ or finalized demo.hdf5 files are considered. Failed
folders are never read. Native tidy_table aliases are unambiguous (one instance
per category); dishes_away is NEVER repaired by guessing which plate an alias is.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import h5py
import numpy as np
from state_policy_features import feature_keys

TIDY_ALIASES = {'task::teacup.n.02_1':'object::teacup', 'task::sink.n.01_1':'object::sink',
                'task::countertop.n.01_1':'object::countertop', 'task::floor.n.01_1':'object::floor'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',action='append',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--min-demos',type=int,default=4)
    p.add_argument('--stable-seconds',type=float,default=10)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    partial=a.output.with_suffix('.hdf5.partial')
    if partial.exists(): raise FileExistsError(partial)
    provenance=[]; fingerprints=set(); expected_meta=None; expected_keys=None
    skipped=[]
    with h5py.File(partial,'w') as out:
        data=out.create_group('data')
        for run in a.run_dir:
            for task_dir in sorted((run/'generated').glob('*')):
                final=task_dir/'demo.hdf5'
                paths=[final] if final.is_file() else sorted((task_dir/'tmp').glob('*.hdf5'))
                for path in paths:
                    if time.time()-path.stat().st_mtime<a.stable_seconds: continue
                    try:
                        with h5py.File(path,'r') as f:
                            meta=json.loads(f['data'].attrs['env_args'])
                            if expected_meta and meta['env_name']!=expected_meta['env_name']:
                                raise ValueError('Cannot mix different tasks in one policy dataset')
                            for name in sorted((n for n in f['data'] if n.startswith('demo_')),key=lambda n:int(n.split('_')[-1])):
                                g=f['data'][name]
                                if bool(g.attrs.get('partial',False)): continue
                                if not all(k in g for k in ('actions','states','obs')): continue
                                actions=g['actions'][()]; states=g['states'][()]
                                if len(actions)<2 or len(actions)!=len(states) or actions.shape[1]!=21: continue
                                digest=hashlib.sha256(actions.tobytes()+states[0].tobytes()).hexdigest()
                                if digest in fingerprints: continue
                                obs={k:g['obs'][k] for k in g['obs'] if len(g['obs'][k].shape)==2 and g['obs'][k].shape[1]<=2048}
                                alias_used=False
                                if not any(k.startswith('task::') for k in obs):
                                    if meta['env_name']!='r1_tidy_table_D0':
                                        skipped.append({'path':str(path),'reason':'No unique task instances; native aliases unsafe for this task'})
                                        continue
                                    if not all(v in obs for v in TIDY_ALIASES.values()): continue
                                    obs.update({k:obs[v] for k,v in TIDY_ALIASES.items()});alias_used=True
                                keys=feature_keys(obs)
                                if expected_keys and keys!=expected_keys: raise ValueError('Observation feature schema mismatch')
                                if not all(k in obs for k in keys+['prop_state']): continue
                                if not np.isfinite(actions).all(): continue
                                episode=data.create_group(f'demo_{len(provenance)}')
                                episode.create_dataset('actions',data=actions,compression='gzip')
                                episode.create_dataset('states',data=states,compression='gzip')
                                target=episode.create_group('obs')
                                for key,value in obs.items(): target.create_dataset(key,data=value[()],compression='gzip')
                                for key,value in g.attrs.items(): episode.attrs[key]=value
                                episode.attrs['source_file']=str(path);episode.attrs['source_demo']=name
                                fingerprints.add(digest);expected_keys=keys;expected_meta=meta
                                provenance.append({'demo':episode.name.split('/')[-1],'source_file':str(path),'source_demo':name,
                                   'fingerprint':digest,'steps':len(actions),'native_single_instance_aliases_expanded':alias_used})
                    except OSError as error:
                        skipped.append({'path':str(path),'reason':str(error)})
        if expected_meta: data.attrs['env_args']=json.dumps(expected_meta)
        data.attrs['total']=sum(x['steps'] for x in provenance)
    report={'num_success_demos':len(provenance),'required':a.min_demos,'provenance':provenance,'skipped':skipped,
            'source_policy':'Only finalized success datasets or closed success tmp files; no tmp_failed/ trajectories',
            'created_at':time.time(),'dataset':str(a.output)}
    if len(provenance)<a.min_demos:
        partial.unlink();print(json.dumps(report),flush=True);raise SystemExit(3)
    os.replace(partial,a.output)
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
