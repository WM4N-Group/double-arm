#!/usr/bin/env python3
"""Independently audit physical validation and the no-update control."""
import argparse
import json
from pathlib import Path

import numpy as np

from collect_x7_transport_bc import layouts, sha_layout
from train_x7_turning_trajectory_bc import digest


def audit_case(run,row,layout,h):
    mode,split,name=row['mode'],layout['split'],layout['case']
    base=run/'validation'/split/mode/name
    if row['checkpoint_sha256']!=h or row['layout_sha256']!=sha_layout(layout) or row['case']!=name:
        raise ValueError(f'Wrong checkpoint/layout: {name}/{mode}')
    if not row['physical_execution']:
        if row['passed']:raise ValueError('Nonphysical row marked passed')
        return dict(case=name,mode=mode,passed=False,physical_execution=False)
    f=base/'trace.npz'
    if digest(f)!=row['trace_sha256']:raise ValueError('Raw validation trace changed')
    if row['review_sha256'] and digest(base/'review.mp4')!=row['review_sha256']:raise ValueError('Review changed')
    with np.load(f,allow_pickle=False) as r:
        a,old,p,keep=(r[k] for k in ('action','nonlearned_anchor_action','phase','live_transport_mask'))
        if (not np.array_equal(a[~keep],old[~keep]) or not np.array_equal(a[:,6:],old[:,6:]) or not np.all(p[keep]==4)):
            raise ValueError('Modified unlearned action block')
        idx=r['live_rgb_action_index']; images=r['live_rgb'];calls=row['actual_inference_calls']
        if mode=='nominal':
            if calls or len(idx) or len(images):raise ValueError('Nominal arm was not an unlearned comparator')
        elif (calls!=len(idx) or images.dtype!=np.uint8 or images.shape!=(calls,128,192,3)
                or not np.array_equal(np.flatnonzero(keep),(idx[:,None]+np.arange(6)).reshape(-1))):
            raise ValueError('Live pre-observation/command alignment broken')
        s,init=r['station_clearance'],r['station_initialization_clearance']
        near=float(min(np.min(s[...,:2]),np.min(init[...,:2])))
        clear=near>=.005 and not np.any(s[...,2:4]) and not np.any(init[...,2:4])
        if near!=row['minimum_robot_station_m'] or row['passed'] and not clear:
            raise ValueError('Disagreement with independent unrounded collision gate')
    return dict(case=name,mode=mode,passed=row['passed'],physical_execution=True,minimum_station_m=near,
                calls=calls,trajectory_sha256=row['trace_sha256'],review_sha256=row['review_sha256'])


def audit(run):
    plan=json.loads((run/'dataset/layouts.json').read_text()); fit=json.loads((run/'training/training-inputs.json').read_text())
    train=json.loads((run/'training/status.json').read_text()); h=digest(run/'training/best.pt')
    if (plan['layouts']!=layouts() or train['stage']!='train_completed' or h!=train['checkpoint_sha256'] or
            digest(run/'dataset/layouts.json')!=fit['layouts_sha256'] or fit['accepted_episodes']!={'train':72,'dev':12} or
            any((run/'dataset'/'shards'/r['case']).exists() for r in layouts() if r['split']=='heldout')):
        raise ValueError('Unexpected training/data state or heldout teacher traces')
    dev=json.loads((run/'validation/dev-gate.json').read_text());split_records={}
    for split in ('dev','heldout'):
        if split=='heldout' and not dev['all_12_candidate_physical_pass']:
            if (run/'validation/heldout-report.json').exists():raise ValueError('Heldout read before authorization')
            break
        rows=[]
        for row in layouts():
            if row['split']!=split:continue
            for mode in ('candidate','nominal'):
                f=run/'validation'/split/mode/row['case']/'report.json'
                rows.append(audit_case(run,json.loads(f.read_text()),row,h))
        split_records[split]=rows
    return dict(checkpoint_sha256=h,optimizer_updates=train['optimizer_updates'],records=split_records,
        counts={split:{mode:dict(total=sum(r['mode']==mode for r in rows),
                                 passed=sum(r['mode']==mode and r['passed'] for r in rows))
                       for mode in ('candidate','nominal')} for split,rows in split_records.items()},
        heldout_teacher_traces_generated=False,hardware_or_dynamic_safety_claimed=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);args=ap.parse_args()
    result=audit(args.run)
    with (args.run/'validation'/'independent-qa.json').open('x') as out:
        out.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['counts']),flush=True)


if __name__=='__main__':main()
