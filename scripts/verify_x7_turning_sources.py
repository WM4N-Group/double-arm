#!/usr/bin/env python3
"""Verify X7 turning archive, split, and independent repeat evidence without running physics."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from run_x7_turning_carry import LAYOUTS, turning_score

EXACT_FIELDS = ('action','phase','base_pre','base_post','arm_pre','gripper_post','object_post',
                'contact_samples','on_table','on_floor','station_clearance')


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def verify(primary, probe):
    report=json.loads((primary/'report.json').read_text())
    benchmark=json.loads((probe/'report.json').read_text())
    assert report['all_pass'] and benchmark['all_pass']
    asset=primary/'model'/'x7_proxy_gripper.urdf'
    if not asset.exists():
        [asset]=list((primary/'model').glob('*.urdf'))
    assert digest(asset)==report['asset_sha256']==benchmark['asset_sha256']
    assert report['training_started'] is False and report['external_rgb_hz']==5
    rows={r['case']:r for r in report['case_reports']}
    repeats={r['case']:r for r in benchmark['case_reports']}
    assert len(rows)==len(repeats)==len(LAYOUTS) and set(rows)==set(repeats)==set(LAYOUTS)
    verified=[]
    for case,layout in LAYOUTS.items():
        r=rows[case]; v=repeats[case]; n=r['steps']; partition=layout['split']
        expected=f'{partition}/accepted/{case}.npz'
        assert r['trace']==expected and r['partition']==partition and r['success']
        p0=primary/expected; p1=probe/f'{case}.npz'
        assert digest(p0)==r['trace_sha256'] and digest(p1)==v['trace_sha256']
        with np.load(p0,allow_pickle=False) as t, np.load(p1,allow_pickle=False) as ref:
            data={k:t[k] for k in t.files}
            assert t['action'].shape==(n,22) and t['base_post'].shape==(n,3,2) and t['arm_pre'].shape==(n,14,2)
            rgb=t['rgb_external']; ids=np.arange(0,n,6)
            assert rgb.shape==(len(ids),128,192,3) and rgb.dtype==np.uint8 and len(ids)==r['rgb_frames']
            assert np.array_equal(t['rgb_action_index'],ids) and np.min(np.ptp(rgb.reshape(len(ids),-1),axis=1))>30
            assert all(np.array_equal(t[k],ref[k]) for k in EXACT_FIELDS)
        score=turning_score(data,data['object_pre'][0],layout)
        assert score['success'] and score['robot_station_clearance_pass'] and score['physical_events']==r['physical_events']
        assert score['station_negative_distance_substeps']==score['robot_station_contact_substeps']==0
        verified.append({'case':case,'split':partition,'steps':n,'rgb_frames':len(ids),
                         'trace_sha256':r['trace_sha256'],'exact_fields_vs_independent_no_camera_probe':list(EXACT_FIELDS)})
    return {'kind':'X7 scripted turning source and heldout geometry checks; not policy performance',
            'source_count':sum(x['split']=='source' for x in verified),
            'heldout_layout_count':sum(x['split']=='heldout' for x in verified),
            'control_steps':sum(x['steps'] for x in verified),'rgb_frames':sum(x['rgb_frames'] for x in verified),
            'records':verified,'report_sha256':digest(primary/'report.json')}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('collection',type=Path); ap.add_argument('probe',type=Path)
    ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    result=verify(args.collection,args.probe)
    with args.out.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
