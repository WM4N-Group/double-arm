#!/usr/bin/env python3
"""Fit an explicitly clocked, state-conditioned X7 joint/base trajectory BC baseline.

Train only on checked episodes. At inference the context is known pickup/tote
and dock geometry and a specified turn-bay height. Position references are
predicted, not supplied by a planner. The low-level motor servos still use
feedback; this is not a reactive or visual policy, and does not read expert
trajectories in evaluation.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from generate_x7_pickup_anchor import PickupWorld, check_pose
from run_x7_mobile_bimanual import CONTROL_HZ, execute, make_model
from run_x7_turning_carry import LAYOUTS, WXY

BASE = np.array([0,1,2]); ARM = np.arange(6,20); JAW=np.array([20,21])
K=41
BLOCK_PHASES=(0,1,2,3,4,4,4,4,4,5,6,7,8,9)
SOURCE_DIRS = ('x7-turning-carry-20261118-192041','x7-turning-se2-20261118-193054','x7-pickup-anchor-20261118-203010')
DEV={
    'dev-cw': {'goal':(3.31,.091,-24.),'bay_y':.645,'tote':(-.004,-.009,-1.1)},
    'dev-ccw': {'goal':(3.27,.134,26.),'bay_y':.717,'tote':(.003,-.010,.8)},
}
HELDOUT={'goal':LAYOUTS['heldout-angle']['goal'],'bay_y':LAYOUTS['heldout-angle']['bay_y'],'tote':(0.,0.,0.)}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def j(path):
    return json.loads(path.read_text())


def load_training(root, model_sha):
    """Read only accepted, audited episodes. Holdout trajectory data is never loaded."""
    turn,se2,pickup=(root/s for s in SOURCE_DIRS)
    turn_report=turn/'collection/report.json'; t=j(turn_report); tq=j(turn/'turning-qa.json')
    assert t['asset_sha256']==model_sha and t['all_pass'] and tq['report_sha256']==digest(turn_report)
    assert tq['source_count']==2 and tq['heldout_layout_count']==1
    sourced=[r for r in t['case_reports'] if r['partition']=='source']
    assert {r['case'] for r in sourced}=={'source-cw','source-ccw'}
    contexts=[]
    for r in sourced:
        assert next(q for q in tq['records'] if q['case']==r['case'])['trace_sha256']==r['trace_sha256']
        contexts.append((turn/'collection',r,LAYOUTS[r['case']],'source'))
    for folder,qaname in ((se2,'qa-se2.json'),(pickup,'qa-pickup.json')):
        s=j(folder/'dataset/summary.json'); qa=j(folder/qaname); f=folder/'dataset/attempts.json'
        assert s['attempts_sha256']==digest(f)
        assert s.get('asset_sha256',s.get('robot_model_sha256'))==model_sha
        assert qa['accepted']==(4 if folder==se2 else 3)
        for r in j(f):
            if r['success']:
                contexts.append((folder/'dataset',r,r['layout'],'se2' if folder==se2 else 'pickup'))
    rows=[]; hashes=set()
    for directory,record,layout,group in contexts:
        path=directory/record['trace']; h=digest(path)
        assert h==record['trace_sha256'] and h not in hashes and path.parent.name=='accepted'
        assert record['success'] and record['robot_station_clearance_pass'] and record['object_attachment_constraints_seen']==0
        assert record['state_writes_during_actions']==0 and record['case']!='heldout-angle'
        layout=layout|{'tote':layout.get('tote',(0.,0.,0.))}
        with np.load(path,allow_pickle=False) as f:
            u,ph=f['action'],f['phase']; assert (u.shape==(len(ph),22) and np.isfinite(u).all())
            base,arm,jaw,n=align(u,ph,record['route_segments'])
        rows.append(dict(case=record['case'],layout=layout,trace=str(path),trace_sha256=h,group=group,
                         base=base,arm=arm,jaw=jaw,durations=n))
        hashes.add(h)
    assert len(rows)==9
    return rows


def align(u,ph,segments):
    intervals=[]
    for k in range(4):
        ids=np.flatnonzero(ph==k); intervals.append((ids[0],ids[-1]+1))
    intervals.extend((r['start_inclusive'],r['end_exclusive']) for r in segments)
    for k in range(5,10):
        ids=np.flatnonzero(ph==k); intervals.append((ids[0],ids[-1]+1))
    assert len(intervals)==len(BLOCK_PHASES) and intervals[0][0]==0 and intervals[-1][1]==len(u)
    assert all(b==c for (_,b),(c,_) in zip(intervals[:-1],intervals[1:]))
    assert all(np.all(ph[a:b]==k) for k,(a,b) in zip(BLOCK_PHASES,intervals))
    samples=[]; lengths=[]
    for a,b in intervals:
        seq=np.vstack([u[max(0,a-1)],u[a:b]]); n=b-a;out_t=np.linspace(0.,1.,K)
        samples.append(np.column_stack([np.interp(out_t,np.linspace(0.,1.,n+1),seq[:,k]) for k in np.r_[BASE,ARM,JAW]]))
        lengths.append(n)
    stack=np.stack(samples)
    return stack[:,:,:3],stack[:,:,3:17],stack[:,:,-2:],np.asarray(lengths)


def context_base(layout):
    x,y,yaw=layout['goal']
    return np.r_[1.,(x-3.26)/.15,(y-.10)/.08,(abs(yaw)-25.)/6.,(layout['bay_y']-.66)/.09]


def context_arm(layout):
    x,y,angle=layout['tote'];return np.r_[1.,x/.01,y/.012,angle/1.8]


def fit_regression(rows,feature,key,penalty):
    x=np.stack([feature(r['layout']) for r in rows]); a=np.stack([r[key] for r in rows]); shape=a.shape[1:]
    z=np.linalg.solve(x.T@x+np.diag(penalty),x.T@a.reshape(len(rows),-1)).reshape((x.shape[1],*shape))
    return z


def fit(rows):
    m={}
    for label,sign in (('cw',-1),('ccw',1)):
        r=[a for a in rows if sign*a['layout']['goal'][2]>0]
        assert len(r)>=4
        m[label+'_base']=fit_regression(r,context_base,'base',[0.,.055,.055,.025,.025])
        m[label+'_dur']=fit_regression(r,context_base,'durations',[0.,.18,.18,.11,.11])
    m['arm']=fit_regression(rows,context_arm,'arm',[0.,.0005,.0005,.0005])
    m['jaw']=np.mean([r['jaw'] for r in rows],axis=0)
    return m


def predict(model,l):
    side='ccw' if l['goal'][2]>0 else 'cw'; f=context_base(l)
    base=np.tensordot(f,model[side+'_base'],axes=1)
    arm=np.tensordot(context_arm(l),model['arm'],axes=1)
    dur=np.tensordot(f,model[side+'_dur'],axes=1)
    jaw=model['jaw']; axes=np.r_[np.array(WXY)*.86,np.full(14,.9),[.15,.15]]
    result=[]; phases=[]; counts=[]
    for i,k in enumerate(BLOCK_PHASES):
        s=np.column_stack([base[i],arm[i],jaw[i]])
        n=max(int(round(dur[i])),int(np.ceil((K-1)*CONTROL_HZ*np.max(abs(np.diff(s,axis=0))/axes))))
        if n>650 or n<2:
            raise ValueError(f'Unexpected inferred trajectory duration {n}')
        t=np.linspace(0.,1.,n+1)
        ref=np.column_stack([np.interp(t,np.linspace(0.,1.,K),s[:,axis]) for axis in range(19)])
        v=np.gradient(ref[:,:3],1/CONTROL_HZ,axis=0);v[[0,-1]]=0.
        out=np.column_stack([ref[1:,:3],v[1:],ref[1:,3:17],ref[1:,-2:]])
        if not np.isfinite(out).all() or np.any(abs(out[:,3:6])>np.array(WXY)*.86+1e-10):
            raise RuntimeError('Unsafe inferred reference')
        result.append(out); phases.extend([k]*n); counts.append(n)
    return np.vstack(result),np.asarray(phases,dtype=np.uint8),counts


def roll_case(model,asset,case,layout,out):
    a,ph,steps=predict(model,layout)
    # Initial arm configuration is predicted by this checkpoint. Setup uses
    # the same reset mechanics, with no teacher/planner calls in this case.
    homes=[SimpleNamespace(home=a[0,6+i*7:13+i*7]) for i in range(2)]
    world=PickupWorld(asset,homes,layout)
    trace=execute(world,a,ph); score=check_pose(trace,world,layout)
    trace['dock_pose']=world.dock; trace['pickup_anchor_delta']=np.asarray(layout['tote'])
    part=out/f'{case}.npz'; np.savez_compressed(part,**trace)
    row=score|{'case':case,'layout':layout,'trace_sha256':digest(part),'phase_durations':steps,
               'base_terminal_command_minus_dock':(a[-1,:3]-world.dock).tolist(),
               'policy_execution':'context/clock ridge BC trajectory then ordinary motor feedback; no expert execution'}
    print(json.dumps(row),flush=True)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('asset',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cad-zip',type=Path,required=True);ap.add_argument('--runs-root',type=Path,required=True)
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':ap.error('Remote CPU only: CUDA_VISIBLE_DEVICES must be empty')
    args.out.mkdir(parents=True);asset=make_model(args.asset,args.out/'model',args.cad_zip);rows=load_training(args.runs_root,digest(asset))
    split={'train':[ {k:r[k] for k in ('case','trace','trace_sha256','group','layout')} for r in rows],
           'development_geometries':DEV,'preserved_unread_heldout_reference':HELDOUT}
    split_file=args.out/'frozen-split.json';split_file.write_text(json.dumps(split,indent=2)+'\n')
    model=fit(rows); ckpt=args.out/'trajectory-bc.npz';np.savez_compressed(ckpt,**model)
    with np.load(ckpt,allow_pickle=False) as c:
        saved={k:c[k] for k in c.files}
    assert set(saved)==set(model) and all(np.array_equal(v,saved[k]) for k,v in model.items())
    print(json.dumps(dict(train_episodes=len(rows),checkpoint_sha256=digest(ckpt),split_sha256=digest(split_file))),flush=True)
    p.connect(p.DIRECT)
    try:
        results=[roll_case(saved,asset,k,l,args.out) for k,l in DEV.items()]
    finally:
        p.disconnect()
    total={'kind':'clocked, context-conditioned supervised trajectory BC pilot (not RGB or reactive)',
           'training_finished':True,'train_count':len(rows),'checkpoint_sha256':digest(ckpt),'frozen_split_sha256':digest(split_file),
           'heldout_executed':False,'heldout_trace_read':False,'development_passes':sum(r['success'] for r in results),
           'results':results}
    (args.out/'report.json').write_text(json.dumps(total,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
