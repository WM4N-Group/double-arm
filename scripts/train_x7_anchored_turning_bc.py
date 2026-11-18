#!/usr/bin/env python3
"""X7 anchored hybrid: fitted timing/transport + measured task-frame constraints.

Use the same nine training traces as the clocked BC baseline. Save coefficients
and a paired proprioceptive grasp template. At inference, a calibrated object
pose fixes the contact segment and the known B fixture pose fixes the terminal
base pose. No source/evaluation episode is read by rollout. This is a hybrid,
not an end-to-end learned, reactive, or visual policy.
"""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from generate_x7_pickup_anchor import PickupWorld, check_pose, new_arm_tape
from run_x7_mobile_bimanual import ArmIK, CONTROL_HZ, GRIP_Z, HZ, PARK_Y, execute, make_model, smooth
from run_x7_turning_carry import HANDLE_HALF_SPAN, WXY, dock_pose, review_turn
from train_x7_turning_trajectory_bc import BLOCK_PHASES, DEV, HELDOUT, digest, fit, load_training, predict

DX_LIMIT_M=.10
DYAW_LIMIT_RAD=math.radians(9.)


def compact_template(rows):
    """Paired arms for all phases, not separate arm retrieval or a base tape.

    The two zero-offset sources store the same bimanual contact pattern. Use
    the first one and store the original phase timing only for its arm knots.
    One sample queue for each arm is packaged in a *single* coupling template.
    """
    a=next(r for r in rows if r['case']=='source-cw' and r['group']=='source')
    path=Path(a['trace']); assert digest(path)==a['trace_sha256']
    with np.load(path,allow_pickle=False) as t:
        u,ph=t['action'],t['phase']
    arm=u[:,6:20]
    # Transport itself has constant chassis-relative arms; keep one point
    # there, to preserve every distinct pickup/release arm sample untouched.
    idx=np.r_[np.ones(1,dtype=bool), (ph[1:]!=4)|(ph[:-1]!=4)]
    return {'template_arm':arm[idx], 'template_phase':ph[idx]}


def constrain_dock(u, counts, l):
    a=u.copy(); target=dock_pose(l); d=target-a[-1,:3]
    if np.linalg.norm(d[:2])>DX_LIMIT_M or abs(d[2])>DYAW_LIMIT_RAD:
        raise ValueError('Refused a dock correction beyond the declared geometry envelope')
    if any(len(np.unique(np.round(a[-counts[-1]:,k],decimals=9)))>1 for k in range(3)):
        raise ValueError('Expected a fixed base reference in the final settle block')
    lo,hi=[int(t) for t in np.r_[0,np.cumsum(counts)][[8,9]]]
    s=np.arange(1,hi-lo+1)/(hi-lo)
    a[lo:hi,:3]+=smooth(s)[:,None]*d; a[hi:,:3]+=d
    v_corr=(30*s*s*(1-s)*(1-s)*CONTROL_HZ/(hi-lo))[:,None]*d
    a[lo:hi,3:6]+=v_corr
    if np.any(abs(a[:,3:6])>np.asarray(WXY)+1e-10):
        raise ValueError('Dock projection would exceed base feedforward limit')
    np.testing.assert_allclose(a[-1,:3],target,atol=1e-9,rtol=0.)
    np.testing.assert_array_equal(a[:,6:],u[:,6:])
    return a,d


def prior_arms(asset, arms, packed, pose):
    ph=packed['template_phase']; original=np.zeros((len(ph),22));original[:,6:20]=packed['template_arm']
    warped,_,stats=new_arm_tape(asset,arms,original,ph,pose)
    return warped[:,6:20],ph,stats


def constrain_arms(u,ph,reference,reference_phase):
    result=u.copy()
    for stage in sorted(set(BLOCK_PHASES)):
        ids=np.flatnonzero(ph==stage); old=reference[reference_phase==stage]
        if not len(ids) or not len(old):
            raise ValueError('Incomplete paired grasp or release')
        t=np.linspace(0,1,len(ids))
        anchor=np.column_stack([np.interp(t,np.linspace(0,1,len(old)),old[:,q]) for q in range(14)])
        mix=np.ones(len(ids))
        if stage==0: mix=smooth(t)
        if stage==8: mix=1.-smooth(t)
        if stage==9: mix[:]=0
        result[ids,6:20]=mix[:,None]*anchor+(1-mix[:,None])*u[ids,6:20]
    np.testing.assert_array_equal(result[:,:6],u[:,:6]);np.testing.assert_array_equal(result[:,-2:],u[:,-2:])
    speed=float(np.max(abs(np.diff(result[:,6:20],axis=0)))*CONTROL_HZ)
    if speed>1.25 or np.any(abs(result[:,6:20])>3.14):
        raise ValueError(f'Contact projection creates discontinuous/invalid arm references: {speed}')
    return result,speed


def make_action(model,asset,homes,layout):
    a,phase,n=predict(model,layout)
    b,d=constrain_dock(a,n,layout)
    prior,ph,ks=prior_arms(asset,homes,model,layout['tote'])
    u,sp=constrain_arms(b,phase,prior,ph)
    return u,phase,dict(fitted_phase_durations=n,terminal_base_projection=d.tolist(),peak_arm_ref_rad_s=sp,
                        paired_grasp_fk_error=ks,learned_and_geometric_command_delta_max=float(np.max(abs(u-a))),
                        calibrated_geometry_input_only=True)


def evaluate(model,asset,homes,case,l,out,checkpoint_hash,review):
    try:
        u,ph,contract=make_action(model,asset,homes,l)
    except (RuntimeError,ValueError,AssertionError) as exc:
        row={'case':case,'success':False,'physical_execution':False,'error':str(exc)}
        print(json.dumps(row),flush=True);return row
    initial=[SimpleNamespace(home=u[0,6+i*7:13+i*7]) for i in range(2)]
    world=PickupWorld(asset,initial,l)
    label='ANCHORED BC HYBRID   /   +34 DEG HELDOUT'
    t=execute(world,u,ph,reviewer=review_turn,review_label=label,movie=out/(case+'.mp4') if review else None)
    metrics=check_pose(t,world,l);t['dock_pose']=world.dock
    path=out/(case+'.npz'); np.savez_compressed(path,**t)
    row=metrics|contract|{'case':case,'layout':l,'trace_sha256':digest(path),'checkpoint_sha256':checkpoint_hash,
                           'physical_execution':True,'trajectory_from_checkpoint_and_calibrated_geometry':True,
                           'used_heldout_expert_actions':False}
    if review: row['video_sha256']=digest(out/(case+'.mp4'))
    print(json.dumps(row,allow_nan=False),flush=True);return row


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('asset',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cad-zip',type=Path,required=True);ap.add_argument('--runs-root',type=Path,required=True)
    ap.add_argument('--heldout-review',action='store_true'); args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':ap.error('Remote CPU only: CUDA_VISIBLE_DEVICES must be empty')
    args.out.mkdir(parents=True); model=make_model(args.asset,args.out/'model',args.cad_zip)
    rows=load_training(args.runs_root,digest(model))
    manifest={'train':[{k:r[k] for k in ('case','trace_sha256','group','layout')} for r in rows],
              'dev':DEV,'heldout_geometry':HELDOUT,'heldout_policy':'only after every fixed dev gate passes; one attempt; no tape read',
              'constraints':{'terminal_translation_m':DX_LIMIT_M,'terminal_yaw_rad':DYAW_LIMIT_RAD,
                             'paired_contact':'grasp template persisted from source-cw train member, object-relative IK, grasp through release'}}
    m=args.out/'frozen-split.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
    weights=fit(rows)|compact_template(rows)
    ckpt=args.out/'anchored-hybrid.npz';np.savez_compressed(ckpt,**weights); fp=digest(ckpt)
    with np.load(ckpt,allow_pickle=False) as f: loaded={k:f[k] for k in f.files}
    assert set(loaded)==set(weights) and all(np.array_equal(loaded[k],weights[k]) for k in loaded)
    print(json.dumps({'train_count':len(rows),'checkpoint_sha256':fp,'frozen_split_sha256':digest(m)}),flush=True)
    arms=[ArmIK(model,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT); hold=None
    try:
        dev=[evaluate(loaded,model,arms,k,l,args.out,fp,review=False) for k,l in DEV.items()]
        pre= args.out/'dev-gate.json'; pre.write_text(json.dumps({'cases':dev,'all_pass':all(r['success'] for r in dev)},indent=2)+'\n')
        if all(r['success'] for r in dev):
            hold=evaluate(loaded,model,arms,'heldout-angle',HELDOUT,args.out,fp,review=args.heldout_review)
    finally:
        p.disconnect()
    result={'kind':'anchored hybrid: fitted task-clock, base transport refs, gripper timing and unloaded motion; object IK and fixture projection',
            'frozen_split_sha256':digest(m),'checkpoint_sha256':fp,'training_count':len(rows),'training_finished':True,
            'development_cases':dev,'heldout_gate_passed':all(r['success'] for r in dev),'heldout':hold,
            'heldout_expert_trace_read':False,'control_hz':CONTROL_HZ,'physics_hz':HZ}
    (args.out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
