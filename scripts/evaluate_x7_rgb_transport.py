#!/usr/bin/env python3
"""Run X7 hybrid transport in physics with a CPU-reloaded GPU checkpoint."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
import torch

from audit_x7_anchor_tolerance import check_footprint
from collect_x7_transport_bc import FROZEN_CHECKPOINT, layouts, sha_layout
from generate_x7_pickup_anchor import PickupWorld, check_pose
from run_x7_mobile_bimanual import ArmIK, CONTROL_HZ, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, WXY, capture_turn_rgb, review_turn
from train_x7_anchored_turning_bc import make_action
from train_x7_rgb_transport import CHUNK, LIMIT, RGBTransport, feature, transport_prior, write_json
from train_x7_turning_trajectory_bc import digest


class LiveTransport(PickupWorld):
    def __init__(self,asset,homes,layout,prior,bounds,net,norm):
        self.prior=prior;self.lo,self.hi=bounds;self.net=net;self.norm=norm;self.layout=layout
        self.applied=[];self.mask=[];self.chunk=None;self.calls=0;self.clipped=0;self.images=[];self.indices=[]
        self.model_delta=[]
        super().__init__(asset,homes,layout)

    def policy_base_observation(self):
        return self.qv(self.base)

    def step(self,a):
        k=len(self.applied); b=a.copy();active=False
        if k%CHUNK==0:
            self.chunk=None
            if self.lo<=k and k+CHUNK<=self.hi:
                nom=self.prior[k:k+CHUNK]
                if self.net is None: delta=np.zeros_like(nom)
                else:
                    img=capture_turn_rgb()
                    x=feature(self.layout,self.policy_base_observation(),self.qv(sum(self.arm_ids,[])),self.qv(sum(self.jaws,[])),nom)
                    xm,xs,ym,ys=self.norm
                    with torch.inference_mode():
                        pred=self.net(torch.from_numpy(img[None]),torch.from_numpy(((x-xm)/xs)[None]))[0].numpy()
                    delta=pred*ys+ym
                    if not np.isfinite(delta).all():raise ValueError('Nonfinite live transport action')
                    self.images.append(img);self.indices.append(k);self.calls+=1
                clipped=np.clip(delta,-LIMIT,LIMIT);self.clipped+=int(np.count_nonzero(delta!=clipped))
                self.chunk=nom+clipped
        if self.chunk is not None:
            b[:6]=self.chunk[k%CHUNK]
            # Apply a fixed rate/velocity limit to learned base targets, not a
            # replacement with a teacher command if prediction is inconvenient.
            cap=np.asarray(WXY)
            last=self.applied[-1][:3] if self.applied else a[:3]
            unclamped=b[:6].copy()
            b[:3]=last+np.clip(b[:3]-last,-cap*1.12/CONTROL_HZ,cap*1.12/CONTROL_HZ)
            b[3:6]=np.clip(b[3:6],-cap,cap)
            self.clipped+=int(np.count_nonzero(abs(b[:6]-unclamped)>1.e-10))
            self.model_delta.append(b[:6]-self.prior[k]);active=True
        self.applied.append(b);self.mask.append(active)
        return super().step(b)


def load_episode_assets(pilot,dataset,training):
    pth=training/'best.pt';st=json.loads((training/'status.json').read_text());info=training/'training-inputs.json'
    if st['stage']!='train_completed' or st['checkpoint_sha256']!=digest(pth) or st['training_input_sha256']!=digest(info):
        raise ValueError('No finalized training artifacts')
    loaded=torch.load(pth,map_location='cpu',weights_only=True)
    if loaded['teacher_sha256']!=FROZEN_CHECKPOINT or loaded['manifest_sha256']!=digest(info) or loaded['smoke']:
        raise ValueError('Non-formal or mismatched checkpoint')
    plan=dataset/'layouts.json';input_plan=json.loads(info.read_text())
    if digest(plan)!=input_plan['layouts_sha256'] or json.loads(plan.read_text())['layouts']!=layouts():
        raise ValueError('Unexpected eval layout manifest')
    asset=pilot/'model/x7_proxy_gripper.urdf';ckpt=pilot/'anchored-hybrid.npz'
    if digest(ckpt)!=FROZEN_CHECKPOINT or digest(asset)!=input_plan['urdf_sha256']:
        raise ValueError('Unexpected frozen geometry or timing anchor')
    with np.load(ckpt,allow_pickle=False) as w:anchor={k:w[k] for k in w.files}
    model=RGBTransport(len(loaded['xmean']));model.load_state_dict(loaded['state_dict'],strict=True);model.eval()
    norms=[np.array(loaded[k],dtype=np.float32) for k in ('xmean','xstd','ymean','ystd')]
    return asset,anchor,model,norms,digest(pth),loaded['epoch']


def evaluate(args):
    if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('Physics evaluation is pinned to CPU')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    asset,anchor,net,norm,h,epoch=load_episode_assets(args.pilot,args.dataset,args.training)
    if args.split=='heldout':
        gate=json.loads((args.out/'dev-gate.json').read_text())
        if not gate['all_12_candidate_physical_pass'] or gate['checkpoint_sha256']!=h:
            raise ValueError('Dev physical gate did not authorize heldout')
    folder=args.out/args.split/args.mode;folder.mkdir(parents=True,exist_ok=True)
    cases=[l for l in layouts() if l['split']==args.split]
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT)
    try:
        for i,layout in enumerate(cases):
            if (args.case and layout['case']!=args.case) or (not args.case and i%args.workers!=args.worker):continue
            name=layout['case'];dest=folder/name;dest.mkdir()
            try:u,ph,meta=make_action(anchor,asset,arms,layout)
            except (ValueError,AssertionError,RuntimeError) as e:
                report=dict(case=name,split=args.split,mode=args.mode,passed=False,physical_execution=False,error=str(e),
                            checkpoint_sha256=h,layout_sha256=sha_layout(layout))
            else:
                homes=[SimpleNamespace(home=u[0,6+7*j:13+7*j]) for j in range(2)]
                prior,lo,hi=transport_prior(layout,meta['fitted_phase_durations'])
                w=LiveTransport(asset,homes,layout,prior,(lo,hi),net if args.mode=='candidate' else None,norm)
                movie=dest/'review.mp4' if args.mode=='candidate' and (i<2 or args.case) else None
                trace=execute(w,u,ph,reviewer=review_turn,movie=movie,
                              review_label='GPU-TRAINED BASE TRANSPORT / GEOMETRIC GRASP')
                actual=np.asarray(w.applied);mask=np.asarray(w.mask,dtype=bool)
                if (len(actual)!=len(u) or not np.array_equal(actual[:,6:],u[:,6:]) or
                        not np.array_equal(actual[~mask],u[~mask]) or not np.all(ph[mask]==4)):
                    raise ValueError('Hidden action changes or incomplete physics execution')
                if args.mode=='candidate' and w.calls*CHUNK!=np.count_nonzero(mask):
                    raise ValueError('Unqualified inference-to-action alignment')
                trace.update(action=actual,nonlearned_anchor_action=u,live_transport_mask=mask,
                             live_rgb=np.asarray(w.images),live_rgb_action_index=np.asarray(w.indices,dtype=np.int32))
                metrics=check_pose(trace,w,layout);foot=check_footprint(trace,layout)
                s,init=trace['station_clearance'],trace['station_initialization_clearance']
                dmin=float(min(np.min(s[...,:2]),np.min(init[...,:2])))
                transport_fraction=float(mask.sum()/(hi-lo));transport_delta=float(np.linalg.norm(np.asarray(w.model_delta))/max(1,mask.sum())**.5)
                qualified=bool(metrics['success'] and foot['full_bottom_supported'] and dmin>=.005 and
                    not np.any(s[...,2:4]) and not np.any(init[...,2:4]) and transport_fraction>=.98 and
                    metrics['object_attachment_constraints_seen']==0 and metrics['state_writes_during_actions']==0 and
                    (args.mode=='nominal' or (w.calls>0 and transport_delta>1.e-5)))
                f=dest/'trace.npz';np.savez_compressed(f,**trace)
                report=dict(case=name,split=args.split,mode=args.mode,checkpoint_sha256=h,checkpoint_epoch=epoch,
                    layout_sha256=sha_layout(layout),passed=qualified,physical_execution=True,
                    minimum_robot_station_m=dmin,metrics=metrics,footprint=foot,trace_sha256=digest(f),
                    actual_inference_calls=w.calls,transport_chunk_fraction=transport_fraction,
                    transport_base_reference_delta_rms_mixed_units=transport_delta,numeric_limits_applied=w.clipped,
                    uses_geometric_grasp_and_dock=True,teacher_episode_tape_loaded=False,review_sha256=digest(movie) if movie else None)
            write_json(dest/'report.json',report)
            print(json.dumps({k:report.get(k) for k in ('case','mode','passed','physical_execution','actual_inference_calls',
                       'minimum_robot_station_m','transport_chunk_fraction','error')},allow_nan=False),flush=True)
    finally:p.disconnect()


def summarize(args):
    h=digest(args.training/'best.pt');group=[r for r in layouts() if r['split']==args.split];data={}
    for mode in ('candidate','nominal') if args.split=='dev' else ('candidate',):
        records=[]
        for layout in group:
            name=layout['case'];path=args.out/args.split/mode/name
            r=json.loads((path/'report.json').read_text())
            if r['checkpoint_sha256']!=h or r['layout_sha256']!=sha_layout(layout) or r['mode']!=mode:
                raise ValueError('Mismatched validation identity')
            if r['physical_execution'] and digest(path/'trace.npz')!=r['trace_sha256']:
                raise ValueError('Changed rollout archive')
            records.append(r)
        data[mode]=records
    out=dict(checkpoint_sha256=h,split=args.split,passed={k:sum(x['passed'] for x in rows) for k,rows in data.items()},
             case_count=len(group),records=data,learned_scope='six base commands for full transport chunks only',
             all_12_candidate_physical_pass=bool(len(group)==12 and all(x['passed'] for x in data['candidate'])),
             train_task_collision_coverage_not_hardware_validation=True)
    write_json(args.out/('dev-gate.json' if args.split=='dev' else 'heldout-report.json'),out)
    print(json.dumps({k:out[k] for k in ('split','passed','case_count','all_12_candidate_physical_pass')}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','dataset','training','out'):ap.add_argument(name,type=Path)
    ap.add_argument('--split',choices=('dev','heldout'),required=True)
    ap.add_argument('--mode',choices=('candidate','nominal'),default='candidate');ap.add_argument('--worker',type=int,default=0)
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--case');ap.add_argument('--summarize',action='store_true')
    a=ap.parse_args()
    if not 0<=a.worker<a.workers or a.split=='heldout' and a.mode!='candidate':ap.error('Invalid shard or split')
    summarize(a) if a.summarize else evaluate(a)


if __name__=='__main__':main()
