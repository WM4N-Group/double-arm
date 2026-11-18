#!/usr/bin/env python3
"""Simulate a masked rear depth curtain for the lower X7 chassis envelope.

Finite rays do not establish free space between beams. For this pilot only,
coverage assumes the relevant obstacle presents a detectable axis-aligned
projected rectangle >=12 mm on both sampled axes. Any unavailable cell in
that envelope makes the rear packet unknown; do not consult simulator IDs or
clearance to authorize retreat. Upper robot and payload are NOT covered.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import CASES, FROZEN_CHECKPOINT, LAYOUTS, perceived_pose
from audit_x7_certified_retreat import REAR_OBSTACLE_SPEED_BOUND
from audit_x7_chassis_range_proxy import (RANGE_M, MOUNT_PAD_M, RangeWorld, gaps_from_fractions,
                                         transform_mounts)
from audit_x7_dock_sensor import BIAS, GUARD_SCAN_M, PERIOD, WAVE
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, r2
from test_x7_dock_guard import guard_metrics
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

PITCH_M = .008
MIN_PROJECTED_DIM_M = .012
OCCLUSION_DELAY = 58
CASES_TO_RUN = ('clear', 'narrow', 'upper-low-chassis', 'partial-before-retreat', 'partial-during-retreat')


def rear_curtain(cad):
    # The lowest CAD slab is the wheeled chassis envelope, not the whole robot.
    row = min(cad['parts_in_cad_frame'], key=lambda item: item['center'][0])
    a, b, c = row['center']; dx, dy, dz = row['size']
    x = np.linspace(-c-dz/2, -c+dz/2, int(np.ceil(dz/PITCH_M))+1)
    z = np.linspace(1.77+a-dx/2, 1.77+a+dx/2, int(np.ceil(dx/PITCH_M))+1)
    xx, zz = np.meshgrid(x,z)
    return np.column_stack([xx.ravel(),np.full(xx.size,b+dy/2+MOUNT_PAD_M),zz.ravel()])


def summarize_cells(fractions, valid):
    """States: unknown (mask/corrupt), no return, or measured return.

    A valid no-return denotes only that the modeled beam returned no hit.
    The limited projected-obstacle contract is external to this operation.
    """
    f=np.asarray(fractions,dtype=float); valid=np.asarray(valid,dtype=bool)
    if not len(f) or f.shape!=valid.shape or f.ndim!=1:
        raise ValueError('incomplete curtain schema')
    known=valid & np.isfinite(f) & (f>=0.) & (f<=1.)
    unknown=int(np.count_nonzero(~known)); hit=int(np.count_nonzero(known & (f<1.)))
    distance=None if unknown else float(gaps_from_fractions(f).min())
    return distance,unknown,hit


class CoverageWorld(RangeWorld):
    def __init__(self,asset,homes,l,obs,arm,span,name):
        self.coverage_case=name
        self.curtain=rear_curtain(json.loads((asset.parent/'base-envelope.json').read_text()))
        self.coverage=[]; self.mask_start=None; self.mask_stop=None
        super().__init__(asset,homes,l,obs,arm,span,
                         'rear-narrow-blind' if name in ('narrow','upper-low-chassis') else 'overtravel-clear')

    def spawn_rear(self,k):
        if self.coverage_case=='narrow':
            return super().spawn_rear(k)
        # At 0.56 m this target is outside every older bumper ray, even if
        # horizontal spacing is dense, but inside the lower chassis slab.
        theta=float(self.qv(self.base)[2,0]); xy=self.qv(self.base)[:2,0]+r2(theta)@[.035,.238+.044+.055]
        half=[.006,.055,.016]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[.78,.24,.13,1.])
        self.obstacle=p.createMultiBody(baseMass=0,basePosition=[*xy,.56],baseOrientation=p.getQuaternionFromEuler([0,0,theta]),
            linkMasses=[1.],linkCollisionShapeIndices=[col],linkVisualShapeIndices=[vis],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth={'injected_substep':k,'half_extents_m':half,'center_z_m':.56}

    def begin_sensor_substep(self,k):
        super().begin_sensor_substep(k)
        if self.mask_start is None:
            if self.coverage_case=='partial-before-retreat' and self.latch_k is not None:
                self.mask_start=k
            if self.coverage_case=='partial-during-retreat' and self.started is not None:
                self.mask_start=self.started+OCCLUSION_DELAY

    def acquire_packet(self,k,ignored_truth_rear):
        q=self.qv(self.base)[:,0]
        a,b=transform_mounts(self.mounts[0],q,-1)
        front=gaps_from_fractions([r[2] for r in p.rayTestBatch(a.tolist(),b.tolist())]).min()
        a,b=transform_mounts(self.curtain,q,1)
        fraction=np.array([r[2] for r in p.rayTestBatch(a.tolist(),b.tolist(),numThreads=1)])
        valid=np.ones(len(fraction),dtype=bool)
        if self.mask_start is not None and self.mask_start<=k<self.mask_start+80:
            valid &= abs(self.curtain[:,0]-.035)>.012
        rear,unknown,hits=summarize_cells(fraction,valid)
        wave=np.sin(.77*(k//PERIOD)+np.array([.43,2.24]))*WAVE+BIAS
        if unknown and self.mask_stop is None: self.mask_stop=k
        self.coverage.append([k,len(fraction),unknown,hits,-1. if rear is None else rear])
        return np.array([GUARD_SCAN_M,min(front+wave[0],GUARD_SCAN_M),
                         -1. if rear is None else min(rear+wave[1],GUARD_SCAN_M)]),q.copy()

    def lower_from_packet(self,raw,pose,age):
        observed,low,move=super().lower_from_packet(raw,pose,age)
        if raw[2]<0.: low[2]=-1.
        return observed,low,move


    def observe_active_packet(self,k,sample_k,observed,lower,qv):
        # Retain *delivered* partial coverage faults once braking has begun;
        # a valid later packet alone must not restore autonomous retreat.
        g=self.guard
        if observed is None or observed[2]>=0. or g.state not in ('brake','retreat'):
            return
        if not g.cannot_retreat:
            self.rear_events.append({'substep':k,'sample_k':sample_k,'event':'rear_coverage_lost',
                                     'state_when_received':g.state})
        g.cannot_retreat=True
        if g.state=='retreat':
            g.hold=qv[:,0].copy()
            g.switch(k,'hold')


def run_case(model,asset,arms,name,out):
    true=LAYOUTS['north']; obs=perceived_pose(true,*CASES['dock-in-20mm'])
    u,ph,meta=make_action(model,asset,arms,obs); n=meta['fitted_phase_durations']
    init=[SimpleNamespace(home=u[0,6+7*i:13+7*i]) for i in range(2)]
    world=CoverageWorld(asset,init,true,obs,sum(n[:8]),n[8],name)
    t=execute(world,u,ph); score=guard_metrics(world,t,true)
    rear=np.asarray(world.obstacle_trace); coverage=np.asarray(world.coverage); stream=np.asarray(world.sensor)
    hits=int(np.count_nonzero(rear[:,1])); low=float(rear[:,0].min()) if world.obstacle is not None else None
    assert np.max(abs(np.asarray(world.rear_position)[:,2]))<=REAR_OBSTACLE_SPEED_BOUND
    states=[event['state'] for event in score['guard_states'] if 'state' in event]
    move=score['retreat_measured_m']; unknown=bool(np.any(coverage[:,2]>0))
    if name=='clear': expected=abs(move-.014)<.001 and not unknown
    elif name in ('narrow','upper-low-chassis'):
        expected=low is not None and low>=.005 and 0.<move<.013 and np.any(coverage[:,3]>0) and any(
            event['event']=='rear_dynamic_braking' for event in world.rear_events)
    elif name=='partial-before-retreat':
        expected=unknown and 'retreat' not in states and any(
            event['event']=='rear_coverage_lost' and event['state_when_received']=='brake' for event in world.rear_events)
    else:
        expected=unknown and 'retreat' in states and 0.<move<.013 and any(
            event['event']=='rear_coverage_lost' and event['state_when_received']=='retreat' for event in world.rear_events)
    record=score|{'case':name,'coverage_case_pass':bool(expected and score['safe_holding_interrupt'] and hits==0),
        'rear_min_actual_m':low,'rear_contact_substeps':hits,'rear_events':world.rear_events,
        'rear_curtain_cells':len(world.curtain),'masked_packets':int(np.count_nonzero(coverage[:,2])),
        'first_masked_acquisition':world.mask_stop,'full_robot_coverage':False,'checkpoint_sha256':FROZEN_CHECKPOINT}
    path=out/(name+'.npz')
    np.savez_compressed(path,**t,coverage_packets=coverage,sensor_telemetry=stream,
                        rear_geometry_audit=rear,actual_motor_references=np.asarray(world.applied))
    return record|{'trace_sha256':digest(path)}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',choices=CASES_TO_RUN,nargs='+',default=CASES_TO_RUN);a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(set(a.cases))!=len(a.cases):ap.error('Remote CPU only; unique cases')
    a.out.mkdir(parents=True); c=a.pilot/'anchored-hybrid.npz';assert digest(c)==FROZEN_CHECKPOINT
    with np.load(c,allow_pickle=False) as data: model={key:data[key] for key in data.files}
    asset=a.pilot/'model'/'x7_proxy_gripper.urdf'
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    results=[]; p.connect(p.DIRECT)
    try:
        for name in a.cases:
            row=run_case(model,asset,arms,name,a.out);results.append(row)
            print(json.dumps({k:row[k] for k in ('case','coverage_case_pass','success','rear_curtain_cells',
              'masked_packets','rear_min_actual_m','rear_contact_substeps','rear_events','retreat_measured_m')},allow_nan=False),flush=True)
    finally: p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'cases':results,'checkpoint_sha256':FROZEN_CHECKPOINT,
        'max_grid_pitch_m':PITCH_M,'assumed_minimum_projected_width_and_height_m':MIN_PROJECTED_DIM_M,
        'scope':'lower-chassis only, idealized detectable rectangle; unknown cells veto movement; not real safety validation',
        'training_started':False,'holdout_reused':False},indent=2,allow_nan=False)+'\n')
    return 0 if all(row['coverage_case_pass'] for row in results) else 1


if __name__=='__main__':raise SystemExit(main())
