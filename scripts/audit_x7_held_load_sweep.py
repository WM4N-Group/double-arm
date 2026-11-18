#!/usr/bin/env python3
"""Measure the held X7 envelope and probe a blind zone outside fixed-CAD sensing.

Remote simulated physics only. Collision queries for arms and payload are
independent audit observations, never inputs to the frozen retreat guard.
Each world AABB is transformed conservatively into the frozen dock frame;
axes extents are not a collision proof between integration samples.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import CASES, FROZEN_CHECKPOINT, LAYOUTS, perceived_pose
from audit_x7_certified_retreat import REAR_OBSTACLE_SPEED_BOUND, measure
from audit_x7_upper_rear_coverage import ExtendedWorld, cad_boxes, digest
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, r2
from test_x7_dock_guard import STATION_CLEARANCE_M, guard_metrics
from train_x7_anchored_turning_bc import make_action

SCENARIOS=('clear','payload-edge')
CORNER=np.array([[a,b,c] for a in (0,1) for b in (0,1) for c in (0,1)])
FIXTURE_HALF=(.006,.018,.016)
FIXTURE_X=.399
INITIAL_FRONT_GAP=.043


def local_aabb(box,frame):
    """Bounds of the full world AABB rotated into the chosen fixed frame."""
    low,high=np.asarray(box,dtype=float)
    if not np.isfinite([low,high]).all() or np.any(low>high):
        raise ValueError('Invalid collision AABB')
    corners=low+CORNER*(high-low)
    corners[:,:2]=(corners[:,:2]-frame[:2])@r2(frame[2])
    return np.stack((corners.min(axis=0),corners.max(axis=0)))


def union_bounds(samples):
    a=np.asarray(samples)
    return np.stack([a[...,0,:].reshape(-1,3).min(axis=0),a[...,1,:].reshape(-1,3).max(axis=0)])


class LoadAuditWorld(ExtendedWorld):
    def __init__(self,asset,home,true,obs,arm,span,case):
        self.case_name=case;self.audit=[];self.bounds=[];self.frame=None
        super().__init__(asset,home,true,obs,arm,span,'fixed-cad-clear')
        self.config['rear']=case=='payload-edge'
        named=[[f'Joint{i}' for i in range(1,8)],[f'Joint{i}' for i in range(8,15)]]
        self.arm_links=[]
        for names,suffix in zip(named,('','_B')):
            names += [f'proxy_finger_{side}_slide{suffix}' for side in ('a','b')]+[f'proxy_palm_mount{suffix}']
            self.arm_links.append([self.joints[n] for n in names if p.getCollisionShapeData(self.robot,self.joints[n])])
        if min(map(len,self.arm_links))==0:
            raise RuntimeError('Expected arm and gripper collision shapes')

    def spawn_rear(self,k):
        q=self.qv(self.base)[:,0]; pose=self.state_object()
        xy=q[:2]+r2(q[2])@[FIXTURE_X,0.]
        # Independent fixture placement in the simulated world, not a guard input.
        tote_y=(pose[:2]-q[:2])@r2(q[2])[:,1]
        xy+=r2(q[2])@[0.,tote_y+.095+INITIAL_FRONT_GAP+FIXTURE_HALF[1]]
        z=float(pose[2]+.049)
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=FIXTURE_HALF)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=FIXTURE_HALF,rgbaColor=[.84,.20,.13,1.])
        self.obstacle=p.createMultiBody(baseMass=0,basePosition=[*xy,z],
            baseOrientation=p.getQuaternionFromEuler([0,0,q[2]]),linkMasses=[1.],
            linkCollisionShapeIndices=[col],linkVisualShapeIndices=[vis],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],linkParentIndices=[0],
            linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth={'substep':k,'position':[*xy,z],'half_extents':FIXTURE_HALF,'base_x_projection_m':FIXTURE_X,
                    'initial_robot_gap_m':measure(self.robot,self.obstacle),
                    'initial_payload_gap_m':measure(self.tote,self.obstacle)}

    def audit_held(self,k):
        if self.latch_k is None:return
        if self.frame is None:self.frame=self.qv(self.base)[:,0].copy()
        ids=[local_aabb(p.getAABB(self.robot,self.base_link),self.frame)]
        for group in self.arm_links:
            ids.append(union_bounds([local_aabb(p.getAABB(self.robot,i),self.frame) for i in group]))
        ids.append(local_aabb(p.getAABB(self.tote),self.frame))
        self.bounds.append(np.asarray(ids));self.audit.append([k,1 if self.guard.state=='retreat' else 0,
                      .32 if self.obstacle is None else measure(self.tote,self.obstacle),
                      0 if self.obstacle is None else len(p.getContactPoints(self.tote,self.obstacle))])

    def begin_sensor_substep(self,k):
        super().begin_sensor_substep(k)
        self.audit_held(k)


def case_run(weights,asset,arms,case,out):
    true=LAYOUTS['north']; obs=perceived_pose(true,*CASES['dock-in-20mm']);u,ph,meta=make_action(weights,asset,arms,obs)
    n=meta['fitted_phase_durations']; home=[SimpleNamespace(home=u[0,6+i*7:13+i*7]) for i in range(2)]
    w=LoadAuditWorld(asset,home,true,obs,sum(n[:8]),n[8],case)
    t=execute(w,u,ph);w.audit_held(len(w.applied)); r=guard_metrics(w,t,true)
    audit=np.asarray(w.audit);sweep=np.asarray(w.bounds);mask=audit[:,1]==1
    if not np.any(mask): raise RuntimeError('Run never entered the intended straight retreat')
    physical=np.asarray(w.obstacle_trace);packets=np.asarray(w.coverage)
    rear_robot=float(physical[:,0].min()) if w.obstacle is not None else None
    rear_payload=float(audit[:,2].min()) if w.obstacle is not None else None
    contacts=int(np.count_nonzero(audit[:,3]));robot_contacts=int(np.count_nonzero(physical[:,1]))
    motion=np.asarray(w.rear_position); assert np.max(abs(motion[:,2]))<=REAR_OBSTACLE_SPEED_BOUND
    parts=('fixed_cad','arm_A_with_gripper','arm_B_with_gripper','held_payload')
    ranges={name:{'min_xyz_m':sweep[mask,i,0,:].min(axis=0).tolist(),
                  'max_xyz_m':sweep[mask,i,1,:].max(axis=0).tolist()}
            for i,name in enumerate(parts)}
    # Static nominal CAD reference, not the more conservative rotated world AABB.
    a,b=cad_boxes(w.cad); cad_span=[float(a[:,0].min()),float(b[:,0].max())]
    # Compare like against like: the time-union of rotated world AABBs
    # is conservative for every part, including the fixed body.
    left=min(v['min_xyz_m'][0] for v in ranges.values())
    right=max(v['max_xyz_m'][0] for v in ranges.values())
    fixed=ranges['fixed_cad']
    base_screen_ok=r['safe_holding_interrupt'] and robot_contacts==0 and (rear_robot is None or rear_robot>=STATION_CLEARANCE_M)
    payload_ok=contacts==0 and (rear_payload is None or rear_payload>=STATION_CLEARANCE_M)
    expected=bool(base_screen_ok and abs(r['retreat_measured_m']-.014)<.001 and not np.any(packets[:,3]) and
                  (payload_ok if case=='clear' else not payload_ok))
    contact_indices=audit[audit[:,3]>0,0]
    passed=bool(base_screen_ok and payload_ok)
    dst=out/(case+'.npz');np.savez_compressed(dst,**t,held_aabb_dock_frame=sweep,payload_audit=audit,
         fixed_aabb_frame=w.frame,coverage_packets=packets,independent_robot_rear=physical,rear_actor_motion=motion)
    return r|{'case':case,'controlled_outcome_verified':expected,'tested_clearance_pass':passed,
              'safe_holding_interrupt':passed,
              'tested_payload_clearance_pass':bool(payload_ok),'robot_only_guard_pass':bool(base_screen_ok),
              'held_aabb_dock_frame':ranges,'fixed_CAD_local_x_span_m':cad_span,
              'conservative_held_system_extension_vs_fixed_aabb_x_m':[fixed['min_xyz_m'][0]-left,right-fixed['max_xyz_m'][0]],
              'arm_collision_link_ids':w.arm_links,'retreat_sweep_samples':int(np.count_nonzero(mask)),
              'payload_obstacle_minimum_m':rear_payload,'payload_obstacle_contact_samples':contacts,
              'first_payload_obstacle_contact_substep':int(contact_indices[0]) if contacts else None,
              'last_payload_obstacle_contact_substep':int(contact_indices[-1]) if contacts else None,
              'robot_obstacle_minimum_m':rear_robot,'robot_obstacle_contact_samples':robot_contacts,
              'max_rear_beam_hits':int(packets[:,3].max()),'obstacle_reset':w.birth,'trace_sha256':digest(dst)}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',choices=SCENARIOS,nargs='+',default=SCENARIOS);a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(a.cases)!=len(set(a.cases)):ap.error('Remote CPU only; each case once')
    a.out.mkdir(parents=True); f=a.pilot/'anchored-hybrid.npz'; assert digest(f)==FROZEN_CHECKPOINT
    with np.load(f,allow_pickle=False) as data:weights={k:data[k] for k in data.files}
    asset=a.pilot/'model'/'x7_proxy_gripper.urdf'
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[];p.connect(p.DIRECT)
    try:
        for case in a.cases:
            row=case_run(weights,asset,arms,case,a.out);rows.append(row)
            print(json.dumps({k:row[k] for k in ('case','controlled_outcome_verified','tested_clearance_pass',
                 'retreat_measured_m','payload_obstacle_minimum_m','robot_obstacle_minimum_m','conservative_held_system_extension_vs_fixed_aabb_x_m',
                 'payload_obstacle_contact_samples','max_rear_beam_hits','trace_sha256')},allow_nan=False),flush=True)
    finally:p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'checkpoint_sha256':FROZEN_CHECKPOINT,'mode':'independent swept AABB measurement and payload counterexample',
        'trained':False,'heldout_reused':False,'cases':rows},indent=2,allow_nan=False)+'\n')
    return 0 if all(r['controlled_outcome_verified'] for r in rows) else 1


if __name__=='__main__':raise SystemExit(main())
