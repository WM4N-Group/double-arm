#!/usr/bin/env python3
"""Compare a lower-chassis curtain with an all-fixed-CAD rear ray profile.

Physics-only experiment; articulated arms, carried load, lateral swept
volume, and real sensor detection are outside the coverage claim.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import CASES, FROZEN_CHECKPOINT, LAYOUTS, perceived_pose
from audit_x7_certified_retreat import REAR_OBSTACLE_SPEED_BOUND, measure
from audit_x7_chassis_range_proxy import MOUNT_PAD_M
from audit_x7_rear_coverage import CoverageWorld, PITCH_M
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, r2
from test_x7_dock_guard import STATION_CLEARANCE_M, guard_metrics
from train_x7_anchored_turning_bc import make_action
from x7_station_collision import CAD_MOUNT_Z

CASES_TO_RUN = ('lower-only-counterexample', 'fixed-cad-upper-block', 'fixed-cad-clear', 'fixed-cad-masked')
RAY_BATCH = 8000
OBSTACLE_Z = 1.46
OBSTACLE_HALF_EXTENTS = (.006,.055,.016)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def cad_boxes(cad):
    rows=[]
    for r in cad['parts_in_cad_frame']:
        a,b,c=r['center'];dx,dy,dz=r['size']
        center=np.array([-c,b,CAD_MOUNT_Z+a]);half=np.array([dz,dy,dx])/2.
        rows.append((center-half,center+half))
    lo,hi=np.array(rows).transpose((1,0,2))
    if not len(lo) or not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(hi<=lo):
        raise ValueError('Malformed fixed CAD envelopes')
    return lo,hi


def envelope_grid(cad, pitch=PITCH_M):
    # Use each CAD box's own closed X/Z interval. At points shared by two
    # slabs emit the ray behind the outermost rear surface, not inside a slab.
    lo,hi=cad_boxes(cad)
    planes=[]
    for lower,upper in zip(lo,hi):
        xz=[np.linspace(lower[k],upper[k],int(np.ceil((upper[k]-lower[k])/pitch))+1) for k in (0,2)]
        x,z=np.meshgrid(*xz);planes.append(np.column_stack([x.ravel(),z.ravel()]))
    xz=np.unique(np.vstack(planes),axis=0)
    mask=np.all((lo[None,:,[0,2]]<=xz[:,None,:]+1.e-12)&(hi[None,:,[0,2]]>=xz[:,None,:]-1.e-12),axis=2)
    aft=np.max(np.where(mask,hi[None,:,1],-np.inf),axis=1)
    if not np.isfinite(aft).all(): raise RuntimeError('Ray missing its fixed CAD envelope')
    return np.column_stack([xz[:,0],aft+MOUNT_PAD_M,xz[:,1]])


class ExtendedWorld(CoverageWorld):
    def __init__(self,asset,homes,l,obs,arm,span,name):
        self.extension=name
        self.cad=json.loads((asset.parent/'base-envelope.json').read_text())
        super().__init__(asset,homes,l,obs,arm,span,
            {'fixed-cad-clear':'clear','fixed-cad-masked':'partial-before-retreat'}.get(name,'narrow'))

    def make_curtain(self,cad):
        return super().make_curtain(cad) if self.extension=='lower-only-counterexample' else envelope_grid(cad)

    def rear_fractions(self,a,b):
        entries=[]
        for k in range(0,len(a),RAY_BATCH):
            start,end=a[k:k+RAY_BATCH],b[k:k+RAY_BATCH]
            batch=p.rayTestBatch(start.tolist(),end.tolist(),numThreads=1)
            if len(batch)!=len(start):
                entries.extend([np.nan]*len(start))  # no silent drop or max-range substitute
            else:
                entries.extend(r[2] for r in batch)
        return np.array(entries)

    def spawn_rear(self,k):
        q=self.qv(self.base)[:,0]; yaw=float(q[2]);lo,hi=cad_boxes(self.cad)
        # Validate the obstacle height has positive overlap only with an
        # upper fixed-body slab, not the lower chassis sampled in the control.
        span=(lo[:,2]<OBSTACLE_Z+OBSTACLE_HALF_EXTENTS[2]) & (hi[:,2]>OBSTACLE_Z-OBSTACLE_HALF_EXTENTS[2])
        assert span.any() and not span[lo[:,2].argmin()]
        aft=float(hi[span,1].max()); xy=q[:2]+r2(yaw)@[.035,aft+.044+OBSTACLE_HALF_EXTENTS[1]]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=OBSTACLE_HALF_EXTENTS)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=OBSTACLE_HALF_EXTENTS,rgbaColor=[.76,.22,.15,1.])
        self.obstacle=p.createMultiBody(baseMass=0,basePosition=[*xy,OBSTACLE_Z],
            baseOrientation=p.getQuaternionFromEuler([0,0,yaw]),linkMasses=[1.],
            linkCollisionShapeIndices=[col],linkVisualShapeIndices=[vis],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],linkParentIndices=[0],
            linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth={'substep':k,'position':[*xy,OBSTACLE_Z],'yaw':yaw,'half_extents':OBSTACLE_HALF_EXTENTS,
                    'initial_clearance_m':measure(self.robot,self.obstacle)}


def run(model,asset,arms,case,out):
    geom=LAYOUTS['north'];obs=perceived_pose(geom,*CASES['dock-in-20mm'])
    actions,ph,meta=make_action(model,asset,arms,obs);n=meta['fitted_phase_durations']
    starts=[SimpleNamespace(home=actions[0,6+7*i:13+7*i]) for i in range(2)]
    world=ExtendedWorld(asset,starts,geom,obs,sum(n[:8]),n[8],case)
    trace=execute(world,actions,ph);stats=guard_metrics(world,trace,geom)
    sensors=np.asarray(world.sensor);packets=np.asarray(world.coverage);rear=np.asarray(world.obstacle_trace)
    motion=np.asarray(world.rear_position); cmd=np.asarray(world.applied)
    assert np.max(abs(motion[:,2]))<=REAR_OBSTACLE_SPEED_BOUND
    rear_min=float(rear[:,0].min()) if world.obstacle is not None else None
    # Qualifying the expected counterexample is NOT declaring the control safe.
    audit_ok=bool(stats['robot_station_clearance_pass'] and not np.any(rear[:,1]) and
                  (rear_min is None or rear_min>=STATION_CLEARANCE_M))
    safe=bool(audit_ok and stats['safe_holding_interrupt']);dist=stats['retreat_measured_m']
    if case=='lower-only-counterexample':
        expected=(not safe and rear_min is not None and rear_min<STATION_CLEARANCE_M and
                  np.all(packets[:,3]==0) and abs(dist-.014)<.001)
    elif case=='fixed-cad-upper-block':
        expected=(safe and 0.<dist<.013 and any(e['event']=='rear_dynamic_braking' for e in world.rear_events))
    elif case=='fixed-cad-clear':
        expected=safe and abs(dist-.014)<.001
    else:
        expected=(safe and abs(dist)<1.e-6 and np.any(packets[:,2]>0) and not np.any(sensors[:,2]==2) and
                  any(e['event']=='rear_coverage_lost' for e in world.rear_events))
    path=out/(case+'.npz')
    np.savez_compressed(path,**trace,sensor_telemetry=sensors,coverage_packets=packets,rear_geometry_audit=rear,
                        actual_motor_references=cmd,rear_obstacle_motion=motion)
    return stats|{'case':case,'expected_outcome_verified':bool(expected),'safe_holding_interrupt':safe,
         'rear_min_actual_m':rear_min,'rear_contact_substeps':int(np.count_nonzero(rear[:,1])),
         'rear_events':world.rear_events,'ray_count':len(world.curtain),'obstacle_reset_spec':world.birth,
         'obstacle_travel_m':float(np.ptp(motion[:,1])),'maximum_obstacle_speed_m_s':float(np.max(abs(motion[:,2]))),
         'trace_sha256':digest(path)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',choices=CASES_TO_RUN,nargs='+',default=CASES_TO_RUN);a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(a.cases)!=len(set(a.cases)):ap.error('Remote CPU only; no repeat cases')
    a.out.mkdir(parents=True);weights=a.pilot/'anchored-hybrid.npz';assert digest(weights)==FROZEN_CHECKPOINT
    with np.load(weights,allow_pickle=False) as f:model={k:f[k] for k in f.files}
    asset=a.pilot/'model'/'x7_proxy_gripper.urdf'
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[]; p.connect(p.DIRECT)
    try:
        for case in a.cases:
            row=run(model,asset,arms,case,a.out);rows.append(row)
            print(json.dumps({k:row[k] for k in ('case','expected_outcome_verified','safe_holding_interrupt',
               'retreat_measured_m','station_min_clearance_m','rear_contact_substeps','rear_min_actual_m','rear_events','ray_count')},allow_nan=False),flush=True)
    finally:p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'scope':'fixed CAD rear envelope only; straight retreat; simulation experiment',
        'checkpoint_sha256':FROZEN_CHECKPOINT,'train_started':False,'heldout_reused':False,'cases':rows},indent=2,allow_nan=False)+'\n')
    return 0 if all(r['expected_outcome_verified'] for r in rows) else 1


if __name__=='__main__':raise SystemExit(main())
