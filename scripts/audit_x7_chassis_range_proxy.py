#!/usr/bin/env python3
"""X7 simulated chassis-only range head experiment, without hit or link identities.

The controller consumes fractions from virtual rays and planar odometry. Robot
station and obstacle proximity/contact queries remain independent scoring
only. A deliberate between-beam blind target is a coverage counterexample.
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import CASES,FROZEN_CHECKPOINT,LAYOUTS,perceived_pose
from audit_x7_certified_retreat import (OBSTACLE_TRAVEL,OBSTACLE_VMAX,REAR_OBSTACLE_SPEED_BOUND,RetreatWorld,
                                       correct_moving_rear)
from audit_x7_dock_sensor import BIAS,WAVE,PERIOD,POS_ERROR_BOUND,measure
from run_x7_mobile_bimanual import ArmIK,GRIP_Z,HZ,PARK_Y,execute
from run_x7_turning_carry import HANDLE_HALF_SPAN,dock_pose,r2,review_turn
from test_x7_dock_guard import GUARD_SCAN_M,guard_metrics
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

RANGE_M=.32
MOUNT_PAD_M=.0025
RADIUS_M=.83
PROXY_GUARD_M=.0014
FRONT_X=np.linspace(-.16,.16,9)
REAR_X=np.arange(-.28,.281,.07)
FRONT_Z=(1.204,1.225,1.246)
REAR_Z=(.22,.27,.35)
BASE_NAMES=('nominal','overtravel-clear','rear-wide','rear-narrow-blind','near-occlusion')
NAMES=(*BASE_NAMES,'rear-narrow-dense')
DENSE_REAR_X=np.linspace(-.28,.28,71)


def make_mounts(cad,rear_x=REAR_X):
    # Same fixed geometry manifest as the chassis model; no simulator IDs or
    # queried body poses are used to choose a point on the chassis template.
    boxes=[]
    for row in cad['parts_in_cad_frame']:
        a,b,c=row['center'];dx,dy,dz=row['size'];center=np.array([-c,b,1.77+a]);half=np.array([dz,dy,dx])/2
        boxes.append((center-half,center+half))
    sensors=[]
    for sign,xs,zs in [(-1,FRONT_X,FRONT_Z),(1,rear_x,REAR_Z)]:
        group=[]
        for z in zs:
            for x in xs:
                ends=[(a[1] if sign<0 else b[1])*sign for a,b in boxes if a[0]<=x<=b[0] and a[2]<=z<=b[2]]
                if not ends:raise ValueError('Configured beam is not supported by chassis template')
                group.append([x,sign*(max(ends)+MOUNT_PAD_M),z])
        sensors.append(np.asarray(group))
    return sensors


def transform_mounts(mounts,q,sign):
    d=np.asarray(mounts).copy();d[:,:2]=d[:,:2]@r2(q[2]).T+q[:2];v=np.r_[r2(q[2])@[0.,sign*RANGE_M],0.]
    return d,d+v


def gaps_from_fractions(frac):
    f=np.asarray(frac)
    if not np.isfinite(f).all() or np.any((f<0.)|(f>1.)):
        raise ValueError('Missing/invalid simulated range fraction')
    return f*RANGE_M+MOUNT_PAD_M


def odom_correction(q0,q1):
    # Bound only chassis motion in this range proxy. Cannot cover moving arms.
    dyaw=(q1[2]-q0[2]+np.pi)%(2*np.pi)-np.pi
    return float(np.linalg.norm(q0[:2]-q1[:2])+RADIUS_M*abs(dyaw))


class RangeWorld(RetreatWorld):
    def __init__(self,asset,homes,l,obs,arm,span,name):
        self.range_case=name
        self.mounts=make_mounts(json.loads((asset.parent/'base-envelope.json').read_text()),
                                DENSE_REAR_X if name=='rear-narrow-dense' else REAR_X)
        self.beamtrace=[]
        super().__init__(asset,homes,l,obs,arm,span,'moving-rear' if name in ('rear-wide','rear-narrow-blind','rear-narrow-dense') else 'clear')
        if name=='near-occlusion':self.occlude_from=8*(arm+int(.72*span));self.occlude_to=self.occlude_from+80

    def acquire_packet(self,k,ignored_truth_rear):
        q=self.qv(self.base)[:,0];values=[]
        for i,(sign,loc) in enumerate(zip((-1,1),self.mounts)):
            a,b=transform_mounts(loc,q,sign)
            # Only [2], the optical-style hit fraction. The hit object/link IDs,
            # point and surface normal from Bullet are not read or used.
            gaps=gaps_from_fractions([result[2] for result in p.rayTestBatch(a.tolist(),b.tolist())])
            n=float(np.sin(.77*(k//PERIOD)+(.43,2.24)[i])*WAVE+BIAS)
            noisy=np.minimum(gaps+n,GUARD_SCAN_M);values.append(noisy)
        self.beamtrace.append(np.concatenate(([k], *values))); return np.r_[GUARD_SCAN_M,values[0].min(),values[1].min()],q.copy()

    def lower_from_packet(self,raw,sampled_state,age):
        x=odom_correction(sampled_state,self.qv(self.base)[:,0]); low=np.array(raw)
        # Channel zero denotes no alternate forward surface. The physical
        # evaluator still audits BOTH stations independently at every substep.
        low[1:]=low[1:]*math.cos(math.radians(3.))-x-POS_ERROR_BOUND-PROXY_GUARD_M
        low[0]=GUARD_SCAN_M-x
        return raw,correct_moving_rear(low,age),x

    def spawn_rear(self,k):
        # For the narrow counterexample, the 12 mm width sits between 70 mm
        # rear beams. Both rear objects still exist as real collision bodies.
        narrow=self.range_case.startswith('rear-narrow-'); x=.035 if narrow else 0.
        q=self.qv(self.base)[:,0];theta=float(q[2]); xy=q[:2]+r2(theta)@[x,.238+.044+.055]
        half=[.006 if narrow else .12,.055,.14]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half);vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[.78,.24,.13,1.])
        self.obstacle=p.createMultiBody(baseMass=0,basePosition=[*xy,.27],baseOrientation=p.getQuaternionFromEuler([0,0,theta]),
            linkMasses=[1.],linkCollisionShapeIndices=[col],linkVisualShapeIndices=[vis],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth={'injected_substep':k,'simulated_body_local_offset_m':x,'half_extents_m':half,
                    'independent_initial_clearance_m':measure(self.robot,self.obstacle)}


def run_case(weights,asset,arms,name,out,archive):
    l=LAYOUTS['north'];obs=perceived_pose(l,*CASES['exact' if name=='nominal' else 'dock-in-20mm']); u,ph,m=make_action(weights,asset,arms,obs)
    n=m['fitted_phase_durations']; homes=[SimpleNamespace(home=u[0,6+7*a:13+7*a]) for a in range(2)]
    w=RangeWorld(asset,homes,l,obs,sum(n[:8]),n[8],name);kwargs={}
    if archive:
        k=archive['guard_states'][0]['substep']//8; start,end=max(0,k-60),min(len(u),k+210)
        def window(path,mode,label):
            enc,draw=review_turn(path,mode,label)
            def conditional(i,ph,*a):
                if start<=i<end:draw(i,ph if w.guard.state=='pass' else 5,*a)
            return enc,conditional
        kwargs=dict(movie=out/(name+'.mp4'),reviewer=window,review_label='SPARSE RANGE GRID / MARGIN FAILED' if not archive['safe_holding_interrupt'] else 'RANGE GRID / COVERAGE TEST')
    t=execute(w,u,ph,**kwargs);r=guard_metrics(w,t,l); rear=np.asarray(w.obstacle_trace); obs_motion=np.asarray(w.rear_position)
    min_rear=float(rear[:,0].min()) if w.obstacle is not None else None; hits=int(np.count_nonzero(rear[:,1])); ref=np.asarray(w.applied)
    if w.obstacle is not None:assert np.max(abs(obs_motion[:,2]))<=REAR_OBSTACLE_SPEED_BOUND
    # Keep the unmitigated outcome visible; failing geometry isn't made safe
    # by a nominal-looking guard state or by absence of actual contact.
    safe=bool(r['safe_holding_interrupt'] and hits==0 and (min_rear is None or min_rear>=.005))
    r|={'case':name,'guard_station_only_pass':r['safe_holding_interrupt'],'safe_holding_interrupt':safe,
        'extra_obstacle_min_clearance_m':min_rear,'extra_obstacle_contacts':hits,'extra_obstacle':w.birth,
        'rear_events':w.rear_events,'obstacle_travel_m':float(np.ptp(obs_motion[:,1])),
        'full_geometry_sensor_claim':False,'checkpoint_sha256':FROZEN_CHECKPOINT}
    t|={'beam_gap_packets':np.asarray(w.beamtrace),'range_telemetry':np.asarray(w.sensor),
        'real_motor_references':ref,'independent_rear_distance_and_contacts':rear,'rear_obstacle_motion':obs_motion}
    p0=out/(name+'.npz');np.savez_compressed(p0,**t); r['trace_sha256']=digest(p0)
    if archive:assert r['trace_sha256']==archive['trace_sha256'];r['video_sha256']=digest(out/(name+'.mp4'))
    return r


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',choices=NAMES,nargs='+',default=list(BASE_NAMES));ap.add_argument('--review-archive',type=Path);a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.review_archive and len(a.cases)!=1:ap.error('Remote CPU only; one case per review')
    a.out.mkdir(parents=True);p1=a.pilot/'anchored-hybrid.npz';assert digest(p1)==FROZEN_CHECKPOINT
    with np.load(p1,allow_pickle=False) as f:weights={k:f[k] for k in f.files}
    asset=a.pilot/'model/x7_proxy_gripper.urdf';arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    archived=None if not a.review_archive else next(r for r in json.loads(a.review_archive.read_text())['cases'] if r['case']==a.cases[0])
    rows=[];p.connect(p.DIRECT)
    try:
        for name in a.cases:
            r=run_case(weights,asset,arms,name,a.out,archived);rows.append(r)
            keys=('case','success','safe_holding_interrupt','guard_reason','station_min_clearance_m','extra_obstacle_min_clearance_m','extra_obstacle_contacts','rear_events','retreat_measured_m')
            print(json.dumps({k:r[k] for k in keys},allow_nan=False),flush=True)
    finally:
        p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'cases':rows,'mode':'range fractions + base encoder pose; chassis only; no hit IDs used in policy',
        'heldout_reused':False,'weights_updated':False,'checkpoint_sha256':FROZEN_CHECKPOINT,'sparse_rear_beam_columns_m':REAR_X.tolist(),'dense_rear_beam_columns_m':DENSE_REAR_X.tolist()},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
