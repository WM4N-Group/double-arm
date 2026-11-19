#!/usr/bin/env python3
"""Opt-in X7 dynamics pilot for a dense transverse edge witness.

The sensor uses planned offsets, fixed CAD and sampled ray fractions only.
Fixture truth and distances are confined to the independent audit.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pybullet as p

from audit_x7_certified_retreat import measure
from audit_x7_chassis_range_proxy import MOUNT_PAD_M, RANGE_M, transform_mounts
from audit_x7_handle_rear_budget import (
    ArmIK, FROZEN_CHECKPOINT, GRIP_Z, HANDLE_HALF_SPAN, PARK_Y, digest, run_case,
)
from audit_x7_split_rear_rays import GAP_FIXTURE_TRAVEL_M, SCENARIOS as SPLIT_SCENARIOS, SplitRearWorld
from audit_x7_upper_rear_coverage import cad_boxes
from run_x7_turning_carry import r2

EDGE_BAND_M = .00125
WITNESS_Y_PITCH_M = .001
WITNESS_INNER_TRIM_M = .00025
WITNESS_OUTER_M = .045
FRONT_ECHO_TOL_M = .004
WEST = SPLIT_SCENARIOS['clear-with-handles'][3]
MIRROR = WEST | {'tote':(-WEST['tote'][0],WEST['tote'][1],-WEST['tote'][2])}
FIXTURES = {
    # x/y/z in base frame at spawn, half-lengths below. Both skinny obstacles
    # are separate from the chassis and move on the original independent clock.
    'edge-right-thin': ((.16895,-.2894,1.36175),(.00045,.0014,.006)),
    'edge-left-high-thin': ((-.16895,-.2894,1.40050),(.00045,.0014,.006)),
    'edge-right-cube': ((.17435,-.2894,1.36175),(.006,.006,.006)),
}
CASES = {k:SPLIT_SCENARIOS[k] for k in (
    'clear-with-handles','fresh-clear','main-wall-with-budget','handle-with-coverage',
    'inner-gap-actor','outer-cad-actor','masked-handle','masked-gap','masked-exterior')}
for key,layout in [('mirror-clear',MIRROR),('masked-edge-right',WEST),('masked-edge-left',MIRROR),
                   ('edge-right-thin',WEST),('edge-left-high-thin',MIRROR),('edge-right-cube',WEST)]:
    CASES[key] = (key,True,True,layout,(0.,0.,0.),.020,FIXTURES.get(key))


def witness_plan(mounts, lo, hi, direct):
    """Build only outside-X edge paths with one unambiguous nominal slab."""
    rows = []
    for i in direct:
        x,y,z = mounts[i]
        slabs = np.flatnonzero((lo[:,2]<=z)&(hi[:,2]>=z))
        if len(slabs)!=1:
            continue
        s = slabs[0]
        sign = -1 if x<lo[s,0] else 1 if x>hi[s,0] else 0
        if sign==0:
            continue
        edge = lo[s,0] if sign<0 else hi[s,0]
        offset = sign*(x-edge)
        first = lo[s,1]-y
        if not WITNESS_INNER_TRIM_M<offset<=EDGE_BAND_M or not FRONT_ECHO_TOL_M<first<RANGE_M-FRONT_ECHO_TOL_M:
            continue
        # Verify the whole sampled longitudinal span, not just the expected
        # CAD front face. Slices are the existing main-box Z rows, not a claim
        # of dense Z or continuous full-body coverage.
        ys = np.linspace(y-MOUNT_PAD_M,y+RANGE_M+MOUNT_PAD_M,
                         int(np.ceil((RANGE_M+2*MOUNT_PAD_M)/WITNESS_Y_PITCH_M))+1)
        a = np.column_stack([np.full(len(ys),edge+sign*WITNESS_OUTER_M),ys,np.full(len(ys),z)])
        b = a.copy()
        b[:,0] = edge+sign*WITNESS_INNER_TRIM_M
        rows.append((int(i),sign,float(first),a,b))
    return rows


def witness_fusion(longitudinal, expected, side_fractions):
    """Only a valid complete no-return side path can lift a grazing veto."""
    f = np.asarray(side_fractions)
    if f.ndim!=1 or not len(f) or not np.isfinite(f).all() or np.any((f<0.)|(f>1.)):
        return np.nan, 'missing'
    if np.any(f<1.):
        return np.nan, 'occupied'
    if not np.isfinite(longitudinal) or not 0.<=longitudinal<=1.:
        return np.nan,'missing'
    if longitudinal<1. and abs(RANGE_M*longitudinal-expected)<=FRONT_ECHO_TOL_M:
        return 1.,'verified_side_clear'
    return float(longitudinal),'preserved'


def world_transform(mounts,pose):
    a = mounts.copy()
    a[:,:2] = a[:,:2] @ r2(pose[2]).T + pose[:2]
    return a


class EdgeWorld(SplitRearWorld):
    def __init__(self,*a):
        self.witness_trace = []
        super().__init__(*a)
        lo,hi=cad_boxes(self.cad)
        (direct,_,_),_=self.split_profiles['main_box']
        self.witness_profiles=witness_plan(self.load_mounts,lo,hi,direct)

    def fixture_travel(self):
        return GAP_FIXTURE_TRAVEL_M if self.panel_case in FIXTURES else super().fixture_travel()

    def spawn_rear(self,k):
        if self.panel_case not in FIXTURES:
            return super().spawn_rear(k)
        (x,y,z),half=self.fixture
        q=self.qv(self.base)[:,0]
        xy=q[:2]+r2(q[2])@[x,y]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        self.obstacle=p.createMultiBody(baseMass=0,basePosition=[*xy,z],
            baseOrientation=p.getQuaternionFromEuler([0,0,q[2]]),linkMasses=[1.],
            linkCollisionShapeIndices=[col],linkVisualShapeIndices=[-1],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth={'substep':k,'half_extents':half,'position':[*xy,z],
                    'initial_box_gap_m':measure(self.tote,self.obstacle),'initial_robot_gap_m':measure(self.robot,self.obstacle)}

    def payload_fractions(self,mounts,pose):
        result=super().payload_fractions(mounts,pose)
        if mounts is not self.load_mounts or self.latch_k is None or not self.witness_profiles:
            return result
        a=world_transform(np.concatenate([p[3] for p in self.witness_profiles]),pose)
        b=world_transform(np.concatenate([p[4] for p in self.witness_profiles]),pose)
        samples=self.rear_fractions(a,b)
        pos=0
        for i,sign,y,orig,_ in self.witness_profiles:
            f=samples[pos:pos+len(orig)].copy()
            pos+=len(orig)
            side='left' if sign<0 else 'right'
            if self.panel_case==f'masked-edge-{side}' and self.latch_k<self.sample_k<self.latch_k+80:
                f[0]=np.nan
            base=float(result[i])
            value,status=witness_fusion(base,y,f)
            result[i]=value
            self.witness_trace.append([self.sample_k,i,sign,self.load_mounts[i,2],len(orig),
                int(np.count_nonzero(np.isfinite(f)&(f<1.))),
                ['preserved','verified_side_clear','missing','occupied'].index(status),
                base if np.isfinite(base) else -1.,value if np.isfinite(value) else -1.])
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(set(a.cases))!=len(a.cases):
        ap.error('remote CPU only; distinct cases')
    f=a.pilot/'anchored-hybrid.npz'
    if digest(f)!=FROZEN_CHECKPOINT:
        raise ValueError('Unrecognized checkpoint')
    a.out.mkdir(parents=True)
    with np.load(f,allow_pickle=False) as ck:m={k:ck[k] for k in ck.files}
    asset=a.pilot/'model/x7_proxy_gripper.urdf'
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[]
    p.connect(p.DIRECT)
    try:
        for name in a.cases:
            worlds=[]
            def factory(*args):
                w=EdgeWorld(*args);worlds.append(w);return w
            row=run_case(m,asset,arms,name,a.out,world_type=factory,scenario=CASES[name])
            w=worlds[0];trace=np.asarray(w.witness_trace).reshape(-1,9)
            f=a.out/(name+'-witness.npz')
            np.savez_compressed(f,witness=trace,fields=['sample_k','mount_index','side_sign','z_m','beam_count',
                'short_hits','code_0_preserve_1_cleared_2_missing_3_occupied','longitudinal_fraction','fused_fraction'])
            counts=[int(np.count_nonzero(trace[:,6]==k)) for k in range(4)]
            row.update(witness_sha256=digest(f),witness_mounts=len(w.witness_profiles),
                edge_preserved=counts[0],edge_released_by_witness=counts[1],edge_unknown=counts[2],edge_occupied=counts[3],
                hit_body_ids_used_for_control=False,
                guard_distance_source='virtual rays with odometry and age correction; simulator contacts for audit only')
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('case','success','safe_holding_interrupt','retreat_measured_m',
                'rear_min_box_m','rear_min_robot_m','box_contact_samples','robot_contact_samples','witness_mounts',
                'edge_released_by_witness','edge_unknown','edge_occupied','rear_events')},allow_nan=False),flush=True)
    finally:p.disconnect()
    (a.out/'report.json').write_text(json.dumps(dict(cases=rows,checkpoint_sha256=FROZEN_CHECKPOINT,
        bench_only=True,trained=False,heldout_angle_reused=False,y_witness_pitch_m=WITNESS_Y_PITCH_M),
        indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
