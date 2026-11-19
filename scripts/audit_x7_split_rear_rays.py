#!/usr/bin/env python3
"""Experimental complementary rear rays; fixed CAD only, no hit-ID filtering."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from audit_x7_certified_retreat import measure

from audit_x7_chassis_range_proxy import MOUNT_PAD_M, RANGE_M, transform_mounts
from audit_x7_handle_rear_budget import (
    ArmIK, FROZEN_CHECKPOINT, GRIP_Z, HANDLE_HALF_SPAN, MIX, PARK_Y, SCENES,
    HandleBudgetWorld, digest, run_case,
)
from audit_x7_upper_rear_coverage import cad_boxes
from run_x7_turning_carry import r2

# Small unobserved strips at the collision-proxy faces. This is an ideal-CAD
# experiment, not certification for small/thin obstacles or calibration error.
FACE_TRIM_M = .0005
GAP_FIXTURE_TRAVEL_M = .0075
FRESH = {'goal': (3.337, .137, -28.2), 'tote': (-.004, -.010, -.8), 'bay_y': .685}
ZERO = (0., 0., 0.)
GAP_FIXTURE = (-.120, -.003, .0955, .016, .010, .010, .004)
OUTER_FIXTURE = (-.120, 1.355, .063, .035, .010, .010, .009)


def legacy(key, handles=True):
    scene, bias, overshoot, fixture = MIX[key]
    return key, handles, True, SCENES[scene], bias, overshoot, fixture


SCENARIOS = {
    'clear-with-handles': legacy('west-clear'),
    'main-wall-with-budget': legacy('west-main-box', handles=False),
    'handle-with-coverage': legacy('west-handle-edge'),
    'masked-handle': legacy('west-clear'),
    'inner-gap-actor': ('west-gap', True, True, SCENES['west'], ZERO, .020, GAP_FIXTURE),
    'outer-cad-actor': ('west-cad-outside', True, True, SCENES['west'], ZERO, .020, OUTER_FIXTURE),
    'masked-gap': legacy('west-clear'),
    'masked-exterior': legacy('west-clear'),
    'fresh-clear': ('fresh-clear', True, True, FRESH, ZERO, .020, None),
    'fresh-wall': ('fresh-wall', True, True, FRESH, ZERO, .021, MIX['west-main-box'][3]),
    'fresh-handle': ('fresh-handle', True, True, FRESH, ZERO, .020, MIX['west-handle-edge'][3]),
}


def partition_mounts(mounts, lo, hi):
    """+Y rays in the odom frame, split only around one continuous CAD solid.

    Return direct and certified-shape indices, CAD entrance/exit Y and unknown
    indices. If a cell starts inside a solid, has a cavity between two slabs,
    do not substitute an exterior ray. The exterior emitter has its own range.
    """
    direct, split, faces, unknown = [], [], [], []
    for idx, (x, y, z) in enumerate(mounts):
        projected = (lo[:,0] <= x) & (hi[:,0] >= x) & (lo[:,2] <= z) & (hi[:,2] >= z) & (hi[:,1] >= y)
        intervals = sorted(zip(lo[projected,1], hi[projected,1]))
        if not intervals:
            direct.append(idx)
            continue
        enter, leave = intervals[0]
        disconnected = False
        for a, b in intervals[1:]:
            if a > leave:
                disconnected = True
            leave = max(leave, b)
        if enter-y >= RANGE_M:
            direct.append(idx)
        elif disconnected or enter-y <= FACE_TRIM_M:
            unknown.append(idx)
        else:
            split.append(idx)
            faces.append((enter, leave))
    return tuple(np.asarray(v,dtype=int) for v in (direct,split,unknown)), np.asarray(faces).reshape(-1,2)


def combine_split(prefix_fraction, exterior_fraction, length):
    """Metric-equivalent fractions; missing either observation stays unknown."""
    prefix, exterior, length = np.broadcast_arrays(prefix_fraction, exterior_fraction, length)
    valid = (np.isfinite(prefix) & np.isfinite(exterior) & (prefix >= 0.) & (prefix <= 1.) &
             (exterior >= 0.) & (exterior <= 1.) & (length > 0.) & (length <= RANGE_M))
    prefix_equiv = np.where(prefix < 1., prefix * length / RANGE_M, 1.)
    return np.where(valid, np.minimum(prefix_equiv, exterior), np.nan)


class SplitRearWorld(HandleBudgetWorld):
    def __init__(self, *args):
        self.split_packets = []
        self.split_profiles = {}
        self.sample_k = -1
        super().__init__(*args)
        lo, hi = cad_boxes(self.cad)
        for label, mounts in [('main_box', self.load_mounts), ('handles', self.handle_origins)]:
            indices, faces = partition_mounts(mounts, lo, hi)
            self.split_profiles[label] = indices, faces

    def fixture_travel(self):
        return GAP_FIXTURE_TRAVEL_M if self.panel_case == 'inner-gap-actor' else super().fixture_travel()

    def spawn_rear(self,k):
        if self.panel_case not in ('inner-gap-actor','outer-cad-actor'):
            return super().spawn_rear(k)
        # Fixture reset poses only; the sensor and guard never read them.
        x,z,face_y,gap,hx,hz,hy = self.fixture
        if self.panel_case == 'inner-gap-actor':
            pose = self.state_object()
            yaw = Rotation.from_quat(pose[3:7]).as_euler('zyx')[0]
            origin = pose[:2]
            z += pose[2]
        else:
            q = self.qv(self.base)[:,0]
            origin, yaw = q[:2], q[2]
        xy = origin+r2(yaw)@[x,face_y+gap+hy]
        half = (hx,hy,hz)
        col = p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        self.obstacle = p.createMultiBody(baseMass=0,basePosition=[*xy,z],
            baseOrientation=p.getQuaternionFromEuler([0,0,yaw]),linkMasses=[1.],
            linkCollisionShapeIndices=[col],linkVisualShapeIndices=[-1],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth = {'substep':k,'half_extents':half,'position':[*xy,z],
                      'initial_box_gap_m':measure(self.tote,self.obstacle),'initial_robot_gap_m':measure(self.robot,self.obstacle)}

    def acquire_packet(self,k,truth_rear):
        self.sample_k = k
        return super().acquire_packet(k,truth_rear)

    def payload_fractions(self,mounts,pose):
        channel = 'main_box' if mounts is self.load_mounts else 'handles'
        (direct, split, unknown), faces = self.split_profiles[channel]
        out = np.full(len(mounts),np.nan)
        if len(direct):
            a,b = transform_mounts(mounts[direct],pose,1)
            out[direct] = self.rear_fractions(a,b)
        n_prefix = n_exterior = 0
        if len(split):
            origins = mounts[split]
            a,b = transform_mounts(origins,pose,1)
            length = faces[:,0]-FACE_TRIM_M-origins[:,1]
            f = self.rear_fractions(a,a+(b-a)*(length[:,None]/RANGE_M))
            exterior = origins.copy()
            exterior[:,1] = faces[:,1]+MOUNT_PAD_M
            a,b = transform_mounts(exterior,pose,1)
            g = self.rear_fractions(a,b)
            fail = (channel == 'main_box' and self.latch_k is not None and
                    self.latch_k < self.sample_k < self.latch_k+80)
            if fail and self.panel_case == 'masked-gap':
                f[0] = np.nan
            if fail and self.panel_case == 'masked-exterior':
                g[0] = np.nan
            n_prefix = int(np.count_nonzero(np.isfinite(f) & (f<1.)))
            n_exterior = int(np.count_nonzero(np.isfinite(g) & (g<1.)))
            out[split] = combine_split(f,g,length)
        self.split_packets.append([self.sample_k,int(channel=='handles'),len(direct),len(split),len(unknown),
                                   n_prefix,n_exterior,int(np.count_nonzero(~np.isfinite(out)))])
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pilot',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('--cases',nargs='+',choices=SCENARIOS,default=list(SCENARIOS))
    args = ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != '' or len(set(args.cases))!=len(args.cases):
        ap.error('remote CPU only; distinct cases')
    args.out.mkdir(parents=True)
    f = args.pilot/'anchored-hybrid.npz'
    if digest(f)!=FROZEN_CHECKPOINT:
        raise ValueError('Unrecognized checkpoint')
    with np.load(f,allow_pickle=False) as src:
        m = {k:src[k] for k in src.files}
    asset = args.pilot/'model/x7_proxy_gripper.urdf'
    arms = [ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[]
    p.connect(p.DIRECT)
    try:
        for case in args.cases:
            worlds=[]

            def factory(*a):
                world = SplitRearWorld(*a)
                worlds.append(world)
                return world

            row=run_case(m,asset,arms,case,args.out,world_type=factory,scenario=SCENARIOS[case])
            w=worlds[0]
            packets=np.asarray(w.split_packets,dtype=int)
            f=args.out/(case+'-split.npz')
            np.savez_compressed(f,packets=packets,fields=['sample_k','handle_channel','direct','split','uncovered',
                                                        'prefix_hits','exterior_hits','unknown'])
            after=packets[packets[:,0]>w.latch_k] if w.latch_k is not None else packets[:0]
            row.update(split_trace_sha256=digest(f),prefix_hit_cells_after_latch=int(after[:,5].sum()),
                       exterior_hit_cells_after_latch=int(after[:,6].sum()),unknown_cells_after_latch=int(after[:,7].sum()),
                       calibrated_cad_only=True,hit_body_ids_used_for_control=False,
                       guard_distance_source='sampled virtual rays with odometry and age correction; contact/proximity audit only')
            rows.append(row)
            keys=('case','success','safe_holding_interrupt','retreat_measured_m','rear_min_box_m','rear_min_robot_m',
                  'box_contact_samples','robot_contact_samples','prefix_hit_cells_after_latch','exterior_hit_cells_after_latch',
                  'unknown_cells_after_latch','rear_events')
            print(json.dumps({k:row[k] for k in keys},allow_nan=False),flush=True)
    finally:
        p.disconnect()
    (args.out/'report.json').write_text(json.dumps(dict(cases=rows,checkpoint_sha256=FROZEN_CHECKPOINT,
        face_trim_m=FACE_TRIM_M,heldout_angle_reused=False,trained=False),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
