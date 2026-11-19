#!/usr/bin/env python3
"""Remote X7 payload-facing range gate for a specific held tote and straight retreat.

Uses the *commanded, calibrated* held-tote transform, motor odometry and
simulated ray fractions, not actual payload or obstacle poses for control.
The actual held box and arms remain independent collision audit subjects.
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
from audit_x7_certified_retreat import REAR_OBSTACLE_SPEED_BOUND
from audit_x7_chassis_range_proxy import MOUNT_PAD_M,RANGE_M,transform_mounts
from audit_x7_dock_sensor import BIAS,GUARD_SCAN_M,PERIOD,WAVE
from audit_x7_held_load_sweep import LoadAuditWorld,digest
from audit_x7_rear_coverage import PITCH_M,summarize_cells
from run_x7_mobile_bimanual import ArmIK,GRIP_Z,PARK_Y,SUPPORT_Z,execute
from run_x7_turning_carry import HANDLE_HALF_SPAN,r2
from test_x7_dock_guard import STATION_CLEARANCE_M,guard_metrics
from train_x7_anchored_turning_bc import make_action

SCENARIOS=('clear','payload-edge','masked-payload')
MODEL_ERROR_M=.005
HELD_RISE_M=.104


def payload_mounts(tote):
    """Rear-wall projection of the main box; excludes arms and outer handles."""
    x=np.linspace(-.50,.50,int(np.ceil(1./PITCH_M))+1)
    z=np.linspace(-.011,.082,int(np.ceil(.093/PITCH_M))+1)
    x,z=np.meshgrid(x,z)
    local=np.column_stack([x.ravel(),np.full(x.size,.0955+MOUNT_PAD_M),z.ravel()])
    dx,dy,deg=tote
    local[:,:2]=local[:,:2]@r2(math.radians(deg)).T+[dx,PARK_Y+dy]
    local[:,2]+=SUPPORT_Z+HELD_RISE_M
    return local


def merged_rear(fixed,payload,wave):
    if fixed<0. or payload is None:return -1.
    # Initially the sampled geometry can differ from the calibrated transform.
    # Penalize only the extra plane; the old range/time/odom margin still applies.
    return min(fixed,min(payload+wave,GUARD_SCAN_M)-MODEL_ERROR_M)


class PayloadGateWorld(LoadAuditWorld):
    def __init__(self,asset,homes,l,obs,arm,span,case):
        self.payload_case=case;self.load_mounts=payload_mounts(obs['tote']);self.load_packets=[]
        self.payload_mask_start=None
        super().__init__(asset,homes,l,obs,arm,span,'payload-edge' if case=='payload-edge' else 'clear')

    def begin_sensor_substep(self,k):
        super().begin_sensor_substep(k)
        if self.payload_case=='masked-payload' and self.latch_k is not None and self.payload_mask_start is None:
            self.payload_mask_start=k

    def payload_fractions(self,mounts,pose):
        a,b=transform_mounts(mounts,pose,1)
        return self.rear_fractions(a,b)

    def acquire_packet(self,k,ignored_truth_rear):
        raw,q=super().acquire_packet(k,ignored_truth_rear)
        fraction=self.payload_fractions(self.load_mounts,q)
        valid=np.ones(len(fraction),dtype=bool)
        if self.payload_mask_start is not None and self.payload_mask_start<=k<self.payload_mask_start+80:
            valid &= abs(self.load_mounts[:,0]-.399)>.012
        gap,unknown,hits=summarize_cells(fraction,valid)
        wave=float(np.sin(.77*(k//PERIOD)+2.24)*WAVE+BIAS)
        fused=merged_rear(raw[2],gap,wave)
        self.load_packets.append([k,len(fraction),unknown,hits,-1. if gap is None else gap,raw[2],fused])
        raw[2]=fused
        return raw,q


def run_case(weights,asset,arms,case,out):
    true=LAYOUTS['north'];obs=perceived_pose(true,*CASES['dock-in-20mm'])
    actions,ph,meta=make_action(weights,asset,arms,obs);n=meta['fitted_phase_durations']
    homes=[SimpleNamespace(home=actions[0,6+7*i:13+7*i]) for i in range(2)]
    w=PayloadGateWorld(asset,homes,true,obs,sum(n[:8]),n[8],case)
    t=execute(w,actions,ph);w.audit_held(len(w.applied)); r=guard_metrics(w,t,true)
    packets=np.asarray(w.load_packets); audit=np.asarray(w.audit); rear=np.asarray(w.obstacle_trace)
    motion=np.asarray(w.rear_position);assert np.max(abs(motion[:,2]))<=REAR_OBSTACLE_SPEED_BOUND
    gap=float(audit[:,2].min()) if w.obstacle is not None else None
    robot_gap=float(rear[:,0].min()) if w.obstacle is not None else None
    hits=int(np.count_nonzero(audit[:,3])); robot_hits=int(np.count_nonzero(rear[:,1]))
    safe=bool(r['safe_holding_interrupt'] and not hits and not robot_hits and
              (gap is None or min(gap,robot_gap)>=STATION_CLEARANCE_M))
    dist=r['retreat_measured_m']; sensors=np.asarray(w.sensor)
    if case=='clear':expected=safe and abs(dist-.014)<.001
    elif case=='payload-edge':expected=(safe and dist<.013 and np.any(packets[:,3]>0) and
          any(e['event']=='rear_dynamic_braking' for e in w.rear_events))
    else:expected=(safe and abs(dist)<1.e-6 and np.any(packets[:,2]>0) and not np.any(sensors[:,2]==2) and
          any(e['event']=='rear_coverage_lost' for e in w.rear_events))
    path=out/(case+'.npz');np.savez_compressed(path,**t,payload_packets=packets,fixed_cad_packets=np.asarray(w.coverage),
      sensor_telemetry=sensors,robot_rear_audit=rear,payload_audit=audit,actual_motor_references=np.asarray(w.applied),rear_actor_motion=motion)
    return r|{'case':case,'expected_outcome_verified':bool(expected),'safe_holding_interrupt':safe,
      'payload_obstacle_minimum_m':gap,'payload_obstacle_contact_substeps':hits,'robot_obstacle_minimum_m':robot_gap,
      'robot_obstacle_contact_substeps':robot_hits,'payload_ray_count':len(w.load_mounts),
      'payload_max_ray_hits':int(packets[:,3].max()),'payload_masked_packets':int(np.count_nonzero(packets[:,2])),
      'obstacle_reset':w.birth,'rear_events':w.rear_events,'trace_sha256':digest(path),
      'coverage':'fixed chassis + calibrated rear wall of main payload; not handles/arms or lateral turning sweep'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',choices=SCENARIOS,nargs='+',default=SCENARIOS);a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(set(a.cases))!=len(a.cases):ap.error('Remote CPU only, distinct cases')
    a.out.mkdir(parents=True);f=a.pilot/'anchored-hybrid.npz';assert digest(f)==FROZEN_CHECKPOINT
    with np.load(f,allow_pickle=False) as d:m={k:d[k] for k in d.files}
    asset=a.pilot/'model'/'x7_proxy_gripper.urdf'
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[];p.connect(p.DIRECT)
    try:
        for case in a.cases:
            row=run_case(m,asset,arms,case,a.out);rows.append(row)
            print(json.dumps({k:row[k] for k in ('case','expected_outcome_verified','safe_holding_interrupt',
              'retreat_measured_m','payload_obstacle_minimum_m','payload_obstacle_contact_substeps',
              'payload_max_ray_hits','payload_masked_packets','rear_events')},allow_nan=False),flush=True)
    finally:p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'checkpoint_sha256':FROZEN_CHECKPOINT,'train_started':False,
      'heldout_reused':False,'cases':rows},indent=2,allow_nan=False)+'\n')
    return 0 if all(row['expected_outcome_verified'] for row in rows) else 1


if __name__=='__main__':raise SystemExit(main())
