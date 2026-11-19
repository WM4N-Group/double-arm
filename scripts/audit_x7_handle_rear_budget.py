#!/usr/bin/env python3
"""X7 straight-retreat pilot: handle-facing rays and finite obstacle-travel budget.

The added retreat check uses only the sampled distance lower bound and an
external bound on total actor travel. Physics distances are for scoring.
The fixture moves on its own clock even if the robot refuses to retreat.
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import FROZEN_CHECKPOINT
from audit_x7_certified_retreat import OBSTACLE_TRAVEL,OBSTACLE_VMAX, REAR_OBSTACLE_SPEED_BOUND
from audit_x7_chassis_range_proxy import MOUNT_PAD_M,transform_mounts
from audit_x7_dock_sensor import BIAS,PERIOD,WAVE,GUARD_SCAN_M
from audit_x7_payload_generalization import MIX,SCENES,PanelWorld,pose_for_control
from audit_x7_payload_range_gate import HELD_RISE_M,merged_rear
from audit_x7_rear_coverage import PITCH_M,summarize_cells
from run_x7_mobile_bimanual import ArmIK,GRIP_Z,PARK_Y,SUPPORT_Z,execute
from run_x7_turning_carry import HANDLE_HALF_SPAN,r2
from test_x7_dock_guard import RETREAT_M,STATION_CLEARANCE_M,guard_metrics
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

ACTOR_DELAY_SUBSTEPS=145
APPROACH_BUDGET_M=.028
BUFFER_M=.003
PANEL={
    'clear-with-handles':('west-clear',True,True),
    'main-wall-no-budget':('west-main-box',False,False),
    'main-wall-with-budget':('west-main-box',False,True),
    'handle-no-coverage':('west-handle-edge',False,True),
    'handle-with-coverage':('west-handle-edge',True,True),
    'masked-handle':('west-clear',True,True),
    'clear-no-budget':('west-clear',True,False),
    'handle-no-coverage-no-budget':('west-handle-edge',False,False),
    'handle-with-coverage-no-budget':('west-handle-edge',True,False),
    'masked-handle-no-budget':('west-clear',True,False),
}


def entry_margin():
    return RETREAT_M+STATION_CLEARANCE_M+APPROACH_BUDGET_M+BUFFER_M


def handle_mounts(tote):
    """Nominal rear projection of the two grip blocks/stems, not arm solids."""
    xmin,xmax=HANDLE_HALF_SPAN-.020,HANDLE_HALF_SPAN+.020
    x=np.linspace(xmin,xmax,int(np.ceil((xmax-xmin)/PITCH_M))+1)
    zlo,zhi=.029,.098
    z=np.linspace(zlo,zhi,int(np.ceil((zhi-zlo)/PITCH_M))+1)
    local=[]
    for sign in (-1,1):
        xx,zz=np.meshgrid(sign*x,z)
        local.extend(zip(xx.ravel(),np.full(xx.size,.023+MOUNT_PAD_M),zz.ravel()))
    a=np.array(local);dx,dy,yaw=tote
    a[:,:2]=a[:,:2]@r2(math.radians(yaw)).T+[dx,PARK_Y+dy]
    a[:,2]+=SUPPORT_Z+HELD_RISE_M
    return a


class HandleBudgetWorld(PanelWorld):
    def __init__(self,asset,homes,true,est,arm,span,fixture,case,handles,budget):
        self.panel_case=case;self.handles=handles;self.budget=budget
        self.handle_origins=handle_mounts(est['tote']);self.handle_packets=[];self.entry_packets=[]
        super().__init__(asset,homes,true,est,arm,span,fixture)

    def begin_sensor_substep(self,k):
        super().begin_sensor_substep(k)
        if self.obstacle is not None:
            start=self.birth['substep']+ACTOR_DELAY_SUBSTEPS
            # Supersedes the original fixture driver before this physics step.
            p.setJointMotorControl2(self.obstacle,0,p.POSITION_CONTROL,
                targetPosition=OBSTACLE_TRAVEL if k>=start else 0.,force=12.,maxVelocity=OBSTACLE_VMAX)

    def acquire_packet(self,k,truth_rear):
        raw,pose=super().acquire_packet(k,truth_rear)
        if not self.handles:return raw,pose
        a,b=transform_mounts(self.handle_origins,pose,1)
        f=self.rear_fractions(a,b);valid=np.ones(len(f),dtype=bool)
        if self.panel_case.startswith('masked-handle') and self.latch_k is not None and self.latch_k<k<self.latch_k+80:
            valid[0]=False
        nearest,unknown,hits=summarize_cells(f,valid)
        wave=float(np.sin(.77*(k//PERIOD)+2.24)*WAVE+BIAS)
        before=raw[2];raw[2]=merged_rear(before,nearest,wave)
        self.handle_packets.append([k,len(f),unknown,hits,-1. if nearest is None else nearest,before,raw[2]])
        return raw,pose

    def observe_active_packet(self,k,sample_k,observed,lower,qv):
        super().observe_active_packet(k,sample_k,observed,lower,qv)
        if not self.budget or self.guard.state!='brake' or self.guard.cannot_retreat or self.latch_k is None:
            return
        if sample_k<self.latch_k or lower is None:return
        gap=lower[2]
        self.entry_packets.append([k,sample_k,gap,entry_margin()])
        if gap>=0. and gap<entry_margin():
            self.guard.cannot_retreat=True
            self.rear_events.append({'substep':k,'sample_k':sample_k,'event':'bounded_rear_travel_veto',
                                     'rear_lower_bound_m':float(gap),'required_m':entry_margin()})


def run_case(m,asset,arms,name,out):
    base,handles,budget=PANEL[name];scene,bias,overshoot,fixture=MIX[base]; true=SCENES[scene];obs=pose_for_control(true,bias,overshoot)
    u,ph,meta=make_action(m,asset,arms,obs);n=meta['fitted_phase_durations'];homes=[SimpleNamespace(home=u[0,6+7*i:13+7*i]) for i in range(2)]
    w=HandleBudgetWorld(asset,homes,true,obs,sum(n[:8]),n[8],fixture,name,handles,budget)
    t=execute(w,u,ph);w.audit_held(len(w.applied));r=guard_metrics(w,t,true)
    box=np.asarray(w.audit);robot=np.asarray(w.obstacle_trace);motion=np.asarray(w.rear_position); sensor=np.asarray(w.sensor)
    travel=float(np.ptp(motion[:,1]));peak=float(np.max(abs(motion[:,2])))
    assert travel<=APPROACH_BUDGET_M and peak<=REAR_OBSTACLE_SPEED_BOUND
    box_hits=int(np.count_nonzero(box[:,3]));robot_hits=int(np.count_nonzero(robot[:,1]));
    box_min=float(np.min(box[:,2])) if fixture else None;robot_min=float(np.min(robot[:,0])) if fixture else None
    safe=bool(r['safe_holding_interrupt'] and not (box_hits or robot_hits) and
              (box_min is None or min(box_min,robot_min)>=STATION_CLEARANCE_M))
    f=out/(name+'.npz');np.savez_compressed(f,**t,raw_handle_packets=np.asarray(w.handle_packets),
        budget_entry_packets=np.asarray(w.entry_packets),payload_audit=box,robot_rear_audit=robot,
        rear_obstacle_motion=motion,sensor_telemetry=sensor,actual_motor_references=np.asarray(w.applied))
    return r|{'case':name,'comparison_fixture':base,'handle_coverage_enabled':handles,'bounded_entry_budget_enabled':budget,
        'safe_holding_interrupt':safe,'robot_only_guard_interruption':r['safe_holding_interrupt'],
        'rear_min_box_m':box_min,'rear_min_robot_m':robot_min,'box_contact_samples':box_hits,'robot_contact_samples':robot_hits,
        'obstacle_reset':w.birth,'measured_obstacle_travel_m':travel,'measured_peak_actor_speed_m_s':peak,
        'entry_budget_total_m':entry_margin(),'rear_events':w.rear_events,'full_arm_coverage':False,'trace_sha256':digest(f)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',nargs='+',choices=PANEL,default=tuple(PANEL));a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(set(a.cases))!=len(a.cases):ap.error('remote CPU only; distinct cases')
    a.out.mkdir(parents=True);f=a.pilot/'anchored-hybrid.npz';assert digest(f)==FROZEN_CHECKPOINT
    with np.load(f,allow_pickle=False) as d:m={k:d[k] for k in d.files}
    asset=a.pilot/'model/x7_proxy_gripper.urdf'
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[];p.connect(p.DIRECT)
    try:
        for name in a.cases:
            row=run_case(m,asset,arms,name,a.out);rows.append(row)
            keys=('case','success','safe_holding_interrupt','retreat_measured_m','rear_min_box_m','rear_min_robot_m',
                  'box_contact_samples','measured_obstacle_travel_m','rear_events')
            print(json.dumps({k:row[k] for k in keys},allow_nan=False),flush=True)
    finally:p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'cases':rows,'checkpoint_sha256':FROZEN_CHECKPOINT,'trained':False,
        'holdout_reused':False,'externally_assumed_approach_budget_m':APPROACH_BUDGET_M},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
