#!/usr/bin/env python3
"""Test a conservative front/rear distance envelope during X7 retreat.

Uses the same delayed link-associated simulator sensing as the previous pilot.
Nothing here certifies physical hardware. The moving rear box is a separate
simulation actor, driven once by a prismatic motor, not robot/object writes.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import CASES, FROZEN_CHECKPOINT, LAYOUTS, perceived_pose
from audit_x7_dock_sensor import MAX_AGE, SampledWorld, measure
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, HZ, PARK_Y, execute, smooth
from run_x7_turning_carry import HANDLE_HALF_SPAN, dock_pose, r2, review_turn
from test_x7_dock_guard import DockGuard, GUARD_SCAN_M, RETREAT_M, STATION_CLEARANCE_M, clearance_trigger, guard_metrics
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

DURATION_STEPS=240
REAR_OBSTACLE_SPEED_BOUND=.043
OBSTACLE_TRAVEL=.026
OBSTACLE_VMAX=.036
SCENARIOS=('clear','moving-rear','occluded-during-retreat')


def correct_moving_rear(lower,age):
    result=np.asarray(lower).copy(); result[2]-=max(0.,age/HZ)*REAR_OBSTACLE_SPEED_BOUND
    return result


class EnvelopeGuard(DockGuard):
    def update(self,k,qv,dist,hits,a):
        if self.state!='retreat':
            return super().update(k,qv,dist,hits,a)
        # Sampled lower bounds need not be monotonic. Retire the former
        # "below historical peak" test, not the absolute clearance gate.
        valid=bool(dist is not None and len(dist)==2 and np.isfinite(dist).all())
        dmin=float(min(dist)) if valid else -1.
        if not valid or sum(hits) or dmin<=STATION_CLEARANCE_M+.001:
            self.hold=qv[:,0].copy();self.cannot_retreat=True
            self.events.append({'substep':k,'type':'retreat_stop','reason':'front_bound_or_missing_reading'})
            self.switch(k,'hold')
            return super().update(k,qv,dist,hits,a)
        self.s+=1; u=min(1.,self.s/DURATION_STEPS)
        move=np.r_[RETREAT_M*r2(self.angle)[:,1],0.]
        b=self.held_cmd.copy(); b[:3]=self.retreat_origin+move*smooth(u)
        b[3:6]=move*30*u*u*(1-u)*(1-u)*HZ/DURATION_STEPS
        if self.s>=DURATION_STEPS:
            self.hold=b[:3].copy();self.switch(k,'hold')
        return b,clearance_trigger(qv[:,1],self.angle)


class RetreatWorld(SampledWorld):
    def __init__(self,asset,homes,true,obs,arm,span,case):
        self.case=case;self.started=None;self.drop_start=None;self.rear_position=[];self.last_retreat_commanded=None
        super().__init__(asset,homes,true,dock_pose(obs)[2],arm,span,dict(rear=case=='moving-rear',linkwise=True))
        self.guard=EnvelopeGuard(dock_pose(obs)[2])

    def sensor_sample_valid(self,k):
        reject=self.case=='occluded-during-retreat' and self.drop_start is not None and self.drop_start<=k<self.drop_start+80
        return not reject and super().sensor_sample_valid(k)

    def rear_bound(self,lower,age):
        # No use of the actual simulated obstacle displacement in the guard.
        # One configured speed bound is applied for every case, occupied or not.
        return correct_moving_rear(lower,age)

    def spawn_rear(self,k):
        q=self.qv(self.base)[:,0];theta=float(q[2]); xy=q[:2]+r2(theta)@[0.,.238+.044+.055]
        half=[.12,.055,.14]; body=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[.78,.22,.15,1.])
        self.obstacle=p.createMultiBody(baseMass=0,basePosition=[*xy,.27],
            baseOrientation=p.getQuaternionFromEuler([0,0,theta]),linkMasses=[1.],
            linkCollisionShapeIndices=[body],linkVisualShapeIndices=[vis],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],
            linkInertialFrameOrientations=[[0,0,0,1]],linkParentIndices=[0],
            linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth={'substep':k,'body_model':'simulated rear obstacle injected on brake; physically slides in the subsequent retreat',
                    'max_planned_travel_m':OBSTACLE_TRAVEL,'max_motor_velocity_m_s':OBSTACLE_VMAX,
                    'true_initial_clearance_m':measure(self.robot,self.obstacle)}

    def begin_sensor_substep(self,k):
        if self.guard.state=='retreat' and self.started is None:
            self.started=k
            if self.case=='occluded-during-retreat':self.drop_start=k+58
        if self.obstacle is not None:
            pos,vel=p.getJointState(self.obstacle,0)[:2]
            ref=0. if self.started is None else OBSTACLE_TRAVEL
            p.setJointMotorControl2(self.obstacle,0,p.POSITION_CONTROL,targetPosition=ref,
                                   force=12.,maxVelocity=OBSTACLE_VMAX)
            self.rear_position.append([k,pos,vel])
        else:self.rear_position.append([k,0.,0.])

    def drive(self,a):
        if self.armed and self.guard.state=='retreat' and self.delivered is not None:
            # Original guard uses only 5 mm for the rear substep veto. Add a
            # separate braking buffer from the previous *valid stale sample*,
            # updated with current odometry *before* applying another command.
            t,raw,old=self.delivered; age=len(self.applied)-t
            if raw is not None:
                if age<=MAX_AGE:
                    _,low,_=self.lower_from_packet(raw,old,age)
                    q=self.qv(self.base); v=max(0.,float(q[:2,1]@r2(self.guard.angle)[:,1]))
                    brake=STATION_CLEARANCE_M+.0012+(v+REAR_OBSTACLE_SPEED_BOUND)*.035+v*v/(2*.60)
                    if low[2]<=brake:
                        self.guard.hold=q[:,0].copy(); self.guard.cannot_retreat=True
                        self.rear_events.append({'substep':len(self.applied),'event':'rear_dynamic_braking',
                                                 'rear_lower_bound_m':float(low[2]),'braking_threshold_m':brake})
                        self.guard.switch(len(self.applied),'hold')
        return super().drive(a)


def scenario(weights,asset,arms,case,out,archive=None):
    l=LAYOUTS['north']; obs=perceived_pose(l,*CASES['dock-in-20mm']);u,phase,r=make_action(weights,asset,arms,obs);dur=r['fitted_phase_durations']
    homes=[SimpleNamespace(home=u[0,6+i*7:13+i*7]) for i in range(2)]
    w=RetreatWorld(asset,homes,l,obs,sum(dur[:8]),dur[8],case);kwargs={}
    if archive:
        k=archive['guard_states'][0]['substep']//8; a=max(0,k-60);b=min(len(u),k+210)
        def review_window(path,mode,label):
            enc,draw=review_turn(path,mode,label)
            def render(i,ph,*x):
                if a<=i<b:draw(i,ph if w.guard.state=='pass' else 5,*x)
            return enc,render
        kwargs=dict(movie=out/(case+'.mp4'),reviewer=review_window,review_label='RETREAT SENSOR TEST / HOLDING BOX')
    tr=execute(w,u,phase,**kwargs); result=guard_metrics(w,tr,l)
    rear=np.asarray(w.obstacle_trace);extra=np.asarray(w.rear_position);ph=w.guard
    move=float(result['retreat_measured_m']); body_clear=float(rear[:,0].min()) if w.obstacle is not None else None
    obstacle_speed=float(np.max(abs(extra[:,2]))); obstacle_range=float(np.ptp(extra[:,1])); contact=int(np.count_nonzero(rear[:,1]))
    assert obstacle_speed <= REAR_OBSTACLE_SPEED_BOUND
    complete=abs(move-RETREAT_M)<.001 and ph.s>=DURATION_STEPS
    clearance_ok=result['robot_station_clearance_pass'] and (body_clear is None or body_clear>=.005) and contact==0
    if case=='clear': scenario_ok=clearance_ok and complete
    elif case=='moving-rear':
        scenario_ok=clearance_ok and .001<move<.013 and obstacle_range>.015 and any(
            e['event']=='rear_dynamic_braking' and e['substep']>w.started for e in w.rear_events)
    else:
        sensor=np.asarray(w.sensor)
        missing_after_start=bool(w.drop_start is not None and np.any(
            (sensor[:,3]>=w.drop_start)&(sensor[:,6]<0.)&(sensor[:,2]==3)))
        scenario_ok=clearance_ok and .0001<move<.013 and ph.state=='hold' and missing_after_start
    result|={'case':case,'scenario_guard_pass':bool(scenario_ok and result['safe_holding_interrupt']),
             'retreat_completed':bool(complete),'extra_obstacle_min_clearance_m':body_clear,'extra_obstacle_contact_substeps':contact,
             'extra_obstacle_measured_travel_m':obstacle_range,'extra_obstacle_peak_measured_speed_m_s':obstacle_speed,
             'rear_brake_events':w.rear_events,'retreat_start_substep':w.started,'occlusion_injected_at_substep':w.drop_start,
             'never_count_retreat_as_carry_task_completion':True,'checkpoint_sha256':FROZEN_CHECKPOINT}
    tr|={'raw_and_corrected_proximity_by_substep':np.asarray(w.sensor),'actual_motor_references':np.asarray(w.applied),
          'rear_obstacle_clearance_contacts':rear,'rear_obstacle_motion':extra}
    dest=out/(case+'.npz');np.savez_compressed(dest,**tr);result['trace_sha256']=digest(dest)
    if archive: assert archive['trace_sha256']==result['trace_sha256'];result['video_sha256']=digest(out/(case+'.mp4'))
    return result


def main():
    p0=argparse.ArgumentParser(description=__doc__);p0.add_argument('pilot',type=Path);p0.add_argument('out',type=Path)
    p0.add_argument('--cases',nargs='+',choices=SCENARIOS,default=list(SCENARIOS));p0.add_argument('--review-archive',type=Path);a=p0.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or a.review_archive and len(a.cases)!=1:p0.error('Remote CPU only; one archived case per review')
    a.out.mkdir(parents=True);path=a.pilot/'anchored-hybrid.npz';assert digest(path)==FROZEN_CHECKPOINT
    asset=a.pilot/'model'/'x7_proxy_gripper.urdf'
    with np.load(path,allow_pickle=False) as f:model={k:f[k] for k in f.files}
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    archive=None if not a.review_archive else next(r for r in json.loads(a.review_archive.read_text())['cases'] if r['case']==a.cases[0])
    results=[];p.connect(p.DIRECT)
    try:
        for case in a.cases:
            r=scenario(model,asset,arms,case,a.out,archive);results.append(r)
            fields=('case','scenario_guard_pass','retreat_completed','retreat_measured_m','station_min_clearance_m','extra_obstacle_min_clearance_m',
                    'extra_obstacle_measured_travel_m','extra_obstacle_peak_measured_speed_m_s','extra_obstacle_contact_substeps','guard_states','rear_brake_events')
            print(json.dumps({k:r[k] for k in fields},allow_nan=False),flush=True)
    finally:
        p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'checkpoint_sha256':FROZEN_CHECKPOINT,'cases':results,'retrained':False,'heldout_reused':False,
      'guard':'absolute clearance, delayed simulator link-associated measurements, configured obstacle speed ceiling'},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
