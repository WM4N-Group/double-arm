#!/usr/bin/env python3
"""Fault-inject a sampled proximity stream into the X7 dock guard in Bullet.

No real sensing: archive delay/noise/occlusion separately from true collision,
and bound robot motion between samples with simulator link transforms.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p

from audit_x7_anchor_tolerance import CASES, FROZEN_CHECKPOINT, LAYOUTS, perceived_pose
from generate_x7_pickup_anchor import PickupWorld
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, HZ, PARK_Y, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, dock_pose, r2, review_turn
from test_x7_dock_guard import (CODE, GUARD_SCAN_M, RETREAT_M, STABLE_SUBSTEPS,
                               GuardWorld, STATION_CLEARANCE_M, guard_metrics)
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

PERIOD=8             # 30 Hz with the 240 Hz engine
LAG=16               # deterministic 66.7 ms transport delay
MAX_AGE=30           # maximum usable age 125 ms
BIAS=.0011
WAVE=.0012
POS_ERROR_BOUND=.0024
REAR_MARGIN=.006
SCENARIOS={
    'nominal-delay-noise': {'bias':'exact'},
    'in20-delay-noise-raw': {'bias':'dock-in-20mm','raw':True},
    'in20-delay-noise-robust': {'bias':'dock-in-20mm'},
    'dock-occlusion': {'bias':'exact','occluded':True},
    'rear-obstacle': {'bias':'dock-in-20mm','rear':True},
    'nominal-link-association': {'bias':'exact','linkwise':True},
    'in20-link-association': {'bias':'dock-in-20mm','linkwise':True},
    'rear-obstacle-link-association': {'bias':'dock-in-20mm','rear':True,'linkwise':True},
}
BASE_SCENARIOS=('nominal-delay-noise','in20-delay-noise-raw','in20-delay-noise-robust','dock-occlusion','rear-obstacle')


def link_motion(origin, current, radius):
    """Upper bound for the change of any simulated robot surface point."""
    delta=np.linalg.norm(origin[:,:3]-current[:,:3],axis=1)
    # Quaternion sign is immaterial. The chord formula uses the shortest arc.
    dot=np.clip(abs(np.sum(origin[:,3:]*current[:,3:],axis=1)),0.,1.)
    return delta+2.*radius*np.sqrt(1.-dot**2)


def movement_bound(origin,current,radius):
    return float(np.max(link_motion(origin,current,radius)))


def correct_linkwise(raw,origin,now,radius):
    displacement=link_motion(origin,now,radius)
    return np.min(raw-POS_ERROR_BOUND-displacement[None,:],axis=1), float(np.max(displacement))


def correct_sample(raw, origin, now, radius, raw_ablation):
    movement=movement_bound(origin,now,radius)
    # When no hit was within the radius, the reported range is a lower
    # bound on the geometric return before the positive sensor error.
    return (raw.copy() if raw_ablation else raw-POS_ERROR_BOUND-movement),movement


def measure(robot, body):
    return min((float(q[8]) for q in p.getClosestPoints(robot,body,GUARD_SCAN_M)),default=GUARD_SCAN_M)


class SampledWorld(GuardWorld):
    def __init__(self,asset,homes,true,angle,arm_at,approach_length,config):
        super().__init__(asset,homes,true,angle,arm_at,-1)
        self.config=config; self.packets=deque();self.delivered=None;self.sensor=[];self.latch_k=None
        self.obstacle=None;self.birth=None;self.obstacle_trace=[];self.rear_events=[]
        self.link_ids=list(range(p.getNumJoints(self.robot)))
        live=self.link_poses(); radius=[]
        for i,pose in zip(self.link_ids,live):
            a,b=[np.array(x) for x in p.getAABB(self.robot,i)]; ext=np.maximum(abs(a-pose[:3]),abs(b-pose[:3]))
            radius.append(float(np.linalg.norm(ext)))
        self.radii=np.array(radius)
        self.occlude_from=8*(arm_at+int(.70*approach_length)) if config.get('occluded') else -10000
        self.occlude_to=self.occlude_from+80

    def link_poses(self):
        return np.array([(*s[4],*s[5]) for s in p.getLinkStates(self.robot,self.link_ids,computeForwardKinematics=True)])

    def measure_links(self,body):
        distances=np.full(len(self.link_ids),GUARD_SCAN_M)
        if body is not None:
            for hit in p.getClosestPoints(self.robot,body,GUARD_SCAN_M):
                idx=hit[3]
                if idx not in self.link_ids:
                    raise RuntimeError("Distance cannot be associated to one verified link")
                distances[idx]=min(distances[idx],float(hit[8]))
        return distances

    def spawn_rear(self,k):
        # Inject a real obstacle once after the forward-approach latch; it is
        # not seen by the guard until a new delayed rear packet is delivered.
        # Static after it appears. No pose writes to robot or carried object.
        q=self.qv(self.base)[:,0]; theta=float(q[2]); xy=q[:2]+r2(theta)@[0.,.238+.009+.055]
        half=[.13,.055,.14]; shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[.75,.23,.18,1.])
        self.obstacle=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,baseVisualShapeIndex=vis,
                       basePosition=[*xy,.27],baseOrientation=p.getQuaternionFromEuler([0.,0.,theta]))
        self.birth={'substep':k,'world_center_m':[*xy,.27],'half_extent_m':half,'yaw_rad':theta,
                    'true_initial_clearance_m':measure(self.robot,self.obstacle)}

    def begin_sensor_substep(self,k):
        """Optional physical fixture driver, separate from sensing and control."""

    def rear_bound(self,lower,age):
        """Optionally account for independently specified obstacle motion bounds."""
        return lower

    def sensor_sample_valid(self,k):
        return not self.occlude_from<=k<self.occlude_to

    def drive(self,a):
        if not self.armed:return PickupWorld.drive(self,a)
        k=len(self.applied); self.begin_sensor_substep(k); active=self.tick>=self.arm_stage or self.guard.state!='pass'; observed=None; low=None
        age=-1;truth_rear=GUARD_SCAN_M;rear_hits=0; trigger=-1.; move=-1.; sample_k=-1; e=a
        if self.obstacle is not None:
            truth_rear=measure(self.robot,self.obstacle);rear_hits=len(p.getContactPoints(self.robot,self.obstacle))
        if self.tick>=self.arm_stage-10:
            if k%PERIOD==0:
                valid=self.sensor_sample_valid(k)
                n=np.sin(0.77*(k//PERIOD)+np.array([.43,1.55,2.24]))*WAVE+BIAS
                if self.config.get("linkwise"):
                    raw=np.minimum(np.stack([self.measure_links(z) for z in [*self.tables,self.obstacle]])+n[:,None],GUARD_SCAN_M)
                else:
                    raw=np.minimum(np.array([*(measure(self.robot,z) for z in self.tables),truth_rear])+n,GUARD_SCAN_M)
                self.packets.append((k,raw if valid else None,self.link_poses()))
            while self.packets and k-self.packets[0][0]>=LAG:
                self.delivered=self.packets.popleft()
            if self.delivered is not None:
                sample_k,raw,sampled_links=self.delivered;age=k-sample_k
                if raw is not None and age<=MAX_AGE:
                    if self.config.get('linkwise'):
                        observed=raw.min(axis=1);low,move=correct_linkwise(raw,sampled_links,self.link_poses(),self.radii)
                    else:
                        observed=raw;low,move=correct_sample(raw,sampled_links,self.link_poses(),self.radii,self.config.get('raw',False))
                    low=self.rear_bound(low,age)
            if active:
                g=self.guard; before=g.state;q=self.qv(self.base)
                # Enforce a rear reading definitively acquired after latching
                # prior to automatic retreat. Only the measured lower bound,
                # not the actual obstacle geometry/event, decides motion.
                if before=='brake' and g.still>=STABLE_SUBSTEPS-1 and not g.cannot_retreat:
                    if low is None or sample_k<self.latch_k or low[2]<RETREAT_M+REAR_MARGIN:
                        g.cannot_retreat=True;self.rear_events.append({'substep':k,'event':'retreat_veto',
                            'sample_k':sample_k,'rear_lower_bound_m':None if low is None else float(low[2])})
                if before=='retreat' and (low is None or low[2]<STATION_CLEARANCE_M):
                    g.hold=q[:,0].copy();g.cannot_retreat=True;g.switch(k,'hold')
                    self.rear_events.append({'substep':k,'event':'retreat_aborted'})
                # No collision-contact oracle is passed as backup to the guard.
                e,trigger=g.update(k,q,None if low is None else list(map(float,low[:2])),[0,0],a)
                if before=='pass' and g.state!='pass':
                    self.latch_k=k
                    if self.config.get('rear'):
                        self.spawn_rear(k)
        mark=lambda a:([-1.]*3 if a is None else a.tolist())
        self.applied.append(e.copy());self.sensor.append([k,int(active),CODE[self.guard.state],sample_k,age,trigger,
                    *mark(observed),*mark(low),move]);self.obstacle_trace.append([truth_rear,rear_hits]);
        # Compatibility with the core guard metric: this is the distance it consumed.
        self.timelog.append([k,int(active),CODE[self.guard.state],trigger,
                             *([-1.,-1.] if low is None else list(low[:2])),int(low is not None)])
        return PickupWorld.drive(self,e)

    def step(self,a):
        # Include a truly simulated rear obstacle collision occurring at the
        # final post-integration sample. Earlier samples are audited at the
        # next substep, so a substep hit cannot vanish between control ticks.
        result=super().step(a)
        if self.obstacle is not None:
            self.obstacle_trace[-1]=[min(self.obstacle_trace[-1][0],measure(self.robot,self.obstacle)),
                                     max(self.obstacle_trace[-1][1],len(p.getContactPoints(self.robot,self.obstacle)))]
        return result


def evaluate(weights,asset,arms,case,l,out,archived):
    cfg=SCENARIOS[case];obs=perceived_pose(l,*CASES[cfg['bias']]);u,ph,r=make_action(weights,asset,arms,obs);n=r['fitted_phase_durations']
    homes=[SimpleNamespace(home=u[0,6+i*7:13+i*7]) for i in range(2)]
    world=SampledWorld(asset,homes,l,dock_pose(obs)[2],sum(n[:8]),n[8],cfg)
    video=None; kwargs={}
    if archived is not None:
        k=int(archived['guard_states'][0]['substep']//8); begin,end=max(0,k-90),min(len(u),k+180)
        video=out/(case+'.mp4')
        def window(path,mode,label):
            enc,draw=review_turn(path,mode,label)
            def tick(i,ph,*a):
                if begin<=i<end:draw(i,ph if world.guard.state=='pass' else 5,*a)
            return enc,tick
        label='RAW DELAYED SENSOR / MARGIN FAILED' if cfg.get('raw') else 'SAMPLED DOCK GUARD / RETREAT VETO' if cfg.get('rear') else 'SAMPLED DOCK GUARD / STOP'
        kwargs=dict(movie=video,reviewer=window,review_label=label)
    t=execute(world,u,ph,**kwargs);score=guard_metrics(world,t,l)
    cmd=np.asarray(world.applied); sensor=np.asarray(world.sensor);rear=np.asarray(world.obstacle_trace)
    assert cmd.shape==(len(u)*8,22) and sensor.shape==(len(u)*8,13)
    hits=int(np.count_nonzero(rear[:,1])); min_rear=float(rear[:,0].min()) if world.obstacle is not None else None
    score|={'case':case,'sensor_model':cfg,'safe_holding_interrupt':bool(score['safe_holding_interrupt'] and hits==0),
            'extra_obstacle_contacts':hits,'extra_obstacle_min_clearance_m':min_rear,'obstacle_injection':world.birth,
            'rear_events':world.rear_events,'max_observation_age_substeps':int(sensor[(sensor[:,1]>0)&(sensor[:,5]>0),4].max()),
            'physical_execution':True,'base_checkpoint_sha256':FROZEN_CHECKPOINT}
    t|={'sensor_telemetry':sensor,'actual_motor_references':cmd,'extra_obstacle_true_samples':rear}
    path=out/(case+'.npz');np.savez_compressed(path,**t);score['trace_sha256']=digest(path)
    if archived is not None:
        assert archived['trace_sha256']==score['trace_sha256'];score['video_sha256']=digest(video)
    return score


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',choices=SCENARIOS,nargs='+',default=BASE_SCENARIOS);ap.add_argument('--review-archive',type=Path)
    a=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or len(set(a.cases))!=len(a.cases) or a.review_archive and len(a.cases)!=1:
        ap.error('Remote CPU only; distinct cases; one case per review')
    a.out.mkdir(parents=True);path=a.pilot/'anchored-hybrid.npz';assert digest(path)==FROZEN_CHECKPOINT
    asset=a.pilot/'model'/'x7_proxy_gripper.urdf'
    with np.load(path,allow_pickle=False) as f:weights={k:f[k] for k in f.files}
    archived=None
    if a.review_archive:
        archived=next(r for r in json.loads(a.review_archive.read_text())['cases'] if r['case']==a.cases[0])
        assert archived['base_checkpoint_sha256']==FROZEN_CHECKPOINT and archived['guard_triggered']
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    results=[];p.connect(p.DIRECT)
    try:
        for name in a.cases:
            r=evaluate(weights,asset,arms,name,LAYOUTS['north'],a.out,archived);results.append(r)
            fields=('case','success','guard_reason','safe_holding_interrupt','station_min_clearance_m',
                    'robot_station_contact_substeps','extra_obstacle_min_clearance_m','extra_obstacle_contacts','rear_events','retreat_measured_m')
            print(json.dumps({k:r[k] for k in fields},allow_nan=False),flush=True)
    finally:
        p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'model':'frozen X7 anchored guard; simulated sensor and obstacle only',
        'sample_period_substeps':PERIOD,'transport_delay_substeps':LAG,'ttl_substeps':MAX_AGE,'positive_error_bound_m':POS_ERROR_BOUND,
        'bias_and_noise_m':[BIAS,WAVE],'checkpoint_sha256':FROZEN_CHECKPOINT,'cases':results},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
