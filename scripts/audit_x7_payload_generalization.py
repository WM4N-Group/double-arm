#!/usr/bin/env python3
"""Frozen X7 payload guard: independent straight-retreat cases and physical audit.

The older controller and ray geometry are used unchanged. Actual box pose is
used only by the physical fixture placement and independent audit, never to
correct the pose estimate passed to the controller or to the ray model.
"""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from audit_x7_anchor_tolerance import FROZEN_CHECKPOINT
from audit_x7_certified_retreat import REAR_OBSTACLE_SPEED_BOUND, measure
from audit_x7_payload_range_gate import MODEL_ERROR_M, PayloadGateWorld
from run_x7_mobile_bimanual import ArmIK, GRIP_Z, PARK_Y, execute
from run_x7_turning_carry import HANDLE_HALF_SPAN, r2
from test_x7_dock_guard import STATION_CLEARANCE_M, guard_metrics
from train_x7_anchored_turning_bc import make_action
from train_x7_turning_trajectory_bc import digest

SCENES={
    'east':{'goal':(3.345,.167,26.9),'tote':(.006,-.010,1.10),'bay_y':.687},
    'west':{'goal':(3.323,.106,-27.1),'tote':(-.005,-.011,-1.15),'bay_y':.683},
}
ZERO=(0.,0.,0.)
# Offsets are (x, y, degrees), and obstacle coordinates are tote-local.
# This case panel is fixed before execution; no min-clearance tuning afterward.
MIX={
    'east-clear':('east',ZERO,.020,None),
    'west-clear':('west',ZERO,.020,None),
    'east-main-box':('east',ZERO,.020,(.429,.061,.0955,.034,.010,.014)),
    'west-main-box':('west',ZERO,.021,(-.437,.040,.0955,.034,.009,.016)),
    'east-pose-error':('east',(.002,.004,.40),.020,(.456,.060,.0955,.034,.008,.015)),
    'west-handle-edge':('west',ZERO,.020,(-.599,.081,.023,.036,.008,.013)),
}


def pose_for_control(true,bias,overshoot):
    x,y,deg=true['goal']; p0=np.array([x,y])+r2(math.radians(deg))@[0.,-overshoot]
    return true|{'goal':[float(p0[0]),float(p0[1]),deg], 'tote':np.add(true['tote'],bias).tolist()}


class PanelWorld(PayloadGateWorld):
    def __init__(self,asset,homes,true,estimate,arm_at,span,fixture):
        self.fixture=fixture
        super().__init__(asset,homes,true,estimate,arm_at,span,'payload-edge' if fixture else 'clear')

    def spawn_rear(self,k):
        x,z,face_y,gap,half_x,half_z=self.fixture
        # Align a simulated moving obstacle with the actual box at injection.
        # No access to this reset transform in the sensor or controller.
        pose=self.state_object();yaw=Rotation.from_quat(pose[3:7]).as_euler('zyx')[0]
        half=(half_x,.017,half_z); xy=pose[:2]+r2(yaw)@[x,face_y+gap+half[1]]
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=[.81,.20,.13,1.])
        self.obstacle=p.createMultiBody(baseMass=0,basePosition=[*xy,pose[2]+z],
            baseOrientation=p.getQuaternionFromEuler([0,0,yaw]),linkMasses=[1.],
            linkCollisionShapeIndices=[col],linkVisualShapeIndices=[vis],linkPositions=[[0,0,0]],
            linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],
            linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[0,-1,0]])
        self.birth={'substep':k,'half_extents':half,'position':[*xy,pose[2]+z],
                    'initial_box_gap_m':measure(self.tote,self.obstacle),'initial_robot_gap_m':measure(self.robot,self.obstacle)}


def run_case(weights,asset,arms,name,out):
    scene,bias,overshoot,fixture=MIX[name];true=SCENES[scene];est=pose_for_control(true,bias,overshoot)
    try:
        action,phase,meta=make_action(weights,asset,arms,est)
    except (RuntimeError,ValueError,AssertionError) as err:
        return {'case':name,'executed':False,'error':str(err)}
    n=meta['fitted_phase_durations']; home=[SimpleNamespace(home=action[0,6+7*i:13+7*i]) for i in range(2)]
    w=PanelWorld(asset,home,true,est,sum(n[:8]),n[8],fixture);tr=execute(w,action,phase);w.audit_held(len(w.applied))
    score=guard_metrics(w,tr,true); qa=np.asarray(w.audit); rear=np.asarray(w.obstacle_trace)
    packets=np.asarray(w.load_packets); sensors=np.asarray(w.sensor); motion=np.asarray(w.rear_position)
    m={'z_hold':score['held_box_after_stop'],'guard_latched':score['guard_triggered'],
       'clearance_score_with_box':bool(score['safe_holding_interrupt'] and np.min(qa[:,2])>=STATION_CLEARANCE_M
                                        and np.min(rear[:,0])>=STATION_CLEARANCE_M and not (np.any(qa[:,3]) or np.any(rear[:,1]))),
       'table_min_m':float(np.min(tr['station_clearance'][...,:2])), 'robot_station_contact_substeps':score['robot_station_contact_substeps'],
       'robot_obstacle_min_m':float(np.min(rear[:,0])) if fixture else None,'box_obstacle_min_m':float(np.min(qa[:,2])) if fixture else None,
       'robot_obstacle_contact_substeps':int(np.count_nonzero(rear[:,1])), 'box_obstacle_contact_substeps':int(np.count_nonzero(qa[:,3])),
       'retreat_m':score['retreat_measured_m'],'guard_states':score['guard_states'],'rear_events':w.rear_events,
       'obstacle_reset':w.birth,'obstacle_travel_m':float(np.ptp(motion[:,1])), 'maximum_obstacle_speed_m_s':float(np.max(abs(motion[:,2])))}
    assert m['maximum_obstacle_speed_m_s']<=REAR_OBSTACLE_SPEED_BOUND
    f=out/(name+'.npz');np.savez_compressed(f,**tr,payload_packets=packets,station_packets=np.asarray(w.coverage),
        sensor_telemetry=sensors,robot_rear_audit=rear,payload_audit=qa,rear_actor_motion=motion,
        actual_motor_references=np.asarray(w.applied))
    return m|{'case':name,'executed':True,'trace_sha256':digest(f),'estimated_pose':est,'true_pose':true,
              'is_task_completion':False,'checkpoint_sha256':FROZEN_CHECKPOINT}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('pilot',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--cases',nargs='+',choices=MIX,default=tuple(MIX));a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or len(set(a.cases))!=len(a.cases):ap.error('remote CPU only; no duplicate case')
    a.out.mkdir(parents=True);c=a.pilot/'anchored-hybrid.npz';assert digest(c)==FROZEN_CHECKPOINT
    if MODEL_ERROR_M!=.005:raise ValueError('Frozen measurement margin changed')
    with np.load(c,allow_pickle=False) as f:weights={k:f[k] for k in f.files}
    asset=a.pilot/'model'/'x7_proxy_gripper.urdf'
    arms=[ArmIK(asset,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    rows=[];p.connect(p.DIRECT)
    try:
        for n in a.cases:
            row=run_case(weights,asset,arms,n,a.out);rows.append(row)
            keys=('case','executed','error','z_hold','clearance_score_with_box','retreat_m','box_obstacle_min_m',
                  'robot_obstacle_min_m','box_obstacle_contact_substeps','robot_obstacle_contact_substeps','rear_events','obstacle_travel_m')
            print(json.dumps({k:row[k] for k in keys if k in row},allow_nan=False),flush=True)
    finally:p.disconnect()
    (a.out/'report.json').write_text(json.dumps({'cases':rows,'frozen_margin_m':MODEL_ERROR_M,
        'checkpoint_sha256':FROZEN_CHECKPOINT,'trained':False,'holdout_reused':False},indent=2,allow_nan=False)+'\n')
    # A physical failure is a measured result, not a tool execution failure.
    return 0 if all(r['executed'] for r in rows) else 1


if __name__=='__main__': raise SystemExit(main())
