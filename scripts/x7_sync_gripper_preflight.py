#!/usr/bin/env python3
"""X7 v2: single jaw actuator with an internal rack synchronizer; remote CPU only.

Uses the v1 tabletop and motor interface for comparison. The only allowed
user constraint is the robot-to-itself gear coupling its two jaw coordinates.
The block is moved only by contact after reset. Never re-label v1 data as v2.
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p

from x7_pick_place_pilot import CONTROL_HZ, HZ, Kinematics, World, build_model, plan, render_review, run, scenario

MASTER_FORCE_N = 28.0
SYNC_FORCE_N = 60.0
SYNC_MAX_OFFSET_M = .0005
GEAR_RATIO = -1.0


class SynchronizedWorld(World):
    """Prismatic jaws have opposite local axes but equal opening coordinates."""
    def setup_jaw_actuation(self):
        self.diffs = []
        self.gear_count = 0
        self.max_center_offset_m = 0.0
        self.max_loaded_center_offset_m = 0.0
        a, b = self.jaw
        # Bullet's default velocity servo on the follower must not resist the rack.
        p.setJointMotorControl2(self.robot, b, p.VELOCITY_CONTROL, targetVelocity=0., force=0.)
        self.rack = p.createConstraint(self.robot, a, self.robot, b, p.JOINT_GEAR, [1., 0., 0.], [0.]*3, [0.]*3)
        p.changeConstraint(self.rack, gearRatio=GEAR_RATIO, relativePositionTarget=0., maxForce=SYNC_FORCE_N, erp=.55)

    def drive_jaws(self, jaw_target):
        p.setJointMotorControl2(self.robot, self.jaw[0], p.POSITION_CONTROL, targetPosition=jaw_target,
                                force=MASTER_FORCE_N, positionGain=.22, velocityGain=1.)

    def verify_rack(self):
        if p.getNumConstraints() != 1 or p.getConstraintUniqueId(0) != self.rack:
            raise RuntimeError('Unexpected, missing, or substituted user constraint')
        info = p.getConstraintInfo(self.rack)
        if info[:5] != (self.robot, self.jaw[0], self.robot, self.jaw[1], p.JOINT_GEAR):
            raise RuntimeError(f'Constraint does not connect the exact two jaw slides: {info[:5]}')
        if info[10:15] != (SYNC_FORCE_N, GEAR_RATIO, 0, 0., .55):
            raise RuntimeError(f'Unexpected rack parameters: {info[10:15]}')

    def audit_constraints(self):
        self.verify_rack()
        q0, q1 = [s[0] for s in p.getJointStates(self.robot, self.jaw)]
        center = .5*(q0-q1)
        if not np.isfinite(center):
            raise RuntimeError('Non-finite simulated jaw state')
        self.max_center_offset_m = max(self.max_center_offset_m, abs(center))
        if any(c[9]>.06 for c in p.getContactPoints(self.robot, self.block)):
            self.max_loaded_center_offset_m = max(self.max_loaded_center_offset_m, abs(center))
        self.diffs.append(center)
        self.gear_count += 1


def v2_model(source, out):
    model = build_model(source, out)
    tree = ET.parse(model)
    tree.getroot().set('name', 'x7_duo_sync_gripper_v2')
    limit = tree.find("joint[@name='proxy_finger_a_slide']/limit")
    limit.set('effort', str(MASTER_FORCE_N))
    tree.write(model, encoding='utf-8', xml_declaration=True)
    return model


def checked_run(world, action, phase, video=None):
    trace, summary = run(world, action, phase, video=video)
    substeps = len(action)*HZ//CONTROL_HZ
    if world.gear_count != substeps:
        raise RuntimeError(f'Missing substep mechanical audit, expected {substeps} got {world.gear_count}')
    sync_pass = world.max_center_offset_m <= SYNC_MAX_OFFSET_M
    legacy_result = summary.pop('success')
    summary.pop('constraints_seen')  # In v2 there IS one internal drivetrain constraint.
    summary |= dict(success=legacy_result and sync_pass, task_gate_without_symmetry=legacy_result,
                    symmetry_pass=bool(sync_pass), max_center_offset_m=world.max_center_offset_m,
                    max_loaded_center_offset_m=world.max_loaded_center_offset_m,
                    gear_constraint_verified_substeps=world.gear_count,
                    gear_scope='robot jaw A to robot jaw B only', object_assistance_constraint_substeps=0,
                    source='synchronized gripper v2')
    trace['center_offset_m_at_substep'] = np.asarray(world.diffs)
    return trace, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('asset',type=Path,help='X7 fixed-base derivative URDF from run_x7_fixed_base_demo.py')
    ap.add_argument('out',type=Path)
    ap.add_argument('--tape',type=Path,help='Optionally replay a previous action tape on the new mechanics (not a policy run)')
    ap.add_argument('--seed',type=int,default=700)
    ap.add_argument('--offset-mm',type=float,nargs='+',default=[0.,-8.,8.],help='Move block along the jaw closing axis at reset, without changing planned grasp target')
    ap.add_argument('--review',action='store_true')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(not np.isfinite(o) or abs(o)>15 for o in args.offset_mm):
        ap.error('CPU-only; set CUDA_VISIBLE_DEVICES to empty. Offsets must be finite and <=15 mm.')
    if args.review and len(args.offset_mm)!=1:
        ap.error('--review requires one case')
    args.out.mkdir(parents=True)
    model=v2_model(args.asset,args.out/'model')
    kin=Kinematics(model)
    if args.tape:
        with np.load(args.tape,allow_pickle=False) as f:
            initial,goal,action,phase=[f[k] for k in ('start','goal','action','phase')]
    else:
        initial,goal=scenario(args.seed)
        action,phase=plan(kin,initial,goal)
    if (action.ndim != 2 or action.shape[1] != 8 or phase.shape != (len(action),)
        or initial.shape != (3,) or goal.shape != (3,) or not np.isfinite(action).all()
        or not np.isfinite(np.r_[initial,goal]).all() or np.any(abs(action[:,:7])>3.14)
        or np.any((action[:,7]<0.) | (action[:,7]>.036))):
        raise ValueError('Malformed trajectory or invalid joint target')
    manifest=[]
    p.connect(p.DIRECT)
    try:
        for offset in args.offset_mm:
            x=initial+np.array([offset/1000.,0.,0.])
            w=SynchronizedWorld(model,kin,x,goal)
            w.verify_rack()
            label = 'RECORDED COMMANDS' if args.tape else 'SCRIPTED EXPERT'
            enc,video=render_review(args.out/'synchronized-review.mp4', title=f'X7 SYNC V2  /  {label}',
                                    subtitle='One driven jaw + mechanical follower  /  no object attachment',
                                    show_jaw_center=True) if args.review else (None,None)
            try:
                data,report=checked_run(w,action,phase,video)
            finally:
                if enc:
                    enc.stdin.close()
                    if enc.wait()!=0:
                        raise RuntimeError('Video encoding failed')
            f=args.out/f'case-{offset:+.3f}mm.npz'
            np.savez_compressed(f,**data,start=x,goal=goal)
            report |= dict(reset_block_offset_mm=offset,trace_sha256=sha256(f.read_bytes()).hexdigest(), trace=f.name)
            manifest.append(report)
            print(json.dumps(report),flush=True)
    finally:
        p.disconnect()
    m=dict(model='x7_proxy_gripper_sync_v2', jaw_drive='master A only; mirrored follower B through gear',
           source='recorded-command mechanics intervention; not a learned closed-loop rollout' if args.tape else 'scripted expert; not learned',
           original_tape_sha256=sha256(args.tape.read_bytes()).hexdigest() if args.tape else None,
           derived_model_sha256=sha256(model.read_bytes()).hexdigest(), symmetry_pass_threshold_m=SYNC_MAX_OFFSET_M,
           cases=manifest, passed=all(r['success'] for r in manifest))
    (args.out/'report.json').write_text(json.dumps(m,indent=2)+'\n')
    return 0 if m['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
