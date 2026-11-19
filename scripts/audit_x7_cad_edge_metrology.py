#!/usr/bin/env python3
"""Offline metrology for a grazing X7 rear return. Never drives the robot."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pybullet as p

from audit_x7_upper_rear_coverage import cad_boxes, digest
from run_x7_mobile_bimanual import AXES
from run_x7_turning_carry import r2

# Predeclared scan: includes both sides of nominal X edge, the archived pose,
# and independent yaw/translation to separate local geometry from world pose.
OFFSETS_M = np.array([-3., -2., -1., -.5, 0., .25, .5, .75, 1., 1.25, 1.5,
                      1.75, 2., 2.5, 3., 4., 5., 6.]) / 1000
FORWARD_M = (.130, .320, .550, .750)
PROBE_RADIUS_M = .0001
Y_VIEWS_M = (-.300, -.275, -.250, -.220, -.170, -.100, -.040, .010, .050, .080)
GRID_PITCH_M = .008


def frame(xyz, q):
    result = np.asarray(xyz, dtype=float).copy()
    result[..., :2] = result[..., :2] @ r2(q[2]).T + q[:2]
    return result


def unframe(xyz, q):
    result = np.asarray(xyz, dtype=float).copy()
    result[..., :2] = (result[..., :2] - q[:2]) @ r2(q[2])
    return result


def first_nominal_entry(a, b, lo, hi):
    """Inclusive segment–CAD intersection, in normalized ray parameter units."""
    a, b = np.asarray(a), np.asarray(b)
    delta = b - a
    moving = abs(delta) > 1e-12
    d = np.where(moving, delta, 1.)
    left, right = (lo - a) / d, (hi - a) / d
    start = np.where(moving, np.minimum(left, right), -np.inf).max(axis=1)
    stop = np.where(moving, np.maximum(left, right), np.inf).min(axis=1)
    misses_parallel = np.any(~moving & ((a < lo) | (a > hi)), axis=1)
    start, stop = np.maximum(start, 0.), np.minimum(stop, 1.)
    good = ~misses_parallel & (start <= stop)
    return float(start[good].min()) if good.any() else None


def scan_rays(a, b, q, lo, hi, world):
    robot, base, actor = world
    rows = p.rayTestBatch(frame(a,q).tolist(),frame(b,q).tolist(),numThreads=1)
    if len(rows) != len(a):
        raise RuntimeError('Missing ray observation')
    out = []
    for source, target, (body, link, frac, hit, normal) in zip(a,b,rows):
        kind = ('none' if body < 0 else 'cad_link' if body == robot and link == base else
                'robot_other_link' if body == robot else 'probe_actor' if body == actor else 'other')
        out.append({'a_m':source.tolist(),'b_m':target.tolist(),'kind':kind,'fraction':float(frac),
                    'cad_nominal_fraction':first_nominal_entry(source,target,lo,hi),
                    'hit_m':unframe(hit,q).tolist() if body>=0 else None})
    return out


def closest_cad_probe(probe, robot, base, xyz, q):
    p.resetBasePositionAndOrientation(probe, frame(xyz,q).tolist(), [0,0,0,1])
    pair = p.getClosestPoints(robot,probe,.04,linkIndexA=base)
    # Keep raw signed distance. Sphere-adjusted center clearance is used
    # outside the surface only, not as an exact interior signed distance.
    separation = min((float(row[8]) for row in pair),default=None)
    return {'center_m':list(xyz),'sphere_signed_distance_m':separation}


def world_forward(a, q, robot, base, actor=None):
    d = np.r_[r2(q[2])@[0.,.320],0.]
    a = np.asarray(a)
    body, link, fraction, hit, _ = p.rayTestBatch([a.tolist()],[(a+d).tolist()],numThreads=1)[0]
    return {'cad_hit':body==robot and link==base,'actor_hit':body==actor,'fraction':float(fraction),
            'hit_world_m':list(hit) if body>=0 else None}


def raw_ray_fractions(rows):
    """Identity-independent range payload, used for the ambiguity check."""
    return np.array([row['fraction'] for row in rows])


def scan_pose(asset, q, lo, hi, suspect, *, exact_world_origin=None):
    p.resetSimulation()
    robot = p.loadURDF(str(asset),useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION)
    joints = {p.getJointInfo(robot,i)[1].decode():i for i in range(p.getNumJoints(robot))}
    base = joints['cad_base_mount']
    for axis, value in zip(AXES,q):
        p.resetJointState(robot,joints[axis],float(value))
    x, y, z = suspect
    cross = (lo[:,2]<=z) & (hi[:,2]>=z)
    assert cross.sum()==1
    edges = np.array([lo[cross,0][0],hi[cross,0][0]])
    probes = np.concatenate([edges[0]-OFFSETS_M,edges[1]+OFFSETS_M,[x]])
    sagittal = np.column_stack([probes,np.full(len(probes),y),np.full(len(probes),z)])
    records = {}
    for n in FORWARD_M:
        d = np.array([0.,n,0.])
        records[f'forward_{n:.3f}m'] = scan_rays(sagittal,sagittal+d,q,lo,hi,(robot,base,None))
    # Complete through-view, begins beyond the furthest CAD surface (not
    # inside it). Cross-views are transverse at probe depths, same pose.
    d = np.array([0.,.50,0.])
    records['reverse_full'] = scan_rays(sagittal+d,sagittal,q,lo,hi,(robot,base,None))
    side_start = np.array([(x+.045, yi, z) for yi in Y_VIEWS_M])
    side_stop = side_start.copy()
    side_stop[:,0] = x-.030
    records['transverse_from_outside'] = scan_rays(side_start,side_stop,q,lo,hi,(robot,base,None))
    # A fixed witness grid, independent of actor placement. Compare both
    # registrations; direct-to-actor transverse view below is diagnostic only.
    grids = {}
    for name, pitch, shift in [('phase_0.000m',GRID_PITCH_M,0.), ('phase_0.004m',GRID_PITCH_M,GRID_PITCH_M/2),
                               ('dense_1mm_for_diagnosis',.001,0.)]:
        ys = y+shift+np.arange(int(.512/pitch))*pitch
        a = np.column_stack([np.full(len(ys),x+.045),ys,np.full(len(ys),z)])
        b = a.copy()
        b[:,0] = x-.030
        grids[name] = a,b
    witness_clear = {name:scan_rays(a,b,q,lo,hi,(robot,base,None)) for name,(a,b) in grids.items()}
    # Retain the archive's exact world-space start and construct its end using
    # the original transform order; local round trips can shift a grazing ray.
    exact = None if exact_world_origin is None else world_forward(exact_world_origin,q,robot,base)

    # Contact probe is deliberately in a separate sensor-free phase; no point
    # or body identity is available as a steering input in this tool.
    col = p.createCollisionShape(p.GEOM_SPHERE,radius=PROBE_RADIUS_M)
    body = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,basePosition=[-8,-8,-8])
    distance = [closest_cad_probe(body,robot,base,(float(xi),float(yi),float(z)),q)
                for xi in [edges[1]-.001,edges[1],x,edges[1]+.002] for yi in Y_VIEWS_M]
    p.removeBody(body)

    # Counterexamples to filtering the observed X band or using a second
    # view as permission to ignore the first: place a real narrow obstacle
    # straddling the exact suspect beam, with independently queried clearance.
    actors = {}
    for label, position_y, half, dx in [
        ('near_face',-.2735,(.00045,.0015,.006),0.),
        ('gap',-.296,(.00045,.0025,.006),0.),
        ('alongside',-.170,(.00045,.004,.006),0.),
        ('three_mm_between_witness_rows',y+6.25*GRID_PITCH_M,(.00045,.0014,.006),0.),
        # A 12 mm cube fits outside of the CAD, but clips the suspect ray.
        ('twelve_mm_cube',y+6.25*GRID_PITCH_M,(.006,.006,.006),.0054),
    ]:
        shape = p.createCollisionShape(p.GEOM_BOX,halfExtents=half)
        center = [x+dx,position_y,z]
        actor = p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,basePosition=frame(center,q).tolist(),
                                  baseOrientation=p.getQuaternionFromEuler([0,0,float(q[2])]))
        clearance = min((float(row[8]) for row in p.getClosestPoints(robot,actor,.10,linkIndexA=base)),default=None)
        a = np.array([suspect]); b = a+[0.,.320,0.]
        side_a, side_b = np.array([[x+.045,position_y,z]]),np.array([[x-.030,position_y,z]])
        w = robot,base,actor
        witness = {name:scan_rays(a0,b0,q,lo,hi,w) for name,(a0,b0) in grids.items()}
        actors[label] = dict(center_m=center,half_m=half,cad_signed_clearance_m=clearance,
            forward=scan_rays(a,b,q,lo,hi,w)[0],reverse=scan_rays(a+[0.,.50,0.],a,q,lo,hi,w)[0],
            exact_world_forward=None if exact_world_origin is None else world_forward(exact_world_origin,q,robot,base,actor),
            transverse_to_actor_center=scan_rays(side_a,side_b,q,lo,hi,w)[0],
            fixed_witness_hits={name:sum(r['kind']=='probe_actor' for r in rows) for name,rows in witness.items()},
            witness_raw_equals_empty={name:bool(np.array_equal(raw_ray_fractions(rows),
                                        raw_ray_fractions(witness_clear[name]))) for name,rows in witness.items()})
        p.removeBody(actor)
    return dict(q=q.tolist(),right_edge_m=float(edges[1]),rays=records,sphere_probes=distance,actors=actors,
                fixed_witness_clear=witness_clear,exact_world_replay=exact)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model',type=Path); ap.add_argument('diagnosis',type=Path); ap.add_argument('out',type=Path)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        ap.error('remote CPU only')
    archive = json.loads(args.diagnosis.read_text())
    event = archive['case']['rear_events'][0]
    packet = archive['details'][str(event['sample_k'])]
    shot = packet['channels']['main_direct']['hits_by_body_link'][0]
    old = np.array(packet['pose']); suspect = unframe(shot['origin_m'],old)
    asset,cad = args.model/'x7_proxy_gripper.urdf',args.model/'base-envelope.json'
    lo,hi = cad_boxes(json.loads(cad.read_text()))
    q = {'archived-west':old,'same-yaw-origin':np.array([0.,0.,old[2]]),'stationary-axes':np.zeros(3),
         'fresh-yaw':np.array([3.337,.137,math.radians(-28.2)]), 'reverse-facing':np.array([-.50,1.20,math.radians(61.)])}
    p.connect(p.DIRECT)
    try:
        data = {name:scan_pose(asset,pose,lo,hi,suspect,
                               exact_world_origin=shot['origin_m'] if name=='archived-west' else None)
                for name,pose in q.items()}
    finally:
        p.disconnect()
    west = data['archived-west']
    expected = float(np.linalg.norm(np.array(shot['hit_m'])-shot['origin_m'])/.320)
    replay = west['exact_world_replay']
    front_match = replay['cad_hit'] and math.isclose(replay['fraction'],expected,abs_tol=1.e-7,rel_tol=0.)
    variants = {label:[name for name,rays in row['rays'].items() if name.startswith('forward_') and
                       rays[-1]['kind']=='cad_link' and rays[-1]['cad_nominal_fraction'] is None]
                for label,row in data.items()}
    # Keep the exact replay status distinct from alternate-length/pose
    # observations; the latter are an independent metrology experiment.
    report = dict(source_sha256=digest(args.diagnosis),cad_sha256=digest(cad),asset_sha256=digest(asset),
                  suspect_ray_origin_m=suspect.tolist(),expected_original_fraction=expected,
                  exact_world_ray_reproduced=front_match,cad_outside_nominal_returns=variants,
                  sampled_frame_count=len(data),probe_radius_m=PROBE_RADIUS_M,poses=data,
                  changes_controller=False,continuous_collision_proof=False)
    with args.out.open('x') as out:
        out.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not variants['archived-west']:
        raise SystemExit('Isolated scene did not reproduce the outside-CAD disagreement')
    print('exact_world_ray_reproduced',front_match)
    for label,row in data.items():
        print(label,variants[label],{name:[a['forward']['kind'],a['reverse']['kind'],a['fixed_witness_hits']]
                                     for name,a in row['actors'].items()})


if __name__=='__main__':
    main()
