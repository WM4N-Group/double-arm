#!/usr/bin/env python3
"""Remote CPU-only X7 single-arm, contact-only pick/place demonstration pilot."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HZ, CONTROL_HZ = 240, 30
VISUAL_HZ = 5
BASE = np.array([0., 0., 1.77])
BASE_R = Rotation.from_euler('y', -math.pi/2)
ROT = Rotation.from_euler('x', math.pi)
TCP = .218
TABLE = 1.30
HALF = .022
OPEN, CLOSE = .034, .006
PARK = [0, .45, 0, -.87, 0, .33, 0]
HOME = np.array([-.69, -.18, TABLE + HALF + .105])
# Absolute joint position targets for Joint1..7 and a shared jaw target in meters.
ACTION_NAMES = [f'Joint{i}_target_rad' for i in range(1, 8)] + ['jaw_half_opening_motor_target_m']
PHASES = ['approach', 'descend', 'close', 'lift', 'carry', 'lower', 'release', 'retreat', 'settle']


def box_part(link, size, xyz, rgba, mass):
    inertial = ET.SubElement(link, 'inertial')
    ET.SubElement(inertial, 'origin', xyz=xyz)
    ET.SubElement(inertial, 'mass', value=str(mass))
    x, y, z = map(float, size.split())
    ET.SubElement(inertial, 'inertia', ixy='0', ixz='0', iyz='0',
                  ixx=str(mass*(y*y+z*z)/12), iyy=str(mass*(x*x+z*z)/12), izz=str(mass*(x*x+y*y)/12))
    for tag in ('visual', 'collision'):
        g = ET.SubElement(link, tag)
        ET.SubElement(g, 'origin', xyz=xyz)
        ET.SubElement(ET.SubElement(g, 'geometry'), 'box', size=size)
        if tag == 'visual':
            ET.SubElement(ET.SubElement(g, 'material', name=link.get('name')), 'color', rgba=rgba)


def build_model(source, out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    shutil.copytree(source.parent / 'meshes', out / 'meshes')
    root = ET.parse(source).getroot()
    # Retain the vendor Link7 inertial properties; these jaws are an explicitly
    # synthetic tool, not a claim about the installed X7 hardware.
    for side, sgn in [('a', 1), ('b', -1)]:
        name = 'proxy_finger_' + side
        link = ET.SubElement(root, 'link', name=name)
        box_part(link, '.010 .032 .069', '0 0 .023', '.14 .22 .28 1', .048)
        joint = ET.SubElement(root, 'joint', name=name + '_slide', type='prismatic')
        ET.SubElement(joint, 'parent', link='Link7')
        ET.SubElement(joint, 'child', link=name)
        ET.SubElement(joint, 'origin', xyz=f'{sgn * .009} 0 .195')
        ET.SubElement(joint, 'axis', xyz=f'{sgn} 0 0')
        ET.SubElement(joint, 'limit', lower='0', upper='.036', effort='14', velocity='.2')
    # Fixed tool housing at the end of the CAD wrist.
    palm = ET.SubElement(root, 'link', name='proxy_palm')
    box_part(palm, '.104 .055 .027', '0 0 0', '.23 .45 .50 1', .09)
    j = ET.SubElement(root, 'joint', name='proxy_palm_mount', type='fixed')
    ET.SubElement(j, 'parent', link='Link7')
    ET.SubElement(j, 'child', link='proxy_palm')
    ET.SubElement(j, 'origin', xyz='0 0 .164')
    f = out / 'x7_proxy_gripper.urdf'
    ET.ElementTree(root).write(f, xml_declaration=True, encoding='utf-8')
    return f


class Kinematics:
    """Independent FK/IK on the URDF chain; no physics state writes for planning."""
    def __init__(self, urdf):
        root = ET.parse(urdf).getroot()
        d = {j.get('name'): j for j in root.findall('joint')}
        self.chain = [(np.fromstring(j.find('origin').get('xyz'), sep=' '),
                       Rotation.from_euler('xyz', np.fromstring(j.find('origin').get('rpy'), sep=' ')),
                       np.fromstring(j.find('axis').get('xyz'), sep=' '))
                      for j in (d[f'Joint{i}'] for i in range(1, 8))]
        self.lower = np.repeat(-3.12, 7)
        self.upper = -self.lower
        rng = np.random.default_rng(201)
        starts = [np.zeros(7), PARK, [-.7, -.8, -.7, 1.5, .9, 1.1, -.3],
                  [.7, .8, .7, -1.5, -.9, -1.1, .3], *rng.uniform(-2, 2, (19, 7))]
        sols = [self.solve(HOME, np.asarray(q), continuity=.001, max_eval=175, validate=False) for q in starts]
        good = [q for q in sols if self.err(HOME, q)[0] < .0015 and self.err(HOME, q)[1] < .012]
        if not good:
            raise RuntimeError('HOME has no usable IK solution')
        # Prefer the elbow out from the torso, not an elbow wrapping behind it.
        self.home = min(good, key=lambda q: np.linalg.norm(q) + 4*max(0., self.fk(q)[2][2][0]+.50))

    def fk(self, q):
        x, r = BASE.copy(), BASE_R
        positions = []
        for (xyz, rot, axis), theta in zip(self.chain, q):
            x = x + r.apply(xyz)
            r = r * rot * Rotation.from_rotvec(axis * theta)
            positions.append(x)
        return x + r.apply([0, 0, TCP]), r, positions

    def err(self, xyz, q):
        x, r, _ = self.fk(q)
        return float(np.linalg.norm(x-xyz)), float((r.inv() * ROT).magnitude())

    def solve(self, xyz, previous, continuity=.012, max_eval=55, validate=True):
        def residual(q):
            x, r, _ = self.fk(q)
            return np.r_[7*(x-xyz), (r.inv()*ROT).as_rotvec(), continuity*(q-previous)]
        opt = least_squares(residual, np.clip(previous, self.lower, self.upper), bounds=(self.lower, self.upper),
                            max_nfev=max_eval, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        ep, er = self.err(xyz, opt.x)
        if validate and (ep > .002 or er > .02):
            raise ValueError(f'bad IK at {xyz.tolist()}: {ep:.4f} m {er:.4f} rad')
        return opt.x


def scenario(seed, partition="probe"):
    rng = np.random.default_rng(seed)
    initial = np.array([rng.uniform(-.717, -.675), rng.uniform(-.201, -.151), TABLE + HALF + .0005])
    goal = np.array([rng.uniform(-.717, -.675), rng.uniform(-.018, .022), TABLE + HALF])
    # Freeze disjoint initial-Y bands before looking at outcomes. The heldout
    # interval tests interpolation between training bands, not novel scenes.
    if partition == "train":
        initial[1] = rng.uniform(*([-.202,-.188] if seed % 2 == 0 else [-.166,-.151]))
    elif partition == "heldout":
        initial[1] = rng.uniform(-.181,-.173)
    return initial, goal


def plan(kin, initial, goal):
    pick = initial.copy(); pick[2] = TABLE + HALF + .019
    place = goal.copy(); place[2] = TABLE + HALF + .018
    pre = pick + [0, 0, .098]
    high = place + [0, 0, .086]
    positions = [pre, pick, pick, pre, high, place, place, high, high]
    lengths = [36, 36, 24, 36, 59, 36, 24, 36, 30]
    jaws = [OPEN, OPEN, CLOSE, CLOSE, CLOSE, CLOSE, OPEN, OPEN, OPEN]
    xyz0, previous, g0 = HOME, kin.home.copy(), OPEN
    actions, phases = [], []
    for k, (xyz1, n, g1) in enumerate(zip(positions, lengths, jaws)):
        # IK knots at up to 7.5 Hz; joint interpolation stays continuous.
        hs = sorted(set(range(0, n, 4)) | {n})
        qs = [previous]
        for h in hs[1:]:
            u = h / n
            u = u*u*(3 - 2*u)
            a = (1-u)*xyz0 + u*xyz1
            qs.append(kin.solve(a, qs[-1]) if np.linalg.norm(xyz1-xyz0) > 1e-8 else qs[-1])
        q = np.stack([np.interp(np.arange(1,n+1), hs, np.array(qs)[:,i]) for i in range(7)], axis=1)
        u = np.linspace(1/n, 1., n)
        w = g0 + (g1-g0)*u*u*(3 - 2*u)
        actions.extend(np.column_stack([q,w])); phases.extend([k]*n)
        xyz0, previous, g0 = xyz1, qs[-1], g1
    return np.asarray(actions), np.asarray(phases, dtype=np.uint8)


def box(xyz, dims, rgba, mass=0., collision=True):
    dims = np.asarray(dims)/2
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims) if collision else -1
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=dims, rgbaColor=rgba)
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis, basePosition=xyz)


class World:
    def __init__(self, urdf, kin, initial, goal):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1,
                                   fixedTimeStep=1./HZ, numSubSteps=0, enableFileCaching=0)
        p.setGravity(0, 0, -9.81)
        self.robot = p.loadURDF(str(urdf), BASE, BASE_R.as_quat(), useFixedBase=True,
                               flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION)
        self.idx = {p.getJointInfo(self.robot, i)[1].decode(): i for i in range(p.getNumJoints(self.robot))}
        self.arm = [self.idx[f'Joint{i}'] for i in range(1, 8)]
        self.park = [self.idx[f'Joint{i}'] for i in range(8, 15)]
        self.jaw = [self.idx[f'proxy_finger_{s}_slide'] for s in ('a','b')]
        self.arm_effort = [p.getJointInfo(self.robot, i)[10] for i in self.arm]
        palm = self.idx['proxy_palm_mount']
        for group in (self.arm, self.park):
            for a, i in enumerate(group):
                for j in group[max(0,a-2):a]:
                    p.setCollisionFilterPair(self.robot, self.robot, i, j, 0)
                p.changeVisualShape(self.robot, i, rgbaColor=(.23,.63,.78,1) if group is self.arm else (.59,.66,.74,1))
        for i in [*self.jaw, palm]:
            for j in [*self.arm[-2:], *self.jaw, palm]:
                if i != j:
                    p.setCollisionFilterPair(self.robot, self.robot, i,j,0)
        for i in self.jaw:
            p.changeDynamics(self.robot, i, lateralFriction=1.15, restitution=0.)
        for i,q in zip(self.arm + self.park + self.jaw, [*kin.home, *PARK, OPEN, OPEN]):
            p.resetJointState(self.robot, i, q)
        self.setup_jaw_actuation()
        self.table = box([-.72,-.07,TABLE-.022], [.45,.56,.044], [.67,.73,.76,1])
        box([-.72,-.07,(TABLE-.044)*.5],[.29,.43,TABLE-.044],[.30,.36,.44,1], collision=False)
        box([0,0,-.029],[8,8,.018],[.28,.34,.42,1], collision=False)
        gx, gy = goal[:2]
        for x,y,sx,sy in [(gx+.045,gy,.004,.094),(gx-.045,gy,.004,.094),(gx,gy+.045,.094,.004),(gx,gy-.045,.094,.004)]:
            box([x,y,TABLE+.0004],[sx,sy,.0008],[.18,.60,.29,1], collision=False)
        self.block = box(initial,[2*HALF]*3,[.98,.40,.16,1], mass=.05)
        p.changeDynamics(self.block,-1,lateralFriction=.87, spinningFriction=.001, rollingFriction=.001, restitution=0.)
        self.goal = goal
        self.initial = initial
        for _ in range(72):
            self.drive(np.r_[kin.home, OPEN]); p.stepSimulation()
        self.x0 = np.asarray(p.getBasePositionAndOrientation(self.block)[0])
        wrist = p.getLinkState(self.robot,self.arm[-1],computeForwardKinematics=True)
        pos = np.asarray(wrist[4])+Rotation.from_quat(wrist[5]).apply([0,0,TCP])
        if np.linalg.norm(pos-HOME) > .006:
            raise RuntimeError(f'Home TCP check failed: {pos}, expected {HOME}; q={self.q()[:7]}')
        self.view = p.computeViewMatrix([-.95,-.87,1.92],[-.67,-.065,1.392],[0.,0.,1.])

    def setup_jaw_actuation(self):
        """Hook for the separate mechanically synchronized model variant."""

    def drive(self, u):
        p.setJointMotorControlArray(self.robot,self.arm,p.POSITION_CONTROL,targetPositions=u[:7],forces=self.arm_effort,
                                    positionGains=[.29]*7,velocityGains=[1.]*7)
        p.setJointMotorControlArray(self.robot,self.park,p.POSITION_CONTROL,targetPositions=PARK,forces=[45.]*7,
                                    positionGains=[.23]*7,velocityGains=[1.]*7)
        self.drive_jaws(u[7])

    def drive_jaws(self, jaw_target):
        p.setJointMotorControlArray(self.robot,self.jaw,p.POSITION_CONTROL,targetPositions=[jaw_target]*2,forces=[14.]*2,
                                    positionGains=[.22]*2,velocityGains=[1.]*2)

    def q(self):
        return np.asarray([(s[0],s[1]) for s in p.getJointStates(self.robot, self.arm + self.jaw)]).T

    def audit_constraints(self):
        if p.getNumConstraints() != 0:
            raise RuntimeError('Unexpected grasp constraint')

    def step(self, u):
        both = False
        for _ in range(HZ//CONTROL_HZ):
            self.drive(u); p.stepSimulation()
            contacts = p.getContactPoints(self.robot,self.block)
            pressed = {c[3] for c in contacts if c[9] > .06}
            both |= all(i in pressed for i in self.jaw)
            self.audit_constraints()
        contacts = p.getContactPoints(self.robot,self.block)
        forces = np.asarray([sum(c[9] for c in contacts if c[3]==i) for i in self.jaw])
        other = [c for c in contacts if c[3] not in self.jaw and c[9] > .06]
        table_forces = [c for c in p.getContactPoints(self.robot,self.table) if c[9] > .5]
        block_sitting = bool(p.getContactPoints(self.table,self.block))
        return forces, int(both), len(other), len(table_forces), int(block_sitting)

    def camera(self,w,h):
        view = p.computeProjectionMatrixFOV(43,w/h,.08,6)
        rgba = p.getCameraImage(w,h,self.view,view,renderer=p.ER_TINY_RENDERER,shadow=0,
                                lightDirection=[1,-2,5],lightAmbientCoeff=.62)[2]
        return np.asarray(rgba, dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy()

    def obj(self):
        pos, ori = p.getBasePositionAndOrientation(self.block)
        vel, ang = p.getBaseVelocity(self.block)
        return np.r_[pos,ori,vel,ang]


def summarize(trace, world):
    pos = trace['object_post'][:, :3]
    z = pos[:,2]-world.x0[2]
    phase = trace['phase']
    final = phase==8
    lift_contact = int(np.count_nonzero((z>.045) & (trace['opposed_contact']>0)))
    retain = float(np.linalg.norm(pos[phase==4][-1,:2]-world.goal[:2]))
    settled = int(np.count_nonzero((np.linalg.norm(pos[final,:2]-world.goal[:2],axis=1)<.029) &
                                    (abs(pos[final,2]-world.x0[2]) < .007) &
                                    (np.linalg.norm(trace['object_post'][final,7:10],axis=1)<.02) &
                                    (trace['finger_force'][final].max(axis=1)<.06) &
                                    (trace['on_table'][final]>0)))
    other, table = [int(np.count_nonzero(trace[name])) for name in ('other_robot_object_contacts','robot_table_contacts')]
    success = bool(z.max()>.065 and lift_contact>=9 and retain<.035 and settled>=25 and not other and not table)
    return {'success':success, 'max_lift_m':float(z.max()), 'opposed_contact_while_lifted_frames':lift_contact,
            'carry_end_goal_xy_error_m':retain, 'final_goal_xy_error_m':float(np.linalg.norm(pos[-1,:2]-world.goal[:2])),
            'settled_on_goal_released_frames':settled, 'other_robot_object_contact_frames':other,
            'robot_table_contact_frames':table, 'constraints_seen':0, 'object_pose_writes_after_reset':0,
            'steps':int(len(phase)), 'source':'scripted Cartesian IK expert (not learned)',
            'control_hz':CONTROL_HZ, 'physics_hz':HZ}


def run(world, actions, phases, images=False, video=None):
    out = dict(action=[], phase=[], q=[], dq=[], object_pre=[], object_post=[], finger_force=[], opposed_contact=[],
               other_robot_object_contacts=[], robot_table_contacts=[], on_table=[])
    pixels, at = [], []
    for i,(u,ph) in enumerate(zip(actions,phases)):
        if i % (CONTROL_HZ//VISUAL_HZ)==0 and images:
            pixels.append(world.camera(160,120)); at.append(i)
        q,dq = world.q(); obj = world.obj()
        f,b,o,t,sit = world.step(u)
        for k,v in zip(out,(u,ph,q,dq,obj,world.obj(),f,b,o,t,sit)):
            out[k].append(v)
        if video and i%2==0:
            video(world,i,int(ph),f)
    data = {k:np.asarray(v) for k,v in out.items()}
    if images:
        data |= {'rgb':np.asarray(pixels), 'rgb_action_index':np.asarray(at,dtype=np.int32)}
    return data, summarize(data,world)


def render_review(out, *, title='X7 PICK & PLACE  /  SCRIPTED EXPERT',
                  subtitle='Simulation proxy gripper  •  Contact only  •  Not a learned policy', show_jaw_center=False):
    w,h=800,520
    enc = subprocess.Popen(['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
                            '-framerate','15','-i','pipe:0','-c:v','libx264','-threads','1','-preset','fast','-crf','19',
                            '-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',21)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    def write(world,k,phase,force):
        img=Image.fromarray(world.camera(w,h)); d=ImageDraw.Draw(img)
        right = int(50+max(d.textlength(title,font=font),d.textlength(subtitle,font=small)))
        d.rounded_rectangle((14,14,right,86),10,fill='#152336')
        d.text((28,23),title,font=font,fill='#f4f8fe')
        d.text((29,56),subtitle,font=small,fill='#aabed6')
        d.rectangle((0,h-37,w,h),fill='#152336')
        status = f'{k/CONTROL_HZ:04.1f}s    {PHASES[phase].upper()}    |    Finger normal force: {force[0]:.1f} / {force[1]:.1f} N'
        if show_jaw_center:
            a,b = world.q()[0][-2:]
            status += f'    |    Jaw center {(500*(a-b)):+.3f} mm'
        d.text((16,h-28),status,font=small,fill='#f4f8fe')
        enc.stdin.write(img.tobytes())
    return enc,write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('asset',type=Path,help='Derived fixed-base demo URDF (asset meshes beside it)')
    ap.add_argument('out',type=Path)
    ap.add_argument('--start',type=int,default=700)
    ap.add_argument('--count',type=int,default=1)
    ap.add_argument('--rgb',action='store_true')
    ap.add_argument('--review',action='store_true')
    ap.add_argument('--partition',choices=('probe','train','heldout'),default='probe')
    ap.add_argument('--replay',type=Path,help='Use saved actions and initial state; verify every transition')
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.count<=0 or (args.review or args.replay) and args.count!=1:
        ap.error('Requires explicit CUDA_VISIBLE_DEVICES="" and >0 episodes; review is single-episode')
    args.out.mkdir(parents=True)
    urdf=build_model(args.asset,args.out/'model')
    kin=Kinematics(urdf)
    print('home',kin.home.tolist(),flush=True)
    (args.out/'schema.json').write_text(json.dumps({'version':'x7_proxy_pick_place_v1', 'action':ACTION_NAMES,
        'q':ACTION_NAMES[:7]+['jaw_a_pos_m','jaw_b_pos_m'], 'observations':'q,dq,object_pre before applying same-row action',
        'camera':'fixed external rendered RGB; pre-action, aligned by rgb_action_index', 'rgb_hz':VISUAL_HZ,
        'control_hz':CONTROL_HZ,'physics_hz':HZ,'phase_order':PHASES,'eval_only':'metrics, contacts and phases not deployable input',
        'asset_urdf_sha256':hashlib.sha256(urdf.read_bytes()).hexdigest()},indent=2)+'\n')
    p.connect(p.DIRECT)
    manifest=[]
    try:
        for seed in range(args.start,args.start+args.count):
            if args.replay:
                archive = np.load(args.replay, allow_pickle=False)
                seed, initial, goal = int(archive["seed"]), archive["start"], archive["goal"]
                u, ph = archive["action"], archive["phase"]
            else:
                initial, goal = scenario(seed, args.partition)
                u, ph = plan(kin,initial,goal)
            world = World(urdf,kin,initial,goal)
            enc,video = render_review(args.out/'expert-review.mp4') if args.review else (None,None)
            try:
                trace,res = run(world,u,ph,args.rgb,video)
            finally:
                if enc:
                    enc.stdin.close()
                    if enc.wait()!=0:
                        raise RuntimeError('ffmpeg encoding failed')
            if args.replay:
                checks = ('action','q','dq','object_pre','object_post','finger_force','opposed_contact','on_table')
                res['replay_exact'] = all(np.array_equal(archive[k],trace[k]) for k in checks)
                archive.close()
                if not res['replay_exact']:
                    raise RuntimeError('Recorded action tape did not reproduce the saved physics trajectory')
            episode = args.out/f'episode-{seed}.npz'
            np.savez_compressed(episode,**trace,seed=seed,start=initial,goal=goal)
            res |= {'seed':seed,'start':initial.tolist(),'goal':goal.tolist(),'rgb_frames':len(trace.get('rgb',[])),
                    'partition':args.partition,'file':episode.name, 'sha256':hashlib.sha256(episode.read_bytes()).hexdigest()}
            manifest.append(res); print(json.dumps(res),flush=True)
        (args.out/'results.json').write_text(json.dumps(manifest,indent=2)+'\n')
    finally:
        p.disconnect()


if __name__=='__main__':
    main()
