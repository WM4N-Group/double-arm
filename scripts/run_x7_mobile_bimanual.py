#!/usr/bin/env python3
"""Remote CPU-only X7 two-hand tote carry, contact-only with planar drive proxy.

The inherited X7 CAD base has no wheel model. Three internal URDF joints
constrain its chassis to the floor plane and drive it through simulated motor
forces, not pose resets. Fixed storage stations do not move with the robot.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pybullet as p
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from x7_pick_place_pilot import CONTROL_HZ, HZ, OPEN, CLOSE, Kinematics, box
from x7_sync_gripper_preflight import GEAR_RATIO, MASTER_FORCE_N, SYNC_FORCE_N, SYNC_MAX_OFFSET_M, v2_model
from x7_station_collision import (CAD_MOUNT_Z, STATION_CLEARANCE_M, STATION_SCAN_M, cad_envelopes,
                                  compound_station, station_clearance, station_parts)

TABLE_Z = 1.25
PARK_Y = -.400
CARRY_X = 2.65
TRAY_MASS = .64
SUPPORT_Z = TABLE_Z + .011
GRIP_Z = SUPPORT_Z + .081 + .0095
FRAME_NAMES = ('approach', 'descend', 'close', 'lift', 'drive', 'stop', 'lower', 'release', 'retreat', 'settle')
COUNTS = [25, 43, 30, 44, 340, 32, 48, 28, 38, 32]
CASE_NAMES = ('fixed-lift', 'mobile', 'base-locked', 'only-A', 'only-B')
CONTACT_THRESHOLD_N = .12
AXES = ('base_x', 'base_y', 'base_yaw')


def make_model(source, out, cad_zip):
    model = v2_model(source, out)
    xml = ET.parse(model)
    root = xml.getroot()
    root.set('name', 'x7_duo_mobile_carry_collision_v2')
    # Mirror the complete v2 gripper, retaining both original arm chains.
    for name in ('proxy_finger_a', 'proxy_finger_b', 'proxy_palm'):
        link = root.find(f"link[@name='{name}']")
        joint = next(j for j in root.findall('joint') if j.find('child').get('link') == name)
        lj, jj = [ET.fromstring(ET.tostring(e)) for e in (link, joint)]
        lj.set('name', name + '_B')
        jj.set('name', joint.get('name') + '_B')
        jj.find('child').set('link', lj.get('name'))
        jj.find('parent').set('link', 'Link14')
        root.extend([lj, jj])
    for name in ('carry_anchor', 'carry_x', 'carry_y', 'carry_yaw'):
        link = ET.SubElement(root, 'link', name=name)
        inert = ET.SubElement(link, 'inertial')
        ET.SubElement(inert, 'mass', value='1')
        ET.SubElement(inert, 'inertia', ixx='.1', iyy='.1', izz='.1', ixz='0', ixy='0', iyz='0')
    def joint(name, parent, child, kind, axis=None, xyz='0 0 0', rpy='0 0 0'):
        j = ET.SubElement(root, 'joint', name=name, type=kind)
        ET.SubElement(j, 'parent', link=parent)
        ET.SubElement(j, 'child', link=child)
        ET.SubElement(j, 'origin', xyz=xyz, rpy=rpy)
        if axis:
            ET.SubElement(j, 'axis', xyz=axis)
            ET.SubElement(j, 'limit', lower='-3.14' if kind=='revolute' else '-5',
                          upper='3.14' if kind=='revolute' else '5', effort='1500', velocity='.65')
    joint('base_x', 'carry_anchor', 'carry_x', 'prismatic', '1 0 0')
    joint('base_y', 'carry_x', 'carry_y', 'prismatic', '0 1 0')
    joint('base_yaw', 'carry_y', 'carry_yaw', 'revolute', '0 0 1')
    joint('cad_base_mount', 'carry_yaw', 'base_link', 'fixed', xyz=f'0 0 {CAD_MOUNT_Z}', rpy=f'0 {-math.pi/2} 0')
    # Conservative boxes bound both the raw CAD and the mesh shown in the video.
    # Collision shapes and floating-point hash are part of the new model identity.
    proxy=cad_envelopes(cad_zip,model.parent/'meshes'/'base_link.STL')
    base=root.find("link[@name='base_link']")
    for part in proxy['parts_in_cad_frame']:
        col=ET.SubElement(base,'collision')
        ET.SubElement(col,'origin',xyz=' '.join(map(str,part['center'])))
        ET.SubElement(ET.SubElement(col,'geometry'),'box',size=' '.join(map(str,part['size'])))
    (out/'base-envelope.json').write_text(json.dumps(proxy,indent=2)+'\n')
    xml.write(model, encoding='utf-8', xml_declaration=True)
    return model


class ArmIK(Kinematics):
    """Solve independent arm IK in the chassis frame, with nearby orientations."""
    def __init__(self, model, arm, home):
        joints = {j.get('name'): j for j in ET.parse(model).getroot().findall('joint')}
        self.chain = [(np.fromstring(j.find('origin').get('xyz'), sep=' '),
                       Rotation.from_euler('xyz', np.fromstring(j.find('origin').get('rpy'),sep=' ')),
                       np.fromstring(j.find('axis').get('xyz'),sep=' '))
                      for j in [joints[f'Joint{i}'] for i in range(7*arm+1,7*arm+8)]]
        self.lower = np.full(7,-3.11); self.upper = -self.lower
        self.home_xyz = home
        self.orientation = Rotation.from_euler('xyz',[math.pi,0.,math.pi*arm])
        rng = np.random.default_rng(501+arm)
        hints = [np.zeros(7), [-.45,.13,.16,-1.68,1.58,-.99,1.55],
                 [.45,-.13,-.16,1.68,-1.58,.99,-1.55], *rng.uniform(-2.5,2.5,(21,7))]
        sols = [self.solve(home,np.asarray(h),.001,140,False) for h in hints]
        sols = [s for s in sols if self.err(home,s)[0]<.002 and self.err(home,s)[1]<.014]
        if not sols:
            raise RuntimeError(f'Top-down home not reachable for arm {arm}: {home.tolist()}')
        # Keep the elbow forward/outboard of the torso; keep clear of joint limits.
        sign = (-1,1)[arm]
        def cost(q):
            elbow = self.fk(q)[2][2]
            return np.linalg.norm(q) + 9*max(0.,.39-sign*elbow[0])+2*max(0.,elbow[1]+.1)
        self.home = min(sols,key=cost)

    def err(self, xyz, q):
        x,r,_ = self.fk(q)
        return float(np.linalg.norm(x-xyz)),float((r.inv()*self.orientation).magnitude())

    def solve(self, xyz, previous, continuity=.014, max_eval=55, validate=True):
        def residual(q):
            x,r,_ = self.fk(q)
            return np.r_[7.5*(x-xyz),(r.inv()*self.orientation).as_rotvec(),continuity*(q-previous)]
        result=least_squares(residual,np.clip(previous,self.lower,self.upper),bounds=(self.lower,self.upper),
                             max_nfev=max_eval,ftol=1e-8,xtol=1e-8,gtol=1e-8).x
        err = self.err(xyz,result)
        if validate and (err[0]>.002 or err[1]>.017):
            raise RuntimeError(f'IK failed at {xyz.tolist()}, errors {err}')
        return result


def scene(seed):
    # Seed 0 is the centered geometry bring-up. Other seeds are separately labeled trials.
    rng=np.random.default_rng(seed)
    if seed==0:
        return .589,PARK_Y,0.,CARRY_X
    return .589, float(PARK_Y+rng.uniform(-.008,.007)), float(rng.uniform(-.018,.018)), float(rng.uniform(2.57,2.79))


def smooth(u):
    return u*u*u*(10.+u*(-15.+6.*u))


def make_plan(arms, handle_x, pick_y, object_x, base_goal):
    grip = [np.array([object_x+s*handle_x,pick_y,GRIP_Z]) for s in (-1,1)]
    high = [g + [0.,0.,.094] for g in grip]
    carry = [g + [0.,0.,.104] for g in grip]
    place = [g + [0.,0.,.006] for g in grip]
    places = [high,grip,grip,carry,carry,carry,place,place,high,high]
    jaw = [OPEN,OPEN,CLOSE,CLOSE,CLOSE,CLOSE,CLOSE,OPEN,OPEN,OPEN]
    result=[]; phases=[]; q0=[a.home for a in arms]; x0=[a.home_xyz for a in arms]; j0=OPEN
    for ph,(n,pos,jnext) in enumerate(zip(COUNTS,places,jaw)):
        knots=sorted(set(range(0,n,4))|{n}); chunks=[]
        u=np.arange(1,n+1)/n
        for a,xprev,xnext,qprev in zip(arms,x0,pos,q0):
            pts=[qprev]
            for k in knots[1:]:
                s=smooth(k/n)
                pts.append(a.solve((1-s)*xprev+s*xnext,pts[-1]) if np.linalg.norm(xnext-xprev)>1e-10 else pts[-1])
            chunks.append(np.stack([np.interp(np.arange(1,n+1),knots,np.asarray(pts)[:,d]) for d in range(7)],axis=1))
        if ph==4:
            bx=base_goal*smooth(u); bv=(base_goal*CONTROL_HZ/n)*30*u*u*(1-u)*(1-u)
        else:
            bx=np.full(n,base_goal if ph>4 else 0.); bv=np.zeros(n)
        j = j0 + (jnext-j0)*smooth(u)
        result.append(np.column_stack([bx,bv,*chunks,j,j])); phases.extend([ph]*n)
        q0=[q[-1] for q in chunks]; x0=pos; j0=jnext
    return np.concatenate(result,axis=0),np.asarray(phases,dtype=np.uint8)


def create_tote(center, handle_x):
    # Handle rails are part of one rigid tote; no robot/object attachment.
    # The fingers pinch short blocks widened away from the rim.
    parts = [(0.,0.,0.,1.0,.19,.022,[.34,.58,.21,1]),
             (0.,-.089,.045,1.,.013,.074,[.35,.59,.22,1]),(0.,.089,.045,1.,.013,.074,[.35,.59,.22,1]),
             (0.,0.,.043,.47,.14,.064,[.81,.80,.71,1])]
    for sign in (-1,1):
        parts += [(sign*.493,0.,.042,.014,.17,.069,[.35,.59,.22,1]),
                  (sign*(handle_x+.485)/2,0.,.023,handle_x-.456,.04,.021,[.30,.46,.18,1]),
                  (sign*handle_x,0.,.052,.032,.043,.044,[.96,.57,.15,1]),
                  (sign*handle_x,0.,.081,.039,.046,.034,[.96,.57,.15,1])]
    half=np.asarray([r[3:6] for r in parts])/2
    xyz=[r[:3] for r in parts]; types=[p.GEOM_BOX]*len(parts)
    col=p.createCollisionShapeArray(shapeTypes=types,halfExtents=half,collisionFramePositions=xyz)
    vis=p.createVisualShapeArray(shapeTypes=types,halfExtents=half,visualFramePositions=xyz,rgbaColors=[r[-1] for r in parts])
    body=p.createMultiBody(baseMass=TRAY_MASS,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,basePosition=center)
    p.changeDynamics(body,-1,lateralFriction=.86,spinningFriction=.001,restitution=0.)
    return body


class World:
    def __init__(self, model, arms, geometry, mode):
        handle_x, pick_y, obj_x, goal_x = geometry
        self.mode=mode
        self.start=np.array([obj_x,pick_y,SUPPORT_Z+.0007]); self.goal=np.array([goal_x+obj_x,pick_y,SUPPORT_Z]);
        if mode=='fixed-lift': self.goal[0]=obj_x
        p.resetSimulation()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1,fixedTimeStep=1/HZ,numSubSteps=0,numSolverIterations=130,enableFileCaching=0)
        p.setGravity(0,0,-9.81)
        self.robot=p.loadURDF(str(model),useFixedBase=True,flags=p.URDF_USE_INERTIA_FROM_FILE|p.URDF_USE_SELF_COLLISION)
        self.joints={p.getJointInfo(self.robot,i)[1].decode():i for i in range(p.getNumJoints(self.robot))}
        self.arm_ids=[[self.joints[f'Joint{i}'] for i in range(7*a+1,7*a+8)] for a in range(2)]
        self.jaws=[[self.joints[f'proxy_finger_{s}_slide{suffix}'] for s in ('a','b')] for suffix in ('','_B')]
        mounts=[self.joints[f'proxy_palm_mount{suffix}'] for suffix in ('','_B')]
        self.base=[self.joints[n] for n in AXES]; self.base_link=self.joints['cad_base_mount']
        self.forces=[[p.getJointInfo(self.robot,i)[10] for i in ids] for ids in self.arm_ids]
        for a,ids in enumerate(self.arm_ids):
            for k,i in enumerate(ids):
                for j in ids[max(0,k-2):k]: p.setCollisionFilterPair(self.robot,self.robot,i,j,0)
                if k==0: p.setCollisionFilterPair(self.robot,self.robot,i,self.base_link,0)
                p.changeVisualShape(self.robot,i,rgbaColor=((.25,.66,.80,1),(.74,.39,.18,1))[a])
            for i in [*self.jaws[a],mounts[a]]:
                for j in [*ids[-2:],*self.jaws[a],mounts[a]]:
                    if i!=j: p.setCollisionFilterPair(self.robot,self.robot,i,j,0)
            for i,q in zip(ids,arms[a].home): p.resetJointState(self.robot,i,q)
        self.racks={}
        for a,b in self.jaws:
            for i in (a,b):
                p.resetJointState(self.robot,i,OPEN)
                p.changeDynamics(self.robot,i,lateralFriction=1.15,restitution=0.)
            p.setJointMotorControl2(self.robot,b,p.VELOCITY_CONTROL,targetVelocity=0,force=0)
            rack=p.createConstraint(self.robot,a,self.robot,b,p.JOINT_GEAR,[1.,0.,0.],[0.]*3,[0.]*3)
            p.changeConstraint(rack,gearRatio=GEAR_RATIO,maxForce=SYNC_FORCE_N,relativePositionTarget=0.,erp=.55)
            self.racks[rack]=(self.robot,a,self.robot,b,p.JOINT_GEAR)
        p.changeVisualShape(self.robot,self.base_link,rgbaColor=(.76,.82,.89,1))
        # Table deck, legs, cantilevers and braces share identical visible/collision geometry.
        self.station_parts=[station_parts(x,PARK_Y,TABLE_Z,rgba) for x,rgba in
                            zip((0.,goal_x),([.47,.58,.68,1],[.41,.61,.52,1]))]
        self.tables=[compound_station(parts) for parts in self.station_parts]
        self.floor=box([.90,0.,-.023],[16.,16.,.046],[.23,.30,.39,1])
        self.station_init=[]
        self.tote=create_tote(self.start,handle_x)
        for _ in range(96):
            self.drive(np.r_[0.,0.,arms[0].home,arms[1].home,OPEN,OPEN]); p.stepSimulation()
            self.check_constraints()
            dist,hits,_=station_clearance(self.robot,self.tables)
            self.station_init.append([*dist,*hits])
        self.initial_object=self.state_object()
        self.settle_contacts=sum(len(p.getContactPoints(self.robot,t)) for t in self.tables)
        self.settle_self_contacts=len(p.getContactPoints(self.robot,self.robot))
        self.state_writes_during_actions=0

    def check_constraints(self):
        current={p.getConstraintUniqueId(i) for i in range(p.getNumConstraints())}
        if current!=set(self.racks):
            raise RuntimeError('Unexpected constraint IDs (only two internal jaw racks are permitted)')
        for rack,expect in self.racks.items():
            info=p.getConstraintInfo(rack)
            if info[:5]!=expect or info[10:15]!=(SYNC_FORCE_N,GEAR_RATIO,0,0.,.55):
                raise RuntimeError('Invalid constraints: a rack does not match its two sliders and approved parameters')

    def drive(self, a):
        xref, vref = a[:2]
        x=p.getJointState(self.robot,self.base[0])[0]
        v=float(np.clip(vref+3.5*(xref-x),-.5,.5))
        p.setJointMotorControl2(self.robot,self.base[0],p.VELOCITY_CONTROL,targetVelocity=v,force=1100.)
        p.setJointMotorControlArray(self.robot,self.base[1:],p.POSITION_CONTROL,targetPositions=[0.,0.],forces=[1000.,1000.],
                                    positionGains=[.18,.18],velocityGains=[1.,1.])
        for i,ids in enumerate(self.arm_ids):
            p.setJointMotorControlArray(self.robot,ids,p.POSITION_CONTROL,targetPositions=a[2+i*7:9+i*7],forces=self.forces[i],
                                        positionGains=[.29]*7,velocityGains=[1.]*7)
        for i,(master,_) in enumerate(self.jaws):
            p.setJointMotorControl2(self.robot,master,p.POSITION_CONTROL,targetPosition=a[-2+i],force=MASTER_FORCE_N,
                                    positionGain=.22,velocityGain=1.)

    def state_object(self):
        pos,r=p.getBasePositionAndOrientation(self.tote); v,w=p.getBaseVelocity(self.tote)
        return np.r_[pos,r,v,w]

    def qv(self,ids):
        return np.asarray([s[:2] for s in p.getJointStates(self.robot,ids)],dtype=float)

    def step(self,a):
        finger_indices=sum(self.jaws,[])
        frames=[]; clear=[]
        for _ in range(HZ//CONTROL_HZ):
            self.drive(a); p.stepSimulation(); self.check_constraints()
            contacts=p.getContactPoints(self.robot,self.tote)
            f=[sum(c[9] for c in contacts if c[3]==k) for k in finger_indices]
            rogue=sum(c[3] not in finger_indices and c[9]>.25 for c in contacts)
            dist,hits,links=station_clearance(self.robot,self.tables)
            table=sum(hits)
            self_touch=sum(c[8]<-.0005 for c in p.getContactPoints(self.robot,self.robot))
            clear.append([*dist,*hits,*links])
            q=self.qv(finger_indices)[:,0]
            centers=.5*(q[::2]-q[1::2])
            frames.append([*f,rogue,table,self_touch,*centers])
        return np.asarray(frames),np.asarray(clear)


def capture_rgb():
    # Same external camera for every sample, before applying the indexed action.
    view=p.computeViewMatrix([2.84,-5.68,3.04],[1.29,-.19,.99],[0.,0.,1.])
    proj=p.computeProjectionMatrixFOV(44.,192/128,.11,20.)
    rgba=p.getCameraImage(192,128,view,proj,renderer=p.ER_TINY_RENDERER,shadow=0)[2]
    return np.asarray(rgba,dtype=np.uint8).reshape(128,192,4)[:,:,:3].copy()


def review(path, mode, label):
    wh = (1040,600)
    w,h=wh
    enc=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate','15','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    fpath='/usr/share/fonts/truetype/dejavu/DejaVuSans'
    title=ImageFont.truetype(fpath+'-Bold.ttf',24); font=ImageFont.truetype(fpath+'.ttf',17)
    cam=p.computeViewMatrix([2.84,-5.68,3.04],[1.29,-.19,.99],[0.,0.,1.])
    proj=p.computeProjectionMatrixFOV(44.,w/h,.11,20.)
    def draw(k, ph, base, obj, forces):
        _,_,rgb,_,seg=p.getCameraImage(w,h,cam,proj,renderer=p.ER_TINY_RENDERER,shadow=0,
                                      lightDirection=[1,-3,5],lightAmbientCoeff=.61)
        im=np.asarray(rgb,dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy(); im[np.asarray(seg).reshape(h,w)<0]=(212,225,240)
        out=Image.fromarray(im); d=ImageDraw.Draw(out)
        d.rounded_rectangle((18,15,614,99),12,fill='#142235')
        d.text((31,24),'X7  /  TWO-HAND MOBILE TOTE CARRY',font=title,fill='white')
        d.text((31,60),f'{mode.replace("-"," ").upper()}   /   {label}   /   PLANAR BASE PROXY',font=font,fill='#afc4dd')
        d.rectangle((0,h-95,w,h),fill='#142235')
        d.text((25,h-74),f'{(k+1)/CONTROL_HZ:04.1f} s    {FRAME_NAMES[ph].upper()}      BASE X {base[0,0]:+.2f} m      TOTE Z {obj[2]:.2f} m',font=font,fill='white')
        d.text((25,h-39),f'Finger normal forces [N]     A  {forces[0]:.1f} / {forces[1]:.1f}       B  {forces[2]:.1f} / {forces[3]:.1f}',font=font,fill='#afc4dd')
        enc.stdin.write(out.tobytes())
    return enc,draw


def execute(world, action, phase, *, movie=None, rgb=False, review_label="SCRIPTED EXPERT"):
    data={k:[] for k in ('base_pre','base_post','arm_pre','gripper_pre','gripper_post','object_pre','object_post','contact_samples','on_table','on_floor','station_clearance')}
    # Isolate abort paths in the simulated scene; every reset occurs before the first recorded command.
    enc,draw=review(movie,world.mode,review_label) if movie else (None,None)
    frames,indices=[],[]
    try:
        for k,(a,ph) in enumerate(zip(action,phase)):
            if rgb and k%6==0:
                frames.append(capture_rgb()); indices.append(k)
            base_pre=world.qv(world.base); arm_pre=world.qv(sum(world.arm_ids,[])); grip_pre=world.qv(sum(world.jaws,[])); o0=world.state_object()
            sample,clear=world.step(a); q1=world.qv(world.base); o1=world.state_object()
            hit=[int(bool(p.getContactPoints(t,world.tote))) for t in world.tables]
            floor=int(bool(p.getContactPoints(world.floor,world.tote)))
            grip_post=world.qv(sum(world.jaws,[]))
            for n,v in zip(data,(base_pre,q1,arm_pre,grip_pre,grip_post,o0,o1,sample,hit,floor,clear)): data[n].append(v)
            if draw and k%2==0: draw(k,int(ph),q1,o1,sample[-1,:4])
    finally:
        if enc:
            enc.stdin.close()
            if enc.wait()!=0: raise RuntimeError('Video encode failed')
    trace={k:np.asarray(v) for k,v in data.items()}
    trace |= {'action':action,'phase':phase,'initial_object_xyz':world.start,'target_xyz':world.goal,
              'station_initialization_clearance':np.asarray(world.station_init)}
    if rgb:
        trace|={'rgb_external':np.asarray(frames),'rgb_action_index':np.asarray(indices,dtype=np.int32)}
    return trace


def score(trace, mode, initial):
    o=trace['object_post']; ph=trace['phase']; b=trace['base_post']; c=trace['contact_samples']; target=trace['target_xyz']
    if not all(np.isfinite(a).all() for a in trace.values()): raise RuntimeError('Non-finite trace')
    delta=o[:,2]-initial[2]; z_axis=Rotation.from_quat(o[:,3:7]).apply([0,0,1.]); tilt=np.degrees(np.arccos(np.clip(z_axis[:,2],-1.,1.)))
    airborne=delta>.055; voyage=(ph==4)&(np.abs(b[:,0,1])>.055); both=np.all(c[:,:,:4]>CONTACT_THRESHOLD_N,axis=2)
    both_frames=np.count_nonzero(np.mean(both,axis=1)>=.75)
    four_transit=np.mean(both[voyage]) if np.any(voyage) else 0.
    dist=float(np.ptp(b[:,0,0])); t_idx=0 if mode=='fixed-lift' else 1
    final=ph==9
    xy=np.linalg.norm(o[:,:2]-target[:2],axis=1)
    jaws_open=np.min(trace['gripper_post'][:,:,0],axis=1)>.030
    stable=jaws_open & (abs(b[:,0,1])<.015) & (xy<.055) & (abs(o[:,2]-SUPPORT_Z)<.018) & (tilt<6.) & (np.linalg.norm(o[:,7:10],axis=1)<.025) &(np.max(c[:,:,:4],axis=(1,2))<.12) & (trace['on_table'][:,t_idx]>0)
    place=bool(np.count_nonzero(stable&final)>=25)
    self_touch=int(np.count_nonzero(c[:,:,6])); nonjaw=int(np.count_nonzero(c[:,:,4])); table_touch=int(np.count_nonzero(c[:,:,5]));
    clear=trace['station_clearance']; init_clear=trace['station_initialization_clearance']
    all_dist=np.vstack((clear[:,:,:2].reshape(-1,2),init_clear[:,:2]))
    near=all_dist.min(axis=0)
    warm_contacts=int(np.count_nonzero(init_clear[:,2:4]))
    below_clearance=int(np.count_nonzero(np.any(all_dist<STATION_CLEARANCE_M,axis=1)))
    penetration=int(np.count_nonzero(np.any(all_dist<0.,axis=1)))
    collision_pass=below_clearance==0 and table_touch==0 and warm_contacts==0
    jaw_center=float(np.max(abs(c[:,:,7:9]))); tilt_air=float(tilt[airborne].max()) if any(airborne) else None
    fallen=bool(np.any(trace['on_floor']))
    plateau=not fallen and -.005<=delta.min()
    carried=voyage & airborne & ~np.any(trace['on_table'],axis=1)
    unsupported_fraction=np.count_nonzero(carried)/max(1,np.count_nonzero(voyage))
    grip_lift=int(np.count_nonzero(airborne & (np.mean(both,axis=1)>.75)))
    object_carry_x=float(o[ph==5][-1,0]-initial[0]); obj_dist=float(np.ptp(o[:,0]));
    mechanical=jaw_center<=SYNC_MAX_OFFSET_M and collision_pass and self_touch==0 and nonjaw==0 and grip_lift>=40 and tilt_air is not None and tilt_air<10. and plateau
    passes=bool(place and mechanical and ((dist<.008) if mode=='fixed-lift' else (dist>=2.5 and object_carry_x>=2.45 and four_transit>=.90 and unsupported_fraction>=.95)))
    return dict(mode=mode,success=passes,final_placement_pass=place,steps=int(len(ph)),
                max_lift_m=float(delta.max()),lowest_relative_height_m=float(delta.min()),
                base_travel_x_m=dist,tote_x_at_carry_end_rel_m=object_carry_x,tote_x_excursion_m=obj_dist,
                simultaneous_four_finger_contact_frames=int(both_frames),dual_grip_while_lifted_frames=grip_lift,
                four_finger_contact_fraction_while_base_moving=float(four_transit),unsupported_airborne_fraction_while_base_moving=float(unsupported_fraction),
                touched_floor=fallen,moving_frames=int(np.count_nonzero(voyage)),
                airborne_tilt_max_deg=tilt_air,final_tilt_deg=float(tilt[-1]),final_goal_xy_error_m=float(xy[-1]),
                settled_released_target_frames=int(np.count_nonzero(stable&final)),
                nonfinger_tote_contact_substeps=nonjaw,robot_station_contact_substeps=table_touch,self_contact_substeps=self_touch,
                robot_station_clearance_pass=bool(collision_pass), station_min_clearance_m=near.tolist(),
                station_query_radius_m=STATION_SCAN_M, station_clearance_threshold_m=STATION_CLEARANCE_M,
                station_clearance_audited_substeps=len(all_dist), station_negative_distance_substeps=penetration,
                station_below_clearance_substeps=below_clearance, station_initialization_contact_substeps=warm_contacts,
                max_gripper_center_offset_mm=1000*jaw_center,rack_check_count=int(c.shape[0]*c.shape[1]),
                object_attachment_constraints_seen=0,state_writes_during_actions=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('asset',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('--cad-zip',type=Path,required=True,help='Original X7 ZIP with raw base CAD for conservative collision envelopes')
    ap.add_argument('--seed',type=int,nargs='+',default=[0]); ap.add_argument('--cases',nargs='+',choices=CASE_NAMES,default=list(CASE_NAMES))
    ap.add_argument('--review',choices=CASE_NAMES)
    ap.add_argument('--rgb',action='store_true')
    args=ap.parse_args()
    if (os.environ.get('CUDA_VISIBLE_DEVICES')!='' or len(set(args.seed))!=len(args.seed)
        or len(set(args.cases))!=len(args.cases) or args.review and (args.review not in args.cases or len(args.seed)!=1)):
        ap.error('Explicitly set CUDA_VISIBLE_DEVICES=""; provide unique cases and seeds; review is one seed/case')
    args.out.mkdir(parents=True)
    asset=make_model(args.asset,args.out/'model',args.cad_zip); reports=[]
    p.connect(p.DIRECT)
    try:
        for seed in args.seed:
            geom=scene(seed); hx,py,ox,base_goal=geom
            homes=[np.array([ox+s*hx,py+.015,GRIP_Z+.105]) for s in (-1,1)]
            arms=[ArmIK(asset,i,home) for i,home in enumerate(homes)]
            commands,phases=make_plan(arms,*geom)
            for case in args.cases:
                u=commands.copy()
                if case in ('fixed-lift','base-locked'): u[:,:2]=0.
                if case=='only-A': u[:,-1]=OPEN
                if case=='only-B': u[:,-2]=OPEN
                w=World(asset,arms,geom,case)
                f=args.out/f'seed-{seed}-{case}.npz'
                trace=execute(w,u,phases,rgb=args.rgb,movie=args.out/f'{case}-expert.mp4' if case==args.review else None)
                np.savez_compressed(f,**trace)
                report=score(trace,case,w.initial_object)
                report |= dict(seed=seed,initial_robot_table_contacts=w.settle_contacts,initial_self_contacts=w.settle_self_contacts,
                               trace=f.name,trace_sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                print(json.dumps(report),flush=True); reports.append(report)
    finally:
        p.disconnect()
    result={"mode":"scripted expert joint trajectories and simulated planar motor drive, not a learned policy",
            "asset_sha256":hashlib.sha256(asset.read_bytes()).hexdigest(), "case_reports":reports, "physics_hz":HZ,"control_hz":CONTROL_HZ,
            "contact_schema":["four finger forces A1 A2 B1 B2", "nonfinger-tote contacts","robot-station contacts","self contacts","two jaw center offsets m"],
            "action_schema":["chassis X reference m", "X velocity feedforward m/s", "arm A Joint1-7 targets rad", "arm B Joint8-14 targets rad", "jaw A and B half travel target m"],
            "move_motors":"force-limited planar joints, no wheels/slip required; Y and yaw held at zero",
            "collision_contract":"v2 fully collidable station compounds, CAD/render envelopes; no contact and >=5 mm from each station",
            "station_clearance_schema":["A signed distance m","B signed distance m","A contact count","B contact count","A nearest robot link index","B nearest robot link index"]}
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    okay=all(r['success'] if r['mode'] in ('fixed-lift','mobile') else not r['final_placement_pass'] for r in reports)
    return 0 if okay else 1


if __name__=='__main__':
    raise SystemExit(main())
