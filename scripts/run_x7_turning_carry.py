#!/usr/bin/env python3
"""X7 turning source expert; force-driven planar base with rotated physical stations.

This is a source-collection experiment, not a learned policy or official
MoMaGen execution. The common held-tote frame turns with the robot; only
motor commands are applied after reset. The stricter v2 station checks stay on.
"""
import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from run_x7_mobile_bimanual import (ArmIK, CONTROL_HZ, FRAME_NAMES, GRIP_Z, HZ, PARK_Y,
                                    SUPPORT_Z, TABLE_Z, World, execute, make_model, make_plan,
                                    score, smooth)
from x7_station_collision import compound_station, station_parts
from x7_sync_gripper_preflight import MASTER_FORCE_N

HANDLE_HALF_SPAN = .589
WXY = (.43,.31,.23)
EFFORT = (1100.,1050.,330.)
LAYOUTS = {
    'source-cw': {'split':'source', 'goal':(3.26,.06,-25.), 'bay_y':.60},
    'source-ccw': {'split':'source', 'goal':(3.22,.10,25.), 'bay_y':.68},
    'heldout-angle': {'split':'heldout', 'goal':(3.53,.23,34.), 'bay_y':.73},
}


def r2(yaw):
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([[c,-s],[s,c]])


def dock_pose(layout):
    x,y,deg=layout['goal']; return np.array([x,y,math.radians(deg)])


def waypoints(layout):
    goal=dock_pose(layout); bay=layout['bay_y']; inner=np.array([1.55,bay])
    # Clear rack A in its own frame before rotating. Then reach a point
    # behind B and drive along B's local negative Y into its dock pose.
    rear=goal[:2]+r2(goal[2])@np.array([0.,.59])
    return [('clear A',[0.,0.,0.]),('turn bay entry',[1.48,0.,0.]),('turn bay', [*inner,0.]),
            ('turn in place',[*inner,goal[2]]),('approach B',[*rear,goal[2]]),('dock B',goal.tolist())]


def drive_segment(start,end):
    a,b=np.array(start),np.array(end); diff=b-a
    n=max(2,int(np.ceil(CONTROL_HZ * max(1.875*abs(diff)/(np.array(WXY)*.83)))))
    t=np.arange(1,n+1)/n
    pos=a+smooth(t)[:,None]*diff
    vel=(30*t*t*(1-t)*(1-t)*CONTROL_HZ/n)[:,None]*diff
    return np.column_stack([pos,vel])


def commands(arms, layout):
    home,phase=make_plan(arms,HANDLE_HALF_SPAN,PARK_Y,0.,dock_pose(layout)[0]); base_seg=waypoints(layout)
    drive=np.concatenate([drive_segment(a,b) for (_,a),(_,b) in zip(base_seg,base_seg[1:])])
    t0,t1=np.flatnonzero(phase==4)[[0,-1]]; before,after=home[:t0],home[t1+1:]
    # Both jaws and both arms reuse one common held-box segment; relative
    # end-effector posture is fixed while the base turns and drives.
    before=np.column_stack([np.zeros((len(before),6)),before[:,2:]])
    during=np.column_stack([drive,np.tile(home[t0,2:],(len(drive),1))])
    after=np.column_stack([np.tile(np.r_[dock_pose(layout),np.zeros(3)],(len(after),1)),after[:,2:]])
    # Extra settle before lowering; no fake pose snaps at a route waypoint.
    u=np.vstack([before,during,after]); ph=np.r_[phase[:t0],np.full(len(during),4,dtype=np.uint8),phase[t1+1:]]
    widths=[len(drive_segment(a,b)) for (_,a),(_,b) in zip(base_seg,base_seg[1:])]
    edges=np.r_[t0,t0+np.cumsum(widths)]
    route=[{'name':name,'start_inclusive':int(i),'end_exclusive':int(j)}
           for (name,_),i,j in zip(base_seg[1:],edges[:-1],edges[1:])]
    return u,ph,route


class TurningWorld(World):
    def __init__(self,model,arms,layout):
        self.dock=dock_pose(layout)
        super().__init__(model,arms,(HANDLE_HALF_SPAN,PARK_Y,0.,float(self.dock[0])),'turning')
        self.goal=np.r_[self.dock[:2]+r2(self.dock[2])@[0.,PARK_Y],SUPPORT_Z]

    def setup_stations(self, ignored_x):
        self.station_parts=[station_parts(0.,PARK_Y,TABLE_Z,rgba) for rgba in
                            ([.47,.58,.68,1],[.41,.61,.52,1])]
        self.tables=[compound_station(self.station_parts[0]),compound_station(self.station_parts[1],
                     position=[*self.dock[:2],0.],yaw=float(self.dock[2]))]

    def drive(self, action):
        if len(action)==18:  # unchanged blank base command during inherited initialization
            action=np.r_[np.zeros(6),action[2:]]
        if len(action)!=22:
            raise ValueError('Invalid turning command width')
        target=action[:3]; feedforward=action[3:6]
        present=self.qv(self.base)[:,0]; error=target-present
        error[2]=(error[2]+np.pi)%(2*np.pi)-np.pi
        v=np.clip(feedforward+3.5*error,-np.array(WXY),np.array(WXY))
        p.setJointMotorControlArray(self.robot,self.base,p.VELOCITY_CONTROL,targetVelocities=v,forces=EFFORT)
        for i,ids in enumerate(self.arm_ids):
            p.setJointMotorControlArray(self.robot,ids,p.POSITION_CONTROL,targetPositions=action[6+i*7:13+i*7],
                                        forces=self.forces[i],positionGains=[.29]*7,velocityGains=[1.]*7)
        for i,(master,_) in enumerate(self.jaws):
            p.setJointMotorControl2(self.robot,master,p.POSITION_CONTROL,targetPosition=action[-2+i],
                                    force=MASTER_FORCE_N,positionGain=.22,velocityGain=1.)


def turning_score(t, start, layout):
    # The straight score still checks contact, released goal, tilt and ground.
    base=score(t,'mobile',start); ph,pb,obj=t['phase'],t['base_post'],t['object_post']; drive=ph==4
    moving=drive & ((np.linalg.norm(pb[:,:2,1],axis=1)>.045) | (abs(pb[:,2,1])>.04))
    turning=drive & (abs(pb[:,2,1])>.06)
    contacts=np.all(t['contact_samples'][:,:,:4]>.12,axis=2)
    airborne=(obj[:,2]-start[2]>.055)&~np.any(t['on_table'],axis=1)
    final_frame=float(np.sum(np.linalg.norm(np.diff(pb[:,:2,0],axis=0),axis=1)))
    degree=np.degrees(np.unwrap(pb[:,2,0])); actual_yaw=float(degree[-1]); wanted=dock_pose(layout)
    yaw=Rotation.from_quat(obj[:,3:7]).as_euler('zyx',degrees=True)[:,0]
    yerr=abs((yaw-layout['goal'][2]+180)%360-180)
    end_base_error=float(np.linalg.norm(pb[-1,:2,0]-wanted[:2]))
    turn_contact=float(contacts[turning].mean()) if turning.any() else 0.
    move_contact=float(contacts[moving].mean()) if moving.any() else 0.
    mobile_air=float(airborne[moving].mean()) if moving.any() else 0.
    turn_air=float(airborne[turning].mean()) if turning.any() else 0.
    zturn_error=np.max(abs((yaw[turning]-degree[turning]+180)%360-180)) if turning.any() else None
    success=bool(base['success'] and final_frame>3.0 and abs(actual_yaw-layout['goal'][2])<1.5 and end_base_error<.020
                 and np.ptp(degree[drive])>20 and moving.sum()>100 and turning.sum()>25 and move_contact>=.98
                 and turn_contact>=.98 and mobile_air>=.98 and turn_air>=.98 and zturn_error is not None and zturn_error<5. and yerr[-1]<5.)
    first=lambda mask: int(np.flatnonzero(mask)[0]) if np.any(mask) else None
    bilateral=np.mean(contacts,axis=1)>=.75
    released=(ph>=7)&(np.min(t['gripper_post'][:,:,0],axis=1)>.030)&(t['on_table'][:,1]>0)
    events={'both_grips':first((ph==2)&bilateral),'unsupported_lift':first((ph==3)&airborne&bilateral),
            'holding_while_turning':first(turning & airborne & bilateral),'B_release':first(released)}
    success=success and all(v is not None for v in events.values()) and list(events.values())==sorted(events.values())
    return base|dict(success=success, planar_path_m=final_frame, base_final_xy_error_m=end_base_error,
                     base_yaw_final_deg=actual_yaw, base_yaw_excursion_deg=float(np.ptp(degree[drive])),
                     tote_final_dock_yaw_error_deg=float(yerr[-1]), max_tote_base_yaw_error_during_turn_deg=float(zturn_error),
                     both_grip_substep_fraction_turn=turn_contact,both_grip_substep_fraction_motion=move_contact,
                     unsupported_fraction_turn=turn_air,unsupported_fraction_motion=mobile_air,turning_control_frames=int(turning.sum()),
                     physical_events=events)


def camera_view(width,height):
    view=p.computeViewMatrix([3.40,-6.27,3.44],[1.66,.12,.96],[0,0,1])
    proj=p.computeProjectionMatrixFOV(42,width/height,.11,20.)
    return view,proj


def capture_turn_rgb():
    w,h=192,128
    view,proj=camera_view(w,h)
    rgba=p.getCameraImage(w,h,view,proj,renderer=p.ER_TINY_RENDERER,shadow=0)[2]
    return np.asarray(rgba,dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy()


def review_turn(path, mode, label):
    w,h=1040,600
    enc=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}',
        '-framerate','15','-i','pipe:0','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    f='/usr/share/fonts/truetype/dejavu/DejaVuSans'
    title=ImageFont.truetype(f+'-Bold.ttf',24); font=ImageFont.truetype(f+'.ttf',16)
    view,proj=camera_view(w,h)
    def frame(k,ph,base,obj,force):
        _,_,rgb,_,seg=p.getCameraImage(w,h,view,proj,renderer=p.ER_TINY_RENDERER,shadow=0,lightDirection=[1,-3,5],lightAmbientCoeff=.61)
        im=np.asarray(rgb,dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy(); im[np.asarray(seg).reshape(h,w)<0]=(212,225,240)
        image=Image.fromarray(im); d=ImageDraw.Draw(image)
        d.rounded_rectangle((18,15,662,94),10,fill='#142235')
        d.text((32,23),'X7  /  BIMANUAL TURN & DOCK',font=title,fill='white')
        banner='SCRIPTED SOURCE DEMO' if label=='SCRIPTED EXPERT' else label
        d.text((33,59),f'{banner}   /   CONTACT ONLY   /   PLANAR BASE PROXY',font=font,fill='#aec4dd')
        d.rectangle((0,h-71,w,h),fill='#142235')
        d.text((22,h-59),f'{(k+1)/CONTROL_HZ:05.1f}s   {FRAME_NAMES[ph].upper()}   BASE XY {base[0,0]:.2f}, {base[1,0]:.2f} m   YAW {np.degrees(base[2,0]):+.1f}°',font=font,fill='white')
        d.text((22,h-32),f'Finger forces (N)   A {force[0]:.1f}/{force[1]:.1f}    B {force[2]:.1f}/{force[3]:.1f}',font=font,fill='#aec4dd')
        enc.stdin.write(image.tobytes())
    return enc,frame


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('asset',type=Path); ap.add_argument('out',type=Path)
    ap.add_argument('--cad-zip',type=Path,required=True); ap.add_argument('--cases',nargs='+',choices=LAYOUTS,default=list(LAYOUTS))
    ap.add_argument('--review',choices=LAYOUTS); ap.add_argument('--rgb',action='store_true'); args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or len(set(args.cases))!=len(args.cases) or args.review and args.review not in args.cases:
        ap.error('Remote CPU only. Set CUDA_VISIBLE_DEVICES empty. Select unique cases; review must be among them.')
    args.out.mkdir(parents=True); model=make_model(args.asset,args.out/'model',args.cad_zip)
    arms=[ArmIK(model,i,np.array([s*HANDLE_HALF_SPAN,PARK_Y+.015,GRIP_Z+.105])) for i,s in enumerate((-1,1))]
    p.connect(p.DIRECT); rows=[]
    try:
        for case in args.cases:
            layout=LAYOUTS[case]; u,ph,routes=commands(arms,layout); w=TurningWorld(model,arms,layout)
            movie=args.out/f'{case}.mp4' if case==args.review else None
            trace=execute(w,u,ph,movie=movie,reviewer=review_turn,rgb=args.rgb,camera=capture_turn_rgb)
            trace['dock_pose']=w.dock
            if args.rgb and (not np.array_equal(trace['rgb_action_index'],np.arange(0,len(u),6)) or
                             trace['rgb_external'].shape!=(len(trace['rgb_action_index']),128,192,3)):
                raise RuntimeError('External RGB/action alignment failed')
            result=turning_score(trace,w.initial_object,layout)
            before=np.flatnonzero(ph==3)[0]; place=np.flatnonzero(ph==6)[0]
            result|={'case':case,'partition':layout['split'],'target_dock_pose':dock_pose(layout).tolist(),
                     'waypoints':waypoints(layout),'route_segments':routes,
                     'subtask_intervals':[['tote_at_A',0,int(before)],['joint_transport_frame',int(before),int(place)],
                                          ['rotated_station_B',int(place),len(ph)]],
                     'rgb_frames':len(trace.get('rgb_external', []))}
            folder=args.out/layout['split']/('accepted' if result['success'] else 'rejected'); folder.mkdir(parents=True,exist_ok=True)
            f=folder/f'{case}.npz'; np.savez_compressed(f,**trace)
            result|={'trace':str(f.relative_to(args.out)),'trace_sha256':sha256(f.read_bytes()).hexdigest()}
            rows.append(result); print(json.dumps(result),flush=True)
            (args.out/'attempts.json').write_text(json.dumps(rows,indent=2)+'\n')
    finally:
        p.disconnect()
    report={'workflow':'scripted X7 turning sources; heldout is a physically separate docking layout, not a trained policy test',
            'action_schema':['world base x,y,yaw references','world x,y,yaw velocity feedforward','joint targets A[7], B[7]','jaw widths A,B'],
            'asset_sha256':sha256(model.read_bytes()).hexdigest(),'case_reports':rows,
            'all_pass':all(x['success'] for x in rows),'training_started':False,
            'physics_hz':HZ,'control_hz':CONTROL_HZ,'external_rgb_hz':5 if args.rgb else 0,
            'rgb_alignment':'pre-action external observation matched by rgb_action_index'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['all_pass'] else 1


if __name__=='__main__':
    raise SystemExit(main())
