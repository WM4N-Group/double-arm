#!/usr/bin/env python3
"""CPU state-feedback imitation pilot for the fixed-layout X7 proxy-gripper task.

A separate ridge regressor per control tick predicts absolute motor targets.
It is intentionally small and time-indexed; camera learning and broad task
transfer are out of scope. Training sees only whole episodes in the train set.
"""
import argparse
from hashlib import sha256
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

from x7_pick_place_pilot import CONTROL_HZ, PHASES, Kinematics, World, run

N_STEPS = 317
FEATURE_NAMES = ['initial_block_x','initial_block_y','goal_x','goal_y',
                 *[f'sensed_pos_{k}' for k in range(9)], *[f'sensed_vel_{k}' for k in range(9)],
                 'observed_block_x','observed_block_y','observed_block_z']
# Predictions use simulated ground-truth object position, not image detections.
# Penalize reliance on the narrow training distribution of dynamic features.
SCALE = np.array([.025]*4 + [.13]*7 + [.012]*2 + [.45]*7 + [.045]*2 + [.025]*3)
PENALTY = np.r_[[.014]*4, [175.]*21]


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_manifest(folder, role, expected_seeds):
    reports = json.loads((folder/'results.json').read_text())
    if [r['seed'] for r in reports]!=list(expected_seeds) or any(
            not r['success'] or r['partition']!=role or r['constraints_seen'] or r['object_pose_writes_after_reset']
            for r in reports):
        raise ValueError(f'Unverified or incomplete {role} partition: {folder}')
    return reports


def check_episodes(folder, reports):
    samples = []
    for r in reports:
        name = f'episode-{r["seed"]}.npz'
        f = folder/name
        if r['file']!=name or r['sha256']!=digest(f):
            raise ValueError(f'Unverified file: {f}')
        with np.load(f,allow_pickle=False) as d:
            times = d['rgb_action_index']
            if (int(d['seed'])!=r['seed'] or d['action'].shape!=(N_STEPS,8)
                or d['q'].shape!=(N_STEPS,9) or d['dq'].shape!=(N_STEPS,9)
                or d['object_pre'].shape!=(N_STEPS,13) or d['phase'].shape!=(N_STEPS,)
                or d['rgb'].dtype!=np.uint8 or d['rgb'].shape!=(53,120,160,3)
                or not np.array_equal(times,np.arange(0,N_STEPS,6))
                or np.min(np.ptp(d['rgb'].reshape((53,-1)),axis=1))<32):
                raise ValueError(f'Data schema or RGB alignment check failed: {f}')
            vals = {k:d[k] for k in ('q','dq','object_pre','action','phase','start','goal')}
            if not all(np.isfinite(v).all() for v in vals.values()):
                raise ValueError(f'Non-finite data: {f}')
            samples.append(vals)
    return samples


def features(q, dq, obj, start, goal):
    ctx = np.broadcast_to(np.r_[start[:2],goal[:2]],q.shape[:-1]+(4,))
    return np.concatenate((ctx,q,dq,obj[...,:3]), axis=-1)


def fit(samples):
    """Fit exactly one deterministic model using TRAIN state/actions only."""
    x = np.stack([features(d['q'],d['dq'],d['object_pre'],d['start'],d['goal']) for d in samples],axis=1)
    y = np.stack([d['action'] for d in samples],axis=1)
    xmean, ymean = x.mean(axis=1),y.mean(axis=1)
    xt = (x-xmean[:,None,:])/SCALE
    yt = y-ymean[:,None,:]
    design = np.swapaxes(xt,1,2)
    weights = np.linalg.solve(design@xt+np.diag(PENALTY),design@yt)
    yh = ymean[:,None,:]+xt@weights
    return {'xmean':xmean,'ymean':ymean,'weights':weights,'scale':SCALE,
            'phases':samples[0]['phase']}, error(yh,y)


def error(pred, reference):
    d = np.asarray(pred)-reference
    return {'joint_rmse_deg':float(np.rad2deg(np.sqrt(np.mean(d[...,:7]**2)))),
            'jaw_target_rmse_mm':float(1000*np.sqrt(np.mean(d[...,7]**2)))}


def predict(model, k, f):
    norm = np.clip((f-model['xmean'][k])/model['scale'],-5.,5.)
    a = model['ymean'][k] + norm@model['weights'][k]
    return np.clip(a,[-3.12]*7+[.004],[3.12]*7+[.036])


def policy_video(path, seed):
    w,h=800,520
    encoder=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pixel_format','rgb24',
        '-video_size',f'{w}x{h}','-framerate','15','-i','pipe:0','-c:v','libx264','-threads','1',
        '-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    base='/usr/share/fonts/truetype/dejavu/DejaVuSans'
    title=ImageFont.truetype(base+'-Bold.ttf',21)
    font=ImageFont.truetype(base+'.ttf',15)
    def write(world,k,stage,force):
        img=Image.fromarray(world.camera(w,h)); d=ImageDraw.Draw(img)
        d.rounded_rectangle((14,14,530,86),10,fill='#152336')
        d.text((28,23),f'X7  /  LEARNED STATE BC  /  HELDOUT {seed}',font=title,fill='#f4f8fe')
        d.text((29,56),'Live simulated state  •  No RGB input  •  No expert actions',font=font,fill='#aabed6')
        d.rectangle((0,h-37,w,h),fill='#152336')
        d.text((16,h-28),f'{k/CONTROL_HZ:04.1f}s    {PHASES[stage].upper()}    |    Finger normal force: {force[0]:.1f} / {force[1]:.1f} N',
               font=font,fill='#f4f8fe')
        encoder.stdin.write(img.tobytes())
    return encoder,write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('train',type=Path)
    ap.add_argument('heldout',type=Path)
    ap.add_argument('out',type=Path)
    ap.add_argument('--review-seed',type=int,choices=range(900,906))
    args=ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        ap.error('Set CUDA_VISIBLE_DEVICES to the empty string for this CPU-only pilot')
    args.out.mkdir(parents=True)
    tr=check_manifest(args.train,'train',range(800,824))
    train=check_episodes(args.train,tr)
    if not all(-.202<=s['start'][1]<=-.188 or -.166<=s['start'][1]<=-.151 for s in train):
        raise ValueError('Unexpected training layout bands')
    model,loss=fit(train)
    ckpt=args.out/'state-feedback-bc.npz'
    np.savez_compressed(ckpt,**model,feature_names=FEATURE_NAMES)
    model_sha=digest(ckpt)
    with np.load(ckpt,allow_pickle=False) as f:
        fresh={k:f[k] for k in model}
    if not all(np.array_equal(v,fresh[k]) for k,v in model.items()):
        raise RuntimeError('Independent checkpoint reload mismatch')
    print(json.dumps({'checkpoint_sha256':model_sha,'training_error':loss,'heldout_seen_by_trainer':False}),flush=True)
    # No heldout obs or labels are opened before the model is committed.
    te=check_manifest(args.heldout,'heldout',range(900,906))
    heldout=check_episodes(args.heldout,te)
    if not all(-.181<=s['start'][1]<=-.173 for s in heldout):
        raise ValueError('Train/heldout layout regimes overlap or changed')
    u_offline=[]; u_expert=[]; outcomes=[]
    urdf=args.train/'model'/'x7_proxy_gripper.urdf'
    kin=Kinematics(urdf)
    p.connect(p.DIRECT)
    try:
        for d,r in zip(heldout,te):
            seed=r['seed']
            off=features(d['q'],d['dq'],d['object_pre'],d['start'],d['goal'])
            u_offline.append([predict(fresh,k,f) for k,f in enumerate(off)]); u_expert.append(d['action'])
            world=World(urdf,kin,d['start'],d['goal'])
            def learned_targets():
                for k in range(N_STEPS):
                    q,dq=world.q()
                    yield predict(fresh,k,features(q,dq,world.obj(),world.x0,d['goal']))
            enc,write=policy_video(args.out/f'learned-heldout-{seed}.mp4',seed) if seed==args.review_seed else (None,None)
            try:
                trace,res=run(world,learned_targets(),fresh['phases'],video=write)
            finally:
                if enc:
                    enc.stdin.close()
                    if enc.wait()!=0:
                        raise RuntimeError('Failed to encode live learned-rollout review')
            res |= {'seed':seed,'source':'trained time-indexed state-feedback ridge BC (no teacher/controller plan in rollout)',
                    'checkpoint_sha256':model_sha,'actual_policy_calls':len(trace['action']),
                    'expert_actions_used_for_control':0,'rgb_used_for_control':False}
            np.savez_compressed(args.out/f'learned-rollout-{seed}.npz',**trace,seed=seed,start=d['start'],goal=d['goal'])
            outcomes.append(res); print(json.dumps(res),flush=True)
    finally:
        p.disconnect()
    report={'status':'trained_reloaded_and_evaluated','checkpoint_sha256':model_sha,
            'training_partition_manifest_sha256':digest(args.train/'results.json'),
            'heldout_partition_manifest_sha256':digest(args.heldout/'results.json'),
            'train_episodes':len(train),'heldout_episodes':len(heldout),'train_error':loss,
            'offline_heldout_error':error(u_offline,np.asarray(u_expert)),
            'heldout_physical_successes':sum(r['success'] for r in outcomes),
            'heldout_scope':'new seeds and a disjoint initial Y band within same tabletop scene; interpolation only',
            'policy_scope':'time + known simulated start/goal + live proprioception and simulator ground-truth object position; not a camera policy',
            'condition':'one arm, fixed base, synthetic parallel jaw, small cube, yaw fixed, contact only',
            'rollouts':outcomes}
    (args.out/'training-and-eval.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rollouts'}),flush=True)


if __name__=='__main__':
    main()
