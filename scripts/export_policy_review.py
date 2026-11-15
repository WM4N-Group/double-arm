#!/usr/bin/env python3
"""SERVER ONLY: export an explicitly labelled learned-policy video review pack.

Failures may be exported for diagnosis but never labelled as successful demos.
No datasets, weights, assets or license material are copied into the pack.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from export_demo_review import read_video, sha256_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy-run',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--demo')
    p.add_argument('--require-success',action='store_true')
    p.add_argument('--require-physical',action='store_true')
    args=p.parse_args()
    train=json.loads((args.policy_run/'training-result.json').read_text())
    result=json.loads((args.policy_run/'evaluation/result.json').read_text())
    manifest=json.loads((args.policy_run/'manifest.json').read_text())
    if train.get('status')!='training_completed': raise ValueError('Training did not finish')
    if result.get('mode') not in ('closed_loop_neural_policy','closed_loop_learned_policy') or result.get('planner_enabled') is not False or result.get('teacher_action_reads')!=0:
        raise ValueError('Not a teacher-free closed-loop policy evaluation')
    grasping_mode=result.get('grasping_mode',manifest['env_args']['env_kwargs']['robots'][0].get('grasping_mode','unspecified'))
    if args.require_physical and (grasping_mode!='physical' or result.get('assisted_constraint_seen') is not False or result.get('object_pose_write_count')!=0):
        raise ValueError('No explicit physical-only / zero-assisted-constraint evidence')
    episodes=result.get('episodes',[])
    candidates=[e for e in episodes if e['demo']==args.demo] if args.demo else sorted(episodes,
           key=lambda e:(e.get('sustained_success',False),e.get('split')=='validation'),reverse=True)
    if not candidates: raise ValueError('No completed evaluation episode')
    episode=candidates[0]
    if args.require_success and not episode.get('sustained_success',False): raise ValueError('No sustained task success')
    if episode.get('initial_state_load_count')!=1: raise ValueError('Unexpected initial state restoration count')
    source=args.policy_run/'evaluation'/f'{episode["demo"]}.mp4'
    trace=args.policy_run/'evaluation'/f'{episode["demo"]}-trace.npz'
    with np.load(trace) as file:
        if any(len(file[k])!=episode['steps'] for k in ('actions','observations','base_positions')):
            raise ValueError('Policy trace length differs from evaluation')
        if not np.isfinite(file['actions']).all():raise ValueError('Nonfinite actions in trace')
    args.output.mkdir(parents=True,exist_ok=False)
    dest=args.output/'demo.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-v','error','-n','-i',str(source),'-c:v','libx264',
           '-preset','medium','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(dest)],check=True)
    video=read_video(dest)
    action_hz=manifest['env_args']['env_kwargs']['env']['action_frequency']
    if abs(video['duration']-episode['steps']/action_hz)>1.0:
        raise ValueError('Video timebase differs from simulation action rate')
    subprocess.run(['ffmpeg','-v','error','-i',str(dest),'-f','null','-'],check=True)
    try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    except OSError:font=ImageFont.load_default()
    thumbs=[]
    policy_label='MEMORY BC / NOT NEURAL' if manifest['algorithm']=='nearest_state_memory_BC' else 'LEARNED POLICY'
    numerator,denominator=map(float,video['r_frame_rate'].split('/'))
    last_frame_time=max(0,video['duration']-2*denominator/numerator)
    for fraction in (0.02,0.20,0.40,0.60,0.80,1.0):
        second=max(0,min(video['duration']*fraction,last_frame_time))
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(second),'-i',str(dest),
             '-frames:v','1','-vf','scale=640:-2','-f','image2pipe','-vcodec','png','-'])
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        ImageDraw.Draw(image).text((8,32),f'{second:.1f}s / {policy_label}',font=font,fill='yellow',stroke_width=1,stroke_fill='black')
        thumbs.append(image)
    width,height=thumbs[0].size
    sheet=Image.new('RGB',(width*3,2*(height+24)),'#161616')
    for i,image in enumerate(thumbs):sheet.paste(image,((i%3)*width,(i//3)*(height+24)))
    sheet.save(args.output/'contact-sheet.jpg',quality=88)
    review={'task':manifest['env_args']['env_name'],'method':manifest['algorithm'],
          'training_method':manifest.get('training_method',manifest['algorithm']),'privileged_state':True,'single_demo_smoke':manifest.get('single_demo_smoke',False),
          'train_demos':manifest['train_demos'],'val_demos':manifest['val_demos'],'rendering_mode':result['mode'],
          'policy_training':'completed','planner_enabled':False,'teacher_action_reads':0,
          'grasping_mode':grasping_mode,'assisted_constraint_seen':result.get('assisted_constraint_seen'),'object_pose_write_count':result.get('object_pose_write_count'),'upstream_finger_friction':result.get('upstream_finger_friction'),'physical_parameters_calibrated_to_real_robot':False,'task_success':episode.get('sustained_success',False),'episode':episode,
          'source_video':str(source),'checkpoint':result['checkpoint'],
          'checkpoint_sha256':sha256_file(result['checkpoint']),'training_dataset_sha256':manifest['dataset_sha256'],
          'video':video,'caveat':('Explicit single-demonstration overfit smoke test, not a generalization claim. Single active arm R1 task.' if manifest.get('single_demo_smoke') else 'Generated source distribution, not a new-scene generalization claim. Single active arm R1 task.')}
    if manifest['algorithm']=='nearest_state_memory_BC':
        review['caveat']='Nonparametric nearest-state memory BC, NOT a gradient-trained neural policy. '+review['caveat']
    if grasping_mode=='physical':
        review['caveat']+=' Original task gripper friction is 4.0; not calibrated to real hardware.'
    if grasping_mode!='physical':
        review['caveat']+=' Assisted grasp attachment enabled; NOT verified as pure contact/friction grasping.'
    (args.output/'review.json').write_text(json.dumps(review,indent=2))
    print(json.dumps(review,indent=2),flush=True)


if __name__=='__main__':main()
