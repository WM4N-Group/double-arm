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
    args=p.parse_args()
    train=json.loads((args.policy_run/'training-result.json').read_text())
    result=json.loads((args.policy_run/'evaluation/result.json').read_text())
    manifest=json.loads((args.policy_run/'manifest.json').read_text())
    if train.get('status')!='training_completed': raise ValueError('Training did not finish')
    if result.get('mode')!='closed_loop_neural_policy' or result.get('planner_enabled') is not False or result.get('teacher_action_reads')!=0:
        raise ValueError('Not a teacher-free closed-loop policy evaluation')
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
    for fraction in (0.02,0.20,0.40,0.60,0.80,0.98):
        second=max(0,min(video['duration']*fraction,video['duration']-0.05))
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(second),'-i',str(dest),
             '-frames:v','1','-vf','scale=640:-2','-f','image2pipe','-vcodec','png','-'])
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        ImageDraw.Draw(image).text((8,32),f'{second:.1f}s / LEARNED POLICY',font=font,fill='yellow',stroke_width=1,stroke_fill='black')
        thumbs.append(image)
    width,height=thumbs[0].size
    sheet=Image.new('RGB',(width*3,2*(height+24)),'#161616')
    for i,image in enumerate(thumbs):sheet.paste(image,((i%3)*width,(i//3)*(height+24)))
    sheet.save(args.output/'contact-sheet.jpg',quality=88)
    review={'task':manifest['env_args']['env_name'],'method':'learned state-history action-chunk BC',
          'privileged_state':True,'rendering_mode':'closed_loop_neural_policy',
          'policy_training':'completed','planner_enabled':False,'teacher_action_reads':0,
          'task_success':episode.get('sustained_success',False),'episode':episode,
          'source_video':str(source),'checkpoint':result['checkpoint'],
          'checkpoint_sha256':sha256_file(result['checkpoint']),'training_dataset_sha256':manifest['dataset_sha256'],
          'video':video,'caveat':'Generated source distribution, not a new-scene generalization claim. Single active arm R1 task.'}
    (args.output/'review.json').write_text(json.dumps(review,indent=2))
    print(json.dumps(review,indent=2),flush=True)


if __name__=='__main__':main()
