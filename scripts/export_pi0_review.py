#!/usr/bin/env python3
"""SERVER ONLY: compact real pi0 rollout video, hashes and honest review labels.

Export both successful and failed completed episodes, never relabel a failed
attempt as success. Weights, datasets, simulator assets and tokens are not copied.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
from export_demo_review import read_video,sha256_file
from pi0_review_contract import fingerprint_checkpoint,physical_task_passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--require-success',action='store_true')
    args=parser.parse_args()
    root=Path('/mnt/yuanyue/double-arm')
    if not args.output.resolve().is_relative_to((root/'runs').resolve()):raise ValueError('Remote review output required')
    if args.output.exists():raise FileExistsError(args.output)
    training=json.loads((args.campaign/'status.json').read_text())
    assessed=json.loads((args.evaluation/'status.json').read_text())
    result=json.loads((args.evaluation/'physical-rollout/result.json').read_text())
    if training.get('training')!='completed':raise ValueError('Formal training is not completed')
    if (result.get('mode')!='closed_loop_neural_policy' or result.get('algorithm')!='official_openpi_pi0_lora32'
        or result.get('privileged_state') is not False or result.get('planner_enabled') is not False
        or result.get('teacher_action_reads')!=0):raise ValueError('Not an authentic teacher-free visual pi0 rollout')
    episodes=result.get('episodes',[])
    if len(episodes)!=1:raise ValueError('No single completed episode to review')
    episode=episodes[0]
    if episode.get('initial_state_load_count')!=1:raise ValueError('Unexpected state replay/reset')
    checkpoint=Path(assessed['checkpoint'])
    if not checkpoint.resolve().is_relative_to((root/'runs').resolve()):raise ValueError('Project checkpoint required')
    if str(checkpoint)!=result.get('checkpoint'):raise ValueError('Checkpoint provenance mismatch')
    passed=physical_task_passed(result) and assessed.get('task_success') is True
    if args.require_success and not passed:raise ValueError('Physical task acceptance did not pass')
    label='PI0 PHYSICAL SUCCESS - MANUAL REVIEW PENDING' if passed else 'PI0 FAILED ATTEMPT - NOT A SUCCESS DEMO'
    rollout=args.evaluation/'physical-rollout'
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    with np.load(rollout/(episode['demo']+'-trace.npz'),allow_pickle=False) as trace:
        for key,width in (('actions',21),('observations',21),('base_positions',3)):
            values=trace[key]
            if values.shape!=(episode['steps'],width) or not np.isfinite(values).all():
                raise ValueError('Invalid or misaligned rollout trace')
    fingerprint=fingerprint_checkpoint(checkpoint)
    args.output.mkdir()
    source=rollout/(episode['demo']+'.mp4');destination=args.output/'demo.mp4'
    subprocess.run(['ffmpeg','-hide_banner','-v','error','-n','-i',str(source),'-vf',
        "drawbox=x=0:y=ih-30:w=iw:h=30:color=black:t=fill,drawtext=text='"+label+"':x=8:y=h-23:fontsize=16:fontcolor=yellow",
        '-c:v','libx264','-preset','medium','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart','-an',str(destination)],check=True)
    video=read_video(destination)
    if abs(video['duration']-episode['steps']/30)>1:raise ValueError('Video timebase differs from 30Hz simulator')
    subprocess.run(['ffmpeg','-v','error','-i',str(destination),'-f','null','-'],check=True)
    try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    except OSError:font=ImageFont.load_default()
    frames=[]
    for fraction in (.02,.20,.40,.60,.80,.98):
        second=max(0,min(video['duration']*fraction,video['duration']-.1))
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(second),'-i',str(destination),'-frames:v','1',
            '-vf','scale=640:-2','-f','image2pipe','-vcodec','png','-'])
        image=Image.open(io.BytesIO(raw)).convert('RGB')
        ImageDraw.Draw(image).text((8,32),f'{second:.1f}s / {result.get("inference_platform","unknown").upper()}',font=font,fill='yellow',stroke_width=1,stroke_fill='black')
        frames.append(image)
    width,height=frames[0].size;sheet=Image.new('RGB',(width*3,height*2),'#161616')
    for index,image in enumerate(frames):sheet.paste(image,((index%3)*width,(index//3)*height))
    sheet.save(args.output/'contact-sheet.jpg',quality=88)
    review={'label':label,'algorithm':result['algorithm'],'task':'r1_tidy_table_D0','physical_task_acceptance':passed,
        'manual_video_review_required':True,'inference_platform':result.get('inference_platform'),
        'real_time_control_claim':False,'checkpoint_fingerprint':fingerprint['sha256'],'checkpoint':str(checkpoint),
        'formal_training_completed':True,'train_demonstrations':training.get('train_demonstrations'),
        'train_frames':training.get('train_frames'),'source_data_grasping':'author assisted attachment',
        'evaluation':result,'video':video,'video_sha256':sha256_file(destination),
        'source_video_sha256':sha256_file(source),'trace_sha256':sha256_file(rollout/(episode['demo']+'-trace.npz')),
        'source_campaign':str(args.campaign),'source_evaluation':str(args.evaluation),
        'physical_parameters_calibrated_to_real_robot':False,'upstream_finger_friction':4.0,
        'not_cooperative_bimanual':True,'not_cross_room_navigation':True,
        'local_only_restricted_assets':'BEHAVIOR noncommercial academic provenance; no public dataset/media upload'}
    (args.output/'checkpoint-fingerprint.json').write_text(json.dumps(fingerprint,indent=2))
    (args.output/'review.json').write_text(json.dumps(review,indent=2))
    print(json.dumps({'review_dir':str(args.output),'label':label,'physical_task_acceptance':passed},indent=2),flush=True)


if __name__=='__main__':main()
