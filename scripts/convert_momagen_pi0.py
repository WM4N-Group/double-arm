#!/usr/bin/env python3
"""SERVER ONLY: successful official MoMaGen RGB demonstrations -> local LeRobot.

Uses all three REAL RGB views plus prop_state; excludes object poses, source
phase, time and serialized scene states from policy features. Keeps one entire
native seed0 demonstration held out. Never uploads restricted BEHAVIOR data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
from momagen_pi0_schema import camera_keys,rgb_image,native_actions,PROMPT,STATE_NAMES,ACTION_NAMES


def successful_sources(run,stable_seconds=15):
    for task in sorted((Path(run)/'generated').glob('*')):
        final=task/'demo.hdf5'
        candidates=[final] if final.is_file() else sorted((task/'tmp').glob('*.hdf5'))
        for candidate in candidates:
            if candidate.stat().st_mtime+stable_seconds<=time.time():yield candidate


def inspect_episode(group):
    if bool(group.attrs.get('partial',False)):raise ValueError('Partial episode')
    obs=group['obs'];cameras=camera_keys(obs);actions=native_actions(group['actions'][()]);state=obs['prop_state'][()]
    if state.shape!=(len(actions),21) or not np.isfinite(state).all():raise ValueError('Invalid proprioception')
    for name in cameras.values():
        if obs[name].shape[:1]!=(len(actions),):raise ValueError('Camera / action lengths differ')
        samples=[rgb_image(obs[name][index]) for index in sorted({0,len(actions)//2,len(actions)-1})]
        if max(float(image.std()) for image in samples)<1.0:
            raise ValueError(f'Camera is blank / constant in sampled frames: {name}')
    if len(actions)<30:raise ValueError('Too short for a full demonstration')
    digest=hashlib.sha256(actions.tobytes()+state[0].tobytes()).hexdigest()
    return cameras,actions,state,digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,action='append',required=True)
    p.add_argument('--output-root',type=Path,default=Path('/mnt/yuanyue/double-arm/data/pi0/lerobot'))
    p.add_argument('--train-repo',default='momagen/r1_tidy_table_train_v1')
    p.add_argument('--validation-repo',default='momagen/r1_tidy_table_holdout_v1')
    p.add_argument('--min-train-episodes',type=int,default=4)
    args=p.parse_args()
    if not str(args.output_root).startswith('/mnt/'):raise ValueError('Remote output required')
    for repo in (args.train_repo,args.validation_repo):
        if repo.startswith('/') or '..' in Path(repo).parts:raise ValueError('Unsafe dataset id')
        if (args.output_root/repo).exists():raise FileExistsError(args.output_root/repo)
    metadata=[];seen=set();rejected=[]
    for index,run in enumerate(args.run_dir):
        for path in successful_sources(run):
            try:
                with h5py.File(path,'r') as file:
                    env=json.loads(file['data'].attrs['env_args'])
                    if env['env_name']!='r1_tidy_table_D0':raise ValueError('Wrong task')
                    for name in sorted(file['data'],key=lambda x:int(x.split('_')[-1]) if x.startswith('demo_') else -1):
                        if not name.startswith('demo_'):continue
                        cameras,actions,state,fingerprint=inspect_episode(file['data'][name])
                        if fingerprint in seen:continue
                        split='validation' if index==0 and not any(item['split']=='validation' for item in metadata) else 'train'
                        seen.add(fingerprint);metadata.append({'source':str(path),'source_demo':name,'split':split,
                            'fingerprint':fingerprint,'frames':len(actions),'cameras':cameras,'env_name':env['env_name'],
                            'source_grasping_mode':env['env_kwargs']['robots'][0].get('grasping_mode','unspecified')})
            except (OSError,ValueError,KeyError) as error:rejected.append({'source':str(path),'error':repr(error)})
    train=[entry for entry in metadata if entry['split']=='train'];validation=[entry for entry in metadata if entry['split']=='validation']
    if len(train)<args.min_train_episodes or len(validation)!=1:
        print(json.dumps({'train_episodes':len(train),'validation_episodes':len(validation),'rejected':rejected},indent=2))
        raise SystemExit(3)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    for split,repo,episodes in [('train',args.train_repo,train),('validation',args.validation_repo,validation)]:
        features={key:{'dtype':'image','shape':(256,256,3),'names':['height','width','channel']} for key in ['head_image','left_wrist_image','right_wrist_image']}
        features.update(state={'dtype':'float32','shape':(21,),'names':list(STATE_NAMES)},actions={'dtype':'float32','shape':(21,),'names':list(ACTION_NAMES)})
        dataset=LeRobotDataset.create(repo_id=repo,root=args.output_root/repo,robot_type='momagen_r1',fps=30,features=features,
                                      use_videos=False,image_writer_threads=4,image_writer_processes=0)
        for entry in episodes:
            with h5py.File(entry['source'],'r') as file:
                g=file['data'][entry['source_demo']];cameras,actions,state,_=inspect_episode(g)
                for index in range(len(actions)):
                    frame={key:rgb_image(g['obs'][name][index]) for key,name in cameras.items()}
                    if any(value.shape!=(256,256,3) for value in frame.values()):raise ValueError('Unexpected native camera resolution')
                    frame.update(state=state[index].astype(np.float32),actions=actions[index],task=PROMPT)
                    dataset.add_frame(frame)
                dataset.save_episode()
            print(json.dumps({'split':split,'converted_source':entry['source'],'episode':entry['source_demo'],'frames':entry['frames']}),flush=True)
        dataset.stop_image_writer()
        report={'split':split,'repo_id':repo,'fps':30,'episodes':episodes,'num_episodes':len(episodes),'total_frames':sum(e['frames'] for e in episodes),
                'state_schema':list(STATE_NAMES),'action_schema':list(ACTION_NAMES),'prompt':PROMPT,
                'images':'Three real native RGB views; alpha dropped; no synthetic missing wrist image',
                'object_poses_as_policy_inputs':False,'phase_or_clock_as_policy_inputs':False,
                'grasping_note':'Official author-config generation uses assisted attachment; physical-only evaluation is a separate gate',
                'action_note':'Native local-delta base commands; absolute trunk/arm targets; native jaw commands. No ALOHA conversion / extra delta transform. Not claiming exact author pi0 action encoding.',
                'rejected':rejected,'local_only':True,'restricted_assets':'User-approved BEHAVIOR noncommercial academic terms; not uploaded'}
        (args.output_root/repo/'momagen-provenance.json').write_text(json.dumps(report,indent=2))
    print('PI0_DATASET_CONVERSION_COMPLETE',len(train),len(validation),flush=True)


if __name__=='__main__':main()
