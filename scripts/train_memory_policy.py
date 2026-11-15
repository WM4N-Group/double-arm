#!/usr/bin/env python3
"""SERVER ONLY: nonparametric nearest-state behavior cloning diagnostic.

Fit a memory policy from successful demonstrations, with a whole-episode holdout.
Inference uses measured state history and fitted state/action prototypes only:
no clock, phase, episode identity, trajectory cursor, planner or online teacher.
This is explicitly memory-based BC, not a gradient-trained neural/VLA policy.
"""
import argparse
import json
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch import nn
from policy_action_codecs import encode_world_base_goals
from train_state_policy import load_episodes, digest


class NearestStatePolicy(nn.Module):
    def __init__(self, centers, targets):
        super().__init__()
        self.register_buffer('centers',torch.as_tensor(centers,dtype=torch.float64))
        self.register_buffer('targets',torch.as_tensor(targets,dtype=torch.float32))
        self.register_buffer('center_norm',self.centers.square().sum(-1))

    def forward(self, observations):
        query=observations.flatten(1).to(self.centers.dtype)
        distance=(query.square().sum(-1)[:,None]+self.center_norm[None]-2*query@self.centers.T)
        return self.targets[distance.argmin(-1)][:,None,:]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--val-demo',default='demo_0')
    parser.add_argument('--target-offset',type=int,default=0,help='Training-label lookahead only; no trajectory cursor at inference')
    parser.add_argument('--history',type=int,default=4)
    parser.add_argument('--compress-idle',action='store_true',help='Training-only removal of no-progress plateau samples')
    parser.add_argument('--ignore-grasp-flags',action='store_true',help='Avoid assisted-vs-physical grasp-enum domain mismatch')
    parser.add_argument('--history-stride',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if args.history<1 or args.history_stride<1 or args.target_offset<0:raise ValueError('Positive history dimensions required')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    episodes,keys,meta=load_episodes(args.dataset)
    train=[e for e in episodes if e[0]!=args.val_demo]
    val=[e for e in episodes if e[0]==args.val_demo]
    if len(train)<2 or len(val)!=1:raise ValueError('Need >=2 training demos and an episode-disjoint holdout')
    x=np.concatenate([e[1] for e in train]);actions=np.concatenate([e[2] for e in train])
    mean=x.mean(0);scale=np.maximum(x.std(0),.05)
    if args.ignore_grasp_flags:scale[38:40]=1e6
    scale[3:6]=1e6  # measured velocity is noisy; position history still represents motion
    for lo,hi in [(19,23),(33,37)]+[(43+7*i,47+7*i) for i in range(len(keys)-3)]:
        scale[lo:hi]=np.maximum(scale[lo:hi],.15)
    retention={}
    def pack(episodes):
        centers=[];targets=[]
        with h5py.File(args.dataset) as file:
            for name,observations,commands,_ in episodes:
                pose=file[f'data/{name}/obs/base_link_pose'][()]
                goals=encode_world_base_goals(commands,pose,observations[:,2])
                future=np.minimum(np.arange(len(commands))+args.target_offset,len(commands)-1)
                keep=np.ones(len(observations),dtype=bool)
                if args.compress_idle:
                    if args.target_offset<1:raise ValueError('Idle compression needs a positive training-label lookahead')
                    changes=np.abs((observations[future]-observations)/scale).max(-1)
                    goal_change=np.abs(goals[future]-goals).max(-1)
                    keep=(changes>.02)|(goal_change>.002)
                    keep[-1]=True
                retention[name]={'source_frames':len(observations),'retained_prototypes':int(keep.sum())}
                targets.append(goals[future][keep])
                past=np.maximum(np.arange(len(observations))[:,None]
                    +np.arange(-(args.history-1),1)[None,:]*args.history_stride,0)
                centers.append(((observations[past]-mean)/scale).reshape(len(observations),-1)[keep])
        return np.concatenate(centers),np.concatenate(targets)
    start=time.time();centers,targets=pack(train)
    y_mean=targets.mean(0);y_std=np.maximum(targets.std(0),.002)
    y_mean[[14,21]]=0;y_std[[14,21]]=1
    model=NearestStatePolicy(centers,(targets-y_mean)/y_std)
    low=actions.min(0);high=actions.max(0);limit=np.maximum(np.abs(low[:3]),np.abs(high[:3]));low[:3]=-limit;high[:3]=limit
    norm={'x_mean':mean,'x_std':scale,'y_mean':y_mean,'y_std':y_std,'action_min':low,'action_max':high}
    sha=digest(args.dataset)
    manifest={'algorithm':'nearest_state_memory_BC','training_method':'Nonparametric supervised fit of state-history/action prototypes; no gradient training',
              'dataset':str(args.dataset),'dataset_sha256':sha,'env_args':meta,'keys':keys,
              'train_demos':[e[0] for e in train],'val_demos':[args.val_demo],
              'privileged_state':True,'history':args.history,'history_stride':args.history_stride,
              'action_mode':'world_base_goal','compress_idle':args.compress_idle,'ignore_grasp_flags':args.ignore_grasp_flags,'retention':retention,'target_offset':args.target_offset,'caveat':'Memory-based imitation baseline, not a neural policy or novel-scene generalization claim; no trajectory order at inference'}
    checkpoint={**manifest,'model':model.state_dict(),'obs_dim':len(mean),'horizon':1,'hidden':0,'mixtures':1,
                'norm':{key:value.tolist() for key,value in norm.items()},'step':0}
    torch.save(checkpoint,args.output/'best.pt');torch.save(checkpoint,args.output/'last.pt')
    report={'status':'training_completed','algorithm':manifest['algorithm'],'training_method':manifest['training_method'],
            'num_prototypes':len(centers),'train_demos':len(train),'val_demos':len(val),'elapsed_s':time.time()-start,
            'closed_loop_success':'not_yet_evaluated'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'training-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
