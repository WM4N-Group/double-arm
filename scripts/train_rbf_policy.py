#!/usr/bin/env python3
"""SERVER ONLY: local kernel behavior cloning, fitted from successful state/action data.

A Gaussian radial-basis policy with analytically learned ridge coefficients.
No time index, task-phase rules, trajectory pointer, planner or teacher is used
at inference. This is a small-data privileged-state baseline, NOT a VLA model.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import h5py
from policy_action_codecs import encode_world_base_goals
from torch import nn
from train_state_policy import load_episodes, digest, JOINT, GRIP


class RBFPolicy(nn.Module):
    def __init__(self, centers, weights, bandwidth, normalized=False):
        super().__init__()
        self.register_buffer('centers',torch.as_tensor(centers,dtype=torch.float64))
        self.register_buffer('weights',torch.as_tensor(weights,dtype=torch.float64))
        self.bandwidth=float(bandwidth);self.normalized=bool(normalized)
    def forward(self,obs):
        x=obs.flatten(1).to(dtype=self.centers.dtype)
        distances=(x.square().sum(-1)[:,None]+self.centers.square().sum(-1)[None]-2*x@self.centers.T).clamp_min(0)
        score=-distances/(2*self.bandwidth**2)
        basis=torch.softmax(score,dim=-1) if self.normalized else torch.exp(score)
        return (basis@self.weights).to(dtype=torch.float32)[:,None,:]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--val-demo',default='demo_0');p.add_argument('--bandwidth',type=float,default=0.7)
    p.add_argument('--action-mode',choices=['absolute','delta_joint','world_base_goal'],default='absolute')
    p.add_argument('--ignore-base-velocity',action='store_true')
    p.add_argument('--normalize-basis',action='store_true')
    p.add_argument('--orientation-noise-floor',type=float,default=0.05)
    p.add_argument('--ridge',type=float,default=1e-4);p.add_argument('--history',type=int,default=4)
    p.add_argument('--history-stride',type=int,default=4);p.add_argument('--max-centers',type=int,default=12000)
    a=p.parse_args()
    if a.bandwidth<=0 or a.ridge<=0:raise ValueError('Positive bandwidth and ridge required')
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    episodes,keys,meta=load_episodes(a.dataset)
    train=[e for e in episodes if e[0]!=a.val_demo];val=[e for e in episodes if e[0]==a.val_demo]
    if len(train)<2 or len(val)!=1:raise ValueError('Need >=2 train episodes and one whole-episode validation split')
    x=np.concatenate([e[1] for e in train]);action=np.concatenate([e[2] for e in train]);anchor=np.concatenate([e[3] for e in train])
    x_mean=x.mean(0);x_std=np.maximum(x.std(0),0.05)
    if a.ignore_base_velocity:x_std[3:6]=1e6
    for first,last in [(19,23),(33,37)]+[(40+7*i+3,40+7*i+7) for i in range(len(keys)-3)]:
        x_std[first:last]=np.maximum(x_std[first:last],a.orientation_noise_floor)
    base_poses={}
    if a.action_mode=='world_base_goal':
        with h5py.File(a.dataset) as f:
            base_poses={e[0]:f[f'data/{e[0]}/obs/base_link_pose'][()] for e in episodes}
    def target_actions(episode):
        name,x,act,prop=episode
        if a.action_mode=='world_base_goal':return encode_world_base_goals(act,base_poses[name],x[:,2])
        result=act.copy()
        if a.action_mode=='delta_joint':result[:,JOINT]-=prop[:,JOINT]
        return result
    y=np.concatenate([target_actions(e) for e in train])
    jaw_idx=[14,21] if a.action_mode=='world_base_goal' else GRIP
    joint_idx=list(range(4,14))+list(range(15,21)) if a.action_mode=='world_base_goal' else JOINT
    y_mean=y.mean(0);y_std=np.maximum(y.std(0),0.002);y_mean[jaw_idx]=0;y_std[jaw_idx]=1
    norm={'x_mean':x_mean,'x_std':x_std,'y_mean':y_mean,'y_std':y_std,
          'action_min':action.min(0),'action_max':action.max(0)}
    def data(eps):
        xs=[];ys=[]
        for name,x,act,prop in eps:
            past=np.maximum(np.arange(len(x))[:,None]+np.arange(-(a.history-1),1)[None,:]*a.history_stride,0)
            xs.append(((x[past]-x_mean)/x_std).reshape(len(x),-1))
            target=target_actions((name,x,act,prop))
            ys.append((target-y_mean)/y_std)
        return np.concatenate(xs),np.concatenate(ys)
    tx,ty=data(train);vx,vy=data(val)
    indices=np.linspace(0,len(tx)-1,min(len(tx),a.max_centers),dtype=int)
    # Each Gaussian basis is an observed state-history vector; coefficients are
    # optimized jointly by the supervised kernel ridge solve, not action lookup.
    centers=torch.tensor(tx[indices],device='cuda',dtype=torch.float64)
    labels=torch.tensor(ty[indices],device='cuda',dtype=torch.float64)
    start=time.time()
    distances=(centers.square().sum(-1)[:,None]+centers.square().sum(-1)[None]-2*centers@centers.T).clamp_min_(0)
    kernel=torch.exp_(-distances/(2*a.bandwidth**2));del distances
    if a.normalize_basis:labels=labels*kernel.sum(-1)[:,None]
    kernel.diagonal().add_(a.ridge)
    factor,info=torch.linalg.cholesky_ex(kernel)
    if int(info)!=0:raise RuntimeError(f'Ridge kernel Cholesky failed: {int(info)}')
    weights=torch.cholesky_solve(labels,factor)
    del factor,kernel
    model=RBFPolicy(centers,weights,a.bandwidth,normalized=a.normalize_basis).cuda().eval()
    def metrics(features,target):
        errors=[];jaw=[]
        for first in range(0,len(features),256):
            with torch.inference_mode():pred=model(torch.tensor(features[first:first+256],device='cuda'))[:,0].cpu().numpy()
            truth=target[first:first+256];errors.append(np.abs((pred-truth)*y_std));jaw.append((pred[:,jaw_idx]>0)==(truth[:,jaw_idx]>0))
        errors=np.concatenate(errors)
        return {'base_command_mae':float(errors[:,:3].mean()),'joint_target_mae_rad':float(errors[:,joint_idx].mean()),
                'gripper_accuracy':float(np.concatenate(jaw).mean())}
    report={'status':'training_completed','algorithm':'rbf_kernel_BC','training_method':'closed-form kernel ridge regression',
            'basis_count':len(centers),'train_metrics':metrics(tx,ty),'val_metrics':metrics(vx,vy),
            'elapsed_s':time.time()-start,'closed_loop_success':'not_yet_evaluated'}
    manifest={'dataset':str(a.dataset),'dataset_sha256':digest(a.dataset),'algorithm':'rbf_kernel_BC',
              'privileged_state':True,'train_demos':[e[0] for e in train],'val_demos':[a.val_demo],
              'env_args':meta,'keys':keys,'history':a.history,'history_stride':a.history_stride,
              'bandwidth':a.bandwidth,'ridge':a.ridge,'action_mode':a.action_mode,
              'normalized_basis':a.normalize_basis,'ignore_base_velocity':a.ignore_base_velocity,'orientation_noise_floor':a.orientation_noise_floor,'source_caveat':'Same human source demo family; no new-scene claim'}
    if a.action_mode=='world_base_goal':
        limit=np.maximum(np.abs(norm['action_min'][:3]),np.abs(norm['action_max'][:3]))
        norm['action_min'][:3]=-limit;norm['action_max'][:3]=limit
    checkpoint={'algorithm':'rbf_kernel_BC','model':{k:v.cpu() for k,v in model.state_dict().items()},
       'bandwidth':a.bandwidth,'normalized_basis':a.normalize_basis,'action_mode':a.action_mode,'obs_dim':len(x_mean),'history':a.history,'history_stride':a.history_stride,
       'horizon':1,'hidden':0,'mixtures':1,'keys':keys,'env_args':meta,
       'norm':{k:v.tolist() for k,v in norm.items()},'train_demos':manifest['train_demos'],'val_demos':[a.val_demo],
       'step':1,'dataset_sha256':manifest['dataset_sha256']}
    torch.save(checkpoint,a.output/'best.pt');torch.save(checkpoint,a.output/'last.pt')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));(a.output/'training-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
