#!/usr/bin/env python3
"""SERVER ONLY: recurrent state BC with learned memory for multi-stage control.

Full episode sequences are supervised causally; validation is episode-disjoint.
No clock, phase labels, planner, teacher actions or future state are inputs.
"""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from train_state_policy import load_episodes,digest,JOINT,GRIP


class RecurrentPolicy(nn.Module):
    def __init__(self,obs_dim,hidden=256,layers=2):
        super().__init__();self.rnn=nn.GRU(obs_dim,hidden,layers,batch_first=True)
        self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,21));self._hidden=None
    def reset_memory(self):self._hidden=None
    def forward_sequence(self,obs):return self.head(self.rnn(obs)[0])
    def forward(self,obs):
        out,hidden=self.rnn(obs,self._hidden);self._hidden=hidden.detach();return self.head(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--train-demo',action='append')
    p.add_argument('--single-demo-smoke',action='store_true')
    p.add_argument('--ignore-base-velocity',action='store_true')
    p.add_argument('--joint-precision',type=float,default=0.0)
    p.add_argument('--base-precision',type=float,default=0.0)
    p.add_argument('--transition-weight',type=float,default=1.0)
    p.add_argument('--transition-radius',type=int,default=15)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--min-lr',type=float,default=3e-5)
    p.add_argument('--updates',type=int,default=1000);p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--action-mode',choices=['absolute','delta_joint'],default='absolute')
    p.add_argument('--layers',type=int,default=2);p.add_argument('--val-demo',default='demo_0');p.add_argument('--seed',type=int,default=45)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(a.seed)
    episodes,keys,meta=load_episodes(a.dataset)
    train=[e for e in episodes if e[0]!=a.val_demo and (not a.train_demo or e[0] in a.train_demo)]
    val=[e for e in episodes if e[0]==a.val_demo]
    if not train or len(val)!=1 or (len(train)<2 and not a.single_demo_smoke):raise ValueError('Invalid episode-disjoint split; one demo requires explicit --single-demo-smoke')
    tx=np.concatenate([e[1] for e in train]);ta=np.concatenate([e[2] for e in train]);tp=np.concatenate([e[3] for e in train])
    y=ta.copy()
    if a.action_mode=='delta_joint':y[:,JOINT]-=tp[:,JOINT]
    xm,xs=tx.mean(0),np.maximum(tx.std(0),0.05);ym,ys=y.mean(0),np.maximum(y.std(0),0.002)
    if a.ignore_base_velocity:xs[3:6]=1e6
    ym[GRIP]=0;ys[GRIP]=1
    norm={'x_mean':xm,'x_std':xs,'y_mean':ym,'y_std':ys,'action_min':ta.min(0),'action_max':ta.max(0)}
    def batch(eps):
        length=max(len(e[1]) for e in eps)
        x=np.zeros((len(eps),length,len(xm)),np.float32);target=np.zeros((len(eps),length,21),np.float32);mask=np.zeros((len(eps),length),np.float32);jaw_weight=np.ones((len(eps),length,2),np.float32)
        for i,(_,obs,act,prop) in enumerate(eps):
            n=len(obs);x[i,:n]=(obs-xm)/xs;t=act.copy()
            if a.action_mode=='delta_joint':t[:,JOINT]-=prop[:,JOINT]
            target[i,:n]=(t-ym)/ys;mask[i,:n]=1
            for side,index in enumerate(GRIP):
                for change in np.flatnonzero(np.diff(act[:,index])!=0)+1:
                    jaw_weight[i,max(0,change-a.transition_radius):min(n,change+a.transition_radius+1),side]=a.transition_weight
        return tuple(torch.tensor(v,device='cuda') for v in (x,target,mask,jaw_weight))
    tr,va=batch(train),batch(val);nongrip=[i for i in range(21) if i not in GRIP]
    model=RecurrentPolicy(len(xm),a.hidden,a.layers).cuda();opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5)
    precision=torch.ones(21,device='cuda')
    if a.joint_precision>0:precision[JOINT]=torch.tensor(ys[JOINT]/a.joint_precision,device='cuda')
    if a.base_precision>0:precision[:3]=torch.tensor(ys[:3]/a.base_precision,device='cuda')
    def loss(data):
        x,target,mask,jaw_weight=data;pred=model.forward_sequence(x)
        reg=nn.functional.smooth_l1_loss(pred[:,:,nongrip]*precision[nongrip],target[:,:,nongrip]*precision[nongrip],reduction='none').mean(-1)
        jaw=nn.functional.binary_cross_entropy_with_logits(pred[:,:,GRIP],(target[:,:,GRIP]+1)/2,reduction='none')
        jaw=(jaw*jaw_weight).mean(-1)
        return ((reg+0.25*jaw)*mask).sum()/mask.sum(),pred
    manifest={'dataset':str(a.dataset),'dataset_sha256':digest(a.dataset),'algorithm':'recurrent_GRU_BC',
          'privileged_state':True,'action_mode':a.action_mode,'env_args':meta,'keys':keys,'train_demos':[e[0] for e in train],'val_demos':[a.val_demo],
          'single_demo_smoke':a.single_demo_smoke,'unused_demos':[e[0] for e in episodes if e[0] not in [t[0] for t in train+val]],
          'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
          'training':'causal full-episode sequence supervision; recurrent hidden state reset once per episode','updates':a.updates,
          'caveat':'Same human source demonstration family, not a novel-scene benchmark'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time();best=float('inf')
    with (a.output/'metrics.jsonl').open('w',buffering=1) as log:
        for update in range(1,a.updates+1):
            learning_rate=a.min_lr+(a.lr-a.min_lr)*0.5*(1+math.cos(math.pi*(update-1)/max(1,a.updates-1)))
            for group in opt.param_groups:group['lr']=learning_rate
            model.train();train_loss,_=loss(tr);opt.zero_grad(set_to_none=True);train_loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            if update==1 or update%50==0:
                model.eval()
                with torch.no_grad():
                    value,pred=loss(va);error=(pred-va[1])*torch.tensor(ys,device='cuda')
                    val_base=float(error[:,:,:3].abs().mean());val_joint=float(error[:,:,JOINT].abs().mean())
                    val_jaw=float(((pred[:,:,GRIP]>0)==(va[1][:,:,GRIP]>0)).float().mean())
                record={'update':update,'train_loss':float(train_loss),'val_loss':float(value),'val_base_command_mae':val_base,
                        'val_joint_target_mae_rad':val_joint,'val_gripper_accuracy':val_jaw,'elapsed_s':time.time()-start}
                log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                ckpt={'algorithm':'recurrent_GRU_BC','single_demo_smoke':a.single_demo_smoke,'model':model.state_dict(),'obs_dim':len(xm),'hidden':a.hidden,'layers':a.layers,
                     'history':1,'history_stride':1,'horizon':1,'mixtures':1,'keys':keys,'env_args':meta,
                     'norm':{k:v.tolist() for k,v in norm.items()},'action_mode':a.action_mode,'train_demos':manifest['train_demos'],'val_demos':[a.val_demo],
                     'step':update,'dataset_sha256':manifest['dataset_sha256']}
                if float(value)<best:best=float(value);torch.save(ckpt,a.output/'best.pt')
                torch.save(ckpt,a.output/'last.pt')
    (a.output/'training-result.json').write_text(json.dumps({'status':'training_completed','algorithm':'recurrent_GRU_BC',
               'updates':a.updates,'best_val_loss':best,'elapsed_s':time.time()-start,'closed_loop_success':'not_yet_evaluated'},indent=2))


if __name__=='__main__':main()
