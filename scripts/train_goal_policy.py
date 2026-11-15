#!/usr/bin/env python3
"""SERVER ONLY: denoising neural BC with future world-frame motor goals.

All targets come from successful demonstration frames. Lookahead and noise are
training augmentation, never a runtime clock, phase machine or teacher fallback.
"""
import argparse
import json
import math
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch import nn
from policy_action_codecs import encode_world_base_goals
from train_state_policy import load_episodes, digest


class GoalPolicy(nn.Module):
    def __init__(self, obs_dim, history=4, hidden=768):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(obs_dim*history,hidden),nn.LayerNorm(hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.LayerNorm(hidden),nn.SiLU(),nn.Linear(hidden,hidden),
            nn.SiLU(),nn.Linear(hidden,22))

    def forward(self, observations):
        return self.net(observations.flatten(1))[:,None,:]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--val-demo',default='demo_0')
    parser.add_argument('--train-demo',action='append',default=[])
    parser.add_argument('--single-demo-smoke',action='store_true')
    parser.add_argument('--target-offset',type=int,default=8)
    parser.add_argument('--steps',type=int,default=40000)
    parser.add_argument('--batch',type=int,default=512)
    parser.add_argument('--history',type=int,default=4)
    parser.add_argument('--history-stride',type=int,default=4)
    parser.add_argument('--hidden',type=int,default=768)
    parser.add_argument('--seed',type=int,default=81)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.steps,args.history,args.history_stride,args.batch)<1 or args.target_offset<0:raise ValueError('Invalid dimensions')
    args.output.mkdir(parents=True)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    episodes,keys,meta=load_episodes(args.dataset)
    train=[e for e in episodes if e[0]!=args.val_demo and (not args.train_demo or e[0] in args.train_demo)]
    val=[e for e in episodes if e[0]==args.val_demo]
    if not train or len(val)!=1 or (len(train)<2 and not args.single_demo_smoke):raise ValueError('Episode-disjoint validation required; one demo needs explicit smoke-test flag')
    if args.single_demo_smoke and len(train)!=1:raise ValueError('Single-demo smoke must use exactly one training episode')
    x=np.concatenate([e[1] for e in train]);raw_actions=np.concatenate([e[2] for e in train])
    mean=x.mean(0);scale=np.maximum(x.std(0),.05);scale[3:6]=1e6
    def pack(episodes):
        features=[];targets=[]
        with h5py.File(args.dataset) as file:
            for name,obs,actions,_ in episodes:
                goals=encode_world_base_goals(actions,file[f'data/{name}/obs/base_link_pose'][()],obs[:,2])
                future=np.minimum(np.arange(len(obs))+args.target_offset,len(obs)-1)
                past=np.maximum(np.arange(len(obs))[:,None]+np.arange(-(args.history-1),1)[None,:]*args.history_stride,0)
                features.append((obs[past]-mean)/scale);targets.append(goals[future])
        return np.concatenate(features),np.concatenate(targets)
    tr_x,tr_y=pack(train);va_x,va_y=pack(val)
    ym=tr_y.mean(0);ys=np.maximum(tr_y.std(0),.002);ym[[14,21]]=0;ys[[14,21]]=1
    low=raw_actions.min(0);high=raw_actions.max(0);limit=np.maximum(np.abs(low[:3]),np.abs(high[:3]));low[:3]=-limit;high[:3]=limit
    norm={'x_mean':mean,'x_std':scale,'y_mean':ym,'y_std':ys,'action_min':low,'action_max':high}
    train_x=torch.tensor(tr_x,device='cuda');train_y=torch.tensor((tr_y-ym)/ys,device='cuda')
    val_x=torch.tensor(va_x,device='cuda');val_y=torch.tensor((va_y-ym)/ys,device='cuda')
    # Physical observation perturbations, correlated across the short history.
    # This is observation-noise augmentation, not simulator recovery-label DAgger.
    noise=np.full(len(mean),.001,np.float32);noise[:3]=[.005,.005,.005];noise[3:6]=0
    noise[6:16]=.01;noise[24:30]=.01;noise[16:19]=.003;noise[30:33]=.003
    noise[38:40]=0
    for i in range(len(keys)-3):noise[40+7*i:43+7*i]=.003
    noise=torch.tensor(noise/scale,device='cuda')
    regular=[i for i in range(22) if i not in (14,21)]
    tolerances=np.full(22,.02);tolerances[:2]=.005;tolerances[2:4]=.01
    precision=torch.tensor(ys/tolerances,device='cuda',dtype=torch.float32)
    model=GoalPolicy(len(mean),args.history,args.hidden).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    sha=digest(args.dataset)
    manifest={'algorithm':'denoising_world_goal_MLP_BC','training_method':'Gradient-trained privileged-state BC with noisy observations and demonstration lookahead targets',
        'dataset':str(args.dataset),'dataset_sha256':sha,'keys':keys,'env_args':meta,'privileged_state':True,
        'train_demos':[e[0] for e in train],'val_demos':[args.val_demo],
        'single_demo_smoke':args.single_demo_smoke,'unused_demos':[e[0] for e in episodes if e[0] not in [t[0] for t in train+val]],
        'target_offset':args.target_offset,'observation_noise':True,'history':args.history,'history_stride':args.history_stride,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'caveat':'No simulator recovery labels; not DAgger, no novel-scene generalization claim'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    start=time.time();best=float('inf')
    def save(name,step):
        torch.save({**manifest,'model':model.state_dict(),'obs_dim':len(mean),'hidden':args.hidden,'horizon':1,'mixtures':1,
            'action_mode':'world_base_goal','norm':{k:v.tolist() for k,v in norm.items()},'step':step},args.output/name)
    with (args.output/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            lr=1e-5+(3e-4-1e-5)*.5*(1+math.cos(math.pi*(step-1)/max(1,args.steps-1)))
            for group in optimizer.param_groups:group['lr']=lr
            model.train();index=torch.randint(len(train_x),(args.batch,),device='cuda');target=train_y[index]
            inputs=train_x[index]+torch.randn(args.batch,1,len(mean),device='cuda')*noise[None,None,:]
            pred=model(inputs)[:,0]
            reg=nn.functional.smooth_l1_loss((pred[:,regular]-target[:,regular])*precision[regular],torch.zeros_like(pred[:,regular]))
            jaw=nn.functional.binary_cross_entropy_with_logits(pred[:,[14,21]],(target[:,[14,21]]+1)/2)
            loss=reg+.5*jaw;optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),10);optimizer.step()
            if step==1 or step%500==0 or step==args.steps:
                model.eval()
                with torch.inference_mode():
                    prediction=torch.cat([model(val_x[i:i+256])[:,0] for i in range(0,len(val_x),256)])
                    error=(prediction-val_y).abs()*torch.tensor(ys,device='cuda')
                    metrics={'step':step,'train_loss':float(loss),'val_base_xy_goal_mae_m':float(error[:,:2].mean()),
                        'val_joint_target_mae_rad':float(error[:,[i for i in range(4,22) if i not in (14,21)]].mean()),
                        'val_gripper_accuracy':float(((prediction[:,[14,21]]>0)==(val_y[:,[14,21]]>0)).float().mean()),
                        'elapsed_s':time.time()-start}
                    score=metrics['val_base_xy_goal_mae_m']+metrics['val_joint_target_mae_rad']
                log.write(json.dumps(metrics)+'\n');print(json.dumps(metrics),flush=True)
                save('last.pt',step)
                if score<best:best=score;save('best.pt',step)
    result={'status':'training_completed','algorithm':manifest['algorithm'],'steps':args.steps,'elapsed_s':time.time()-start,
            'closed_loop_success':'not_yet_evaluated','final_metrics':metrics}
    (args.output/'training-result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
