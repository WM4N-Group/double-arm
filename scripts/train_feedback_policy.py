#!/usr/bin/env python3
"""SERVER ONLY: gradient-trained motor-state residual BC, physical data only.

The model predicts a residual from current measured base/joint state to the
supervised motor goals. This is an action codec, NOT a residual on a planner,
source action, phase machine or trajectory replay. No teacher at inference.
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
from policy_action_codecs import encode_world_base_goals, world_goal_anchor
from train_goal_policy import GoalPolicy
from train_state_policy import load_episodes, digest


def anchor_noise_delta(raw_observations, perturbations):
    """Exact anchor change for noisy base/joint features (differentiable)."""
    noise=perturbations
    delta=noise.new_zeros((len(noise),22))
    delta[:,:2]=noise[:,:2]
    theta=raw_observations[:,2]
    delta[:,2]=torch.sin(theta+noise[:,2])-torch.sin(theta)
    delta[:,3]=torch.cos(theta+noise[:,2])-torch.cos(theta)
    delta[:,4:14]=noise[:,6:16]
    delta[:,15:21]=noise[:,24:30]
    return delta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=40000);p.add_argument('--target-offset',type=int,default=0)
    p.add_argument('--history',type=int,default=4);p.add_argument('--history-stride',type=int,default=4)
    p.add_argument('--hidden',type=int,default=512);p.add_argument('--batch',type=int,default=512)
    p.add_argument('--val-demo',default='demo_0');p.add_argument('--seed',type=int,default=132)
    p.add_argument('--noise-multiplier',type=float,default=1.0)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if min(a.steps,a.history,a.history_stride,a.batch)<1 or a.target_offset<0 or a.noise_multiplier<0:raise ValueError('Invalid options')
    with h5py.File(a.dataset) as file:
        if file['data'].attrs.get('demonstration_grasping_mode')!='physical':raise ValueError('Require verified physical demonstration data')
    episodes,keys,meta=load_episodes(a.dataset);train=[e for e in episodes if e[0]!=a.val_demo];val=[e for e in episodes if e[0]==a.val_demo]
    if len(train)<2 or len(val)!=1:raise ValueError('At least 2 train and 1 whole-episode heldout demos required')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.manual_seed(a.seed);np.random.seed(a.seed)
    x=np.concatenate([e[1] for e in train]);native=np.concatenate([e[2] for e in train])
    xm=x.mean(0);xs=np.maximum(x.std(0),.05);xs[3:6]=1e6
    def pack(eps):
        features=[];residuals=[];raws=[]
        with h5py.File(a.dataset) as file:
            for name,obs,actions,joints in eps:
                poses=file[f'data/{name}/obs/base_link_pose'][()]
                goals=encode_world_base_goals(actions,poses,obs[:,2])
                future=np.minimum(np.arange(len(obs))+a.target_offset,len(obs)-1)
                target=goals[future]-world_goal_anchor(poses,obs[:,2],joints)
                past=np.maximum(np.arange(len(obs))[:,None]+np.arange(-(a.history-1),1)[None,:]*a.history_stride,0)
                features.append((obs[past]-xm)/xs);residuals.append(target);raws.append(obs)
        return np.concatenate(features),np.concatenate(residuals),np.concatenate(raws)
    tr_x,tr_y,tr_raw=pack(train);va_x,va_y,_=pack(val)
    ym=tr_y.mean(0);ys=np.maximum(tr_y.std(0),.002);ym[[14,21]]=0;ys[[14,21]]=1
    low=native.min(0);high=native.max(0);limit=np.maximum(np.abs(low[:3]),np.abs(high[:3]));low[:3]=-limit;high[:3]=limit
    norm={'x_mean':xm,'x_std':xs,'y_mean':ym,'y_std':ys,'action_min':low,'action_max':high}
    tx=torch.tensor(tr_x,device='cuda');ty=torch.tensor((tr_y-ym)/ys,device='cuda');raw=torch.tensor(tr_raw,device='cuda')
    vx=torch.tensor(va_x,device='cuda');vy=torch.tensor((va_y-ym)/ys,device='cuda');yst=torch.tensor(ys,device='cuda');xst=torch.tensor(xs,device='cuda')
    noise=np.full(len(xm),.0005,np.float32);noise[:3]=[.002,.002,.003];noise[3:6]=0;noise[6:16]=.003;noise[24:30]=.003
    noise[16:19]=.001;noise[30:33]=.001;noise[38:40]=0
    for index in range(len(keys)-3):noise[40+7*index:43+7*index]=.001
    nt=torch.tensor(noise*a.noise_multiplier,device='cuda')
    regular=[i for i in range(22) if i not in (14,21)];joint=[i for i in range(4,22) if i not in (14,21)]
    model=GoalPolicy(len(xm),a.history,a.hidden).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    manifest={'algorithm':'motor_state_residual_MLP_BC','training_method':'Gradient-trained neural motor-goal residual BC; anchor is measured state, not teacher action',
        'dataset':str(a.dataset),'dataset_sha256':digest(a.dataset),'env_args':meta,'keys':keys,'train_demos':[e[0] for e in train],
        'val_demos':[a.val_demo],'privileged_state':True,'action_mode':'world_base_goal_residual','demonstration_grasping_mode':'physical',
        'history':a.history,'history_stride':a.history_stride,'hidden':a.hidden,'target_offset':a.target_offset,
        'noisy_anchor_target_corrected':True,'upstream_finger_friction':4.0,'physical_parameters_calibrated_to_real_robot':False,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));start=time.time();best=float('inf')
    def save(name,step):
        torch.save({**manifest,'model':model.state_dict(),'obs_dim':len(xm),'horizon':1,'mixtures':1,
                    'norm':{k:v.tolist() for k,v in norm.items()},'step':step},a.output/name)
    with (a.output/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,a.steps+1):
            lr=3e-6+(3e-4-3e-6)*.5*(1+math.cos(math.pi*(step-1)/max(1,a.steps-1)))
            for group in opt.param_groups:group['lr']=lr
            idx=torch.randint(len(tx),(a.batch,),device='cuda');perturb=torch.randn(a.batch,len(xm),device='cuda')*nt
            inputs=tx[idx]+(perturb/xst)[:,None,:]
            # Keep the absolute expert goal fixed under observation noise: a
            # perturbed measured-state anchor must subtract its own perturbation.
            target=ty[idx]-anchor_noise_delta(raw[idx],perturb)/yst
            pred=model(inputs)[:,0]
            reg=nn.functional.mse_loss(pred[:,regular],target[:,regular])
            jaw=nn.functional.binary_cross_entropy_with_logits(pred[:,[14,21]],(target[:,[14,21]]+1)/2)
            loss=reg+.5*jaw;opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
            if step==1 or step%500==0 or step==a.steps:
                with torch.inference_mode():
                    predv=torch.cat([model(vx[i:i+256])[:,0] for i in range(0,len(vx),256)])
                    error=(predv-vy).abs()*yst
                    metrics={'step':step,'train_loss':float(loss),'val_base_goal_mae_m':float(error[:,:2].mean()),
                             'val_joint_goal_mae_rad':float(error[:,joint].mean()),
                             'val_gripper_accuracy':float(((predv[:,[14,21]]>0)==(vy[:,[14,21]]>0)).float().mean()),'elapsed_s':time.time()-start}
                log.write(json.dumps(metrics)+'\n');print(json.dumps(metrics),flush=True);save('last.pt',step)
                score=metrics['val_base_goal_mae_m']+metrics['val_joint_goal_mae_rad']
                if score<best:best=score;save('best.pt',step)
    (a.output/'training-result.json').write_text(json.dumps({'status':'training_completed','algorithm':manifest['algorithm'],
        'steps':a.steps,'final_metrics':metrics,'closed_loop_success':'not_yet_evaluated'},indent=2))


if __name__=='__main__':main()
