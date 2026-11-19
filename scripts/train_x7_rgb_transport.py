#!/usr/bin/env python3
"""GPU BC for base transport, with unchanged geometric grasp/place anchors."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from collect_x7_transport_bc import FROZEN_CHECKPOINT, layouts, sha_layout, validate_trace
from run_x7_mobile_bimanual import CONTROL_HZ, smooth
from run_x7_turning_carry import WXY, waypoints
from train_x7_turning_trajectory_bc import BLOCK_PHASES, digest

CHUNK=6
LIMIT=np.array([.15,.12,.12,.15,.12,.10],dtype=np.float32)
TRAIN_MIN=56
DEV_MIN=8
GPU_UUID='GPU-9eea1085-6cdd-882b-61f3-4ac2c1b10f8b'
ALLOCATOR_BYTES=36*1024**3
PROCESS_LIMIT_MIB=38146  # <= 40 decimal GB; includes non-allocator CUDA memory.


def transport_prior(layout, counts):
    n=np.asarray(counts,dtype=int)
    if len(n)!=len(BLOCK_PHASES) or np.any(n<2):raise ValueError('Invalid timing blocks')
    endpoints=np.asarray([v for _,v in waypoints(layout)])
    prior=np.zeros((sum(n),6),dtype=np.float32)
    i=int(n[:4].sum()); start=i
    for a,b,nk in zip(endpoints[:-1],endpoints[1:],n[4:9]):
        d=b-a;t=np.arange(1,nk+1)/nk
        prior[i:i+nk,:3]=a+smooth(t)[:,None]*d
        prior[i:i+nk,3:]=((30*t*t*(1-t)*(1-t)*CONTROL_HZ/nk)[:,None]*d)
        i+=nk
    return prior,start,i


def feature(layout,base,arms,jaws,prior):
    return np.concatenate((np.asarray([*layout['goal'],layout['bay_y'],*layout['tote']]),
                           np.asarray(base).reshape(-1),np.asarray(arms).reshape(-1),
                           np.asarray(jaws).reshape(-1),prior.reshape(-1))).astype(np.float32)


class RGBTransport(nn.Module):
    def __init__(self,state_dim):
        super().__init__()
        def block(a,b,s):return nn.Sequential(nn.Conv2d(a,b,3,stride=s,padding=1,bias=False),
                                              nn.GroupNorm(8,b),nn.SiLU())
        self.vision=nn.Sequential(block(3,32,2),block(32,48,2),block(48,80,2),
                                  block(80,128,2),block(128,192,2),nn.AdaptiveAvgPool2d((2,3)),nn.Flatten(),
                                  nn.Linear(1152,256),nn.LayerNorm(256),nn.SiLU())
        self.state=nn.Sequential(nn.Linear(state_dim,256),nn.LayerNorm(256),nn.SiLU(),nn.Linear(256,192),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(448,384),nn.LayerNorm(384),nn.SiLU(),nn.Linear(384,192),nn.SiLU(),nn.Linear(192,CHUNK*6))

    def forward(self,rgb,state):
        z=rgb.permute(0,3,1,2).contiguous(memory_format=torch.channels_last).float()/127.5-1
        return self.head(torch.cat([self.vision(z),self.state(state)],dim=1)).reshape(-1,CHUNK,6)


def freeze_and_load(dataset, allow_partial=False):
    file=dataset/'layouts.json';plan=json.loads(file.read_text()); rows=layouts()
    if plan['layouts']!=rows or plan['teacher_sha256']!=FROZEN_CHECKPOINT:
        raise ValueError('Unexpected frozen teacher/layout manifest')
    d={'train':([],[],[]),'dev':([],[],[])}; records=[]; rejected=[]; seen=set()
    for row in rows:
        split=row['split']; name=row['case']; part=dataset/'shards'/name
        if split=='heldout':
            if part.exists():raise ValueError('Heldout trace directory must not exist')
            continue
        rfile=part/'report.json'
        if allow_partial and not rfile.exists():continue
        r=json.loads(rfile.read_text())
        if (r['case']!=name or r['split']!=split or r['layout_sha256']!=sha_layout(row)
                or r['teacher_sha256']!=FROZEN_CHECKPOINT):raise ValueError(f'Unmatched provenance: {name}')
        if not r['accepted']:
            rejected.append(name);continue
        f=part/'trace.npz'; h=digest(f)
        if h!=r['trace_sha256'] or h in seen:raise ValueError(f'Hash/duplicate: {name}')
        seen.add(h)
        with np.load(f,allow_pickle=False) as t:
            if not validate_trace(t,r['metrics'],r['footprint']):raise ValueError(f'Invalid accepted trace: {name}')
            p,lo,hi=transport_prior(row,t['block_counts']);u=t['action'];rgb=t['rgb_external'];idx=t['rgb_action_index']
            if not (np.all(t['phase'][lo:hi]==4) and np.count_nonzero(t['phase']==4)==hi-lo and len(u)==len(p)):
                raise ValueError(f'Phase alignment drift: {name}')
            b,q,j=t['base_pre'],t['arm_pre'],t['gripper_pre']; selected=0
            for image,k in zip(rgb,idx):
                if k<lo or k+CHUNK>hi:continue
                z=p[k:k+CHUNK]; x=feature(row,b[k],q[k],j[k],z)
                y=(u[k:k+CHUNK,:6]-z).astype(np.float32)
                d[split][0].append(image);d[split][1].append(x);d[split][2].append(y);selected+=1
            records.append(dict(case=name,split=split,trace_sha256=h,rows=selected))
    count={key:sum(x['split']==key for x in records) for key in d}
    need={'train':TRAIN_MIN,'dev':DEV_MIN} if not allow_partial else {'train':8,'dev':0}
    if any(count[k]<need[k] for k in need):raise ValueError(f'Insufficient qualified data: {count}')
    manifest=dict(layouts_sha256=digest(file),teacher_sha256=FROZEN_CHECKPOINT,urdf_sha256=plan['urdf_sha256'],
                  accepted_episodes=count,rejected=rejected,records=records,heldout_arrays_loaded=False,
                  partial_smoke_only=allow_partial)
    out={k:tuple(np.stack(a) for a in v) if v[0] else None for k,v in d.items()}
    return out,manifest


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def train(dataset,out,epochs,batch,smoke):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID or torch.cuda.device_count()!=1:
        raise RuntimeError('Only the explicit authorized X7 GPU UUID is allowed')
    if torch.__version__!='2.6.0+cu124':raise RuntimeError('Torch runtime drift')
    torch.manual_seed(271828); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.95,ALLOCATOR_BYTES/total),0)
    out.mkdir(parents=True)
    arrays,manifest=freeze_and_load(dataset,allow_partial=smoke)
    info=out/'training-inputs.json';write_json(info,manifest)
    tr=arrays['train'];xmean=tr[1].mean(0);xstd=np.maximum(tr[1].std(0),.001)
    ymean=tr[2].mean(0);ystd=np.maximum(tr[2].std(0),.0001)
    device='cuda:0'
    def tensors(x):
        if x is None:return None
        return tuple(torch.as_tensor(a,device=device) for a in (x[0],(x[1]-xmean)/xstd,(x[2]-ymean)/ystd))
    train_set=tensors(tr); dev=tensors(arrays['dev']); del arrays,tr
    net=RGBTransport(len(xmean)).to(device=device,memory_format=torch.channels_last)
    opt=torch.optim.AdamW(net.parameters(),lr=.0006,weight_decay=.005,fused=True)
    obj=nn.SmoothL1Loss(); steps=0; best=float('inf')
    batches=(len(train_set[0])+batch-1)//batch
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs*batches),eta_min=.00003)
    write_json(out/'status.json',dict(stage='training',optimizer_updates=0,train_rows=len(train_set[0]),
        dev_rows=0 if dev is None else len(dev[0]),gpu_uuid=GPU_UUID,batch=batch,epochs=epochs,smoke=smoke))
    for epoch in range(epochs):
        net.train(); rolling=[]
        order=torch.randperm(len(train_set[0]),device=device)
        for ids in order.split(batch):
            xi,x,y=(a[ids] for a in train_set); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                loss=obj(net(xi,x),y)
            l=float(loss.detach())
            if not np.isfinite(l):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),2.); opt.step();sched.step()
            rolling.append(l);steps+=1
            if steps==1 or steps%50==0:
                print(json.dumps(dict(epoch=epoch+1,optimizer_updates=steps,loss=l,peak_reserved_mib=torch.cuda.max_memory_reserved()//2**20)),flush=True)
        net.eval(); errors=[]
        if dev is not None:
            with torch.inference_mode():
                for i in range(0,len(dev[0]),batch):
                    a,x,y=(s[i:i+batch] for s in dev)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):errors.append((float(obj(net(a,x),y)),len(a)))
        val=float(np.average([v for v,_ in errors],weights=[n for _,n in errors])) if errors else float(np.mean(rolling))
        if not np.isfinite(val):raise RuntimeError('Nonfinite validation loss')
        if val<best:
            best=val; c=out/'best.pt';tmp=out/'best.pt.tmp'
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},xmean=xmean.tolist(),
                xstd=xstd.tolist(),ymean=ymean.tolist(),ystd=ystd.tolist(),epoch=epoch+1,optimizer_updates=steps,
                validation_loss=val,manifest_sha256=digest(info),teacher_sha256=FROZEN_CHECKPOINT,smoke=smoke),tmp)
            tmp.replace(c)
        m=dict(stage='training',epoch=epoch+1,optimizer_updates=steps,train_loss=float(np.mean(rolling)),
               dev_loss=val,best_dev_loss=best,gpu_uuid=GPU_UUID,peak_reserved_mib=torch.cuda.max_memory_reserved()//2**20)
        write_json(out/'status.json',m)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(m,allow_nan=False)+'\n')
    write_json(out/'status.json',m|{'stage':'smoke_completed' if smoke else 'train_completed',
        'checkpoint_sha256':digest(out/'best.pt'),'training_input_sha256':digest(info),
        'physical_evaluation_completed':False})


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('dataset',type=Path);ap.add_argument('out',type=Path)
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--batch',type=int,default=256);ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args();train(a.dataset,a.out,a.epochs,a.batch,a.smoke)


if __name__=='__main__':main()
