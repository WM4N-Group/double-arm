#!/usr/bin/env python3
"""Fine-tune an X7 transport policy with paired, qualified noisy-odometry traces."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from collect_x7_noisy_odom import task_plan
from collect_x7_transport_bc import validate_trace
from train_x7_rgb_transport import (
    ALLOCATOR_BYTES, CHUNK, FROZEN_CHECKPOINT, GPU_UUID, RGBTransport, digest,
    feature, freeze_and_load, transport_prior, write_json,
)

ORIGINAL='0da95775f33f3b3975d48c01dcf30f3693ea69e28c48baecf70bbcca10ab5065'
UPDATES=640


def noisy_inputs(folder):
    plan=json.loads((folder/'plan.json').read_text());report=json.loads((folder/'qa-noisy-odom.json').read_text())
    if plan['cases']!=task_plan() or plan['teacher_sha256']!=FROZEN_CHECKPOINT or report['accepted']<36 or len(report['records'])!=48:
        raise ValueError('No finalized qualified v2 source data')
    images=[];states=[];targets=[]; info=[]; seen=set(); signs=[]
    for row,r in zip(plan['cases'],report['records']):
        if any(r[k]!=v for k,v in row.items()) or row['layout']['split']!='train':raise ValueError('Noisy layout provenance mismatch')
        if not r['accepted']:continue
        f=folder/'episodes'/r['case']/'trace.npz';h=digest(f)
        if h!=r['trace_sha256'] or h in seen:raise ValueError('Unmatched or duplicate noisy trace')
        seen.add(h)
        with np.load(f,allow_pickle=False) as t:
            if not validate_trace(t,r['metrics'],r['footprint']):raise ValueError('Bad physical gate')
            p,lo,hi=transport_prior(row['layout'],t['block_counts']);a=t['action']
            biased=t['base_policy_pre'];mask=np.any(t['odom_bias_for_audit_only']!=0.,axis=1)
            rgb,idx=t['rgb_external'],t['rgb_action_index'];q,j=t['arm_pre'],t['gripper_pre'];n=0
            for frame,k in zip(rgb,idx):
                if k<lo or k+CHUNK>hi or not np.any(mask[k:k+CHUNK]):continue
                ref=p[k:k+CHUNK]
                images.append(frame);states.append(feature(row['layout'],biased[k],q[k],j[k],ref))
                targets.append((a[k:k+CHUNK,:6]-ref).astype(np.float32));n+=1
                signs.append(1 if row['bias_m']>0 else -1)
        info.append(dict(case=r['case'],parent=r['parent_train_case'],sha256=h,rows=n))
    if min(signs)==max(signs):raise ValueError('Need both measured bias signs')
    return tuple(np.stack(v) for v in (images,states,targets)),info,np.asarray(signs)


def fit(original,noisy,out):
    if os.getenv('CUDA_VISIBLE_DEVICES')!=GPU_UUID or torch.cuda.device_count()!=1 or torch.__version__!='2.6.0+cu124':
        raise RuntimeError('Pin the original authorized GPU and Torch runtime')
    torch.set_num_threads(2);torch.manual_seed(271829);torch.backends.cuda.matmul.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(ALLOCATOR_BYTES/torch.cuda.get_device_properties(0).total_memory,0)
    f=original/'training/best.pt'
    if digest(f)!=ORIGINAL:raise ValueError('Original initializer hash changed')
    packed=torch.load(f,map_location='cpu',weights_only=True)
    clean,manifest=freeze_and_load(original/'dataset');aug,info,sign=noisy_inputs(noisy)
    if packed['manifest_sha256']!=digest(original/'training/training-inputs.json'):
        raise ValueError('Original fit input provenance drift')
    mean,std,ymean,ystd=[np.array(packed[k],dtype=np.float32) for k in ('xmean','xstd','ymean','ystd')]
    device='cuda:0'
    def to(data):
        return tuple(torch.as_tensor(x,device=device) for x in (data[0],(data[1]-mean)/std,(data[2]-ymean)/ystd))
    nominal=to(clean['train']);noisy_gpu=to(aug);del clean,aug
    # Oversample clean terminal contexts plus equally frequent opposite biases.
    # No dev labels, no stress panel, no heldout examples enter the optimizer.
    terms=[];start=0
    for r in manifest['records']:
        if r['split']!='train':continue
        stop=start+r['rows'];terms.extend(range(stop-min(r['rows'],15),stop));start=stop
    term=torch.tensor(terms,device=device);s0=torch.tensor(np.flatnonzero(sign<0),device=device);s1=torch.tensor(np.flatnonzero(sign>0),device=device)
    model=RGBTransport(len(mean));model.load_state_dict(packed['state_dict'],strict=True)
    model.to(device=device,memory_format=torch.channels_last).train();opt=torch.optim.AdamW(model.parameters(),lr=.00012,weight_decay=.005,fused=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=UPDATES,eta_min=.00002);obj=nn.SmoothL1Loss();out.mkdir()
    inputs=out/'training-inputs.json'
    write_json(inputs,manifest|dict(noisy_qualified=len(info),noisy_records=info,noisy_rows=len(sign),initializer_sha256=ORIGINAL,
        noisy_qa_sha256=digest(noisy/'qa-noisy-odom.json'),full_state_for_audit_only=True,dev_labels_in_optimizer=False,
        explicit_recipe='64 clean globally + 64 clean terminal + 64 each noisy sign, fixed final update 640'))
    for step in range(1,UPDATES+1):
        j=torch.cat((torch.randint(len(nominal[0]),(64,),device=device),term[torch.randint(len(term),(64,),device=device)]))
        k=torch.cat((s0[torch.randint(len(s0),(64,),device=device)],s1[torch.randint(len(s1),(64,),device=device)]))
        img,x,y=[torch.cat((a[j],b[k]),dim=0) for a,b in zip(nominal,noisy_gpu)]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=obj(model(img,x),y)
        score=float(loss.detach())
        if not np.isfinite(score):raise RuntimeError('Nonfinite fine-tune loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();sched.step()
        if step==1 or step%40==0:
            r=dict(stage='training',optimizer_updates=step,loss=score,gpu_uuid=GPU_UUID,peak_reserved_mib=torch.cuda.max_memory_reserved()//2**20)
            write_json(out/'status.json',r);print(json.dumps(r),flush=True)
    c=out/'best.pt';tmp=out/'best.pt.tmp'
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
        **{k:packed[k] for k in ('xmean','xstd','ymean','ystd')},epoch=0,optimizer_updates=UPDATES,
        manifest_sha256=digest(inputs),teacher_sha256=FROZEN_CHECKPOINT,smoke=False,selection='fixed_final_640'),tmp)
    tmp.replace(c);write_json(out/'status.json',r|dict(stage='train_completed',checkpoint_sha256=digest(c),
        training_input_sha256=digest(inputs),physical_evaluation_completed=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('original','noisy','out'):ap.add_argument(key,type=Path)
    a=ap.parse_args();fit(a.original,a.noisy,a.out)


if __name__=='__main__':main()
