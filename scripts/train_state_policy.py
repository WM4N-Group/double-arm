#!/usr/bin/env python3
"""Server-only supervised state BC with short observation history/action chunks.

Train/validation split is by whole generated episode, NOT adjacent frames.
No clock, phase labels, teacher actions or future observations are inputs.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import time
import h5py
import numpy as np
import torch
from torch import nn
from state_policy_features import feature_keys

GRIP = [13, 20]
JOINT = list(range(3, 13)) + list(range(14, 20))


class ChunkPolicy(nn.Module):
    def __init__(self, obs_dim, history=4, horizon=16, hidden=512):
        super().__init__()
        self.history, self.horizon = history, horizon
        self.net = nn.Sequential(nn.Linear(obs_dim * history, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, horizon * 21))

    def forward(self, obs):
        return self.net(obs.flatten(1)).reshape(-1, self.horizon, 21)


def joint_anchor(prop):
    """prop_state groups both arms before jaws; action interleaves arm/jaw.

    Action: base3,trunk4,left6,left_jaw1,right6,right_jaw1.
    Proprio: base_vel3,trunk4,left6,right6,left_width1,right_width1.
    Base commands in generated data are local one-step displacement/yaw,
    not world joint targets or the raw human source's velocity controls.
    """
    prop = np.asarray(prop, dtype=np.float32)
    if prop.shape[-1] != 21: raise ValueError('Expected R1 prop_state dimension 21')
    anchor = np.zeros_like(prop)
    anchor[..., 3:13] = prop[..., 3:13]
    anchor[..., 14:20] = prop[..., 13:19]
    return anchor


def load_episodes(path):
    episodes = []
    with h5py.File(path, 'r') as f:
        names = sorted((n for n in f['data'] if n.startswith('demo_')), key=lambda x: int(x.split('_')[-1]))
        if not names:
            raise ValueError('No episodes')
        keys = feature_keys(f['data'][names[0]]['obs'])
        if not any(k.startswith('task::plate.') for k in keys):
            raise ValueError('Unique plate instance poses are required; ambiguous object::plate is insufficient')
        for name in names:
            g = f['data'][name]
            x = np.concatenate([g['obs'][k][()].reshape(len(g['actions']), -1) for k in keys], axis=1).astype('float32')
            a, prop = g['actions'][()].astype('float32'), g['obs/prop_state'][()].astype('float32')
            if a.shape[1] != 21 or not (np.isfinite(x).all() and np.isfinite(a).all()):
                raise ValueError('Invalid state/action data')
            episodes.append((name, x, a, joint_anchor(prop)))
        meta = json.loads(f['data'].attrs['env_args'])
    return episodes, keys, meta


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--history', type=int, default=4)
    p.add_argument('--history-stride', type=int, default=4)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--hidden', type=int, default=512)
    p.add_argument('--val-demo', default=None)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    episodes, keys, meta = load_episodes(args.dataset)
    if len(episodes) < 3:
        raise ValueError('Need >=3 successful episodes for an honest episode-disjoint smoke split')
    val_name = args.val_demo or episodes[-1][0]
    train = [e for e in episodes if e[0] != val_name]
    val = [e for e in episodes if e[0] == val_name]
    if len(val) != 1 or len(train) < 2:
        raise ValueError('Invalid split')
    tx = np.concatenate([e[1] for e in train]); ta = np.concatenate([e[2] for e in train])
    tp = np.concatenate([e[3] for e in train])
    y = ta.copy(); y[:, JOINT] -= tp[:, JOINT]
    x_mean, x_std = tx.mean(0), np.maximum(tx.std(0), 0.05)
    y_mean, y_std = y.mean(0), np.maximum(y.std(0), 0.02)
    y_std[:3] = np.maximum(y.std(0)[:3], 0.002)
    y_mean[GRIP] = 0; y_std[GRIP] = 1
    norm = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
                action_min=ta.min(0), action_max=ta.max(0))
    norm_t = {k: torch.tensor(v, device=device) for k,v in norm.items()}

    def pack(eps):
        xs=[]; acts=[]; props=[]; lows=[]; highs=[]; offset=0
        for _,x,a,prop in eps:
            xs.append(x); acts.append(a); props.append(prop)
            lows.extend([offset]*len(x)); highs.extend([offset+len(x)-1]*len(x)); offset+=len(x)
        return [torch.tensor(v,device=device) for v in (np.concatenate(xs), np.concatenate(acts),
                np.concatenate(props),np.asarray(lows),np.asarray(highs))]
    tr, va = pack(train), pack(val)
    hist_offsets = torch.arange(-(args.history-1),1,device=device)*args.history_stride
    future_offsets = torch.arange(args.horizon,device=device)
    non_grip = [i for i in range(21) if i not in GRIP]

    def batch(data):
        x,a,prop,lo,hi=data
        idx=torch.randint(len(x),(args.batch,),device=device)
        past=(idx[:,None]+hist_offsets).maximum(lo[idx,None])
        future=(idx[:,None]+future_offsets).minimum(hi[idx,None])
        target=a[future].clone()
        target[:,:,JOINT]-=prop[idx][:,None,JOINT]
        target=(target-norm_t['y_mean'])/norm_t['y_std']
        return (x[past]-norm_t['x_mean'])/norm_t['x_std'],target

    def loss_fn(pred,target):
        reg=nn.functional.smooth_l1_loss(pred[:,:,non_grip],target[:,:,non_grip])
        grip=nn.functional.binary_cross_entropy_with_logits(pred[:,:,GRIP],(target[:,:,GRIP]+1)/2)
        return reg + grip * 0.25

    model=ChunkPolicy(len(x_mean),args.history,args.horizon,args.hidden).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    manifest={'dataset':str(args.dataset.resolve()),'dataset_sha256':digest(args.dataset),
              'algorithm':'state_history_chunk_BC','privileged_state':True,'teacher_in_evaluation':False,
              'train_demos':[e[0] for e in train],'val_demos':[val_name],
              'episode_lengths':{e[0]:len(e[1]) for e in episodes},'keys':keys,
              'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'action_schema':'base_local_delta3,trunk4,left_arm6,left_gripper1,right_arm6,right_gripper1',
              'env_args':meta,'split_caveat':'All generated episodes may descend from the same human source demonstration. This split is not a new-scene benchmark.'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    best=float('inf'); start=time.time()
    with (args.output/'metrics.jsonl').open('w',buffering=1) as log:
        for step in range(1,args.steps+1):
            model.train(); x,target=batch(tr); pred=model(x); loss=loss_fn(pred,target)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            if step==1 or step%250==0:
                model.eval()
                with torch.no_grad():
                    vl=float(torch.stack([loss_fn(model(bx),by) for bx,by in [batch(va) for _ in range(8)]]).mean())
                record={'step':step,'train_loss':float(loss),'val_loss':vl,'elapsed_s':time.time()-start}
                log.write(json.dumps(record)+'\n'); print(json.dumps(record),flush=True)
                checkpoint={'model':model.state_dict(),'obs_dim':len(x_mean),'history':args.history,
                     'history_stride':args.history_stride,'horizon':args.horizon,'hidden':args.hidden,
                     'norm':{k:v.tolist() for k,v in norm.items()},'keys':keys,'env_args':meta,
                     'train_demos':manifest['train_demos'],'val_demos':[val_name],
                     'step':step,'val_loss':vl,'dataset_sha256':manifest['dataset_sha256']}
                if vl<best:
                    best=vl; torch.save(checkpoint,args.output/'best.pt')
                torch.save(checkpoint,args.output/'last.pt')
    (args.output/'training-result.json').write_text(json.dumps({'status':'training_completed',
           'best_val_loss':best,'steps':args.steps,'elapsed_s':time.time()-start,
           'closed_loop_success':'not_yet_evaluated'},indent=2))


if __name__=='__main__':
    main()
