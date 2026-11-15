#!/usr/bin/env python3
"""SERVER ONLY: official openpi inference in its isolated Python3.11 runtime."""
import argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import hashlib
import os
from pathlib import Path
import secrets
import time
from pi0_rpc import decode_observation
from resource_allocation import resolve_gpu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--ready-file',type=Path,required=True)
    p.add_argument('--token-file',type=Path,required=True)
    p.add_argument('--port',type=int,default=18731)
    p.add_argument('--hours',type=float,default=2)
    p.add_argument('--inference-platform',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--denoising-steps',type=int,default=10)
    p.add_argument('--sampling-noise',choices=['gaussian','zero'],default='gaussian')
    p.add_argument('--action-representation',choices=['native','absolute_base'],default='native')
    args=p.parse_args();root=Path(os.environ.get('DOUBLE_ARM_ROOT','/mnt/yuanyue/double-arm'))
    if not 1<=args.denoising_steps<=100:raise ValueError('Invalid denoising step count')
    gpu=resolve_gpu(root,os.environ.get('CUDA_VISIBLE_DEVICES'))
    if not str(args.checkpoint).startswith(str(root/'runs')+'/'):raise ValueError('A project fine-tuned checkpoint is required, not the unfine-tuned base')
    if not (args.checkpoint/'params').is_dir():raise ValueError('Missing openpi checkpoint params')
    token=args.token_file.read_text().strip()
    if len(token)<32:raise ValueError('Strong local authentication token required')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu if args.inference_platform=='gpu' else ''
    os.environ['JAX_PLATFORMS']='cuda' if args.inference_platform=='gpu' else 'cpu'
    os.environ.setdefault('OPENPI_DATA_HOME',str(root/'cache/openpi'))
    from pi0_r1_config import register_config
    from openpi.policies.policy_config import create_trained_policy
    config=register_config(action_representation=args.action_representation)
    contract_path=args.checkpoint/'assets/r1-adapter-contract.json'
    contract=json.loads(contract_path.read_text()) if contract_path.is_file() else None
    if contract is not None:
        if (contract.get('model')!='pi0' or contract.get('config_name')!=config.name
            or contract.get('action_representation')!=args.action_representation
            or contract.get('quantile_normalization')!=(args.action_representation=='absolute_base')):
            raise ValueError('Checkpoint adapter contract mismatch; never interpret world targets as native deltas')
        asset_id=config.data.assets.asset_id or config.data.repo_id
        if contract.get('norm_stats_asset_id')!=asset_id:raise ValueError('Wrong checkpoint normalization asset id')
        norm=args.checkpoint/'assets'/asset_id/'norm_stats.json'
        if hashlib.sha256(norm.read_bytes()).hexdigest()!=contract.get('norm_stats_sha256'):
            raise ValueError('Checkpoint normalization hash does not match its adapter contract')
    elif args.action_representation!='native':raise ValueError('Absolute-base checkpoint requires a verified adapter contract')
    policy=create_trained_policy(config,args.checkpoint,sample_kwargs={'num_steps':args.denoising_steps})
    import jax
    actual_platform=jax.default_backend()
    if (actual_platform=='cpu')!=(args.inference_platform=='cpu'):
        raise ValueError('Inference platform does not match requested device')
    ready={'model':'pi0','policy_framework':'official_openpi','checkpoint':str(args.checkpoint),'gpu_uuid':gpu if args.inference_platform=='gpu' else None,
           'allocation_gpu_uuid':gpu,'inference_platform':args.inference_platform,'actual_jax_platform':actual_platform,
           'native_action_dim':21,'action_horizon':50,'privileged_state':False,'teacher_enabled':False,
           'port':args.port,'inference_warmed_up':False,'denoising_steps':args.denoising_steps,'sampling_noise':args.sampling_noise,'action_representation':args.action_representation,'adapter_contract_verified':contract is not None}
    args.ready_file.write_text(json.dumps(ready,indent=2));deadline=time.monotonic()+args.hours*3600
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/health':self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(ready).encode())
        def do_POST(self):
            if self.path!='/infer' or not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403);return
            length=int(self.headers.get('Content-Length','0'))
            if length<1 or length>3*1024*1024:self.send_error(413);return
            try:
                observation=decode_observation(json.loads(self.rfile.read(length)))
                import numpy as np
                noise=np.zeros((50,config.model.action_dim),np.float32) if args.sampling_noise=='zero' else None
                start=time.monotonic();prediction=policy.infer(observation,noise=noise)
                from momagen_pi0_schema import unpad_pi0_actions
                actions=unpad_pi0_actions(prediction['actions'])
                if actions.shape!=(50,21):raise ValueError('Unexpected pi0 action horizon')
                result={'actions':actions.tolist(),'model':'pi0','teacher_enabled':False,'inference_seconds':time.monotonic()-start,
                        'inference_platform':args.inference_platform,'denoising_steps':args.denoising_steps,'sampling_noise':args.sampling_noise,'action_representation':args.action_representation,'adapter_contract_verified':contract is not None}
                ready['inference_warmed_up']=True;args.ready_file.write_text(json.dumps(ready,indent=2))
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(result).encode())
            except Exception as error:
                import traceback
                traceback.print_exc();self.send_error(500,str(type(error).__name__)+': inference failed, no fallback')
        def log_message(self,format,*args):pass
    server=HTTPServer(('127.0.0.1',args.port),Handler);server.timeout=2
    print('PI0_R1_INFERENCE_READY',json.dumps(ready),flush=True)
    while time.monotonic()<deadline and not (root/'runs/USER_PAUSED.json').exists():server.handle_request()
    server.server_close()


if __name__=='__main__':main()
