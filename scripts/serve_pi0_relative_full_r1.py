#!/usr/bin/env python3
"""SERVER ONLY: explicit saved relative-full-demo profile, genuine original pi0.

Serves native21 commands AFTER inverse using current RPC proprio. No source
HDF5/dataset/teacher/planner reads. CPU inference leaves GPU1 for simulation.
"""
import argparse
from http.server import BaseHTTPRequestHandler,HTTPServer
import json
import os
from pathlib import Path
import secrets
import time
from pi0_relative_full_profile import ROOT,PROFILE,MODEL_ENCODING,make_policy,sha
from resource_allocation import resolve_gpu


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--ready-file',type=Path,required=True)
    p.add_argument('--token-file',type=Path,required=True);p.add_argument('--port',type=int,default=18731)
    p.add_argument('--hours',type=float,default=3);p.add_argument('--inference-platform',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--denoising-steps',type=int,default=10);p.add_argument('--sampling-noise',choices=['gaussian','zero'],default='gaussian')
    p.add_argument('--action-representation',choices=['native'],default='native');a=p.parse_args()
    if a.denoising_steps!=10 or a.sampling_noise!='gaussian':raise ValueError('This diagnostic profile was validated with10/Gaussian, no silent sampler changes')
    gpu=resolve_gpu(ROOT,'1')
    if a.inference_platform=='gpu' and os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:raise ValueError('Explicit GPU1 UUID required')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu if a.inference_platform=='gpu' else ''
    os.environ['JAX_PLATFORMS']='cuda' if a.inference_platform=='gpu' else 'cpu'
    os.environ.setdefault('OPENPI_DATA_HOME',str(ROOT/'cache/openpi'))
    token=a.token_file.read_text().strip()
    if len(token)<32:raise ValueError('Strong existing localhost authentication required')
    policy,contract=make_policy(a.checkpoint)
    import jax
    import numpy as np
    from pi0_rpc import decode_observation
    if (jax.default_backend()=='cpu')!=(a.inference_platform=='cpu'):raise ValueError('Backend mismatch')
    ready={'model':'pi0','policy_framework':'official_openpi','policy_profile':PROFILE,'checkpoint':str(a.checkpoint),
        'serving_source_sha256':sha(Path(__file__)),'model_action_encoding':MODEL_ENCODING,'action_representation':'native','native_action_dim':21,'action_horizon':50,
        'parameter_restore_dtype':'preserve_saved','norm_stats_sha256':contract['norm_stats_sha256'],
        'privileged_state':False,'teacher_enabled':False,'planner_enabled':False,'source_demonstrations_opened':False,
        'inference_platform':a.inference_platform,'actual_jax_platform':jax.default_backend(),
        'allocation_gpu_uuid':gpu,'gpu_uuid':gpu if a.inference_platform=='gpu' else None,'port':a.port,
        'denoising_steps':10,'sampling_noise':'gaussian','inference_warmed_up':False,'real_time_control_claim':False,
        'training_is_diagnostic':True,'paper_reproduction_complete':False}
    a.ready_file.write_text(json.dumps(ready,indent=2));deadline=time.monotonic()+a.hours*3600
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/health':self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(ready).encode())
        def do_POST(self):
            if self.path!='/infer' or not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403);return
            size=int(self.headers.get('Content-Length','0'))
            if not 1<=size<=3*1024*1024:self.send_error(413);return
            try:
                observation=decode_observation(json.loads(self.rfile.read(size)))
                start=time.monotonic();prediction=policy.infer(observation);actions=np.asarray(prediction['actions'],np.float32)
                if actions.shape!=(50,21) or not np.isfinite(actions).all():raise ValueError('Invalid native action chunk; no fallback')
                ready['inference_warmed_up']=True;a.ready_file.write_text(json.dumps(ready,indent=2))
                payload={**ready,'actions':actions.tolist(),'inference_seconds':time.monotonic()-start}
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(json.dumps(payload).encode())
            except Exception as error:
                import traceback
                traceback.print_exc();self.send_error(500,type(error).__name__+': relative pi0 inference failed, no fallback')
        def log_message(self,*_):pass
    server=HTTPServer(('127.0.0.1',a.port),Handler);server.timeout=2
    print('PI0_RELATIVE_FULL_INFERENCE_READY',json.dumps(ready),flush=True)
    while time.monotonic()<deadline and not (ROOT/'runs/USER_PAUSED.json').exists():server.handle_request()
    server.server_close()


if __name__=='__main__':main()
