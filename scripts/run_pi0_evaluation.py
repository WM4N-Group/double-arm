#!/usr/bin/env python3
"""SERVER ONLY: bounded genuine pi0 service + simulator, at most one GPU.

Memory feasibility is checked empirically; failure must not spill to GPU0 or
silently switch to a state BC / teacher. Real head+wrists are the only vision.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from resource_allocation import resolve_gpu
from pi0_review_contract import physical_task_passed


def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--source-dataset',type=Path,required=True)
    p.add_argument('--training-provenance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hours',type=float,default=2)
    p.add_argument('--inference-platform',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--execute-steps',type=int,default=None)
    args=p.parse_args();root=args.root;out=args.output
    execute_steps=args.execute_steps if args.execute_steps is not None else (10 if args.inference_platform=='cpu' else 5)
    if not 1<=execute_steps<=50:raise ValueError('Invalid receding horizon')
    # The lightweight controller may run under system Python (no numerical
    # dependencies). Re-exec this CPU orchestrator in the isolated openpi env;
    # never install h5py/numpy into the simulator or workstation as a workaround.
    expected_env=root/'envs/openpi'
    if Path(sys.prefix).resolve()!=expected_env.resolve():
        interpreter=expected_env/'bin/python'
        if not interpreter.is_file():raise FileNotFoundError(interpreter)
        print('Re-executing pi0 evaluation orchestrator in isolated openpi Python',flush=True)
        os.execv(str(interpreter),[str(interpreter),str(Path(__file__).resolve()),*sys.argv[1:]])
    import h5py
    import numpy as np
    if out.exists():raise FileExistsError(out)
    if not str(out).startswith(str(root/'runs')+'/'):raise ValueError('Remote output required')
    gpu=resolve_gpu(root);out.mkdir();deadline=time.monotonic()+args.hours*3600;script=Path(__file__).resolve().parent
    state={'stage':'preparing_pi0_native_evaluation','checkpoint':str(args.checkpoint),'gpu_uuid':gpu,
           'max_gpu_count':1,'task_success':False,'started_at':datetime.now(timezone.utc).isoformat(),
           'physical_parameters_calibrated_to_real_robot':False,'inference_platform':args.inference_platform,
           'execute_steps':execute_steps,'real_time_control_claim':False}
    def save(**kwargs):
        state.update(kwargs,updated_at=datetime.now(timezone.utc).isoformat());tmp=out/'status.tmp'
        tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'status.json')
    save()
    provenance=read(args.training_provenance,{})
    lows=[];highs=[]
    for item in provenance.get('episodes',[]):
        with h5py.File(item['source']) as file:
            actions=file['data'][item['source_demo']]['actions'][()]
            lows.append(actions.min(0));highs.append(actions.max(0))
    if not lows:raise ValueError('No training-only native control bounds available')
    bounds={'min':np.stack(lows).min(0).tolist(),'max':np.stack(highs).max(0).tolist(),
            'source':'Only the training split; no evaluation teacher actions read'}
    (out/'action-bounds.json').write_text(json.dumps(bounds,indent=2))
    token=out/'rpc-token.local';descriptor=os.open(token,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as handle:handle.write(secrets.token_hex(32)+'\n')
    base={**os.environ,'DOUBLE_ARM_ROOT':str(root),'OPENPI_ROOT':str(root/'src/openpi-215abfb'),
          'PYTHONPATH':str(script)+':'+str(root/'src/openpi-215abfb/src'),
          'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
          'HF_HOME':str(root/'cache/huggingface'),'HF_LEROBOT_HOME':str(root/'data/pi0/lerobot'),
          'OPENPI_DATA_HOME':str(root/'cache/openpi')}
    def launch(command,log,extra=None):
        with log.open('a') as handle:
            return subprocess.Popen(['timeout','--signal=TERM','--kill-after=60s',str(max(1,int(deadline-time.monotonic()))),
                'taskset','-c','88-103']+command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                start_new_session=True,env={**base,**(extra or {})})
    server=None;simulation=None;exporter=None
    try:
        existing=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(line.split(',')[0].strip()==gpu for line in existing.splitlines()):raise RuntimeError('Allocated GPU occupied; refusing to steal it')
        server=launch([str(root/'envs/openpi/bin/python'),'-u',str(script/'serve_pi0_r1.py'),
            '--checkpoint',str(args.checkpoint),'--token-file',str(token),'--ready-file',str(out/'inference-ready.json'),
            '--hours',str(args.hours),'--inference-platform',args.inference_platform],out/'inference-server.log',
            {'XLA_PYTHON_CLIENT_PREALLOCATE':'false','XLA_PYTHON_CLIENT_MEM_FRACTION':'0.32',
             'CUDA_VISIBLE_DEVICES':gpu if args.inference_platform=='gpu' else '',
             'JAX_PLATFORMS':'cuda' if args.inference_platform=='gpu' else 'cpu'})
        save(stage='loading_pi0_for_visual_inference',server_pid=server.pid)
        while time.monotonic()<deadline and server.poll() is None:
            if (out/'inference-ready.json').exists():break
            if (root/'runs/USER_PAUSED.json').exists():raise RuntimeError('User paused project')
            time.sleep(10)
        if not (out/'inference-ready.json').exists():raise RuntimeError('Pi0 inference service failed to load; inspect log')
        from pi0_rpc import infer_local
        from momagen_pi0_schema import camera_keys,rgb_image,PROMPT
        # Warm up with a real heldout reset image, not synthetic replacement.
        with h5py.File(args.source_dataset) as file:
            group=file['data/demo_0'];keys=camera_keys(group['obs'])
            warm={name:rgb_image(group['obs'][key][0]) for name,key in keys.items()}
            warm.update(state=group['obs/prop_state'][0],prompt=PROMPT)
        infer_local(warm,token.read_text().strip(),timeout=min(600,max(1,int(deadline-time.monotonic()))))
        # Default CPU inference is the same checkpoint/model, not a substitute
        # policy: it leaves the allocated GPU to the simulator. Explicit shared
        # GPU inference still needs enough measured free memory. No GPU0.
        query=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        free={line.split(',')[0].strip():int(line.split(',')[1].strip()) for line in query.splitlines()}
        save(pi0_warmup_completed=True,free_gpu_mib_before_simulator=free.get(gpu))
        if free.get(gpu,0)<22000:raise RuntimeError('Insufficient free GPU memory for simulator; reserve 22000MiB, use CPU inference rather than GPU0')
        # A bare system-Python controller does not inherit Isaac/OG asset and
        # headless settings. Source the pinned runtime only for the simulator;
        # the inference service keeps its separate openpi environment.
        simulation=launch(['bash','-c','set -euo pipefail; source "$1"; shift; exec "$@"',
            'pi0-simulator',str(script/'server_env.sh'),str(root/'envs/momagen/bin/python'),'-u',str(script/'evaluate_pi0_r1.py'),
            '--dataset',str(args.source_dataset),'--output',str(out/'physical-rollout'),'--checkpoint',str(args.checkpoint),
            '--token-file',str(token),'--action-bounds',str(out/'action-bounds.json'),'--grasping-mode','physical',
            '--execute-steps',str(execute_steps)],out/'physical-rollout.log')
        save(stage='evaluating_real_visual_pi0_physical',simulation_pid=simulation.pid)
        while simulation.poll() is None:
            if time.monotonic()>deadline or (root/'runs/USER_PAUSED.json').exists():raise TimeoutError('Evaluation deadline or user pause')
            if server.poll() is not None:raise RuntimeError('Pi0 service exited during rollout; no fallback allowed')
            time.sleep(15)
        result=read(out/'physical-rollout/result.json',{})
        eligible=simulation.returncode==0 and physical_task_passed(result)
        save(stage='pi0_physical_closed_loop_success_review_required' if eligible else 'pi0_physical_closed_loop_failed',
             task_success=bool(eligible),result=result,video=str(out/'physical-rollout/demo_0.mp4'))
        # Review export has its own status: artifact errors must not change the
        # actual physical result or quietly turn failed attempts into successes.
        if result.get('episodes') and (out/'physical-rollout/demo_0.mp4').is_file():
            exporter=launch([str(root/'envs/openpi/bin/python'),'-u',str(script/'export_pi0_review.py'),
                '--campaign',str(out.parent),'--evaluation',str(out),'--output',str(out/'review')],
                out/'review-export.log',{'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'})
            try:
                export_code=exporter.wait(timeout=min(600,max(1,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                os.killpg(exporter.pid,signal.SIGTERM)
                try:exporter.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(exporter.pid,signal.SIGKILL);exporter.wait(timeout=10)
                export_code=124
            save(review_export_exit_status=export_code,
                 review_directory=str(out/'review') if export_code==0 and (out/'review/review.json').is_file() else None)
        return 0 if eligible else 7
    except Exception as error:
        save(stage='pi0_deployment_or_evaluation_failed',error=repr(error),task_success=False)
        raise
    finally:
        for process in (exporter,simulation,server):
            if process and process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass


if __name__=='__main__':raise SystemExit(main())
