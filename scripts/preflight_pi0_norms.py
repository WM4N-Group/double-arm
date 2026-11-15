#!/usr/bin/env python3
"""SERVER ONLY: CPU-only real-data loader/normalization contract preflight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--wait-pid',type=int,required=True)
    args=p.parse_args();c=args.campaign;root=Path('/mnt/yuanyue/double-arm');deadline=time.monotonic()+5400
    status=c/'real-data-preflight.json'
    def save(stage,**kw):status.write_text(json.dumps({'stage':stage,'not_a_training_run':True,'gpu_initialized':False,'updated_at':time.time(),**kw},indent=2))
    save('waiting_for_real_data_converter',pid=args.wait_pid)
    while True:
        proc=Path(f'/proc/{args.wait_pid}/stat')
        try:active=proc.read_text().rsplit(')',1)[1].split()[0]!='Z'
        except FileNotFoundError:active=False
        if not active:break
        if time.monotonic()>deadline or (root/'runs/USER_PAUSED.json').exists():save('paused_or_preflight_deadline');return 75
        time.sleep(15)
    repo='momagen/r1_tidy_table_contract_train_v0'
    if not (root/'data/pi0/lerobot'/repo/'momagen-provenance.json').exists():save('preflight_converter_failed');return 2
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu','PI0_DATASET_REPO':repo,
         'PI0_NORM_MAX_FRAMES':'128','PI0_BATCH_SIZE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4',
         'DOUBLE_ARM_ROOT':str(root),'OPENPI_ROOT':str(root/'src/openpi-215abfb'),
         'PYTHONPATH':str(root/'code/scripts')+':'+str(root/'src/openpi-215abfb/src'),
         'HF_HOME':str(root/'cache/huggingface'),'HF_LEROBOT_HOME':str(root/'data/pi0/lerobot'),
         'OPENPI_DATA_HOME':str(root/'cache/openpi')}
    save('checking_real_pi0_data_transforms_on_cpu',sample_frames=128)
    with (c/'real-data-normalization-preflight.log').open('a') as log:
        code=subprocess.call(['timeout','--signal=TERM','--kill-after=60s','1800','taskset','-c','88-103',
            str(root/'envs/openpi/bin/python'),'-u',str(root/'code/scripts/run_pi0_entrypoint.py'),'norms'],stdout=log,stderr=subprocess.STDOUT,env=env)
    save('real_pi0_loader_contract_passed' if code==0 else 'real_pi0_loader_contract_failed',exit_status=code,
         normalization='128-frame preflight only, separate v0 repo; NOT the final train normalization stats')
    return code


if __name__=='__main__':raise SystemExit(main())
