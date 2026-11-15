#!/usr/bin/env python3
"""Bounded single-card fallback only if physical collection has too few successes.

Keeps the primary campaign's deadline; never bypasses a user pause or starts
another simulator while the primary controller is alive.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from resource_allocation import allocation, resolve_gpu


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/mnt/yuanyue/double-arm'))
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--teacher',type=Path,required=True)
    args=parser.parse_args();c=args.campaign;path=c/'fallback-watch.json'
    config=allocation(args.root);gpu=resolve_gpu(args.root,config['allowed_gpu_uuid'])
    original=json.loads((c/'status.json').read_text())
    end=datetime.fromisoformat(original['started_at']).timestamp()+original['max_hours']*3600
    def save(stage,**kwargs):
        path.write_text(json.dumps({'stage':stage,'gpu_uuid':gpu,'original_deadline_unix':end,
            'updated_at':datetime.now(timezone.utc).isoformat(),**kwargs},indent=2))
    while time.time()<end:
        if (args.root/'runs/USER_PAUSED.json').exists():save('paused_by_user');return 0
        if allocation(args.root).get('allowed_gpu_uuid')!=gpu:save('allocation_changed');return 0
        state=json.loads((c/'status.json').read_text());stage=state['stage']
        if stage=='insufficient_physical_demonstrations':
            remaining=end-time.time()
            if remaining<1800:save('insufficient_remaining_time');return 0
            if not (args.teacher/'training-result.json').exists():save('fallback_fit_missing');return 2
            pid=state['controller_pid']
            proc=Path(f'/proc/{pid}/stat')
            if proc.exists() and proc.read_text().rsplit(')',1)[1].split()[0]!='Z':time.sleep(2);continue
            stamp=datetime.now().strftime('%Y%m%d-%H%M%S');target=args.root/'runs'/('single-gpu-physical-fallback-'+stamp)
            log=args.root/'logs'/(target.name+'.log')
            command=[sys.executable,'-u',str(Path(__file__).with_name('run_single_gpu_campaign.py')),
                '--root',str(args.root),'--campaign',str(target),'--source-policy',str(args.teacher),
                '--hours',str(remaining/3600),'--target-successes','6','--prior-physical-data',str(c/'physical-successes.hdf5')]
            with log.open('a') as handle:
                child=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                    start_new_session=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'EVAL_GPU':gpu})
            save('fallback_started',controller_pid=child.pid,campaign=str(target),log=str(log));return 0
        terminal=('failed','ready','deadline','insufficient')
        if any(word in stage for word in terminal):save('primary_finished_no_extra_work',primary_stage=stage);return 0
        pid=state.get('controller_pid')
        if pid and not Path(f'/proc/{pid}').exists():save('primary_controller_exited_unexpectedly');return 2
        save('waiting_for_primary_campaign',primary_stage=stage);time.sleep(30)
    save('original_deadline_reached');return 0


if __name__=='__main__':raise SystemExit(main())
