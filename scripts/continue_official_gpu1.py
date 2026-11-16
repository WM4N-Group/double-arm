#!/usr/bin/env python3
"""SERVER CPU: finish existing auxiliary pool -> global raw QA -> ONE new pool.

Never stops the dependency or GPU0, never changes an in-flight generation job.
Keeps a released authorized GPU useful without another heartbeat delay. This is
finite data orchestration, NOT policy training, acceptance or new app automation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from preflight_pi0_accumulation_gpu import process_alive
from resource_allocation import resolve_gpu

ROOT = Path('/mnt/yuanyue/double-arm')
COMPLETE = 'raw_official_pool_target_complete_alignment_and_split_pending'


def dependency_outcome(state):
    if state.get('error') or state.get('stage') == 'generation_failed_or_interrupted':
        return 'failed'
    if state.get('stage') != COMPLETE:
        return 'pending'
    if (state.get('verified_unique_demos', 0) < 25 or state.get('active_pid') != 0
        or state.get('policy_training') != 'not_run'
        or state.get('source_grasping_mode') != 'assisted'):
        return 'failed'
    return 'ready'


def merge_pool_paths(existing, additional):
    return sorted({str(Path(path)) for path in [*existing, *additional]})


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--dependency', type=Path, required=True)
    parser.add_argument('--next-seed', type=int, default=20001)
    args = parser.parse_args()
    c, dependency = args.campaign, args.dependency
    if (not c.resolve().is_relative_to(ROOT/'runs') or c.exists()
        or not dependency.resolve().is_relative_to(ROOT/'runs')
        or not 20001 <= args.next_seed <= 29999):
        raise ValueError('New bounded in-project data followup required')
    gpu = resolve_gpu(ROOT, '1')
    dependency_state = read(dependency/'status.json')
    if dependency_state.get('gpu_uuid') != gpu or dependency_state.get('target_unique_demos') != 25:
        raise ValueError('Expected existing GPU1 auxiliary25 dependency')
    lock = (ROOT/'runs/paper-gpu1-followup.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c.mkdir()
    script = Path(__file__).resolve().parent
    deadline = time.monotonic() + 8*3600
    child = None
    status = {'stage': 'waiting_for_existing_auxiliary_pool', 'controller_pid': os.getpid(),
              'dependency': str(dependency), 'next_seed': args.next_seed,
              'next_target_unique_demos': 25, 'gpu_uuid': gpu,
              'started_at': datetime.now(timezone.utc).isoformat(),
              'dependency_interrupted': False, 'GPU0_modified': False,
              'policy_training': 'not_run', 'policy_task_success': False,
              'source_grasping_mode': 'assisted', 'app_automation_created': False}
    def save(**updates):
        status.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        tmp = c/'status.tmp'
        tmp.write_text(json.dumps(status, indent=2))
        tmp.replace(c/'status.json')
    def check():
        if (ROOT/'runs/USER_PAUSED.json').exists() or resolve_gpu(ROOT, '1') != gpu:
            raise RuntimeError('User pause or allocation changed')
        if time.monotonic() >= deadline:
            raise RuntimeError('Eight-hour followup watchdog, not a user approval window')
    def interrupted(signum, frame):
        raise InterruptedError(f'Followup signal{signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    (ROOT/'runs/latest-paper-gpu1-followup.txt').write_text(str(c)+'\n')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu',
           'PYTHONPATH': str(script), 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
    def register_and_audit(extra, name):
        primary = Path((ROOT/'runs/latest-paper-generation-campaign.txt').read_text().strip())/'raw-episode-provenance.json'
        registry = ROOT/'runs/official-raw-pool-registry.json'
        registry_lock = (ROOT/'runs/official-raw-pool-registry.lock').open('a')
        with registry_lock:
            fcntl.flock(registry_lock, fcntl.LOCK_EX)
            old = read(registry).get('provenance_paths', []) if registry.exists() else []
            paths = merge_pool_paths(old, [primary, dependency/'raw-episode-provenance.json', *extra])
            for path in paths:
                if not Path(path).resolve().is_relative_to(ROOT/'runs') or Path(path).name != 'raw-episode-provenance.json':
                    raise ValueError('Registry must contain only project raw-pool provenance')
            tmp = registry.with_suffix('.tmp')
            tmp.write_text(json.dumps({'scope': 'UNSPLIT official author-assisted raw sources; no per-worker count summation',
                'updated_at': datetime.now(timezone.utc).isoformat(), 'provenance_paths': paths}, indent=2))
            tmp.replace(registry)
        output = c/name
        command = [str(ROOT/'envs/momagen/bin/python'), str(script/'audit_official_pool_snapshot.py'), '--output', str(output)]
        for path in paths:
            if not Path(path).is_file():
                raise ValueError('Registered pool not finalized into provenance yet')
            command += ['--raw-provenance', path]
        for repo in ('r1_tidy_table_train_v1', 'r1_tidy_table_holdout_v1'):
            command += ['--exclude-provenance', str(ROOT/'data/pi0/lerobot/momagen'/repo/'momagen-provenance.json')]
        with (c/(name+'.log')).open('x') as log:
            subprocess.run(['taskset', '-c', '88-89', *command], env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600)
        report = read(output/'report.json')
        (ROOT/'runs/latest-official-raw-pool-audit.txt').write_text(str(output)+'\n')
        return report, output
    try:
        while True:
            check()
            current = read(dependency/'status.json')
            outcome = dependency_outcome(current)
            if outcome == 'failed':
                raise RuntimeError('Dependency failed validation/generation; no automatic bypass')
            if outcome == 'ready' and not any(process_alive(current.get(k)) for k in ('controller_pid', 'active_pid')):
                break
            time.sleep(10)
        if read(ROOT/'runs/latest-gpu1-job.json').get('campaign') != str(dependency):
            raise RuntimeError('GPU1 pointer changed owner; do not steal it')
        save(stage='cross_pool_readonly_audit')
        report, audit = register_and_audit([], 'before-next-pool-audit')
        aux_path = str(dependency/'raw-episode-provenance.json')
        aux = set()
        others = set()
        for snapshot in report['provenance_snapshots']:
            target = aux if snapshot['source'] == aux_path else others
            target.update(e['fingerprint'] for e in snapshot['snapshot']['episodes'])
        excluded = {fp for entry in report['old_split_overlap_checks'] for fp in entry['matching_raw_fingerprints']}
        fresh = len(aux-others-excluded)
        if fresh < 1:
            raise RuntimeError('Auxiliary pool adds no unique eligible episode; inspect data before scaling')
        save(eligible_global_snapshot_episodes=report['eligible_unique_episodes'],
             completed_dependency_new_eligible_episodes=fresh, combined_audit=str(audit))
        for _ in range(30):
            check()
            query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
            if not any(line.split(',')[0].strip() == gpu for line in query.splitlines()):
                break
            time.sleep(5)
        else:
            raise RuntimeError('GPU1 occupied after dependency; no kill/fallback')
        next_pool = c/'next-auxiliary-pool'
        with (c/'next-pool-controller.log').open('x') as log:
            child = subprocess.Popen(['taskset', '-c', '88-103', str(ROOT/'envs/openpi/bin/python'), '-u',
                str(script/'run_paper_generation_shards.py'), '--campaign', str(next_pool), '--gpu', '1',
                '--target', '25', '--shard-size', '25', '--seed-start', str(args.next_seed),
                '--max-shards', '2', '--shard-timeout', '21600'], env=env, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        save(stage='next_bounded_auxiliary_pool', child_pid=child.pid, next_pool=str(next_pool))
        while child.poll() is None:
            check()
            if (next_pool/'status.json').exists():
                pointer = ROOT/'runs/latest-multi-gpu-jobs.json'
                mapping = read(pointer)
                if mapping['GPU1'].get('campaign') != str(next_pool):
                    mapping['completed_gpu1_auxiliary_pool'] = {'campaign': str(dependency), 'status_file': str(dependency/'status.json')}
                    mapping['GPU1'] = {'job_pointer': str(ROOT/'runs/latest-gpu1-job.json'), 'campaign': str(next_pool),
                        'gpu_uuid': gpu, 'role': 'bounded_next_official_assisted_D0_25_pool_NOT_training'}
                    mapping['updated_at'] = datetime.now(timezone.utc).isoformat()
                    temp = pointer.with_suffix('.tmp'); temp.write_text(json.dumps(mapping, indent=2)); temp.replace(pointer)
            time.sleep(10)
        if child.returncode:
            raise RuntimeError(f'Next bounded pool exited{child.returncode}; inspect preserved artifacts')
        if dependency_outcome(read(next_pool/'status.json')) != 'ready':
            raise RuntimeError('Next pool did not pass its final gate')
        save(stage='final_cross_pool_readonly_audit')
        report, audit = register_and_audit([next_pool/'raw-episode-provenance.json'], 'after-next-pool-audit')
        save(stage='bounded_data_followup_complete', child_pid=0, combined_audit=str(audit),
             eligible_global_snapshot_episodes=report['eligible_unique_episodes'])
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            except ProcessLookupError:
                pass
        save(stage='data_followup_failed_or_interrupted', error=repr(error), child_pid=0)
        raise


if __name__ == '__main__':
    main()
