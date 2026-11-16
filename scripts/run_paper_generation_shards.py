#!/usr/bin/env python3
"""SERVER CPU controller: GPU0 official assisted Tidy Table/D0 data shards.

Does NOT train a policy, modify current v3 data, convert labels, select a split,
or claim learned/physical success. Existing GPU1 workflow is left independent.
Uses existing explicit license file, never creates acceptance flags itself.
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
from resource_allocation import resolve_gpu

ROOT = Path('/mnt/yuanyue/double-arm')


def read(path):
    return json.loads(Path(path).read_text())


def generation_run(log, root=ROOT):
    if not log.exists():
        return None
    with log.open() as stream:
        for _ in range(12):
            line = stream.readline()
            if line.startswith('RUN_DIR='):
                path = Path(line.strip().split('=', 1)[1])
                if not path.resolve().is_relative_to(root / 'runs'):
                    raise ValueError('Generator returned an out-of-project run')
                return path
    return None


def unique_new_episodes(episodes, seen):
    """Pure helper: don't count duplicates within/across finalized shards."""
    result = []
    for episode in episodes:
        fingerprint = episode['fingerprint']
        if fingerprint not in seen:
            seen.add(fingerprint)
            result.append(episode)
    return result


def inspect_finalized_shard(run, required):
    # All imports here are CPU-only; no simulator or policy model in controller.
    import h5py
    from convert_momagen_pi0 import inspect_episode
    verdict = read(run / 'generation-result.json')
    manifest = read(run / 'manifest.json')
    if (verdict.get('state') != 'passed' or verdict.get('generator_exit_status') != 0
        or verdict.get('verified_successful_episodes', 0) < required
        or manifest.get('mode') != 'official_upstream_entrypoint_unmodified_config'
        or manifest.get('task') != 'tidy_table' or manifest.get('randomization') != 'D0'
        or manifest.get('generation_guarantee') is not True):
        raise ValueError('Shard did not pass official finalized-success gate')
    dataset = Path(verdict['dataset'])
    if not dataset.resolve().is_relative_to(run.resolve()):
        raise ValueError('Unexpected shard dataset path')
    records = []
    with h5py.File(dataset, 'r') as file:
        env = json.loads(file['data'].attrs['env_args'])
        if env['env_name'] != 'r1_tidy_table_D0':
            raise ValueError('Wrong task in generated data')
        grasp = env['env_kwargs']['robots'][0].get('grasping_mode', 'unspecified')
        if grasp != 'assisted':
            raise ValueError('Unexpected source grasp configuration; preserve explicit provenance')
        for name in sorted(file['data']):
            if not name.startswith('demo_'):
                continue
            cameras, actions, state, fingerprint = inspect_episode(file['data'][name])
            if any(file['data'][name]['obs'][key].shape[1:3] != (256, 256) for key in cameras.values()):
                raise ValueError('Expected real native 256px three-camera records')
            records.append({'source': str(dataset), 'source_demo': name,
                            'fingerprint': fingerprint, 'frames': len(actions),
                            'cameras': cameras, 'source_grasping_mode': grasp,
                            'seed': manifest['seed'], 'source_sha256': manifest['source_sha256']})
    if len(records) < required:
        raise ValueError('Not enough structurally valid three-camera episodes')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--gpu', default='0', choices=['0'])
    parser.add_argument('--target', type=int, default=1000)
    parser.add_argument('--shard-size', type=int, default=25)
    parser.add_argument('--seed-start', type=int, default=10000)
    parser.add_argument('--max-shards', type=int, default=80)
    parser.add_argument('--shard-timeout', type=int, default=21600)
    args = parser.parse_args()
    if (not 1 <= args.target <= 1000 or not 1 <= args.shard_size <= 50
        or not 1 <= args.max_shards <= 100 or not 600 <= args.shard_timeout <= 43200
        or not 0 <= args.seed_start <= 1000000):
        raise ValueError('Invalid bounded generation plan')
    c = args.campaign
    if not c.resolve().is_relative_to(ROOT / 'runs') or c.is_symlink():
        raise ValueError('Remote new campaign under runs/ required')
    gpu = resolve_gpu(ROOT, args.gpu)
    c.mkdir(exist_ok=False)
    script = Path(__file__).resolve().parent
    lock = (ROOT / 'runs' / f'gpu-{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'stage': 'starting', 'controller_pid': os.getpid(),
             'started_at': datetime.now(timezone.utc).isoformat(),
             'gpu_uuid': gpu, 'max_gpu_count': 1, 'target_unique_demos': args.target,
             'shard_size': args.shard_size, 'source_grasping_mode': 'assisted',
             'policy_training': 'not_run', 'learned_policy_task_success': False,
             'paper_reproduction_complete': False, 'validation_split_created': False,
             'existing_coverage_workflow_unchanged': True,
             'verified_unique_demos': 0, 'verified_frames': 0, 'shards': []}
    child = None
    seen = set()
    episodes = []
    def save(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temp = c / 'status.tmp'
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(c / 'status.json')
    def check():
        if (ROOT / 'runs/USER_PAUSED.json').exists():
            raise RuntimeError('User paused project')
        if resolve_gpu(ROOT, args.gpu) != gpu:
            raise RuntimeError('Explicit GPU0 authorization changed')
    def interrupted(signum, frame):
        raise InterruptedError(f'Generation controller signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    (ROOT / 'runs/latest-paper-generation-campaign.txt').write_text(str(c) + '\n')
    pointer = {'controller_pid': os.getpid(), 'campaign': str(c),
               'status_file': str(c / 'status.json'), 'gpu_uuid': gpu, 'max_gpu_count': 1,
               'role': 'official_assisted_paper_data_generation_not_policy_training'}
    (ROOT / 'runs/latest-gpu0-job.json').write_text(json.dumps(pointer, indent=2))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'JAX_PLATFORMS': 'cpu',
           'DOUBLE_ARM_ROOT': str(ROOT), 'TASK': 'tidy_table',
           'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
           'GENERATION_TIMEOUT_SECONDS': str(args.shard_timeout)}
    try:
        for index in range(args.max_shards):
            check()
            # Never steal another process, even on a now-authorized card.
            query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                             '--format=csv,noheader'], text=True)
            if any(row.split(',')[0].strip() == gpu for row in query.splitlines()):
                raise RuntimeError('GPU0 occupied by another process; no kill or fallback')
            required = min(args.shard_size, args.target - len(episodes))
            seed = args.seed_start + index
            log = c / f'shard-{index:03d}-seed-{seed}.log'
            with log.open('x') as handle:
                child = subprocess.Popen([
                    'timeout', '--signal=TERM', '--kill-after=120s', str(args.shard_timeout + 300),
                    'taskset', '-c', '0-15', 'bash', str(script / 'run_official_generation.sh')],
                    stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    env={**env, 'TARGET_SUCCESSES': str(required), 'GENERATION_SEED': str(seed)},
                    start_new_session=True)
            save(stage='official_shard_generation', active_pid=child.pid, active_log=str(log),
                 active_shard=index, active_seed=seed, current_generation_run=None)
            while child.poll() is None:
                check()
                run = generation_run(log)
                if run is not None and state.get('current_generation_run') != str(run):
                    save(current_generation_run=str(run))
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'Official shard exited {child.returncode}; inspect preserved logs/data')
            run = generation_run(log)
            if run is None:
                raise ValueError('No official run path from launcher')
            save(stage='validating_finalized_shard', active_pid=0)
            inspected = inspect_finalized_shard(run, required)
            fresh = unique_new_episodes(inspected, seen)
            episodes.extend(fresh)
            state['shards'].append({'run': str(run), 'seed': seed, 'requested': required,
                                    'verified': len(inspected), 'new_unique': len(fresh)})
            (c / 'raw-episode-provenance.json.tmp').write_text(json.dumps({
                'scope': 'Raw official author-assisted source pool; NOT train/validation split or converted labels',
                'episodes': episodes, 'fingerprint_scope': 'native action bytes + original initial proprio bytes',
                'existing_datasets_modified': False}, indent=2))
            (c / 'raw-episode-provenance.json.tmp').replace(c / 'raw-episode-provenance.json')
            save(verified_unique_demos=len(episodes), verified_frames=sum(item['frames'] for item in episodes))
            if len(episodes) >= args.target:
                save(stage='raw_official_pool_target_complete_alignment_and_split_pending', active_pid=0)
                return 0
        raise RuntimeError('Shard-count watchdog reached before unique target; preserve partial pool')
    except BaseException as error:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass
        save(stage='generation_failed_or_interrupted', error=repr(error), active_pid=0)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
