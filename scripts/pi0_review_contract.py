"""Small pure-stdlib contracts for genuine pi0 review artifacts."""
import hashlib
import json
from pathlib import Path


def physical_task_passed(result):
    episodes=result.get('episodes',[])
    if len(episodes)!=1:return False
    episode=episodes[0]
    return (result.get('mode')=='closed_loop_neural_policy'
        and result.get('algorithm')=='official_openpi_pi0_lora32'
        and result.get('task_success') is True
        and not result.get('error')
        and episode.get('aborted') is not True
        and result.get('privileged_state') is False
        and result.get('planner_enabled') is False
        and result.get('teacher_action_reads')==0
        and result.get('initial_state_load_count')==1
        and result.get('grasping_mode')=='physical'
        and result.get('assisted_constraint_seen') is False
        and result.get('object_pose_write_count')==0
        and episode.get('initial_state_load_count')==1
        and episode.get('split')=='validation'
        and episode.get('sustained_success') is True
        and episode.get('stable_success_frames',0)>=30
        and episode.get('cup_specific_grasp_seen') is True
        and episode.get('max_cup_lift_m',0)>.03
        and episode.get('base_path_length_m',0)>.5
        and episode.get('object_pose_write_count')==0)


def fingerprint_checkpoint(checkpoint):
    """Hash inference parameters/assets, NOT optimizer tensors or filenames only.

    No data/weights are copied. Reject unfinished, linked or concurrently changing
    files; the result identifies the inference state, not the entire training run.
    """
    checkpoint=Path(checkpoint)
    if checkpoint.is_symlink() or not checkpoint.is_dir() or not checkpoint.name.isdigit():
        raise ValueError('Expected a committed numeric Orbax checkpoint directory')
    metadata=checkpoint/'_CHECKPOINT_METADATA'
    if metadata.is_symlink():raise ValueError('Symlink completion metadata')
    commit=json.loads(metadata.read_text())
    if not commit.get('commit_timestamp_nsecs'):raise ValueError('Uncommitted Orbax checkpoint')
    files=[metadata]
    for name in ('params','assets'):
        component=checkpoint/name
        if component.is_symlink() or not component.is_dir():raise ValueError(f'Invalid checkpoint component: {name}')
        for path in sorted(component.rglob('*')):
            if path.is_symlink():raise ValueError('Symlink in checkpoint inference state')
            if path.is_file():files.append(path)
        if not any(path.is_relative_to(component) for path in files):raise ValueError(f'Empty {name}')
    records=[]
    for path in files:
        if path.is_symlink():raise ValueError('Symlink metadata')
        before=path.stat();digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            raise ValueError('Checkpoint changed while hashing')
        records.append({'path':str(path.relative_to(checkpoint)),'bytes':after.st_size,'sha256':digest.hexdigest()})
    records.sort(key=lambda record:record['path'])
    canonical=json.dumps(records,sort_keys=True,separators=(',',':')).encode()
    return {'checkpoint':str(checkpoint),'step_index':int(checkpoint.name),
            'scope':'All inference params, assets and completion metadata; optimizer state excluded',
            'files':records,'total_bytes':sum(record['bytes'] for record in records),
            'sha256':hashlib.sha256(canonical).hexdigest()}
