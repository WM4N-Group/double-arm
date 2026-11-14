"""Shared privileged state schema. Never uses time, demo phase or future state."""
import numpy as np


def as_numpy(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def is_small(value):
    shape = getattr(value, 'shape', ())
    return len(shape) <= 2 and int(np.prod(shape)) <= 2048


def add_unique_task_poses(env, obs):
    """Upstream object::plate loses instance suffix; retain all three plates."""
    import omnigibson.utils.transform_utils as T
    from omnigibson.systems.system_base import BaseSystem
    base_pose = env.robot.get_position_orientation()
    for name, obj in env.env.task.object_scope.items():
        if isinstance(obj.unwrapped, BaseSystem) or 'agent' in name:
            continue
        pos, quat = T.relative_pose_transform(*obj.get_position_orientation(), *base_pose)
        obs['task::' + name] = np.concatenate([as_numpy(pos), as_numpy(quat)])
    return obs


def feature_keys(obs):
    keys = ['prop_eef_basepose', 'grasp_left', 'grasp_right']
    keys += sorted(k for k in obs if k.startswith('task::'))
    return keys


def vectorize(obs, keys):
    return np.concatenate([as_numpy(obs[k]).reshape(-1) for k in keys])
