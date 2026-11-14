#!/usr/bin/env python3
"""Server-only teacher entrypoint: state IL data with unique task object poses.

Retain segmentation for upstream visibility checks, strip it only on disk.
Teacher controls, visibility and success checks are unchanged. This does NOT
train a policy. Images are not suitable for visual IL in this state-only dataset.
"""
import functools
import os
import runpy
import faulthandler
import signal
from state_policy_features import add_unique_task_poses, is_small


def main():
    import torch
    threads=int(os.environ.get('STATE_CPU_THREADS','4'))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    faulthandler.register(signal.SIGUSR1, all_threads=False)
    print(f'STATE_WORKER_THREAD_LIMIT torch={threads} interop=1',flush=True)
    from robomimic.envs.env_omnigibson import EnvOmniGibson
    import momagen.utils.file_utils as file_utils
    original = EnvOmniGibson.get_obs_IL

    def state_obs(self, *args, **kwargs):
        obs, info = original(self, *args, **kwargs)
        obs = {k: v for k, v in obs.items() if is_small(v) or k.endswith('::seg_instance')}
        return add_unique_task_poses(self, obs), info

    original_write = file_utils.write_demo_to_hdf5

    @functools.wraps(original_write)
    def state_write(*args, **kwargs):
        observations = kwargs.get('observations')
        if observations is not None:
            kwargs['observations'] = [{k: v for k, v in o.items() if is_small(v)} for o in observations]
        return original_write(*args, **kwargs)

    EnvOmniGibson.get_obs_IL = state_obs
    file_utils.write_demo_to_hdf5 = state_write
    runpy.run_module('momagen.scripts.generate_dataset', run_name='__main__')


if __name__ == '__main__':
    main()
