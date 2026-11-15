# Project workflow

- Canonical code repository: `git@github.com:WM4N-Group/double-arm.git`.
- Keep the local checkout lightweight: source, configurations, scripts, documentation only.
- Run all robotics simulation, data generation and model training on the remote server, not this workstation.
- Remote persistent project root: `/mnt/yuanyue/double-arm`.
- Put upstream repositories under `src/`, Python environments under `envs/`, assets and datasets under `data/`, generated runs under `runs/`, caches under `cache/`, and logs under `logs/` on that remote root.
- Never commit passwords, tokens, private keys, Python environments, simulator assets, datasets, checkpoints, videos, or PDFs.
- Preserve existing local reference PDFs; do not delete user files simply to save space.
- Pin the upstream MoMaGen and submodule revisions; do not mix unrelated Isaac Sim / OmniGibson versions.
- Distinguish installed dependencies, successful simulator launch, successful demo generation, and successful policy training in status reports. Do not claim success from package installation alone.
- Do not accept NVIDIA or BEHAVIOR dataset terms on the user's behalf. BEHAVIOR assets are restricted to non-commercial academic research; obtain explicit confirmation before setting acceptance flags.

## Current server allocation (resumed 2026-11-15)

- The user resumed this project with **one GPU only**. We explicitly selected physical GPU1 (second card), UUID `GPU-50e5d38d-2825-e379-f537-7f0bf4609aba`; GPU0 is reserved for higher-priority work.
- Never restore the old dual-GPU controller or launch a job on GPU0. Pin all project GPU jobs to the allowed UUID with `CUDA_VISIBLE_DEVICES`; do not expand to another card when it is idle.
- Remote `runs/RESOURCE_ALLOCATION.json` records the allocation. Old user-pause markers were archived after explicit resumption; saved data, checkpoints and logs remain intact.
- Keep automatic notifications quiet unless a meaningful new completion, failure, or required user action occurs. Do not treat a generic heartbeat as permission to expand GPU allocation.
- Grasp acceptance must distinguish assisted attachment from physical contact/friction. Original MoMaGen uses gripper friction 4.0; do not claim physical grasp results are calibrated to real hardware.
