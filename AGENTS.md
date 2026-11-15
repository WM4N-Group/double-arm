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

## User-requested server pause (2026-11-15)

- The user explicitly requested release of server resources for higher-priority tasks. Do not restart this project's simulation, generation, training, evaluation, or audit jobs until the user explicitly resumes them.
- A generic automation heartbeat is not permission to resume. The `double-arm-46207` automation is paused.
- Remote `runs/USER_PAUSED.json` and campaign `user-paused.json` record the pause. Preserve saved datasets, checkpoints, videos, and logs. After explicit user resumption, archive the pause marker before starting a new bounded campaign.
