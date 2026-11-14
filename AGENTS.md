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
