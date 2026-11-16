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

## Current policy mainline (2026-11-15)

- User selected **official MoMaGen demonstration generation → pretrained π₀ (pi-zero) LoRA fine-tuning → joint navigation + manipulation**. Do not substitute our state MLP, nearest-neighbor memory, π₀.₅, or π₀-FAST and call it this result.
- Use the isolated Python3.11 `envs/openpi`; never upgrade the pinned MoMaGen/Isaac Python3.10 environment for π₀.
- Real head + both wrist RGB images, proprioception and instruction are the π₀ inputs. No simulator object poses, phase/time or future observations as inference inputs.
- Preserve author-config assisted generation provenance. Physical-only closed-loop evaluation and its friction caveat remain separate acceptance gates; do not silently relabel assisted source data as physical.
- Pi0 framework, converter and source versions are in `config/pi0.env`. All checkpoints, RGB datasets and training remain remote. Pilot data scale is not a reproduction of the paper's 1000-demo results.

## Experiment state (2026-11-16)

- First genuine pi0 LoRA run finished 30,000 updates, but the physical held-out rollout FAILED: 3,000 steps, no cup grasp, zero sustained task success. Preserve it as a baseline; the local ignored review video is explicitly marked FAILED, not a successful demo.
- Remote `runs/latest-pi0-baseline-campaign.txt` identifies that original run. An author-config assisted diagnostic is separate; its `author_config_task_success` must never be presented as physical success.
- As of 2026-11-16 08:09 CST, that assisted diagnostic also finished 3,000 steps without cup grasp or sustained success. V2 then passed its own 20-update smoke and entered real 30k training on GPU1. Do not call the failure solely a physical-grasp domain gap; the full v2 policy has not passed physical evaluation yet.
- Remote `runs/latest-pi0-campaign.txt` now points to the **absolute-base v2 candidate**. Its model configuration is `pi0_r1_tidy_table_absolute_base_lora32`, not the native-label baseline configuration. It reuses the same author RGB data, with server-side label overlays under `data/pi0/absolute-base-v2/` and explicit quantile normalization.
- V2 policy inputs remain real head + both wrist RGB, ORIGINAL 21D proprioception and instruction. Dataset index is only a supervised-label lookup and is removed before model input; inference must not open teacher-label overlays. Current robot root odometry is used only by the low-level absolute-world-target to native-motor converter, not as an extra VLA input. No object pose inputs, planner or per-step state/pose restoration.
- V2 base labels are world XY + one fixed wrapped yaw branch `[-pi, pi)`. Do not use the archived per-episode unwrapped prototype: it has inconsistent 2*pi aliases across episodes. Native motor commands remain 21D; each absolute-base target is converted and capped using training-native step bounds before ordinary simulator stepping.
- Serve v2 with `--action-representation absolute_base`; do not interpret its world targets as native deltas or use mean/std normalization. Checkpoint `assets/r1-adapter-contract.json` records representation, normalization hash and actual training source hashes, and the server rejects mismatches. Old native checkpoints without this file are supported only as native legacy runs.
- GPU training waits for the author diagnostic to release the SAME allowed GPU. Preserve the original campaign budget anchor; deadline remains approximately **2026-11-16 11:35 CST**, not a fresh 12 hours after repairs. GPU0 remains reserved regardless of idleness.
- See `docs/pi0-native-failure-and-absolute-base-v2-20261116.md` for evidence, limitations and exact experiment distinctions. Math roundtrips, data-loader tests and training loss do not imply closed-loop task success. Do not silently change core labels/adapters of an in-flight training run; use a new explicit variant for further changes.
