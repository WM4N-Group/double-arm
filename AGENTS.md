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
- As of 2026-11-16 10:15 CST, V2 formal 30k training and final checkpoint29999 adapter contracts are complete. Its first physical rollout ABORTED on the unchanged root-planarity safety check (`Robot root is not approximately planar`); this is NOT task success. The first exception handler did not retain a trace, so exact episode metrics are unknown; preserve its video and empty episode record honestly.
- A separate same-checkpoint physical diagnostic now runs under `pi0-v2-physical-telemetry-recheck/`, with completed-transition preservation and robot-root-pose diagnostics only. Do not weaken the .98 planarity threshold or add odometry/object poses to VLA inputs. Aborted episodes cannot pass acceptance. The original 12-hour budget still ends 2026-11-16 11:35:29 CST; this diagnostic is capped at one hour from 10:15 CST, not a fresh training budget. Inspect its separate status alongside the original failed campaign result.

## Reset-observation blocker (2026-11-16 11:29 CST)

- V2 telemetry recheck FINISHED and FAILED after440 completed steps: base path5.0543m, no cup grasp/lift/transport, zero assisted constraints/object pose writes. Root tilt11.594deg barely crossed unchanged11.478deg planarity threshold; do not call this proof of complete robot overturn. Full failure review exists separately from the first trace-less abort.
- **Confirmed initialization observation mismatch**: a snapshot-only run through the same one-load/three-render initialization has EXACTLY the source21D proprioception, but all three RGBs see stairs/door rather than the source countertop. With identical Gaussian noise, changing only RGB moves the first absolute-base XY target from about(4.95,1.52) to(5.33,-0.55), ~2m from current(5,1.5). See `v2-paired-camera-audit/report.json` and source/live comparison image. This is a reproduced input failure, not a successful rollout or proof that fixing it solves the entire task.
- `--reset-settle-steps` is a new **opt-in, UNVALIDATED** physics/Fabric synchronization candidate (default0, max20) in evaluation/orchestration. It does not copy author's table-teleport workaround. Validate true RGB synchronization before trying the same trained checkpoint closed-loop; NEVER feed stored source RGB to the live policy as a fix.
- All current simulator and CPU model jobs exited by11:29; no training was restarted. The original experiment window ends2026-11-16 11:35:29 CST; do not create a fresh12-hour window from a generic heartbeat. Preserve checkpoints/data/reviews and request a renewed computation window if needed for the next cold-start audit/evaluation. GPU1 remains the sole authorized compute GPU even in a renewed window.


## Current compute authorization (renewed 2026-11-16 14:13 CST)

- The user explicitly said: “这张卡就你用，不用等授权，预算全开放”. **GPU1 compute time is now open-ended; do not keep asking for a new window.** This supersedes the previous per-window approval requirement. GPU0 is still reserved and NEVER authorized for this project.
- Preserve the original completed campaign's historical11:35 deadline; do not rewrite its timestamps. New work has a separate workflow pointer `runs/latest-pi0-resumed-workflow.txt` and uses the existing genuine V2 checkpoint, not a fresh training claim. Allocation metadata records the renewed authorization; per-job watchdogs still bound hangs, not overall authorized compute time.
- New priority: a single-load, no-policy snapshot series after0/1/2/5/20 physics steps, with zero explicit task-object pose writes/assisted constraints. Only after actual source/live RGB and robot-alignment integration checks pass may the same checkpoint run physical closed-loop. Stored source RGB is for offline audit ONLY, never substituted for real live policy images.
- `run_pi0_reset_sync_workflow.py` continues from the snapshot audit into the same-checkpoint physical evaluation automatically if the conservative reset RGB gate passes; otherwise it stops with an explicit audit failure. The reset gate is NOT task acceptance. Preserve real physical task gates and manual video review.
- 2026-11-16 14:24: the live reset sync SERIES actually PASSED: 0-step RGB MAE~88/83/103, 1-step~8.7/10.3/9.7,20-step~2.8/3.3/3.3; robot root/joints differ by<1e-6. Manual contact-sheet inspection confirms stairs->correct countertop. Single load, no policy/env.step/teacher/assistance/object pose setters. The earliest conservative pass selects1 physics refresh.
- Continued workflow `/mnt/yuanyue/double-arm/runs/pi0-reset-sync-20261116-141502`, controllerPID104120, has started same-checkpoint `pi0-synced-physical-evaluation/` with explicit `--reset-settle-steps 1`; at14:24 CPU policy was loading. Verify its ACTUAL standalone initial RGB and final physical result; do not assume snapshot-series pass implies task success. This is not a new training run. See `docs/pi0-reset-observation-sync-20261116.md`.
