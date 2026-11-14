#!/usr/bin/env bash
# Source on the SERVER before running any simulation or training.
export DOUBLE_ARM_ROOT=${DOUBLE_ARM_ROOT:-/mnt/yuanyue/double-arm}
export MOMAGEN_ROOT="$DOUBLE_ARM_ROOT/src/MoMaGen"
export VIRTUAL_ENV="$DOUBLE_ARM_ROOT/envs/momagen"
export PATH="$VIRTUAL_ENV/bin:$DOUBLE_ARM_ROOT/tools/bin:$PATH"
export PIP_CACHE_DIR="$DOUBLE_ARM_ROOT/cache/pip"
export UV_CACHE_DIR="$DOUBLE_ARM_ROOT/cache/uv"
export XDG_CACHE_HOME="$DOUBLE_ARM_ROOT/cache/xdg"
export HF_HOME="$DOUBLE_ARM_ROOT/cache/huggingface"
export TORCH_HOME="$DOUBLE_ARM_ROOT/cache/torch"
export TMPDIR="$DOUBLE_ARM_ROOT/cache/tmp"
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.ustc.edu.cn/pypi/web/simple}
export UV_LINK_MODE=copy
export WANDB_MODE=${WANDB_MODE:-offline}
# Headless mode; do NOT set an EULA acceptance flag here automatically.
export OMNIGIBSON_HEADLESS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$DOUBLE_ARM_ROOT/runs" "$DOUBLE_ARM_ROOT/logs"
