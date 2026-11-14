#!/usr/bin/env bash
# Source on the SERVER before running any simulation or training.
export DOUBLE_ARM_ROOT=${DOUBLE_ARM_ROOT:-/mnt/yuanyue/double-arm}
if [[ ! -d $DOUBLE_ARM_ROOT ]]; then
    echo "Remote persistent root not found: $DOUBLE_ARM_ROOT. Run this script on the server, not locally." >&2
    return 2 2>/dev/null || exit 2
fi
export MOMAGEN_ROOT="$DOUBLE_ARM_ROOT/src/MoMaGen"
export VIRTUAL_ENV="$DOUBLE_ARM_ROOT/envs/momagen"
export PATH="$VIRTUAL_ENV/bin:$DOUBLE_ARM_ROOT/tools/bin:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export OMNIGIBSON_DATA_PATH="$DOUBLE_ARM_ROOT/data"
export OMNI_KIT_ALLOW_ROOT=1
export PIP_CACHE_DIR="$DOUBLE_ARM_ROOT/cache/pip"
export UV_CACHE_DIR="$DOUBLE_ARM_ROOT/cache/uv"
export XDG_CACHE_HOME="$DOUBLE_ARM_ROOT/cache/xdg"
export HF_HOME="$DOUBLE_ARM_ROOT/cache/huggingface"
export TORCH_HOME="$DOUBLE_ARM_ROOT/cache/torch"
export TMPDIR="$DOUBLE_ARM_ROOT/cache/tmp"
export UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX:-https://mirrors.ustc.edu.cn/pypi/web/simple}
export UV_LINK_MODE=copy
# Avoid overwhelming mirrors and the network filesystem with default concurrency.
export UV_CONCURRENT_DOWNLOADS=${UV_CONCURRENT_DOWNLOADS:-4}
export UV_CONCURRENT_BUILDS=${UV_CONCURRENT_BUILDS:-2}
export UV_CONCURRENT_INSTALLS=${UV_CONCURRENT_INSTALLS:-4}
export UV_HTTP_RETRIES=${UV_HTTP_RETRIES:-5}
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-120}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
export WANDB_MODE=${WANDB_MODE:-offline}
# Headless mode; do NOT set an EULA acceptance flag here automatically.
export OMNIGIBSON_HEADLESS=1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/tmp/double-arm-runtime-$(id -u)}
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p "$OMNIGIBSON_DATA_PATH" "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$DOUBLE_ARM_ROOT/runs" "$DOUBLE_ARM_ROOT/logs"
