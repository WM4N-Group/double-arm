#!/usr/bin/env bash
# SERVER ONLY: headless graphics loading, build tools, and resumable downloads.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
[[ $(id -u) == 0 ]] || { echo 'Run as root on the training server' >&2; exit 2; }
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git curl aria2 ffmpeg libvulkan1 vulkan-tools \
    libegl1 libgl1 libglu1-mesa libglib2.0-0 libsm6 libxrender1 libxext6 \
    libxrandr2 libxi6 libxcursor1 libxinerama1 libxxf86vm1 libxkbcommon0 \
    libnss3 libasound2 libxt6
