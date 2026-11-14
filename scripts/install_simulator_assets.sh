#!/usr/bin/env bash
# SERVER ONLY. This stage is explicitly gated on licensing confirmation.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
if [[ ${ACCEPT_NVIDIA_EULA:-NO} != YES || ${ACCEPT_BEHAVIOR_DATASET_TERMS:-NO} != YES ]]; then
    echo 'STOP: Explicit acceptance of NVIDIA EULA and BEHAVIOR dataset terms is required.' >&2
    echo 'BEHAVIOR assets are restricted to non-commercial academic research and may not be redistributed.' >&2
    echo 'See config/licenses.example.env and the pinned BEHAVIOR-1K/setup.sh.' >&2
    exit 3
fi
export OMNI_KIT_ACCEPT_EULA=YES
export PIP_CONSTRAINT="$SCRIPT_DIR/../config/constraints.txt"
# Use the installer from the pinned project; do not substitute current Isaac Sim.
cd "$MOMAGEN_ROOT/BEHAVIOR-1K"
./setup.sh --omnigibson --bddl --joylo --dataset --primitives \
    --confirm-no-conda --accept-nvidia-eula --accept-dataset-tos
