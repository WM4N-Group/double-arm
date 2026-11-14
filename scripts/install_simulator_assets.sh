#!/usr/bin/env bash
# SERVER ONLY. Explicit license gate and resumable, pinned stage entry points.
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
case ${1:-all} in
    simulator) exec bash "$SCRIPT_DIR/install_isaac.sh" ;;
    motion) exec bash "$SCRIPT_DIR/install_motion_planning.sh" ;;
    assets) exec python -u "$SCRIPT_DIR/prepare_asset_bundles.py" --manifest "$SCRIPT_DIR/../config/assets.json" ;;
    all) exec bash "$SCRIPT_DIR/run_server_pipeline.sh" ;;
    *) echo 'Usage: install_simulator_assets.sh [all|simulator|motion|assets]' >&2; exit 2 ;;
esac
