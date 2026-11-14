#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
[[ ${ACCEPT_NVIDIA_EULA:-NO} == YES ]] || { echo 'NVIDIA EULA confirmation required' >&2; exit 3; }
export OMNI_KIT_ACCEPT_EULA=YES
exec 9>"$DOUBLE_ARM_ROOT/cache/python-install.lock"
flock -n 9 || { echo 'Another dependency installation is running' >&2; exit 2; }
WHEELS="$DOUBLE_ARM_ROOT/cache/isaac-4.5-wheels"
python "$SCRIPT_DIR/download_isaac_wheels.py" --setup-script "$MOMAGEN_ROOT/BEHAVIOR-1K/setup.sh" --folder "$WHEELS"
uv pip install --python "$VIRTUAL_ENV/bin/python" -c "$SCRIPT_DIR/../config/constraints.txt" "$WHEELS"/*.whl
python -c 'import isaacsim; print("ISAACSIM_IMPORT_PASSED")'
python -m pip freeze > "$DOUBLE_ARM_ROOT/logs/python-packages.txt"
