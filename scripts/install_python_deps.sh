#!/usr/bin/env bash
# SERVER ONLY. Install open-source Python dependencies, NOT Isaac Sim or datasets.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
CONSTRAINTS="$SCRIPT_DIR/../config/constraints.txt"
exec 9>"$DOUBLE_ARM_ROOT/cache/python-install.lock"
flock -n 9 || { echo "Another dependency installation is running." >&2; exit 2; }
for path in "$MOMAGEN_ROOT/setup.py" "$MOMAGEN_ROOT/BEHAVIOR-1K/OmniGibson/setup.py" "$MOMAGEN_ROOT/robomimic/setup.py"; do
    [[ -f $path ]] || { echo "Missing source: $path" >&2; exit 2; }
done
# Keep robotics packages in their own environment, never the system Python.
uv pip install --python "$VIRTUAL_ENV/bin/python" -c "$CONSTRAINTS" \
    'setuptools==79.0.1' 'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0' \
    -e "$MOMAGEN_ROOT" \
    -e "$MOMAGEN_ROOT/BEHAVIOR-1K/OmniGibson" \
    -e "$MOMAGEN_ROOT/BEHAVIOR-1K/bddl" \
    -e "$MOMAGEN_ROOT/BEHAVIOR-1K/joylo" \
    -e "$MOMAGEN_ROOT/robomimic"
uv pip check --python "$VIRTUAL_ENV/bin/python"
python -m pip freeze > "$DOUBLE_ARM_ROOT/logs/python-packages.txt"
echo 'Python dependencies installed. Isaac Sim, motion-planning extras, assets and simulator launch still need separate verification.'
