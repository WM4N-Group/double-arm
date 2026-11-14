#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
printf '=== GPU ===\n'; nvidia-smi
printf '\n=== Storage ===\n'; df -h / "$DOUBLE_ARM_ROOT"
printf '\n=== Python ===\n'; "$VIRTUAL_ENV/bin/python" --version
printf '\n=== Revisions ===\n'
for dir in "$MOMAGEN_ROOT" "$MOMAGEN_ROOT/BEHAVIOR-1K" "$MOMAGEN_ROOT/robomimic"; do
    printf '%s: ' "$dir"; git -C "$dir" rev-parse HEAD
done
printf '\n=== Installed distributions (does not start simulator) ===\n'
python - <<'PY'
from importlib import metadata
for name in ('momagen', 'omnigibson', 'robomimic', 'torch', 'isaacsim', 'isaacsim-kernel'):
    try:
        print(f'{name}: {metadata.version(name)}')
    except metadata.PackageNotFoundError:
        print(f'{name}: NOT INSTALLED')
PY
printf '\n=== Vulkan ===\n'
if command -v vulkaninfo >/dev/null; then
    vulkaninfo --summary
else
    echo 'vulkaninfo not installed'
fi
