#!/usr/bin/env bash
# SERVER ONLY: finite deployment + first bounded MoMaGen run; not a recurring monitor.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
source "$SCRIPT_DIR/../config/licenses.local.env"
export ACCEPT_NVIDIA_EULA ACCEPT_BEHAVIOR_DATASET_TERMS
[[ $ACCEPT_NVIDIA_EULA == YES && $ACCEPT_BEHAVIOR_DATASET_TERMS == YES ]] || exit 3
export OMNI_KIT_ACCEPT_EULA=YES
# Optional server-local networking configuration; never tracked in Git.
if [[ -f "$SCRIPT_DIR/../config/runtime.local.env" ]]; then
    set -a; source "$SCRIPT_DIR/../config/runtime.local.env"; set +a
fi
exec 8>"$DOUBLE_ARM_ROOT/cache/pipeline.lock"
flock -n 8 || { echo 'A deployment pipeline is already running' >&2; exit 2; }
STAGE=initializing
status() {
    python "$SCRIPT_DIR/pipeline_status.py" --file "$DOUBLE_ARM_ROOT/logs/pipeline-status.json" \
        --stage "$STAGE" --state "$1" --exit-code "${2:-0}"
}
trap 'code=$?; trap - EXIT; if ((code != 0)); then status failed "$code"; fi; exit "$code"' EXIT
STAGE=isaac_install; status running
# Wait for an already-running wheel installer before examining the environment.
flock -w 1800 "$DOUBLE_ARM_ROOT/cache/python-install.lock" true
if ! python - "$SCRIPT_DIR" "$MOMAGEN_ROOT/BEHAVIOR-1K/setup.sh" <<'PY'
import importlib.metadata as md
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from download_isaac_wheels import package_list
for package in package_list(sys.argv[2]):
    name, expected = package.rsplit('-', 1)
    if md.version(name) != expected:
        raise SystemExit(1)
PY
then
    bash "$SCRIPT_DIR/install_isaac.sh" > "$DOUBLE_ARM_ROOT/logs/isaac-install.pipeline.log" 2>&1
fi
status passed
STAGE=isaac_headless_startup; status running
ISAAC_RESULT="$DOUBLE_ARM_ROOT/logs/isaac-smoke-result-$$.json"
timeout --signal=TERM --kill-after=60s 1800 python -u "$SCRIPT_DIR/isaac_smoke.py" --result-file "$ISAAC_RESULT" \
    > "$DOUBLE_ARM_ROOT/logs/isaac-smoke.log" 2>&1
python - "$ISAAC_RESULT" <<'PY_CHECK'
import json, sys
result = json.load(open(sys.argv[1]))
assert result.get('isaac_startup') == 'passed' and result.get('updates') == 30
PY_CHECK
status passed
STAGE=motion_planning_install; status running
bash "$SCRIPT_DIR/install_motion_planning.sh" > "$DOUBLE_ARM_ROOT/logs/motion-install.log" 2>&1
status passed
STAGE=asset_download_wait; status running
# An existing finite download may be in progress. Do not start a duplicate.
for ((attempt=0; attempt<480; attempt++)); do
    if ! pgrep -f '[p]ython -u .*/scripts/prepare_asset_bundles.py .*--download-only' >/dev/null; then
        break
    fi
    sleep 30
done
if pgrep -f '[p]ython -u .*/scripts/prepare_asset_bundles.py .*--download-only' >/dev/null; then
    echo 'Asset download wait exceeded four hours' >&2; exit 124
fi
STAGE=asset_install; status running
python -u "$SCRIPT_DIR/prepare_asset_bundles.py" --manifest "$SCRIPT_DIR/../config/assets.json" \
    > "$DOUBLE_ARM_ROOT/logs/asset-install.log" 2>&1
status passed
STAGE=pick_cup_generation; status running
bash "$SCRIPT_DIR/run_momagen_smoke.sh" > "$DOUBLE_ARM_ROOT/logs/pick-cup-smoke.log" 2>&1
status passed
STAGE=complete; status passed
