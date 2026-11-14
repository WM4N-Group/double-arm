#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "${SERVER_CONFIG:-$SCRIPT_DIR/../config/server.local.env}"
printf -v ROOT_QUOTED '%q' "$REMOTE_ROOT"
exec "$SCRIPT_DIR/remote.sh" bash -c "
  printf '=== Pipeline ===\\n'; cat $ROOT_QUOTED/logs/pipeline-status.json 2>/dev/null || true
  printf '\\n=== Pipeline log ===\\n'; tail -8 $ROOT_QUOTED/logs/pipeline.log 2>/dev/null || true
  printf '\\n=== Assets ===\\n'; tail -7 $ROOT_QUOTED/logs/asset-download.log 2>/dev/null || true
  printf '\\n=== Latest simulator check ===\\n'; tail -10 $ROOT_QUOTED/logs/isaac-smoke.log 2>/dev/null || true
"
