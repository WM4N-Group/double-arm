#!/usr/bin/env bash
# Upload tracked + non-ignored source files, not data, assets or user PDFs.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
source "${SERVER_CONFIG:-$PROJECT_ROOT/config/server.local.env}"
: "${REMOTE_ROOT:?}"
[[ $REMOTE_ROOT == /* && $REMOTE_ROOT != / ]] || { echo 'Unsafe REMOTE_ROOT' >&2; exit 2; }
printf -v ROOT_QUOTED '%q' "$REMOTE_ROOT"
git ls-files --cached --others --exclude-standard -z |
    tar --null -T - -czf - |
    "$PROJECT_ROOT/scripts/remote.sh" bash -c "mkdir -p $ROOT_QUOTED/code && tar -xzf - -C $ROOT_QUOTED/code"
