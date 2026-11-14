#!/usr/bin/env bash
# Upload lightweight source; atomically replace files so running jobs keep old inodes.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
source "${SERVER_CONFIG:-$PROJECT_ROOT/config/server.local.env}"
: "${REMOTE_ROOT:?}"
[[ $REMOTE_ROOT == /* && $REMOTE_ROOT != / ]] || { echo 'Unsafe REMOTE_ROOT' >&2; exit 2; }
# Bootstrap/update the helper via a temporary file and atomic rename.
printf -v ROOT_QUOTED '%q' "$REMOTE_ROOT"
"$PROJECT_ROOT/scripts/remote.sh" bash -c \
    "mkdir -p $ROOT_QUOTED/tools; cat > $ROOT_QUOTED/tools/deploy_code_archive.py.new; mv $ROOT_QUOTED/tools/deploy_code_archive.py.new $ROOT_QUOTED/tools/deploy_code_archive.py" \
    < "$PROJECT_ROOT/scripts/deploy_code_archive.py"
git ls-files --cached --others --exclude-standard -z |
    tar --null -T - -czf - |
    "$PROJECT_ROOT/scripts/remote.sh" python3 "$REMOTE_ROOT/tools/deploy_code_archive.py" --root "$REMOTE_ROOT"
