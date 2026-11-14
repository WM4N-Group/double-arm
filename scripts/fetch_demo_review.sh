#!/usr/bin/env bash
# Fetch just the three small review artifacts, not assets or training data.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "${SERVER_CONFIG:-$PROJECT_ROOT/config/server.local.env}"
REMOTE_REVIEW=${1:?Usage: fetch_demo_review.sh REMOTE_REVIEW_DIR [LOCAL_FOLDER_NAME]}
[[ $REMOTE_REVIEW == "$REMOTE_ROOT/runs/"* && $REMOTE_REVIEW != *'..'* ]] || { echo 'Review must be under the remote runs directory' >&2; exit 2; }
NAME=${2:-$(basename "$(dirname "$REMOTE_REVIEW")")}
[[ $NAME =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Unsafe local output name' >&2; exit 2; }
DEST="$PROJECT_ROOT/outputs/demo-review/$NAME"
mkdir -p "$DEST"
"$SCRIPT_DIR/remote.sh" tar -C "$REMOTE_REVIEW" -czf - demo.mp4 contact-sheet.jpg review.json |
    tar --no-same-owner -xzf - -C "$DEST"
printf 'Review artifacts: %s\n' "$DEST"
