#!/usr/bin/env bash
# Restore the one documentation-tree file that OmniGibson actually uses at runtime.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
source "$SCRIPT_DIR/../config/upstream.env"
REPO="$MOMAGEN_ROOT/BEHAVIOR-1K"
FILE=docs/assets/OmniGibson_logo.png
EXPECTED=$(git -C "$REPO" rev-parse "$BEHAVIOR_REV:$FILE")
if [[ -f "$REPO/$FILE" ]]; then
    [[ $(git -C "$REPO" hash-object "$REPO/$FILE") == "$EXPECTED" ]] || { echo 'Runtime logo differs from pinned source' >&2; exit 2; }
else
    mkdir -p "$(dirname "$REPO/$FILE")"
    git -C "$REPO" show "$BEHAVIOR_REV:$FILE" > "$REPO/$FILE.download-part"
    [[ $(git -C "$REPO" hash-object "$REPO/$FILE.download-part") == "$EXPECTED" ]] || exit 2
    mv "$REPO/$FILE.download-part" "$REPO/$FILE"
fi
printf 'Pinned runtime file verified: %s\n' "$FILE"
