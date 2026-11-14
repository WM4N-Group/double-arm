#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
for name in robomimic BEHAVIOR-1K MoMaGen; do
    folder="$SCRIPT_DIR/../patches/$name"
    [[ -d $folder ]] || continue
    repo="$MOMAGEN_ROOT/$name"
    [[ $name == MoMaGen ]] && repo="$MOMAGEN_ROOT"
    for patch in "$folder"/*.patch; do
        [[ -f $patch ]] || continue
        if git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
            printf 'Already applied: %s\n' "$patch"
        else
            git -C "$repo" apply --check "$patch"
            git -C "$repo" apply "$patch"
            printf 'Applied: %s\n' "$patch"
        fi
    done
done
