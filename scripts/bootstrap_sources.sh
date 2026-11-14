#!/usr/bin/env bash
# SERVER ONLY. Download pinned source trees without keeping any on the workstation.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
source "$SCRIPT_DIR/../config/upstream.env"
mkdir -p "$DOUBLE_ARM_ROOT/src"
export GIT_LFS_SKIP_SMUDGE=1
checkout_revision() {
    local url=$1 rev=$2 path=$3 current
    if [[ ! -e "$path/.git" ]]; then
        if [[ -d "$path" && -n $(ls -A "$path") ]]; then
            echo "Refusing to replace non-Git directory: $path" >&2; return 2
        fi
        git init "$path"
        git -C "$path" remote add origin "$url"
    fi
    current=$(git -C "$path" rev-parse --verify HEAD 2>/dev/null || true)
    if [[ $current == "$rev" ]]; then
        printf 'Already pinned: %s %s\n' "$path" "$rev"
        return
    fi
    [[ -z $(git -C "$path" status --porcelain --untracked-files=no) ]] || {
        echo "Refusing to overwrite source changes in $path" >&2; return 2;
    }
    git -C "$path" config remote.origin.promisor true
    git -C "$path" config remote.origin.partialclonefilter blob:none
    if [[ $path == "$MOMAGEN_ROOT/BEHAVIOR-1K" ]]; then
        # Skip unrelated documentation/media; retain all runtime packages.
        git -C "$path" sparse-checkout init --cone
        git -C "$path" sparse-checkout set OmniGibson/omnigibson bddl/bddl joylo/gello joylo/configs joylo/assets joylo/experiments joylo/scripts
    fi
    if [[ $path == "$MOMAGEN_ROOT/robomimic" ]]; then
        git -C "$path" sparse-checkout init --cone
        git -C "$path" sparse-checkout set robomimic
    fi
    git -C "$path" fetch --filter=blob:none --depth 1 origin "$rev"
    git -C "$path" checkout --detach "$rev"
}
checkout_revision "$MOMAGEN_URL" "$MOMAGEN_REV" "$MOMAGEN_ROOT"
checkout_revision https://github.com/StanfordVL/BEHAVIOR-1K.git "$BEHAVIOR_REV" "$MOMAGEN_ROOT/BEHAVIOR-1K"
checkout_revision https://github.com/ChengshuLi/robomimic.git "$ROBOMIMIC_REV" "$MOMAGEN_ROOT/robomimic"

checkout_revision "$CUROBO_URL" "$CUROBO_REV" "$DOUBLE_ARM_ROOT/src/curobo"
