#!/usr/bin/env bash
# SERVER ONLY. Build exactly the cuRobo revision required by the pinned OmniGibson.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
source "$SCRIPT_DIR/../config/upstream.env"
REPO="$DOUBLE_ARM_ROOT/src/curobo"
[[ $(git -C "$REPO" rev-parse HEAD) == "$CUROBO_REV" ]] || { echo 'cuRobo source is not at the pinned revision' >&2; exit 2; }
OMPL_WHEEL="$DOUBLE_ARM_ROOT/cache/motion-wheels/ompl-1.6.0-cp310-cp310-manylinux_2_28_x86_64.whl"
mkdir -p "$(dirname "$OMPL_WHEEL")"
if ! python -c 'import sys,zipfile; sys.exit(0 if zipfile.is_zipfile(sys.argv[1]) else 1)' "$OMPL_WHEEL"; then
    CURL_PROXY=()
    if [[ -n ${MOTION_DOWNLOAD_PROXY:-} ]]; then
        CURL_PROXY=(--proxy "$MOTION_DOWNLOAD_PROXY")
    fi
    curl "${CURL_PROXY[@]}" -fLsS --retry 3 --retry-all-errors --connect-timeout 25 --max-time 300 --continue-at - \
        -o "$OMPL_WHEEL.part" https://storage.googleapis.com/gibson_scenes/ompl-1.6.0-cp310-cp310-manylinux_2_28_x86_64.whl
    python -c 'import sys,zipfile; assert zipfile.is_zipfile(sys.argv[1]), "Invalid OMPL wheel"' "$OMPL_WHEEL.part"
    mv "$OMPL_WHEEL.part" "$OMPL_WHEEL"
fi
exec 9>"$DOUBLE_ARM_ROOT/cache/python-install.lock"
flock -w 1800 9 || { echo 'Dependency installer lock timed out' >&2; exit 2; }
export MAX_JOBS=${MAX_JOBS:-4}
export TORCH_CUDA_ARCH_LIST=8.9
uv pip install --python "$VIRTUAL_ENV/bin/python" -c "$SCRIPT_DIR/../config/constraints.txt" \
    setuptools_scm wheel ninja yourdfpy warp-lang scikit-image
# Avoid rebuilding CUDA extensions on every pipeline restart. Reuse only a
# clean, pinned checkout already installed in this exact Python environment.
if python - "$REPO" <<'PY_CACHE'
import importlib.metadata as md
import json
from pathlib import Path
import subprocess, sys
repo = Path(sys.argv[1]).resolve()
try:
    dist = md.distribution('nvidia-curobo')
    direct = json.loads(dist.read_text('direct_url.json'))
    assert direct['url'] == repo.as_uri()
    assert md.version('ompl') == '1.6.0'
    assert md.version('torch') == '2.6.0'
    assert not subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no']).strip()
    assert len(list((repo / 'src/curobo/curobolib').glob('*.so'))) >= 5
except Exception:
    raise SystemExit(1)
PY_CACHE
then
    echo 'Reusing installed CUDA extensions from clean pinned source.'
else
    # No build isolation: reuse the existing CUDA 12.4 / Torch 2.6 toolchain.
    uv pip install --python "$VIRTUAL_ENV/bin/python" --no-build-isolation \
        -c "$SCRIPT_DIR/../config/constraints.txt" -e "$REPO" "$OMPL_WHEEL"
fi
python - <<'PY'
from curobo.wrap.reacher.motion_gen import MotionGen
from ompl import base, geometric
print('MOTION_PLANNING_IMPORTS_PASSED')
PY
