#!/usr/bin/env bash
# SERVER ONLY. Isolated Python3.11 environment; never alters MoMaGen / Isaac pins.
set -euo pipefail
S=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$S/server_env.sh"
source "$S/../config/pi0.env"
# Standalone CPython records clang in sysconfig; this server has GCC.
export CC=/usr/bin/gcc CXX=/usr/bin/g++
OPENPI="$DOUBLE_ARM_ROOT/src/$OPENPI_SOURCE_DIR"
LEROBOT="$DOUBLE_ARM_ROOT/src/$LEROBOT_SOURCE_DIR"
PY="$DOUBLE_ARM_ROOT/envs/openpi/bin/python"
[[ $(cat "$OPENPI/UPSTREAM_REVISION") == "$OPENPI_REV" ]] || exit 2
[[ $(cat "$LEROBOT/UPSTREAM_REVISION") == "$LEROBOT_REV" ]] || exit 2
[[ -x $PY ]] || { echo 'Install the isolated CPython3.11 interpreter and uv venv first';exit 2; }
# Export exact versions from the pinned upstream lock. Editable paths replace
# Git network fetches only; source revisions are checked above.
python - "$OPENPI/uv.lock" "$DOUBLE_ARM_ROOT/cache/openpi-constraints.txt" <<'PY'
import sys,pathlib,tomli
from packaging.markers import Marker,default_environment
env=default_environment();env.update(python_version='3.11',python_full_version='3.11.18',extra='')
lock=tomli.loads(pathlib.Path(sys.argv[1]).read_text());versions={}
for p in lock['package']:
    if 'registry' not in p.get('source',{}):continue
    markers=p.get('resolution-markers',[])
    if markers and not any(Marker(marker).evaluate(env) for marker in markers):continue
    name=p['name'];version=p['version']
    if name in versions and versions[name]!=version:raise ValueError(f'Ambiguous target lock version: {name}')
    versions[name]=version
pathlib.Path(sys.argv[2]).write_text('\n'.join(f'{name}=={version}' for name,version in sorted(versions.items()))+'\n')
print('Exact upstream constraints',len(versions))
PY
uv pip install --no-sources --python "$PY" --index-url "${PI0_PACKAGE_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}" \
 -c "$DOUBLE_ARM_ROOT/cache/openpi-constraints.txt" \
 -e "$OPENPI" -e "$OPENPI/packages/openpi-client" -e "$LEROBOT"
uv pip check --python "$PY"
uv pip freeze --python "$PY" > "$DOUBLE_ARM_ROOT/envs/openpi/requirements-frozen.txt"
# CPU only import audit: installation must not initialize JAX on a busy GPU.
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "$PY" - <<'PY'
import sys,jax,flax,torch,h5py
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.models.pi0_config import Pi0Config
assert sys.version_info[:2]==(3,11)
print('OPENPI_IMPORT_OK',sys.version.split()[0],jax.__version__,flax.__version__,torch.__version__)
print('PI0_CONFIG',Pi0Config())
PY
