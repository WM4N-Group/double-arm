#!/usr/bin/env bash
# SERVER ONLY. One bounded pick_cup generation attempt; never delete an existing run.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
if [[ ${ACCEPT_NVIDIA_EULA:-NO} != YES || ${ACCEPT_BEHAVIOR_DATASET_TERMS:-NO} != YES ]]; then
    echo 'STOP: NVIDIA EULA and BEHAVIOR asset terms must be explicitly accepted first.' >&2
    exit 3
fi
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
ASSIGNED_GPU=$(python "$SCRIPT_DIR/resource_allocation.py" --root "$DOUBLE_ARM_ROOT" --requested "${CUDA_VISIBLE_DEVICES:-}")
export CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU"
RUN_DIR=$(mktemp -d "$DOUBLE_ARM_ROOT/runs/pick_cup-smoke-$(date +%Y%m%d-%H%M%S)-XXXXXX")
printf 'Run directory: %s\n' "$RUN_DIR"
bash "$SCRIPT_DIR/ensure_runtime_files.sh"
bash "$SCRIPT_DIR/apply_upstream_patches.sh"
cd "$MOMAGEN_ROOT"
python momagen/scripts/generate_configs.py > "$RUN_DIR/config-generation.log" 2>&1
for scene in Rs_int house_single_floor; do
    src="$MOMAGEN_ROOT/momagen/scene_instances/$scene"
    dest="$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances/scenes/$scene/json"
    [[ -d $src ]] || { echo "Missing scene instances: $src" >&2; exit 2; }
    mkdir -p "$dest"
    # Preserve pre-existing files; check any conflicts rather than overwrite assets.
    for file in "$src"/*; do
        [[ -f $file ]] || continue
        target="$dest/$(basename "$file")"
        if [[ -f $target ]] && ! cmp -s "$file" "$target"; then
            echo "Scene instance conflict: $target (resolve explicitly)" >&2; exit 2
        fi
        [[ -f $target ]] || cp "$file" "$target"
    done
done
python - "$MOMAGEN_ROOT" "$RUN_DIR" <<'PY'
import json
import pathlib
import sys
root, out = map(pathlib.Path, sys.argv[1:])
source = root / 'momagen/datasets/configs/demo_src_r1_pick_cup_task_D0.json'
config = json.loads(source.read_text())
config['experiment']['generation']['guarantee'] = False
config['experiment']['generation']['num_trials'] = 1
config['experiment']['max_num_failures'] = 1
# Live attempt videos are always recorded by the upstream generator. Avoid a
# second, expensive replay; keep one camera name because render() already joins
# the third-person and ego views.
config['experiment']['render_video'] = False
config['obs']['camera_names'] = ['agentview']
config['experiment']['num_demo_to_render'] = 1
config['experiment']['num_fail_demo_to_render'] = 1
(out / 'config.json').write_text(json.dumps(config, indent=2))
PY
# Config generation/scene preparation are not task success. Record the actual run.
set +e
timeout --signal=TERM --kill-after=60s "${SMOKE_TIMEOUT_SECONDS:-1800}" \
    python -u momagen/scripts/generate_dataset.py \
    --config "$RUN_DIR/config.json" --num_demos 1 --bimanual --headless \
    --folder "$RUN_DIR/generated" --seed "${SMOKE_SEED:-0}" \
    2>&1 | tee "$RUN_DIR/generation.log"
STATUS=${PIPESTATUS[0]}
set -e
[[ $STATUS == 0 ]] || { echo "Generation exited $STATUS; inspect $RUN_DIR/generation.log" >&2; exit "$STATUS"; }
python - "$RUN_DIR" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
successes = []
for path in root.rglob('*.json'):
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError):
        continue
    if isinstance(data, dict) and data.get('num_success', 0) > 0:
        successes.append(str(path))
hdf5 = [str(p) for p in root.rglob('*.hdf5') if p.stat().st_size > 0]
report = {'successful_generation_stats': successes, 'hdf5_files': hdf5,
          'policy_training': 'not_run', 'run_dir': str(root)}
(root / 'smoke-result.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if not successes or not hdf5:
    sys.exit('No confirmed successful demonstration; review logs, do not mark the task as passed.')
PY
# A passed generation is not yet a deliverable video: validate and export it too.
python -u "$SCRIPT_DIR/export_demo_review.py" --run-dir "$RUN_DIR" > "$RUN_DIR/review-export.log" 2>&1
python - "$RUN_DIR" <<'PY_REVIEW'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
report = json.loads((root / 'smoke-result.json').read_text())
report['review_dir'] = str(root / 'review')
report['review_video'] = str(root / 'review/demo.mp4')
(root / 'smoke-result.json').write_text(json.dumps(report, indent=2))
print('REVIEW_VIDEO_READY', root / 'review/demo.mp4', flush=True)
PY_REVIEW
