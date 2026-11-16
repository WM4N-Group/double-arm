#!/usr/bin/env bash
# SERVER ONLY: original upstream entrypoint/config, no observation monkeypatch.
# guarantee=True counts successful demos. Outer timeout bounds failed retries.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
if [[ -f "$SCRIPT_DIR/../config/licenses.local.env" ]]; then
    set -a; source "$SCRIPT_DIR/../config/licenses.local.env"; set +a
fi
[[ ${ACCEPT_NVIDIA_EULA:-NO} == YES && ${ACCEPT_BEHAVIOR_DATASET_TERMS:-NO} == YES ]] || exit 3
source "$SCRIPT_DIR/../config/upstream.env"
for spec in "$MOMAGEN_ROOT:$MOMAGEN_REV" "$MOMAGEN_ROOT/BEHAVIOR-1K:$BEHAVIOR_REV" "$MOMAGEN_ROOT/robomimic:$ROBOMIMIC_REV"; do
    directory=${spec%:*}; expected=${spec##*:}
    [[ $(git -C "$directory" rev-parse HEAD) == "$expected" ]] || { echo "Revision mismatch: $directory" >&2; exit 2; }
done
TASK=${TASK:-tidy_table}
case "$TASK" in pick_cup|tidy_table|dishes_away|clean_pan) ;; *) echo 'Unknown official core task' >&2; exit 2;; esac
TARGET_SUCCESSES=${TARGET_SUCCESSES:-1}
[[ $TARGET_SUCCESSES =~ ^[1-9][0-9]*$ ]] || exit 2
ASSIGNED_GPU=$(python "$SCRIPT_DIR/resource_allocation.py" --root "$DOUBLE_ARM_ROOT" --requested "${CUDA_VISIBLE_DEVICES:-}")
export CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU"
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
RUN_DIR=$(mktemp -d "$DOUBLE_ARM_ROOT/runs/${TASK}-official-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "RUN_DIR=$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$DOUBLE_ARM_ROOT/runs/latest-official-generation.txt"
# Serialize shared config/scene preparation across independent GPU workers.
# The simulator uses its private copied config after this lock is released.
exec 9> "$DOUBLE_ARM_ROOT/cache/official-generation-prepare.lock"
flock -x 9
bash "$SCRIPT_DIR/ensure_runtime_files.sh"
cd "$MOMAGEN_ROOT"
python momagen/scripts/generate_configs.py > "$RUN_DIR/config-generation.log" 2>&1
# Use the exact task instance supplied by the pinned repository; reject conflicts.
for scene in Rs_int house_single_floor; do
    target="$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances/scenes/$scene/json"
    mkdir -p "$target"
    for file in "$MOMAGEN_ROOT/momagen/scene_instances/$scene/"*; do
        [[ -f $file ]] || continue
        dest="$target/$(basename "$file")"
        [[ ! -f $dest ]] || cmp -s "$file" "$dest" || { echo "Scene conflict: $dest" >&2; exit 2; }
        [[ -f $dest ]] || cp "$file" "$dest"
    done
done
SOURCE_CONFIG="$MOMAGEN_ROOT/momagen/datasets/configs/demo_src_r1_${TASK}_task_D0.json"
cp "$SOURCE_CONFIG" "$RUN_DIR/config.json"
python - "$RUN_DIR" "$MOMAGEN_ROOT" "$TASK" "$TARGET_SUCCESSES" "${GENERATION_SEED:-0}" <<'PY'
import hashlib,json,pathlib,subprocess,sys
run,root=map(pathlib.Path,sys.argv[1:3]);task,target,seed=sys.argv[3:]
c=json.loads((run/'config.json').read_text())
assert c['experiment']['generation']['guarantee'] is True
source=pathlib.Path(c['experiment']['source']['dataset_path'])
sha=hashlib.sha256(source.read_bytes()).hexdigest()
report={'mode':'official_upstream_entrypoint_unmodified_config','task':task,'randomization':'D0',
        'target_successes':int(target),'seed':int(seed),'source_dataset':str(source),'source_sha256':sha,
        'generation_guarantee':True,'training':'not_started',
        'upstream_revisions':{str(p.relative_to(root)) if p!=root else '.':subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip() for p in [root,root/'BEHAVIOR-1K',root/'robomimic']}}
(run/'manifest.json').write_text(json.dumps(report,indent=2))
PY
flock -u 9
exec 9>&-
set +e
timeout --signal=TERM --kill-after=90s "${GENERATION_TIMEOUT_SECONDS:-3600}" \
    python -u momagen/scripts/generate_dataset.py --config "$RUN_DIR/config.json" \
    --num_demos "$TARGET_SUCCESSES" --bimanual --headless \
    --folder "$RUN_DIR/generated" --seed "${GENERATION_SEED:-0}" \
    2>&1 | tee "$RUN_DIR/generation.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '{"generator_exit_status":%d,"policy_training":"not_run"}\n' "$STATUS" > "$RUN_DIR/process-result.json"
python "$SCRIPT_DIR/check_generation_result.py" --run-dir "$RUN_DIR" --exit-status "$STATUS" --required "$TARGET_SUCCESSES"
