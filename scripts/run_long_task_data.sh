#!/usr/bin/env bash
# SERVER ONLY. Bounded attempts; successful data is never silently overwritten.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
if [[ -f "$SCRIPT_DIR/../config/licenses.local.env" ]]; then
    set -a; source "$SCRIPT_DIR/../config/licenses.local.env"; set +a
fi
[[ ${ACCEPT_NVIDIA_EULA:-NO} == YES && ${ACCEPT_BEHAVIOR_DATASET_TERMS:-NO} == YES ]] || exit 3
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TASK=${TASK:-dishes_away}
[[ $TASK == dishes_away || $TASK == bringing_water ]] || exit 2
RUN_DIR=$(mktemp -d "$DOUBLE_ARM_ROOT/runs/${TASK}-data-$(date +%Y%m%d-%H%M%S)-XXXXXX")
printf 'Run directory: %s\n' "$RUN_DIR"
echo "$RUN_DIR" > "$DOUBLE_ARM_ROOT/runs/latest-long-task-data.txt"
bash "$SCRIPT_DIR/ensure_runtime_files.sh"
cd "$MOMAGEN_ROOT"
python momagen/scripts/generate_configs.py > "$RUN_DIR/config-generation.log" 2>&1
python - "$MOMAGEN_ROOT" "$RUN_DIR" "$TASK" <<'PY'
import json,pathlib,sys
root,out,task=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
p=root/f'momagen/datasets/configs/demo_src_r1_{task}_task_D0.json'
c=json.loads(p.read_text())
c['experiment']['generation']['guarantee']=False
c['experiment']['generation']['keep_failed']=False
c['experiment']['max_num_failures']=30
c['experiment']['render_video']=False
c['experiment']['num_demo_to_render']=0
c['experiment']['num_fail_demo_to_render']=0
c['obs']['camera_names']=['agentview']
(out/'config.json').write_text(json.dumps(c,indent=2))
PY
set +e
timeout --signal=TERM --kill-after=90s "${DATA_TIMEOUT_SECONDS:-10800}" \
 python -u "$SCRIPT_DIR/generate_state_demos.py" --config "$RUN_DIR/config.json" \
 --num_demos "${DATA_ATTEMPTS:-8}" --bimanual --headless \
 --folder "$RUN_DIR/generated" --seed "${DATA_SEED:-10}" 2>&1 | tee "$RUN_DIR/generation.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '{"generator_exit_status":%d,"policy_training":"not_run"}\n' "$STATUS" > "$RUN_DIR/process-result.json"
exit "$STATUS"
