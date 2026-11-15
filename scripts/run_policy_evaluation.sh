#!/usr/bin/env bash
# SERVER ONLY: finite teacher-free policy rollout, live status and review export.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
POLICY_RUN=${1:?Usage: run_policy_evaluation.sh /absolute/policy/run}
[[ $POLICY_RUN == "$DOUBLE_ARM_ROOT/runs/"* ]] || exit 2
[[ ! -e "$POLICY_RUN/evaluation" ]] || { echo 'Evaluation output exists; preserve/rename it explicitly before retrying.' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
DATASET=$(python - "$POLICY_RUN/manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))['dataset'])
PY
)
python - "$POLICY_RUN" <<'PY'
import pathlib,json,sys,time
p=pathlib.Path(sys.argv[1]);r={'stage':'evaluating','training':'completed','evaluation':'running','started_at':time.time()}
(p/'evaluation-process-result.json').write_text(json.dumps(r,indent=2))
PY
DEMO_ARGS=()
for demo in ${EVAL_DEMOS:-demo_1 demo_0}; do
 [[ $demo =~ ^demo_[0-9]+$ ]] || { echo 'Invalid demo selector' >&2; exit 2; }
 DEMO_ARGS+=(--demo "$demo")
done
set +e
timeout --signal=TERM --kill-after=90s "${EVAL_TIMEOUT_SECONDS:-7200}" \
 python -u "$SCRIPT_DIR/evaluate_state_policy.py" --dataset "$DATASET" \
 --checkpoint "$POLICY_RUN/${CHECKPOINT_NAME:-best.pt}" --output "$POLICY_RUN/evaluation" \
 "${DEMO_ARGS[@]}" --camera-layout "${EVAL_CAMERA_LAYOUT:-global_ego}" --grasping-mode "${EVAL_GRASPING_MODE:-physical}" --max-steps "${EVAL_MAX_STEPS:-4000}" 2>&1 | tee "$POLICY_RUN/evaluation.log"
STATUS=${PIPESTATUS[0]}
set -e
python - "$POLICY_RUN" "$STATUS" <<'PY'
import pathlib,json,sys,time
p=pathlib.Path(sys.argv[1]);status=int(sys.argv[2]);e=p/'evaluation/result.json'
r=json.loads(e.read_text()) if e.exists() else {}
episodes=r.get('episodes',[])
report={'stage':'evaluated' if status==0 else 'evaluation_failed','training':'completed',
        'evaluation':'completed' if status==0 else 'failed','exit_status':status,'finished_at':time.time(),
        'episodes':episodes,'any_sustained_success':any(x.get('sustained_success',False) for x in episodes),
        'heldout_sustained_success':any(x.get('sustained_success',False) for x in episodes if x.get('split')=='validation')}
(p/'evaluation-process-result.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
PY
if [[ $STATUS == 0 && ! -e "$POLICY_RUN/review" ]]; then
 python "$SCRIPT_DIR/export_policy_review.py" --policy-run "$POLICY_RUN" --output "$POLICY_RUN/review"
fi
exit "$STATUS"
