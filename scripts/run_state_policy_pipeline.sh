#!/usr/bin/env bash
# SERVER ONLY: wait for bounded generation, then train and evaluate without teacher.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
GENERATION_RUN=${1:?Usage: run_state_policy_pipeline.sh /absolute/generation/run}
[[ $GENERATION_RUN == "$DOUBLE_ARM_ROOT/runs/"* && -d $GENERATION_RUN ]] || exit 2
if [[ -f "$SCRIPT_DIR/../config/licenses.local.env" ]]; then
 set -a; source "$SCRIPT_DIR/../config/licenses.local.env"; set +a
fi
[[ ${ACCEPT_NVIDIA_EULA:-NO} == YES && ${ACCEPT_BEHAVIOR_DATASET_TERMS:-NO} == YES ]] || exit 3
export OMNI_KIT_ACCEPT_EULA=YES OMNI_KIT_ALLOW_ROOT=1
RESULT="$GENERATION_RUN/policy-pipeline-status.json"
on_exit() {
 local status=$?
 if [[ $status != 0 ]]; then
  python - "$RESULT" "$status" <<'PY_ERROR'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text()) if p.exists() else {}
r.update(status='failed',exit_status=int(sys.argv[2]),error='See policy-pipeline.log; no policy task success claimed')
p.write_text(json.dumps(r,indent=2))
PY_ERROR
 fi
}
trap on_exit EXIT
python - "$RESULT" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'status':'waiting_for_generation','training':'not_started','evaluation':'not_started'},indent=2))
PY
# Bounded waiting. A failed generator never launches training on partial data.
DEADLINE=$((SECONDS+${WAIT_SECONDS:-11500}))
while [[ ! -f "$GENERATION_RUN/process-result.json" ]]; do
 [[ $SECONDS -lt $DEADLINE ]] || { echo 'Timed out waiting for generation'; exit 124; }
 sleep 20
done
DATASET=$(python - "$GENERATION_RUN" <<'PY'
import json,pathlib,sys,h5py
root=pathlib.Path(sys.argv[1])
if json.loads((root/'process-result.json').read_text())['generator_exit_status'] != 0:
 raise SystemExit('Generation failed; training not launched')
paths=list(root.glob('generated/*/demo.hdf5'))
if len(paths)!=1: raise SystemExit(f'Expected one finalized dataset, found {len(paths)}')
with h5py.File(paths[0]) as f:
 demos=[k for k in f['data'] if k.startswith('demo_')]
 if len(demos)<3: raise SystemExit(f'Only {len(demos)} successful demos, need >=3')
print(paths[0])
PY
)
POLICY_RUN="$GENERATION_RUN/state-bc-$(date +%Y%m%d-%H%M%S)"
echo "DATASET=$DATASET"
echo "POLICY_RUN=$POLICY_RUN"
python - "$RESULT" "$DATASET" "$POLICY_RUN" <<'PY'
import json,sys,pathlib
pathlib.Path(sys.argv[1]).write_text(json.dumps({'status':'training','dataset':sys.argv[2],'policy_run':sys.argv[3],'evaluation':'not_started'},indent=2))
PY
CUDA_VISIBLE_DEVICES=1 python -u "$SCRIPT_DIR/train_state_policy.py" \
 --dataset "$DATASET" --output "$POLICY_RUN" --steps "${BC_STEPS:-15000}" \
 2>&1 | tee "$GENERATION_RUN/training.log"
python - "$RESULT" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text());r.update(status='evaluating',training='completed');p.write_text(json.dumps(r,indent=2))
PY
# Serialize simulator processes: data generator has exited before evaluation.
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=60s "${EVAL_TIMEOUT_SECONDS:-3600}" \
 python -u "$SCRIPT_DIR/evaluate_state_policy.py" --dataset "$DATASET" \
 --checkpoint "$POLICY_RUN/best.pt" --output "$POLICY_RUN/evaluation" \
 --max-steps "${EVAL_MAX_STEPS:-4000}" 2>&1 | tee "$GENERATION_RUN/evaluation.log"
python - "$RESULT" "$POLICY_RUN" <<'PY'
import json,sys,pathlib
p=pathlib.Path(sys.argv[1]);r=json.loads(p.read_text());e=json.loads((pathlib.Path(sys.argv[2])/'evaluation/result.json').read_text())
r.update(status='evaluated',training='completed',evaluation=e);p.write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
PY
