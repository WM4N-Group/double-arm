#!/usr/bin/env bash
# SERVER ONLY: bounded diagnostic, with automatic scheduler resume on every exit.
set -euo pipefail
S=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$S/server_env.sh"
C=${1:?campaign}; CONTROLLER=${2:?controller pid}; PRIOR_EVALUATOR=${3:?evaluator pid}
[[ $C == "$DOUBLE_ARM_ROOT/runs/"* && $CONTROLLER =~ ^[0-9]+$ && $PRIOR_EVALUATOR =~ ^[0-9]+$ ]] || exit 2
exec 9>"$C/memory-probe.lock"
flock -n 9 || exit 2
cleanup() {
    status=$?
    kill -CONT "$CONTROLLER" 2>/dev/null || true
    python - "$C" "$status" <<'PY'
import pathlib,json,sys,time
path=pathlib.Path(sys.argv[1])/'memory-probe-status.json'
try:report=json.loads(path.read_text())
except (OSError,ValueError):report={}
if report.get('stage')!='completed':
    report.update(stage='probe_failed',exit_status=int(sys.argv[2]),finished_at=time.time())
    path.write_text(json.dumps(report,indent=2))
PY
}
trap cleanup EXIT
kill -STOP "$CONTROLLER"
python - "$C" "$CONTROLLER" "$PRIOR_EVALUATOR" <<'PY'
import pathlib,json,sys,time,os
c=pathlib.Path(sys.argv[1]);pid=int(sys.argv[3]);deadline=time.monotonic()+3600
(c/'memory-probe-status.json').write_text(json.dumps({'stage':'waiting_for_gpu0_evaluator','paused_controller_pid':int(sys.argv[2]),'prior_evaluator_pid':pid,'started_at':time.time()},indent=2))
while pathlib.Path(f'/proc/{pid}/cmdline').exists():
    try:command=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:break
    if b'evaluate_state_policy.py' not in command:break
    if time.monotonic()>deadline:raise TimeoutError('Prior evaluator exceeded probe wait limit')
    time.sleep(10)
PY
python - "$C" <<'PY'
import pathlib,json,sys,time
path=pathlib.Path(sys.argv[1])/'memory-probe-status.json'
report=json.loads(path.read_text());report.update(stage='evaluating_memory_policy',updated_at=time.time())
path.write_text(json.dumps(report,indent=2))
PY
EVAL_GPU=0 EVAL_DEMOS='demo_1 demo_0' EVAL_MAX_STEPS=3500 EVAL_TIMEOUT_SECONDS=7200 \
 bash "$S/run_policy_evaluation.sh" "$C/state-memory-world-goal"
python - "$C" <<'PY'
import pathlib,json,sys,time
c=pathlib.Path(sys.argv[1]);p=c/'state-memory-world-goal';r=json.loads((p/'evaluation-process-result.json').read_text())
(c/'memory-probe-status.json').write_text(json.dumps({'stage':'completed','policy_run':str(p),'algorithm':'nearest_state_memory_BC','not_a_neural_policy':True,'result':r,'finished_at':time.time()},indent=2))
PY
