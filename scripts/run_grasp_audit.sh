#!/usr/bin/env bash
# SERVER ONLY: finite diagnostic; never share GPU0 with another simulator.
set -euo pipefail
S=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$S/server_env.sh"
if [[ -f "$S/../config/licenses.local.env" ]]; then
 set -a; source "$S/../config/licenses.local.env"; set +a
fi
[[ ${ACCEPT_NVIDIA_EULA:-NO} == YES && ${ACCEPT_BEHAVIOR_DATASET_TERMS:-NO} == YES ]] || exit 3
C=${1:?campaign};CONTROLLER=${2:?controller pid};PRIOR=${3:?prior evaluator pid}
[[ $C == "$DOUBLE_ARM_ROOT/runs/"* && $CONTROLLER =~ ^[0-9]+$ && $PRIOR =~ ^[0-9]+$ ]] || exit 2
exec 9>"$C/memory-probe.lock";flock -n 9 || exit 2
trap 'kill -CONT "$CONTROLLER" 2>/dev/null || true' EXIT
kill -STOP "$CONTROLLER"
python - "$C" "$PRIOR" <<'PY'
import pathlib,json,sys,time
c=pathlib.Path(sys.argv[1]);pid=int(sys.argv[2]);end=time.monotonic()+3600
(c/'grasp-audit-status.json').write_text(json.dumps({'stage':'waiting_for_prior_evaluator','prior_pid':pid,'started_at':time.time()},indent=2))
while pathlib.Path(f'/proc/{pid}/cmdline').exists():
    try:cmd=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes()
    except FileNotFoundError:break
    if b'evaluate_state_policy.py' not in cmd:break
    if time.monotonic()>end:raise TimeoutError('GPU0 evaluator wait exceeded 1h')
    time.sleep(10)
(c/'grasp-audit-status.json').write_text(json.dumps({'stage':'running_grasp_counterfactual','started_at':time.time()},indent=2))
PY
ASSIGNED_GPU=$(python "$S/resource_allocation.py" --root "$DOUBLE_ARM_ROOT")
export CUDA_VISIBLE_DEVICES="$ASSIGNED_GPU" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_WAIT_POLICY=PASSIVE
set +e
timeout --signal=TERM --kill-after=90s 2400 python -u "$S/audit_grasp_physics.py" \
 --policy-run "$C/state-memory-world-goal" --output "$C/grasp-physics-counterfactual" --max-steps 1100
STATUS=$?
set -e
python - "$C" "$STATUS" <<'PY'
import pathlib,json,sys,time
c=pathlib.Path(sys.argv[1]);status=int(sys.argv[2]);(c/'grasp-audit-status.json').write_text(json.dumps({'stage':'completed' if status==0 else 'failed','exit_status':status,'finished_at':time.time(),'learned_policy_success_claimed':False},indent=2))
PY
exit "$STATUS"
