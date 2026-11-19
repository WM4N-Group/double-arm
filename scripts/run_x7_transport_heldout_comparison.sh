#!/usr/bin/env bash
# Match the heldout candidate against an unlearned transport prior. No new fit.
set -euo pipefail
R=/mnt/yuanyue/double-arm
PY="$R/envs/x7-gpu-transport/bin/python"
P="$R/runs/x7-anchored-turning-bc-20261118-222957/pilot"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD/code"
start=$SECONDS
until [[ -f validation/dev-gate.json ]]; do
    [[ ! -f logs/validation.exit ]] || exit 41
    ((SECONDS-start<3600)) || exit 42
    sleep 5
done
if ! "$PY" -B -c 'import json,sys;x=json.load(open("validation/dev-gate.json"));sys.exit(0 if x["all_12_candidate_physical_pass"] else 1)'; then
    echo 'Matched heldout not run: dev gate denied'; exit 0
fi
run_worker() {
    local i="$1"
    nice -n 12 taskset -c "$((i+24))" timeout 25m "$PY" -uB -c '
from pathlib import Path
from types import SimpleNamespace
import sys
from evaluate_x7_rgb_transport import evaluate
p=Path(sys.argv[1]);i=int(sys.argv[2]);evaluate(SimpleNamespace(pilot=p,dataset=Path("dataset"),training=Path("training"),
    out=Path("validation"),split="heldout",mode="nominal",worker=i,workers=4,case=None))' "$P" "$i" > "logs/heldout-nominal-$i.log" 2>&1
}
pids=();pass=1
for i in {0..3}; do run_worker "$i" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid" || pass=0; done
[[ "$pass" == 1 ]] || exit 43
start=$SECONDS
until [[ -f validation/heldout-report.json ]]; do
    [[ ! -f logs/validation.exit ]] || exit 44
    ((SECONDS-start<2100)) || exit 45
    sleep 5
done
"$PY" -B - <<'PY'
from pathlib import Path
import json
from evaluate_x7_rgb_transport import digest
p=Path('validation');group=json.loads((p/'heldout-report.json').read_text()); paired=[]
for r in group['records']['candidate']:
    case=r['case']; folder=p/'heldout'/'nominal'/case
    b=json.loads((folder/'report.json').read_text())
    if (b['layout_sha256']!=r['layout_sha256'] or b['checkpoint_sha256']!=r['checkpoint_sha256']
        or b['mode']!='nominal' or b.get('actual_inference_calls',0)!=0 or
        b['physical_execution'] and digest(folder/'trace.npz')!=b['trace_sha256']):
        raise ValueError(f'Invalid matched comparator {case}')
    paired.append(dict(case=case,candidate_pass=r['passed'],nominal_pass=b['passed'],
        candidate_min_station_m=r.get('minimum_robot_station_m'),nominal_min_station_m=b.get('minimum_robot_station_m'),
        candidate_calls=r.get('actual_inference_calls'),candidate_trace=r.get('trace_sha256'),nominal_trace=b.get('trace_sha256')))
result=dict(case_count=len(paired),candidate_pass=sum(r['candidate_pass'] for r in paired),
    nominal_pass=sum(r['nominal_pass'] for r in paired),records=paired,checkpoint_sha256=group['checkpoint_sha256'],
    learned_necessity_claimed=False)
with (p/'heldout-comparison.json').open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k] for k in ('case_count','candidate_pass','nominal_pass')}))
PY
