#!/usr/bin/env bash
# Wait for the formal fit, then evaluate new layouts without taking a GPU.
set -euo pipefail
R=/mnt/yuanyue/double-arm
PY="$R/envs/x7-gpu-transport/bin/python"
P="$R/runs/x7-anchored-turning-bc-20261118-222957/pilot"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
start=$SECONDS
until [[ -f logs/formal.exit ]]; do
    ((SECONDS-start<10300)) || exit 31
    sleep 6
done
[[ $(cat logs/formal.exit) == 0 ]] || { echo 'Training failed; evaluation not run'; exit 32; }

run_shard() {
    local split="$1" mode="$2" shard="$3" core="$4" prefix="logs/$1-$2-$3"
    local rc=0
    nice -n 12 taskset -c "$core" timeout 35m "$PY" -uB code/evaluate_x7_rgb_transport.py \
      "$P" dataset training validation --split "$split" --mode "$mode" --worker "$shard" --workers 4 > "$prefix.log" 2>&1 || rc=$?
    printf '%s\n' "$rc" > "$prefix.exit"
    return "$rc"
}
wait_shards() {
    local pid passed=1
    for pid in "$@"; do wait "$pid" || passed=0; done
    [[ "$passed" == 1 ]]
}
summarize() {
    taskset -c 27 "$PY" -uB code/evaluate_x7_rgb_transport.py "$P" dataset training validation --split "$1" --summarize
}

pids=()
for i in {0..3}; do
    run_shard dev candidate "$i" "$((i+20))" & pids+=("$!")
    run_shard dev nominal "$i" "$((i+24))" & pids+=("$!")
done
wait_shards "${pids[@]}" || { echo 'One or more dev shards did not finish'; exit 33; }
summarize dev
if ! "$PY" -B -c 'import json,sys;x=json.load(open("validation/dev-gate.json"));sys.exit(0 if x["all_12_candidate_physical_pass"] else 1)'; then
    printf '%s\n' 'Not run: at least one dev layout did not pass the frozen physical gate.' > validation/heldout-not-run.txt
    exit 0
fi
pids=()
for i in {0..3}; do run_shard heldout candidate "$i" "$((i+20))" & pids+=("$!"); done
wait_shards "${pids[@]}" || { echo 'One or more heldout shards did not finish'; exit 34; }
summarize heldout
