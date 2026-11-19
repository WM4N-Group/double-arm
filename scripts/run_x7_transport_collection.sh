#!/usr/bin/env bash
# Run from the remote experiment directory after creating dataset/layouts.json.
set -u
R=/mnt/yuanyue/double-arm
P="$R/runs/x7-anchored-turning-bc-20261118-222957/pilot"
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cores=(20 21 22 23 24 25 26 27)
for i in "${!cores[@]}"; do
    (
        nice -n 12 taskset -c "${cores[i]}" timeout 90m "$R/envs/x7-cpu/bin/python" -uB \
            code/collect_x7_transport_bc.py "$P" dataset --worker "$i" --workers 8 > "logs/collect-$i.log" 2>&1
        r=$?
        printf '%s\n' "$r" > "logs/collect-$i.exit"
    ) &
done
wait
