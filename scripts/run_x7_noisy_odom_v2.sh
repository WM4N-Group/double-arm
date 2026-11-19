#!/usr/bin/env bash
# From a completed/queued noisy-odometry collection, train v2 and test only its development cases.
set -euo pipefail
R=/mnt/yuanyue/double-arm
GPU=GPU-9eea1085-6cdd-882b-61f3-4ac2c1b10f8b
P="$R/runs/x7-anchored-turning-bc-20261118-222957/pilot"
F=${1:?original fit directory}
PY="$R/envs/x7-gpu-transport/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
start=$SECONDS
until [[ -f logs/collection.exit ]]; do ((SECONDS-start<3500)) || exit 30; sleep 4; done
[[ $(cat logs/collection.exit) == 0 ]] || exit 31
exec 9>"$R/runs/.x7-gpu-${GPU}.lock"
flock -w 1200 9
start=$SECONDS
while [[ -n $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]]; do
    ((SECONDS-start<1200)) || { echo 'X7 GPU0 still occupied; no fallback'; exit 32; }
    sleep 20
done
export CUDA_VISIBLE_DEVICES="$GPU"
nice -n 11 taskset -c 16-19 "$PY" -uB code/train_x7_noisy_odom.py "$F" dataset training &
pid=$!
printf '%s\n' "$pid" > logs/train-v2.worker.pid
printf 'utc_time, gpu_util_percent, gpu_memory_mib, own_memory_mib\n' > logs/train-v2-gpu.csv
start=$SECONDS
while kill -0 "$pid" 2>/dev/null; do
    if ! own=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v pid="$pid" '$1+0==pid{m+=$2} END{print m+0}'); then
        kill -TERM "$pid" 2>/dev/null || true;wait "$pid" || true;exit 33
    fi
    g=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
    printf '%s, %s, %s\n' "$(date -u +%FT%TZ)" "$g" "$own" >> logs/train-v2-gpu.csv
    if ((own>38146 || SECONDS-start>1800)); then kill -TERM "$pid" 2>/dev/null || true;wait "$pid" || true;exit 34;fi
    sleep 2
done
wait "$pid"
flock -u 9
export CUDA_VISIBLE_DEVICES=''
mkdir -p fit-view
ln -s "$F/dataset" fit-view/dataset
ln -s "$PWD/training" fit-view/training
h=$(sha256sum training/best.pt | cut -d ' ' -f1)
taskset -c 27 "$PY" -B code/stress_x7_rgb_transport.py "$P" fit-view dev-v2 --plan
for i in {0..7}; do
    (s=0;nice -n 13 taskset -c "$((i+20))" timeout 25m "$PY" -uB code/stress_x7_rgb_transport.py "$P" fit-view dev-v2 \
        --checkpoint "$h" --candidate-only --worker "$i" > "logs/dev-v2-$i.log" 2>&1 || s=$?
        printf '%s\n' "$s" > "logs/dev-v2-$i.exit") &
done
wait
for i in {0..7}; do [[ $(cat "logs/dev-v2-$i.exit") == 0 ]] || exit 35; done
