#!/usr/bin/env bash
# Run within the immutable remote X7 run directory; touches only the X7 UUID.
set -euo pipefail
MODE=${1:?smoke or formal}
R=/mnt/yuanyue/double-arm
GPU=GPU-9eea1085-6cdd-882b-61f3-4ac2c1b10f8b
PY="$R/envs/x7-gpu-transport/bin/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
if [[ "$MODE" == smoke ]]; then
    OUT=gpu-smoke; ARGS=(--smoke --epochs 1 --batch 256)
elif [[ "$MODE" == formal ]]; then
    OUT=training; ARGS=(--epochs 60 --batch 256)
    start=$SECONDS
    while :; do
        done_count=0
        for i in {0..7}; do
            if [[ -f logs/collect-$i.exit ]]; then
                [[ $(cat "logs/collect-$i.exit") == 0 ]] || { echo "Collect worker $i failed"; exit 21; }
                ((done_count+=1))
            fi
        done
        [[ "$done_count" != 8 ]] || break
        (( SECONDS-start < 5700 )) || exit 22
        sleep 5
    done
else
    exit 2
fi
exec 9>"$R/runs/.x7-gpu-${GPU}.lock"
flock -w 720 9
[[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]] || { echo 'Selected GPU is busy; no fallback'; exit 23; }
export CUDA_VISIBLE_DEVICES="$GPU"
nice -n 10 taskset -c 16-19 "$PY" -uB code/train_x7_rgb_transport.py dataset "$OUT" "${ARGS[@]}" &
pid=$!
printf '%s\n' "$pid" > "logs/$MODE.worker.pid"
printf 'utc_time, gpu_util_percent, gpu_memory_mib, own_memory_mib\n' > "logs/$MODE-gpu.csv"
start=$SECONDS
while kill -0 "$pid" 2>/dev/null; do
    if ! own=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F, -v pid="$pid" '$1+0==pid {total+=$2} END {print total+0}'); then
        echo 'GPU memory polling failed'; kill -TERM "$pid" 2>/dev/null || true; wait "$pid" || true; exit 24
    fi
    local_gpu=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
    printf '%s, %s, %s\n' "$(date -u +%FT%TZ)" "$local_gpu" "$own" >> "logs/$MODE-gpu.csv"
    if (( own > 38146 || SECONDS-start > 4200 )); then
        echo 'X7 own process exceeded 40 GB or watchdog'; kill -TERM "$pid" 2>/dev/null || true; wait "$pid" || true; exit 25
    fi
    sleep 2
done
wait "$pid"
