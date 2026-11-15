#!/usr/bin/env bash
# SERVER ONLY: fit a multimodal action BC alternative while data generation runs.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/server_env.sh"
CAMPAIGN=${1:?Usage: run_secondary_policy.sh /absolute/campaign}
[[ $CAMPAIGN == "$DOUBLE_ARM_ROOT/runs/"* ]] || exit 2
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
DEADLINE=$((SECONDS+7200))
while [[ ! -f "$CAMPAIGN/training-snapshot.hdf5" ]]; do
 [[ $SECONDS -lt $DEADLINE ]] || exit 124
 sleep 15
done
while true; do
 FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
 [[ $FREE -ge 1500 ]] && break
 [[ $SECONDS -lt $DEADLINE ]] || exit 124
 sleep 15
done
python -u "$SCRIPT_DIR/train_state_policy.py" --dataset "$CAMPAIGN/training-snapshot.hdf5" \
 --output "$CAMPAIGN/state-bc-gmm" --val-demo demo_0 --steps 15000 --seed 43 --mixtures 3
