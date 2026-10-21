#!/usr/bin/env bash
# Supervised-only Restormer variant.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}

DATA_ROOT=${DATA_ROOT:-data}
EXP_ROOT=${EXP_ROOT:-experiments}
NAME=${NAME:-restormer_only}
GPU=${GPU:-0,1,2,3}
PORT=${PORT:-10099}

"${PYTHON}" main.py \
  --size 512 --crop_size 256 --init_method "tcp://localhost:${PORT}" \
  --gpu "${GPU}" --batch_size 32 --lr 1e-4 --epochs 200 \
  --generator reformer --lambda_rec 100.0 --lambda_is 0.0 --lambda_icc 0.0 --lambda_idt 0.0 \
  --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" --name "${NAME}"
