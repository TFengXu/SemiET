#!/usr/bin/env bash
# Train SemiET (Restormer backbone, supervised + adversarial + contrastive).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}

DATA_ROOT=${DATA_ROOT:-data}
EXP_ROOT=${EXP_ROOT:-experiments}
NAME=${NAME:-semiet}
GPU=${GPU:-0,1,2,3}
PORT=${PORT:-10098}

"${PYTHON}" main.py \
  --size 512 --crop_size 256 --init_method "tcp://localhost:${PORT}" \
  --gpu "${GPU}" --batch_size 8 --lr 5e-5 --epochs 200 \
  --generator reformer \
  --lambda_rec 100.0 --lambda_is 0.0 --lambda_idt 1.0 --lambda_icc 1.0 --lambda_gan 1.0 \
  --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" --name "${NAME}"
