#!/usr/bin/env bash
# ResNet baseline (importance-sampling reconstruction, no Transformer).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}

DATA_ROOT=${DATA_ROOT:-data}
EXP_ROOT=${EXP_ROOT:-experiments}
NAME=${NAME:-resnet_gan_cut_resize256}
GPU=${GPU:-0,1}
PORT=${PORT:-10098}

"${PYTHON}" main.py \
  --size 256 --init_method "tcp://localhost:${PORT}" \
  --gpu "${GPU}" --batch_size 16 --lr 1e-4 --epochs 200 \
  --generator resnet --lambda_rec 0.0 --lambda_is 1.0 \
  --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" --name "${NAME}"
