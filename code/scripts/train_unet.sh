#!/usr/bin/env bash
# U-Net + saliency-guided variant.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}

DATA_ROOT=${DATA_ROOT:-data}
EXP_ROOT=${EXP_ROOT:-experiments}
NAME=${NAME:-saliency_guide_512}
GPU=${GPU:-0,1,2,3}
PORT=${PORT:-10098}

"${PYTHON}" main.py \
  --size 512 --init_method "tcp://localhost:${PORT}" \
  --gpu "${GPU}" --batch_size 64 --lr 1e-4 --epochs 200 --n_downs 4 \
  --generator u-net-structure --lambda_rec 1.0 --lambda_saliency_mse 1.0 \
  --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" --name "${NAME}"
