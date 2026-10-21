#!/usr/bin/env bash
# Enhance a folder of low-quality images with the released SemiET checkpoint.
# Usage: bash scripts/test_semiet.sh [TEST_DIR] [OUTPUT_DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}

DATA_ROOT=${DATA_ROOT:-data}
EXP_ROOT=${EXP_ROOT:-experiments}
NAME=${NAME:-semiet}
GPU=${GPU:-0}
PORT=${PORT:-10099}
BATCH=${BATCH:-8}

TEST_DIR=${1:-${DATA_ROOT}/eyeq/test/degrade_good}
OUTPUT_DIR=${2:-${DATA_ROOT}/eyeq/test/degrade_${NAME}_enhance}

"${PYTHON}" main.py --test \
  --size 512 --init_method "tcp://localhost:${PORT}" \
  --gpu "${GPU}" --batch_size "${BATCH}" --generator reformer \
  --name "${NAME}" \
  --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" \
  --test_dir "${TEST_DIR}" --output_dir "${OUTPUT_DIR}"
