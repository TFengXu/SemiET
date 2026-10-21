#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-data}; EXP_ROOT=${EXP_ROOT:-experiments}
NAME=${NAME:-restormer_only}; GPU=${GPU:-0}
TEST_DIR=${1:-${DATA_ROOT}/eyeq/test/degrade_good}
OUTPUT_DIR=${2:-${DATA_ROOT}/eyeq/test/degrade_${NAME}_enhance}
"${PYTHON}" main.py --test --size 512 --crop_size -1 --init_method tcp://localhost:10091 \
  --gpu "${GPU}" --batch_size 32 --generator reformer --name "${NAME}" \
  --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" \
  --test_dir "${TEST_DIR}" --output_dir "${OUTPUT_DIR}"
