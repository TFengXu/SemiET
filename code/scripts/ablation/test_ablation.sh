#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_ROOT=${DATA_ROOT:-data}; EXP_ROOT=${EXP_ROOT:-experiments}; GPU=${GPU:-0}
VARIANTS=(abl_full abl_no_gan abl_no_icc abl_no_idt abl_supervised_only abl_mse_lambda_1 abl_mse_lambda_10)
for v in "${VARIANTS[@]}"; do
  echo ">>> testing ${v}"
  "${PYTHON}" main.py --test --size 512 --init_method tcp://localhost:10130 \
    --gpu "${GPU}" --batch_size 8 --generator reformer \
    --name "ablation/${v}" \
    --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" \
    --test_dir "${DATA_ROOT}/eyeq/test/degrade_good" \
    --output_dir "${DATA_ROOT}/eyeq/test/degrade_${v}_enhance"
done
