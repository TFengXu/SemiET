#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
DATA_ROOT=${DATA_ROOT:-data}
VARIANTS=(abl_full abl_no_gan abl_no_icc abl_no_idt abl_supervised_only abl_mse_lambda_1 abl_mse_lambda_10)
for v in "${VARIANTS[@]}"; do
  echo ">>> evaluating ${v}"
  "${PYTHON}" tools/evaluate.py --test_dir "${DATA_ROOT}/eyeq/test/degrade_${v}_enhance" \
    --gt_dir "${DATA_ROOT}/eyeq/test/crop_good" --mask_dir "${DATA_ROOT}/eyeq/test/mask_good"
done
"${PYTHON}" tools/evaluate_lpips.py --gt_dir "${DATA_ROOT}/eyeq/test/crop_good" \
  --mask_dir "${DATA_ROOT}/eyeq/test/mask_good" --base_test_dir "${DATA_ROOT}/eyeq/test" \
  --variants "${VARIANTS[@]}"
