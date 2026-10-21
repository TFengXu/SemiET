#!/usr/bin/env bash
# Train one ablation variant. Usage: bash scripts/ablation/train_ablation.sh <variant>
set -euo pipefail
cd "$(dirname "$0")/../.."

VARIANT=${1:-abl_full}
DATA_ROOT=${DATA_ROOT:-data}
EXP_ROOT=${EXP_ROOT:-experiments}
GPU=${GPU:-0}
PORT=${PORT:-10120}

REC=100.0; GAN=1.0; ICC=1.0; IDT=1.0
case "${VARIANT}" in
  abl_full)            ;;
  abl_no_gan)          GAN=0.0 ;;
  abl_no_icc)          ICC=0.0 ;;
  abl_no_idt)          IDT=0.0 ;;
  abl_supervised_only) GAN=0.0; ICC=0.0; IDT=0.0 ;;
  abl_mse_lambda_1)    REC=1.0 ;;
  abl_mse_lambda_10)   REC=10.0 ;;
  *) echo "Unknown variant: ${VARIANT}"; exit 1 ;;
esac

"${PYTHON}" main.py \
  --size 512 --crop_size 256 --init_method "tcp://localhost:${PORT}" \
  --gpu "${GPU}" --batch_size 8 --lr 5e-5 --epochs 200 --generator reformer \
  --lambda_rec "${REC}" --lambda_gan "${GAN}" --lambda_icc "${ICC}" --lambda_idt "${IDT}" \
  --data_root_dir "${DATA_ROOT}" --experiment_root_dir "${EXP_ROOT}" \
  --name "ablation/${VARIANT}"
