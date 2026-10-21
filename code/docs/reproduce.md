# Reproducing the paper

All commands assume the repository root as working directory and the environment from
`requirements.txt`.

## Table 1 - datasets and metrics
Descriptive; see the paper.

## Table 2 - PSNR / SSIM / LPIPS on degraded EyeQ, REFUGE and DRIVE

```bash
# 1) enhance
bash scripts/test_semiet.sh data/eyeq/test/degrade_good
# 2) PSNR / SSIM
python tools/evaluate.py --test_dir data/eyeq/test/degrade_semiet_enhance \
    --gt_dir data/eyeq/test/crop_good --mask_dir data/eyeq/test/mask_good
# 3) LPIPS
python tools/evaluate_lpips.py --gt_dir data/eyeq/test/crop_good \
    --mask_dir data/eyeq/test/mask_good --base_test_dir data/eyeq/test --variants semiet
```

For REFUGE / DRIVE replace `data/eyeq/test` with the corresponding folder.

## Table 3 - non-reference quality grading (MCF-Net)
See `downstream/mcf_net/README.md`; enhanced images are the output of the step above.

## Table 6 - ablation study

```bash
bash scripts/ablation/train_ablation.sh <variant>
bash scripts/ablation/test_ablation.sh
bash scripts/ablation/eval_ablation.sh
```

## Table 4 - vessel segmentation
See `downstream/vessel/README.md`.

## Table 5 - OD/OC segmentation and fovea detection
See `downstream/fovea/README.md`.

## Table 7 - computational cost

```bash
python tools/efficiency_compare.py
```

## Table 8 - DDR lesion detection
Code not included in this release.

## Known caveats

- The released checkpoint is the only pretrained model; all other variants must be retrained.
- Downstream tasks require the third-party toolkits listed in `downstream/`.
