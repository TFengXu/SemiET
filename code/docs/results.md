# Results and reproduction status

## Released checkpoint

`b2g_gen_best.pt` (Restormer backbone, `lambda_rec=100`) reaches the following validation
scores on the degraded EyeQ validation split:

| Split | PSNR | SSIM |
|---|---|---|
| EyeQ val (degrade_good) | 37.10 | 0.9674 |

## Paper tables

| Table | Content | Status |
|---|---|---|
| 2 | PSNR/SSIM/LPIPS, EyeQ/REFUGE/DRIVE | to be verified with the released checkpoint |
| 3 | MCF-Net quality grading | requires downstream toolkit |
| 6 | Ablation (losses, `lambda_rec`) | requires retraining |
| 4 | Vessel segmentation | requires downstream toolkit |
| 5 | OD/OC + fovea | requires downstream toolkit |
| 7 | Efficiency (FLOPs/params/time) | `tools/efficiency_compare.py` |
| 8 | DDR lesion detection | code not released |

Numbers will be filled in as each table is reproduced.
