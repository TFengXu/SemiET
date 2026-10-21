# -*- coding: utf-8 -*-
"""
Aggregate ablation experiment results into a LaTeX-ready table.
Computes PSNR, SSIM, LPIPS for all ablation variants and prints a summary table.

Run AFTER test_abl_all.sh completes (enhanced images must exist).
Run from the repository root
Usage: python tools/aggregate_ablation.py
"""

import os
import sys
import glob
import json
import numpy as np
from multiprocessing import Pool
from matplotlib.pyplot import imread
import cv2
import math

BASE = os.environ.get('SEMIET_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GT_DIR = os.path.join(BASE, 'data/eyeq/test/crop_good')
MASK_DIR = os.path.join(BASE, 'data/eyeq/test/mask_good')

VARIANTS = [
    ('abl_full',              'Full Model',           '$\\lambda_{rec}=100$, all losses'),
    ('abl_no_gan',            '$-$GAN',               '$\\lambda_{gan}=0$'),
    ('abl_no_icc',            '$-$ICC',               '$\\lambda_{icc}=0$ (no unpaired)'),
    ('abl_no_idt',            '$-$IDT',               '$\\lambda_{idt}=0$'),
    ('abl_supervised_only',   'MSE-only',             '$\\lambda_{gan}=\\lambda_{icc}=\\lambda_{idt}=0$'),
    ('abl_mse_lambda_1',      '$\\lambda_{rec}=1$',   'MSE weight sensitivity'),
    ('abl_mse_lambda_10',     '$\\lambda_{rec}=10$',  'MSE weight sensitivity'),
]


def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def load_gt():
    """Load all GT images and masks."""
    gt_dict = {}
    mask_dict = {}
    for gt_path in sorted(glob.glob(os.path.join(GT_DIR, '*'))):
        _, name = os.path.split(gt_path)
        gt_dict[name] = imread(gt_path).copy()
    for mask_path in sorted(glob.glob(os.path.join(MASK_DIR, '*'))):
        _, name = os.path.split(mask_path)
        mask = imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) if mask.ndim == 2 else mask
        mask_dict[name] = mask
    return gt_dict, mask_dict


def evaluate_variant(variant_name):
    """Compute PSNR/SSIM for one ablation variant."""
    test_dir = os.path.join(BASE, f'data/eyeq/test/degrade_{variant_name}_enhance')
    if not os.path.exists(test_dir):
        return None, f'test dir not found'

    gt_dict, mask_dict = load_gt()
    image_list = sorted(glob.glob(os.path.join(test_dir, '*')))

    psnr_vals, ssim_vals = [], []
    skipped = 0
    for image_path in image_list:
        try:
            _, name = os.path.split(image_path)
            # Match GT name (handle naming convention)
            image_id_split = name.split('_')[:-1]
            gt_name = '_'.join(image_id_split) + '.jpeg'
            if gt_name not in gt_dict:
                gt_name = '_'.join(image_id_split) + '.png'
            if gt_name not in gt_dict:
                skipped += 1
                continue

            enhanced = imread(image_path).copy()
            gt = gt_dict[gt_name]

            if gt_name in mask_dict:
                mask = mask_dict[gt_name] / 255.0
                enhanced = (enhanced * mask).astype(np.uint8)
                gt = (gt * mask).astype(np.uint8)

            psnr_vals.append(psnr(enhanced, gt))
            ssim_vals.append(ssim(enhanced, gt))
        except Exception:
            skipped += 1

    if not psnr_vals:
        return None, f'no matched images (skipped={skipped})'

    return {
        'psnr': (np.mean(psnr_vals), np.std(psnr_vals)),
        'ssim': (np.mean(ssim_vals), np.std(ssim_vals)),
        'count': len(psnr_vals),
        'skipped': skipped,
    }, None


def add_lpips(results):
    """Add LPIPS results from evaluate_lpips.py output if available."""
    try:
        import lpips
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device)
        loss_fn.eval()

        gt_dict, mask_dict = load_gt()

        for variant_key, _, _ in VARIANTS:
            test_dir = os.path.join(BASE, f'data/eyeq/test/degrade_{variant_key}_enhance')
            if not os.path.exists(test_dir) or variant_key not in results:
                continue

            image_list = sorted(glob.glob(os.path.join(test_dir, '*')))
            lpips_vals = []
            for image_path in image_list:
                try:
                    _, name = os.path.split(image_path)
                    image_id_split = name.split('_')[:-1]
                    gt_name = '_'.join(image_id_split) + '.jpeg'
                    if gt_name not in gt_dict:
                        gt_name = '_'.join(image_id_split) + '.png'
                    if gt_name not in gt_dict:
                        continue

                    enhanced = imread(image_path).copy()
                    gt = gt_dict[gt_name]

                    enhanced_t = torch.from_numpy(enhanced).float().permute(2, 0, 1) / 255.0
                    gt_t = torch.from_numpy(gt).float().permute(2, 0, 1) / 255.0

                    if gt_name in mask_dict:
                        mask_t = torch.from_numpy(mask_dict[gt_name] / 255.0).float().permute(2, 0, 1)
                        enhanced_t = enhanced_t * mask_t
                        gt_t = gt_t * mask_t

                    enhanced_t = enhanced_t * 2.0 - 1.0
                    gt_t = gt_t * 2.0 - 1.0

                    with torch.no_grad():
                        d = loss_fn(enhanced_t.unsqueeze(0).to(device),
                                    gt_t.unsqueeze(0).to(device))
                        lpips_vals.append(d.item())
                except Exception:
                    pass

            if lpips_vals:
                results[variant_key]['lpips'] = (np.mean(lpips_vals), np.std(lpips_vals))
    except ImportError:
        pass


def print_latex_table(results):
    """Print ablation results as LaTeX table."""
    print("\n" + "=" * 80)
    print("Ablation Study Results")
    print("=" * 80)

    # Header
    header = f"{'Variant':<25} {'PSNR':>12} {'SSIM':>10} {'LPIPS':>10}"
    print(header)
    print("-" * 60)
    for variant_key, display_name, desc in VARIANTS:
        if variant_key in results:
            r = results[variant_key]
            psnr_str = f"{r['psnr'][0]:.2f}±{r['psnr'][1]:.2f}"
            ssim_str = f"{r['ssim'][0]:.4f}±{r['ssim'][1]:.4f}"
            lpips_str = f"{r.get('lpips', (0, 0))[0]:.4f}" if 'lpips' in r else 'N/A'
            print(f"{display_name:<25} {psnr_str:>12} {ssim_str:>10} {lpips_str:>10}")

    # LaTeX table
    print("\n\n% === LaTeX Table ===")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation study of loss components and $\\lambda_{rec}$ sensitivity.}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Variant & PSNR (dB) & SSIM & LPIPS \\\\")
    print("\\hline")

    # Group 1: Loss ablation
    loss_variants = [v for v in VARIANTS if v[0] in ['abl_full', 'abl_no_gan', 'abl_no_icc', 'abl_no_idt', 'abl_supervised_only']]
    for variant_key, display_name, _ in loss_variants:
        if variant_key in results:
            r = results[variant_key]
            psnr_s = f"${r['psnr'][0]:.2f} \\pm {r['psnr'][1]:.2f}$"
            ssim_s = f"${r['ssim'][0]:.4f} \\pm {r['ssim'][1]:.4f}$"
            lpips_s = f"${r.get('lpips', (0, 0))[0]:.4f}$" if 'lpips' in r else '--'
            print(f"{display_name} & {psnr_s} & {ssim_s} & {lpips_s} \\\\")
    print("\\hline")

    # Group 2: MSE sensitivity
    print("\\multicolumn{4}{c}{\\textit{$\\lambda_{rec}$ Sensitivity}} \\\\")
    print("\\hline")
    mse_variants = [v for v in VARIANTS if 'mse_lambda' in v[0]] + [v for v in VARIANTS if v[0] == 'abl_full']
    for variant_key, display_name, _ in mse_variants:
        if variant_key in results:
            r = results[variant_key]
            psnr_s = f"${r['psnr'][0]:.2f} \\pm {r['psnr'][1]:.2f}$"
            ssim_s = f"${r['ssim'][0]:.4f} \\pm {r['ssim'][1]:.4f}$"
            lpips_s = f"${r.get('lpips', (0, 0))[0]:.4f}$" if 'lpips' in r else '--'
            print(f"{display_name} & {psnr_s} & {ssim_s} & {lpips_s} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == '__main__':
    print('Loading GT images...')
    gt_dict, mask_dict = load_gt()
    print(f'Loaded {len(gt_dict)} GT, {len(mask_dict)} masks')

    results = {}
    for variant_key, display_name, desc in VARIANTS:
        print(f'\n>>> Evaluating {display_name} ({variant_key})...')
        res, err = evaluate_variant(variant_key)
        if res is None:
            print(f'  [SKIP] {err}')
        else:
            results[variant_key] = res
            print(f'  PSNR: {res["psnr"][0]:.2f}±{res["psnr"][1]:.2f}, '
                  f'SSIM: {res["ssim"][0]:.4f}±{res["ssim"][1]:.4f} '
                  f'(N={res["count"]}, skipped={res["skipped"]})')

    print('\nAdding LPIPS...')
    add_lpips(results)

    print_latex_table(results)

    # Save to JSON for later use
    import json
    # Convert numpy types for JSON serialization
    json_results = {}
    for k, v in results.items():
        json_results[k] = {
            'psnr_mean': float(v['psnr'][0]),
            'psnr_std': float(v['psnr'][1]),
            'ssim_mean': float(v['ssim'][0]),
            'ssim_std': float(v['ssim'][1]),
            'count': v['count'],
        }
        if 'lpips' in v:
            json_results[k]['lpips_mean'] = float(v['lpips'][0])
            json_results[k]['lpips_std'] = float(v['lpips'][1])

    json_path = os.path.join(BASE, 'ISECRET/experiments/eyeq/512/ablation_xuf/ablation_results.json')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f'\nResults saved to {json_path}')
