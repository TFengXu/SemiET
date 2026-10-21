# -*- coding: utf-8 -*-
"""
Evaluate LPIPS (Learned Perceptual Image Patch Similarity) for ablation variants.
Compares enhanced images against ground truth.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import lpips
from matplotlib.pyplot import imread
from tqdm import tqdm

parser = argparse.ArgumentParser(description='LPIPS evaluation for ablation variants')
parser.add_argument('--gt_dir', type=str, required=True,
                    help='Ground truth directory (crop_good)')
parser.add_argument('--mask_dir', type=str, default='',
                    help='Mask directory (optional)')
parser.add_argument('--variants', type=str, nargs='+', required=True,
                    help='List of variant names')
parser.add_argument('--base_test_dir', type=str,
                    default='data/eyeq/test',
                    help='Base directory for enhanced images')
args = parser.parse_args()

# Initialize LPIPS model (AlexNet backbone, more widely used)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device)
loss_fn.eval()

# Load GT images
gt_list = sorted(glob.glob(os.path.join(args.gt_dir, '*')))
gt_images = {}
for gt_path in tqdm(gt_list, desc='Loading GT images'):
    _, name = os.path.split(gt_path)
    img = imread(gt_path).copy()
    gt_images[name] = img

print(f'Loaded {len(gt_images)} GT images')

# Load mask if provided
masks = {}
if args.mask_dir != '':
    mask_list = sorted(glob.glob(os.path.join(args.mask_dir, '*')))
    for mask_path in mask_list:
        _, name = os.path.split(mask_path)
        mask = imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) if mask.ndim == 2 else mask
        masks[name] = mask

# Evaluate each variant
for variant in args.variants:
    test_dir = os.path.join(args.base_test_dir, f'degrade_{variant}_enhance')
    if not os.path.exists(test_dir):
        print(f'[SKIP] {variant}: test dir not found at {test_dir}')
        continue

    image_list = sorted(glob.glob(os.path.join(test_dir, '*')))
    if len(image_list) == 0:
        print(f'[SKIP] {variant}: no images found')
        continue

    lpips_values = []
    matched = 0
    skipped = 0

    for image_path in tqdm(image_list, desc=f'LPIPS {variant}'):
        try:
            _, name = os.path.split(image_path)

            # Handle naming: enhanced images may have naming convention differences
            # Try exact match first, then fuzzy match
            if name in gt_images:
                gt_name = name
            else:
                # Try matching by removing suffix or adjusting name
                image_id_split = name.split('_')[:-1]
                gt_name = ''
                for idx, word in enumerate(image_id_split):
                    gt_name += str(word)
                    if idx != len(image_id_split) - 1:
                        gt_name += '_'
                gt_name += '.jpeg'

                if gt_name not in gt_images:
                    # Try .png
                    gt_name = gt_name.replace('.jpeg', '.png')
                if gt_name not in gt_images:
                    skipped += 1
                    continue

            # Load enhanced image
            enhanced = imread(image_path).copy()

            # Convert to tensor [0, 1] range, [-1, 1] for LPIPS
            enhanced_tensor = torch.from_numpy(enhanced).float().permute(2, 0, 1) / 255.0
            gt_tensor = torch.from_numpy(gt_images[gt_name]).float().permute(2, 0, 1) / 255.0

            # Apply mask if available
            if gt_name in masks:
                mask = masks[gt_name] / 255.0
                mask_tensor = torch.from_numpy(mask).float().permute(2, 0, 1)
                enhanced_tensor = enhanced_tensor * mask_tensor
                gt_tensor = gt_tensor * mask_tensor

            # LPIPS expects [0, 1] → normalize to [-1, 1]
            enhanced_tensor = enhanced_tensor * 2.0 - 1.0
            gt_tensor = gt_tensor * 2.0 - 1.0

            # Compute LPIPS
            with torch.no_grad():
                d = loss_fn(enhanced_tensor.unsqueeze(0).to(device),
                            gt_tensor.unsqueeze(0).to(device))
                lpips_values.append(d.item())

            matched += 1
        except Exception as e:
            skipped += 1
            continue

    if len(lpips_values) > 0:
        lpips_mean = np.mean(lpips_values)
        lpips_std = np.std(lpips_values)
        print(f'{variant}: LPIPS = {lpips_mean:.4f} +- {lpips_std:.4f}  (matched={matched}, skipped={skipped})')
    else:
        print(f'{variant}: No valid LPIPS computed (skipped={skipped})')
