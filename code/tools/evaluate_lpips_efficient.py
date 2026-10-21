#!/usr/bin/env python3
"""
Memory-efficient LPIPS evaluation for ablation variants.
Only processes enhanced images that have GT matches (~8.5k instead of 59k).
Uses GPU batch processing for speed.
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import lpips
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Efficient LPIPS evaluation')
parser.add_argument('--gt_dir', type=str, required=True)
parser.add_argument('--mask_dir', type=str, default='')
parser.add_argument('--variants', type=str, nargs='+', required=True)
parser.add_argument('--base_test_dir', type=str,
                    default='data/eyeq/test')
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
loss_fn = lpips.LPIPS(net='alex', version='0.1').to(device)
loss_fn.eval()

# Build GT filename set (just names, no loading)
gt_names = set()
gt_paths = {}
for gt_path in sorted(glob.glob(os.path.join(args.gt_dir, '*'))):
    _, name = os.path.split(gt_path)
    gt_names.add(name)
    gt_paths[name] = gt_path
print(f'Found {len(gt_names)} GT images')

# Build mask name set
mask_paths = {}
if args.mask_dir:
    for mask_path in sorted(glob.glob(os.path.join(args.mask_dir, '*'))):
        _, name = os.path.split(mask_path)
        mask_paths[name] = mask_path
    print(f'Found {len(mask_paths)} masks')

def match_gt_name(enhanced_name, gt_set):
    """Match enhanced filename to GT filename."""
    if enhanced_name in gt_set:
        return enhanced_name
    # Strip last _XXX segment
    parts = enhanced_name.rsplit('.', 1)[0]  # remove extension
    parts = parts.split('_')[:-1]  # remove last _XXX
    for ext in ['.jpeg', '.png', '.jpg']:
        candidate = '_'.join(parts) + ext
        if candidate in gt_set:
            return candidate
    return None

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0

# Precompute matched pairs for each variant
all_pairs = {}
for variant in args.variants:
    test_dir = os.path.join(args.base_test_dir, f'degrade_{variant}_enhance')
    if not os.path.exists(test_dir):
        print(f'[SKIP] {variant}: test dir not found')
        continue

    pairs = []
    for img_path in sorted(glob.glob(os.path.join(test_dir, '*'))):
        _, ename = os.path.split(img_path)
        gt_name = match_gt_name(ename, gt_names)
        if gt_name:
            pairs.append((img_path, gt_paths[gt_name],
                         mask_paths.get(gt_name, None)))

    if pairs:
        all_pairs[variant] = pairs
        print(f'{variant}: {len(pairs)} matched pairs (from {len(glob.glob(os.path.join(test_dir, "*")))})')
    else:
        print(f'{variant}: 0 matched pairs')

print()

# Evaluate each variant with batch processing
for variant, pairs in all_pairs.items():
    lpips_values = []

    for i in tqdm(range(0, len(pairs), args.batch_size), desc=f'LPIPS {variant}'):
        batch_pairs = pairs[i:i + args.batch_size]
        batch_enhanced = []
        batch_gt = []

        for enh_path, gt_path, mask_path in batch_pairs:
            enh = load_image(enh_path)
            gt = load_image(gt_path)
            if enh is None or gt is None:
                continue

            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = mask.astype(np.float32) / 255.0
                    mask = mask[:, :, np.newaxis]
                    enh = enh * mask
                    gt = gt * mask

            batch_enhanced.append(enh)
            batch_gt.append(gt)

        if not batch_enhanced:
            continue

        enh_tensor = torch.from_numpy(np.stack(batch_enhanced)).permute(0, 3, 1, 2).to(device)
        gt_tensor = torch.from_numpy(np.stack(batch_gt)).permute(0, 3, 1, 2).to(device)

        # Normalize to [-1, 1]
        enh_tensor = enh_tensor * 2.0 - 1.0
        gt_tensor = gt_tensor * 2.0 - 1.0

        with torch.no_grad():
            d = loss_fn(enh_tensor, gt_tensor)
            lpips_values.extend(d.cpu().numpy().tolist())

    if lpips_values:
        mean_v = np.mean(lpips_values)
        std_v = np.std(lpips_values)
        print(f'{variant}: LPIPS = {mean_v:.4f} +- {std_v:.4f}  (n={len(lpips_values)})')
    else:
        print(f'{variant}: No valid LPIPS computed')
    print()
