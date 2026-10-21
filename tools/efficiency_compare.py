# -*- coding: utf-8 -*-
"""
Efficiency Comparison Script
Computes FLOPs, #Params, and Inference Time for all comparison methods.

Run from the repository root
Usage: python tools/efficiency_compare.py
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

# Add project paths
BASE = os.environ.get('SEMIET_ROOT', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

# Try to import fvcore for FLOPs (fallback: thop)
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False
    print('[WARN] fvcore not available, trying thop...')
    try:
        from thop import profile, clever_format
        HAS_THOP = True
    except ImportError:
        HAS_THOP = False
        print('[WARN] thop not available, FLOPs will be skipped')

# ============================================================
# Helper functions
# ============================================================

def count_parameters(model):
    """Count trainable parameters (in millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def compute_flops_fvcore(model, input_tensor):
    """Compute FLOPs using fvcore."""
    flops = FlopCountAnalysis(model, input_tensor)
    return flops.total() / 1e9  # GFLOPs


def compute_flops_thop(model, input_tensor):
    """Compute FLOPs using thop."""
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    return flops / 1e9  # GFLOPs


def measure_inference_time(model, input_tensor, warmup=50, repeats=200, device='cuda'):
    """Measure average inference time (ms)."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(input_tensor)

        # Measure
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.perf_counter()

    avg_time = (end - start) / repeats * 1000  # ms
    return avg_time


def print_table(results):
    """Print results as markdown table."""
    print("\n" + "=" * 80)
    print("Efficiency Comparison Results")
    print("=" * 80)
    print(f"{'Method':<25} {'#Params (M)':>12} {'GFLOPs':>10} {'Time (ms)':>10}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<25} {res['params']:>12.2f} {res['flops']:>10.2f} {res['time_ms']:>10.2f}")
    print("-" * 60)
    print()

    # LaTeX table
    print("LaTeX Table:")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Computational efficiency comparison.}")
    print("\\label{tab:efficiency}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("Method & \\#Params (M) & GFLOPs & Inference Time (ms) \\\\")
    print("\\hline")
    for name, res in results.items():
        print(f"{name} & {res['params']:.2f} & {res['flops']:.2f} & {res['time_ms']:.2f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


# ============================================================
# ISECRET Models (Restormer & ResNet)
# ============================================================

def evaluate_isecret_models(results):
    """Evaluate ISECRET models using BACKBONE_REGISTRY."""
    from isecret.model.backbone import BACKBONE_REGISTRY

    class FakeArgs:
        class model:
            generator = 'reformer'
            n_filters = 64
            n_downs = 2
            n_blocks = 9
            input_nc = 3
            output_nc = 3
            norm = 'in'
            padding = 'reflect'
            dis_n_filters = 64
            n_layers = 3
            discriminator = 'patchgan'
            f_model = 'mlp'
            dr_model = 'resnext101'

    args = FakeArgs()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_size = 512
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # ---- ISECRET Restormer ----
    print("\n>>> ISECRET-Restormer (reformer)")
    args.model.generator = 'reformer'

    # We need to bypass the importance- prefix that _check_network adds
    # Directly instantiate Restormer
    from isecret.model.restormer_network import Restormer
    model = Restormer().to(device)
    model.eval()

    # Load best checkpoint
    ckpt_path = os.path.join(BASE, 'ISECRET/experiments/eyeq/512/restormer_gan_cut_resize512_mse100.0 best/b2g_gen_best.pt')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # Handle DDP wrapping: remove 'module.' prefix
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        model.load_state_dict(new_state, strict=False)
        print(f'  Loaded weights from {ckpt_path}')

    params = count_parameters(model)
    flops = compute_flops_fvcore(model, dummy_input) if HAS_FVCORE else 0.0
    inf_time = measure_inference_time(model, dummy_input, device=str(device))
    results['ISECRET-Restormer'] = {'params': params, 'flops': flops, 'time_ms': inf_time}
    print(f'  Params: {params:.2f}M, GFLOPs: {flops:.2f}, Time: {inf_time:.2f}ms')

    # ---- ISECRET ResNet ----
    print("\n>>> ISECRET-ResNet")
    model_resnet = BACKBONE_REGISTRY.get('resnet')(args).to(device)
    model_resnet.eval()

    ckpt_path_resnet = os.path.join(BASE, 'ISECRET/experiments/eyeq/512/resnet_gan_cut200epoch/b2g_gen_best.pt')
    if os.path.exists(ckpt_path_resnet):
        state = torch.load(ckpt_path_resnet, map_location=device)
        new_state = OrderedDict()
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        model_resnet.load_state_dict(new_state, strict=False)
        print(f'  Loaded weights from {ckpt_path_resnet}')

    params = count_parameters(model_resnet)
    flops = compute_flops_fvcore(model_resnet, dummy_input) if HAS_FVCORE else 0.0
    inf_time = measure_inference_time(model_resnet, dummy_input, device=str(device))
    results['ISECRET-ResNet'] = {'params': params, 'flops': flops, 'time_ms': inf_time}
    print(f'  Params: {params:.2f}M, GFLOPs: {flops:.2f}, Time: {inf_time:.2f}ms')


# ============================================================
# CycleGAN / pix2pix generator
# ============================================================

def evaluate_cyclegan(results):
    """Evaluate CycleGAN generator (usable->good direction = G_A)."""
    print("\n>>> CycleGAN")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Import from CycleGAN repo
    sys.path.insert(0, os.path.join(BASE, 'pytorch-CycleGAN-and-pix2pix'))
    from models.networks import ResnetGenerator

    model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64,
                            norm_layer=nn.InstanceNorm2d, n_blocks=9).to(device)
    model.eval()

    ckpt_path = os.path.join(BASE, 'pytorch-CycleGAN-and-pix2pix/checkpoints/cyclegan/latest_net_G_A.pth')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'  Loaded weights from {ckpt_path}')

    input_size = 512
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    params = count_parameters(model)
    flops = compute_flops_fvcore(model, dummy_input) if HAS_FVCORE else 0.0
    inf_time = measure_inference_time(model, dummy_input, device=str(device))
    results['CycleGAN'] = {'params': params, 'flops': flops, 'time_ms': inf_time}
    print(f'  Params: {params:.2f}M, GFLOPs: {flops:.2f}, Time: {inf_time:.2f}ms')


# ============================================================
# CUT generator
# ============================================================

def evaluate_cut(results):
    """Evaluate CUT generator."""
    print("\n>>> CUT")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Note: CUT uses Python 3.6/PyTorch 1.4 env, but the network definition is compatible
    sys.path.insert(0, os.path.join(BASE, 'contrastive-unpaired-translation'))
    from models.networks import ResnetGenerator

    model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64,
                            norm_layer=nn.InstanceNorm2d, n_blocks=9).to(device)
    model.eval()

    ckpt_path = os.path.join(BASE, 'contrastive-unpaired-translation/checkpoints/cut/latest_net_G.pth')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'  Loaded weights from {ckpt_path}')

    input_size = 512
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    params = count_parameters(model)
    flops = compute_flops_fvcore(model, dummy_input) if HAS_FVCORE else 0.0
    inf_time = measure_inference_time(model, dummy_input, device=str(device))
    results['CUT'] = {'params': params, 'flops': flops, 'time_ms': inf_time}
    print(f'  Params: {params:.2f}M, GFLOPs: {flops:.2f}, Time: {inf_time:.2f}ms')


# ============================================================
# CofeNet generator
# ============================================================

def evaluate_cofenet(results):
    """Evaluate CofeNet generator."""
    print("\n>>> CofeNet")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sys.path.insert(0, os.path.join(BASE, 'Fundus-correction-cofe-Net'))
    from models.networks import ResnetGenerator

    # CofeNet uses 1 input channel (grayscale vessel mask as additional input?)
    # Check actual architecture...
    # Default: input_nc=3, output_nc=3, ngf=64, n_blocks=9
    try:
        model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64,
                                norm_layer=nn.InstanceNorm2d, n_blocks=9).to(device)
        model.eval()
    except Exception:
        # Try with different params
        model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64,
                                norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9).to(device)
        model.eval()

    ckpt_path = os.path.join(BASE, 'Fundus-correction-cofe-Net/encode_decode_vessel_2scale_atten10_dagrad3/experiment_name/latest_net_G.pth')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'  Loaded weights from {ckpt_path}')

    input_size = 512
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    params = count_parameters(model)
    flops = compute_flops_fvcore(model, dummy_input) if HAS_FVCORE else 0.0
    inf_time = measure_inference_time(model, dummy_input, device=str(device))
    results['CofeNet'] = {'params': params, 'flops': flops, 'time_ms': inf_time}
    print(f'  Params: {params:.2f}M, GFLOPs: {flops:.2f}, Time: {inf_time:.2f}ms')


# ============================================================
# PCENet generator
# ============================================================

def evaluate_pcenet(results):
    """Evaluate PCENet generator."""
    print("\n>>> PCENet")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sys.path.insert(0, os.path.join(BASE, 'PCENet-Image-Enhancement'))
    from models.networks import ResnetGenerator

    model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64,
                            norm_layer=nn.InstanceNorm2d, n_blocks=9).to(device)
    model.eval()

    ckpt_path = os.path.join(BASE, 'PCENet-Image-Enhancement/checkpoints/pcenet/latest_net_G.pth')
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'  Loaded weights from {ckpt_path}')

    input_size = 512
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    params = count_parameters(model)
    flops = compute_flops_fvcore(model, dummy_input) if HAS_FVCORE else 0.0
    inf_time = measure_inference_time(model, dummy_input, device=str(device))
    results['PCENet'] = {'params': params, 'flops': flops, 'time_ms': inf_time}
    print(f'  Params: {params:.2f}M, GFLOPs: {flops:.2f}, Time: {inf_time:.2f}ms')


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Efficiency comparison for fundus enhancement methods')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['isecret', 'cyclegan', 'cut', 'cofenet', 'pcenet'],
                        help='Methods to evaluate')
    parser.add_argument('--input_size', type=int, default=512,
                        help='Input image size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f'Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}')

    results = OrderedDict()

    if 'isecret' in args.methods:
        evaluate_isecret_models(results)

    if 'cyclegan' in args.methods:
        try:
            evaluate_cyclegan(results)
        except Exception as e:
            print(f'[SKIP] CycleGAN: {e}')

    if 'cut' in args.methods:
        try:
            evaluate_cut(results)
        except Exception as e:
            print(f'[SKIP] CUT: {e}')

    if 'cofenet' in args.methods:
        try:
            evaluate_cofenet(results)
        except Exception as e:
            print(f'[SKIP] CofeNet: {e}')

    if 'pcenet' in args.methods:
        try:
            evaluate_pcenet(results)
        except Exception as e:
            print(f'[SKIP] PCENet: {e}')

    if results:
        print_table(results)
    else:
        print('\nNo results collected.')
