"""
validate.py
============
PyTorch ↔ ONNX Runtime equivalence validation for Stage 1 Denoiser.

For CNN: asserts strict max absolute difference <= tolerance (1e-4).
For SNN: reports max absolute diff, median diff, and asserts output PSNR >= 40 dB
(accounting for Heaviside threshold floating-point variance between backends).
"""

import argparse
import sys
import numpy as np
import torch
import onnxruntime as ort

from models.stage1_denoiser.model import SpikingUNet, CNNUNet


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return 100.0
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def validate(args):
    device = torch.device("cpu")

    if args.arch.lower() == "snn":
        model = SpikingUNet(
            in_channels=args.in_channels,
            base_ch=args.base_ch,
            num_steps=args.num_steps,
            beta=args.beta,
        ).to(device)
        onnx_path = args.onnx if args.onnx else "exports/sunet_denoiser_static_T4.onnx"
    else:
        model = CNNUNet(
            in_channels=args.in_channels,
            base_ch=args.base_ch,
        ).to(device)
        onnx_path = args.onnx if args.onnx else "exports/cnn_denoiser.onnx"

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded PyTorch checkpoint: {args.checkpoint}")

    model.eval()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"[+] Loaded ONNX model: {onnx_path}")

    rng = np.random.default_rng(args.seed)
    max_diff_seen = 0.0
    psnr_list = []

    for i in range(args.num_samples):
        x_np = rng.random((1, args.in_channels, args.height, args.width)).astype(np.float32)
        x_pt = torch.from_numpy(x_np)

        with torch.no_grad():
            y_pt = model(x_pt).numpy()

        y_ort = session.run([output_name], {input_name: x_np})[0]

        diff = np.abs(y_pt - y_ort)
        max_diff_seen = max(max_diff_seen, float(np.max(diff)))
        psnr_list.append(compute_psnr(y_pt, y_ort))

    avg_psnr = float(np.mean(psnr_list))
    print(f"\n[*] Equivalence Results ({args.arch.upper()} across {args.num_samples} samples):")
    print(f"    Max Absolute Difference: {max_diff_seen:.6e}")
    print(f"    Mean Output PSNR:        {avg_psnr:.2f} dB")

    if args.arch.lower() == "cnn":
        passed = max_diff_seen <= args.tolerance
    else:
        # SNN passes if PSNR >= 40 dB (spiking numerical stability gate)
        passed = avg_psnr >= 40.0

    if passed:
        print(f"[+] VALIDATION PASSED: PyTorch and ONNX Runtime produce equivalent outputs.")
        return 0
    else:
        print(f"[!] VALIDATION FAILED: Difference exceeds acceptable tolerance.")
        return 1


def parse_args():
    p = argparse.ArgumentParser(description="Validate PyTorch ↔ ONNX Equivalence")
    p.add_argument("--arch", type=str, default="snn", choices=["snn", "cnn"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--onnx", type=str, default=None)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--num-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(validate(parse_args()))
