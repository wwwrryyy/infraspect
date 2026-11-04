"""
calibrate_threshold.py
======================
Calibrates the anomaly detection MSE threshold from healthy concrete baselines.

Evaluates Stage 1 + Stage 2 + C++ AnomalyCalculus across healthy frames and computes
the 99.5th percentile pixel MSE, writing exports/calibration.json.
"""

import os
import json
import argparse
import numpy as np
import torch
import onnxruntime as ort

from data.datasets import Stage2HealthyDataset
from src_python.tactical_logic import compute_pixelwise_mse_numpy


def calibrate(args):
    print(f"[*] Starting Anomaly Threshold Calibration ({args.arch.upper()}) ...")

    s1_onnx = args.s1_onnx if args.s1_onnx else (
        "exports/sunet_denoiser_static_T4.onnx" if args.arch == "snn" else "exports/cnn_denoiser.onnx"
    )
    s2_onnx = args.s2_onnx if args.s2_onnx else (
        "exports/snn_autoencoder_static_T4.onnx" if args.arch == "snn" else "exports/cnn_autoencoder.onnx"
    )

    if not os.path.isfile(s1_onnx) or not os.path.isfile(s2_onnx):
        raise FileNotFoundError(f"ONNX models not found. Please run export first ({s1_onnx}, {s2_onnx}).")

    s1_session = ort.InferenceSession(s1_onnx, providers=["CPUExecutionProvider"])
    s2_session = ort.InferenceSession(s2_onnx, providers=["CPUExecutionProvider"])

    s1_in = s1_session.get_inputs()[0].name
    s1_out = s1_session.get_outputs()[0].name
    s2_in = s2_session.get_inputs()[0].name
    s2_out = s2_session.get_outputs()[0].name

    dataset = Stage2HealthyDataset(
        image_dir=args.data_dir,
        num_synthetic=args.num_frames,
        height=args.height,
        width=args.width,
        in_channels=args.in_channels,
        seed=42,
    )

    all_pixel_errors = []
    frame_mean_errors = []

    print(f"[+] Processing {len(dataset)} healthy calibration frames ...")
    for i in range(len(dataset)):
        noisy, clean = dataset[i]
        # [1, C, H, W]
        x_np = noisy.unsqueeze(0).numpy()

        restored = s1_session.run([s1_out], {s1_in: x_np})[0]
        recon = s2_session.run([s2_out], {s2_in: restored})[0]

        # Convert back to HWC for anomaly calculus
        rest_hwc = restored.squeeze(0).transpose(1, 2, 0)
        recon_hwc = recon.squeeze(0).transpose(1, 2, 0)

        mean_mse, heatmap = compute_pixelwise_mse_numpy(rest_hwc, recon_hwc, gaussian_sigma=0.8)
        frame_mean_errors.append(mean_mse)
        all_pixel_errors.append(heatmap.flatten())

    all_pixel_errors = np.concatenate(all_pixel_errors)
    p99_5 = float(np.percentile(all_pixel_errors, 99.5))
    p99_0 = float(np.percentile(all_pixel_errors, 99.0))
    mean_error = float(np.mean(frame_mean_errors))

    calib_data = {
        "arch": args.arch,
        "calibrated_mse_threshold": round(p99_5, 5),
        "p99_threshold": round(p99_0, 5),
        "mean_healthy_mse": round(mean_error, 5),
        "num_calibration_frames": len(dataset),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"calibration_{args.arch}.json")
    with open(out_file, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"[+] Calibration Complete:")
    print(f"    99.5th Percentile Anomaly Threshold: {calib_data['calibrated_mse_threshold']}")
    print(f"    Mean Healthy MSE:                  {calib_data['mean_healthy_mse']}")
    print(f"    Calibration Profile Saved:         {out_file}")
    return calib_data


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate Anomaly Threshold on Healthy Baselines")
    p.add_argument("--arch", type=str, default="snn", choices=["snn", "cnn"])
    p.add_argument("--s1-onnx", type=str, default=None)
    p.add_argument("--s2-onnx", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="exports")
    p.add_argument("--num-frames", type=int, default=30)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--in-channels", type=int, default=3)
    return p.parse_args()


if __name__ == "__main__":
    calibrate(parse_args())
