"""
train_snn_autoencoder.py
========================
Negative-Only Training, AUROC Anomaly Evaluation, and ONNX Export for Stage 2.

Supports both SNN (T=4) and Continuous CNN (T=1) backbones.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from models.stage2_detector.autoencoder import SNNAutoencoder, CNNAutoencoder, ONNXExportableSNN
from data.datasets import Stage2HealthyDataset, generate_concrete_patch, inject_synthetic_defect
from models.stage1_denoiser.train import ssim_loss


def evaluate_pixel_auroc(model, device, in_channels=3, h=128, w=128, num_eval=20):
    """Evaluates pixel-level anomaly detection AUROC against synthetic defects."""
    model.eval()
    rng = np.random.default_rng(777)
    y_true_all = []
    y_score_all = []

    with torch.no_grad():
        for i in range(num_eval):
            patch = generate_concrete_patch(height=h, width=w, in_channels=in_channels, rng=rng)
            defect_type = "crack" if i % 2 == 0 else "spalling"
            corrupted, mask, _ = inject_synthetic_defect(patch, defect_type=defect_type, rng=rng)

            # [C, H, W]
            x_t = torch.from_numpy(corrupted.transpose(2, 0, 1) if in_channels == 3 else corrupted[np.newaxis]).unsqueeze(0).to(device)
            recon_t = model(x_t)

            diff_sq = (x_t - recon_t) ** 2
            if in_channels == 3:
                pixel_mse = diff_sq.mean(dim=1).squeeze(0).cpu().numpy()
            else:
                pixel_mse = diff_sq.squeeze(0).squeeze(0).cpu().numpy()

            binary_mask = (mask > 0).astype(np.int32)
            y_true_all.extend(binary_mask.flatten())
            y_score_all.extend(pixel_mse.flatten())

    # Fast trapezoidal AUROC computation
    y_true = np.array(y_true_all, dtype=np.int32)
    y_score = np.array(y_score_all, dtype=np.float32)

    # Sort descending
    sort_idx = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[sort_idx]

    tps = np.cumsum(y_true_sorted)
    fps = np.cumsum(1 - y_true_sorted)
    total_pos = tps[-1]
    total_neg = fps[-1]

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr = tps / total_pos
    fpr = fps / total_neg
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auroc = float(trapz_fn(tpr, fpr))
    return auroc


def train_and_export(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[*] Stage 2 Anomaly Detector: arch={args.arch.upper()} on device={device}")

    if args.arch.lower() == "snn":
        model = SNNAutoencoder(
            in_channels=args.in_channels,
            beta=args.beta,
            num_steps=args.num_steps,
        ).to(device)
        default_onnx = "exports/snn_autoencoder_static_T4.onnx"
    else:
        model = CNNAutoencoder(in_channels=args.in_channels).to(device)
        default_onnx = "exports/cnn_autoencoder.onnx"

    full_dataset = Stage2HealthyDataset(
        image_dir=args.data_dir,
        num_synthetic=args.num_synthetic,
        height=args.height,
        width=args.width,
        in_channels=args.in_channels,
    )
    # Hold out 20% of healthy samples to select checkpoints on validation (not training) loss
    n_val = max(1, len(full_dataset) // 5) if len(full_dataset) > 1 else 0
    if n_val:
        dataset, val_dataset = random_split(
            full_dataset,
            [len(full_dataset) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        dataset = val_dataset = full_dataset
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_fn = nn.MSELoss()
    cnn_l1_latent_weight = 1e-4

    def reconstruction_loss(noisy, clean):
        if args.arch.lower() == "cnn":
            latent = model.encode(noisy)
            recon = model.decode(latent)
            sparsity = cnn_l1_latent_weight * torch.mean(torch.abs(latent))
        else:
            recon = model(noisy)
            sparsity = 0.0
        return mse_fn(recon, clean) + 0.5 * ssim_loss(recon, clean) + sparsity

    print(f"[+] Training Stage 2 {args.arch.upper()} on NEGATIVE-ONLY healthy baselines ({len(dataset)} train / {len(val_dataset)} val samples)")

    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(args.output_dir, f"stage2_{args.arch.lower()}_best.pt")
    if args.checkpoint and os.path.isfile(args.checkpoint):
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights from: {args.checkpoint}")
    elif args.epochs == 0 and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[+] Loaded weights from: {ckpt_path}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for noisy, clean in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            loss = reconstruction_loss(noisy, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for noisy_v, clean_v in val_loader:
                val_running += reconstruction_loss(noisy_v.to(device), clean_v.to(device)).item()
        avg_val_loss = val_running / max(1, len(val_loader))

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"    Epoch [{epoch}/{args.epochs}] Train Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_path)

    # Evaluate AUROC on synthetic defects
    auroc = evaluate_pixel_auroc(
        model, device, in_channels=args.in_channels, h=args.height, w=args.width, num_eval=15
    )
    print(f"[+] Stage 2 Defect Pixel AUROC: {auroc:.4f}")

    # Export to ONNX
    onnx_path = args.output if args.output else default_onnx
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    model.eval()

    dummy = torch.randn(1, args.in_channels, args.height, args.width, device=device)
    export_wrapper = ONNXExportableSNN(model) if args.arch.lower() == "snn" else model

    print(f"[*] Exporting Stage 2 to ONNX: {onnx_path} ...")
    torch.onnx.export(
        export_wrapper,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["restored_frame"],
        output_names=["reconstructed_baseline"],
        dynamic_axes={
            "restored_frame": {0: "batch_size"},
            "reconstructed_baseline": {0: "batch_size"},
        },
        dynamo=False,
    )
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[+] Exported Stage 2 ONNX: {onnx_path} ({size_kb:.1f} KB)")
    return ckpt_path, onnx_path, auroc


def parse_args():
    p = argparse.ArgumentParser(description="Train and Export Stage 2 Autoencoder")
    p.add_argument("--arch", type=str, default="snn", choices=["snn", "cnn"])
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="exports")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--num-synthetic", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    train_and_export(parse_args())
