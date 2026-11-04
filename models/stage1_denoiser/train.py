"""
train.py
========
Training script for Stage 1 Denoiser / Image Restoration (SNN & CNN backbones).

Loss: MSE + 0.5 * SSIM.
Data: Stage1DenoiseDataset (civil optical corruptions + concrete texture).
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.stage1_denoiser.model import SpikingUNet, CNNUNet
from data.datasets import Stage1DenoiseDataset


# ---------------------------------------------------------------------------
# Differentiable SSIM Loss
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(1) @ g.unsqueeze(0)


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> torch.Tensor:
    channels = pred.size(1)
    kernel = _gaussian_kernel_2d(window_size).to(pred.device, pred.dtype)
    kernel = kernel.expand(channels, 1, window_size, window_size)

    mu_p = nn.functional.conv2d(pred, kernel, padding=window_size // 2, groups=channels)
    mu_t = nn.functional.conv2d(target, kernel, padding=window_size // 2, groups=channels)

    mu_p_sq = mu_p ** 2
    mu_t_sq = mu_t ** 2
    mu_pt = mu_p * mu_t

    sigma_p_sq = nn.functional.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=channels) - mu_p_sq
    sigma_t_sq = nn.functional.conv2d(target ** 2, kernel, padding=window_size // 2, groups=channels) - mu_t_sq
    sigma_pt = nn.functional.conv2d(pred * target, kernel, padding=window_size // 2, groups=channels) - mu_pt

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / (
        (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
    )
    return 1.0 - ssim_map.mean()


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[*] Stage 1 Denoiser Training: arch={args.arch.upper()} on device={device}")

    if args.arch.lower() == "snn":
        model = SpikingUNet(
            in_channels=args.in_channels,
            base_ch=args.base_ch,
            num_steps=args.num_steps,
            beta=args.beta,
        ).to(device)
    else:
        model = CNNUNet(
            in_channels=args.in_channels,
            base_ch=args.base_ch,
        ).to(device)

    dataset = Stage1DenoiseDataset(
        image_dir=args.data_dir,
        num_synthetic=args.num_synthetic,
        height=args.height,
        width=args.width,
        in_channels=args.in_channels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    mse_fn = nn.MSELoss()

    print(f"[+] Training {args.arch.upper()} for {args.epochs} epochs ({len(dataset)} samples, batch_size={args.batch_size})")

    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_name = f"sunet_denoiser_best.pt" if args.arch.lower() == "snn" else f"cnn_denoiser_best.pt"
    ckpt_path = os.path.join(args.output_dir, ckpt_name)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for corrupted, clean in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(corrupted)
            loss = mse_fn(pred, clean) + args.ssim_weight * ssim_loss(pred, clean)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(loader))
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"    Epoch [{epoch}/{args.epochs}] loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"[+] Stage 1 ({args.arch.upper()}) training complete. Best loss: {best_loss:.6f}")
    print(f"[+] Saved to: {ckpt_path}")
    return ckpt_path


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 1 Denoiser (SNN or CNN)")
    p.add_argument("--arch", type=str, default="snn", choices=["snn", "cnn"])
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="exports")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--base-ch", type=int, default=32)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--ssim-weight", type=float, default=0.5)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--num-synthetic", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
