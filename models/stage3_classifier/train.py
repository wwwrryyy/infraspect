"""
train.py
========
Training script for Stage 3 Defect Classifier (SNN and CNN backbones).

Taxonomy: CODEBRIM 6 classes (multi-label).
Loss: BCEWithLogitsLoss.
Calibration: Post-hoc temperature scaling on validation logits.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.stage3_classifier.model import SpikingMobileNetV3Classifier, CNNMobileNetV3Classifier
from data.datasets import Stage3DefectDataset, DEFECT_CLASSES


class TemperatureScaler(nn.Module):
    """Post-hoc calibration: scales logits by 1/T to calibrate confidence probabilities."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.2)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def train(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[*] Stage 3 Defect Classifier: arch={args.arch.upper()} on device={device}")

    if args.arch.lower() == "snn":
        model = SpikingMobileNetV3Classifier(
            in_channels=args.in_channels,
            num_classes=len(DEFECT_CLASSES),
            num_steps=args.num_steps,
            beta=args.beta,
        ).to(device)
    else:
        model = CNNMobileNetV3Classifier(
            in_channels=args.in_channels,
            num_classes=len(DEFECT_CLASSES),
        ).to(device)

    dataset = Stage3DefectDataset(
        image_dir=args.data_dir,
        num_synthetic=args.num_synthetic,
        in_channels=args.in_channels,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"[+] Training Stage 3 {args.arch.upper()} for {args.epochs} epochs ({len(dataset)} samples)")

    best_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"stage3_{args.arch.lower()}_best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for crops, labels in loader:
            crops = crops.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(crops)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            print(f"    Epoch [{epoch}/{args.epochs}] Multi-Label BCE Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

    print(f"[+] Stage 3 {args.arch.upper()} complete. Best loss: {best_loss:.6f}")
    print(f"[+] Checkpoint saved to: {ckpt_path}")
    return ckpt_path


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage 3 MobileNetV3 Classifier")
    p.add_argument("--arch", type=str, default="snn", choices=["snn", "cnn"])
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="exports")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--in-channels", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.85)
    p.add_argument("--num-synthetic", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
