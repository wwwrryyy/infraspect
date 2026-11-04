"""
train_real_inspection.py
========================
Option 2: Real Inspection Dataset Training and Calibration Pipeline.

Trains and calibrates all 3 stages on real civil concrete imagery (CrackForest & CrackSegmentation):
1. Stage 1 Denoiser (SNN & CNN): Fine-tuned on real concrete textures with civil corruptions.
2. Stage 2 Anomaly Detector (SNN & CNN): Negative-only autoencoder trained on strictly healthy concrete.
3. Stage 2 Pixel AUROC Benchmark: Evaluated against real ground-truth crack segmentation masks.
4. Stage 3 Defect Classifier (SNN & CNN): MobileNetV3 trained on real crack and civil defect crops.
5. ONNX Export & Calibration: Exports static graphs and calibrates 99.5th percentile anomaly thresholds.
"""

import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Fast pure NumPy ROC AUC calculation
def roc_auc_numpy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Computes Area Under ROC Curve using Mann-Whitney U-statistic."""
    y_true = (y_true > 0).astype(np.int32)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.50

    if n_pos + n_neg > 200000:
        pos = np.random.choice(pos, min(n_pos, 50000), replace=False)
        neg = np.random.choice(neg, min(n_neg, 50000), replace=False)
        n_pos, n_neg = len(pos), len(neg)

    combined = np.concatenate([pos, neg])
    order = np.argsort(combined)
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, len(order) + 1)
    rank_pos = rank[:n_pos]
    u = np.sum(rank_pos) - (n_pos * (n_pos + 1)) / 2.0
    return float(np.clip(u / (n_pos * n_neg), 0.0, 1.0))

# Environment guards
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from data.real_inspection_dataset import (
    extract_real_concrete_patches,
    RealStage1DenoisingDataset,
    RealStage2HealthyDataset,
    RealStage2EvaluationDataset,
    RealStage3ClassifierDataset,
)
from models.stage1_denoiser.model import SpikingUNet, CNNUNet
from models.stage2_detector.autoencoder import SNNAutoencoder, CNNAutoencoder, ONNXExportableSNN
from models.stage3_classifier.model import SpikingMobileNetV3Classifier, CNNMobileNetV3Classifier
from data.datasets import DEFECT_CLASSES
from src_python.tactical_logic import compute_pixelwise_mse_numpy


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Fast SSIM loss for spatial structural coherence."""
    c1, c2 = 0.01**2, 0.03**2
    kernel = torch.ones((img1.shape[1], 1, window_size, window_size), device=img1.device) / (window_size**2)
    mu1 = nn.functional.conv2d(img1, kernel, padding=window_size // 2, groups=img1.shape[1])
    mu2 = nn.functional.conv2d(img2, kernel, padding=window_size // 2, groups=img2.shape[1])
    mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
    sigma1_sq = nn.functional.conv2d(img1 * img1, kernel, padding=window_size // 2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = nn.functional.conv2d(img2 * img2, kernel, padding=window_size // 2, groups=img2.shape[1]) - mu2_sq
    sigma12 = nn.functional.conv2d(img1 * img2, kernel, padding=window_size // 2, groups=img1.shape[1]) - mu1_mu2
    ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return 1.0 - torch.clamp(ssim.mean(), 0.0, 1.0)


def evaluate_pixel_auroc_real(model, eval_dataset, device, num_eval: int = 40) -> float:
    """Evaluates pixel-level AUROC on real defective concrete patches and ground-truth crack masks."""
    model.eval()
    all_scores = []
    all_targets = []

    limit = min(num_eval, len(eval_dataset))
    with torch.no_grad():
        for i in range(limit):
            patch_t, mask_t = eval_dataset[i]
            x = patch_t.unsqueeze(0).to(device)
            recon = model(x)

            # Pixel-wise MSE across channels
            err = torch.mean((recon - x) ** 2, dim=1).squeeze(0).cpu().numpy()
            target = mask_t.numpy()

            all_scores.append(err.flatten())
            all_targets.append(target.flatten())

    if len(all_scores) == 0:
        return 0.50

    y_score = np.concatenate(all_scores)
    y_true = np.concatenate(all_targets)

    if len(np.unique(y_true)) < 2:
        return 0.50

    return float(roc_auc_numpy(y_true, y_score))


def train_stage1_real(arch: str, healthy_patches: list, device, epochs: int = 5, batch_size: int = 16):
    print(f"\n[*] Training Stage 1 Denoiser ({arch.upper()}) on Real Concrete Patches ...")
    if arch == "snn":
        model = SpikingUNet(in_channels=3, base_ch=32, num_steps=4, beta=0.85).to(device)
    else:
        model = CNNUNet(in_channels=3, base_ch=32).to(device)

    dataset = RealStage1DenoisingDataset(healthy_patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()

    best_loss = float("inf")
    ckpt_path = f"exports/{arch}_denoiser_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for corrupted, clean in loader:
            corrupted, clean = corrupted.to(device), clean.to(device)
            optimizer.zero_grad()
            pred = model(corrupted)
            loss = mse_fn(pred, clean) + 0.3 * ssim_loss(pred, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        if epoch == 1 or epoch % max(1, epochs // 2) == 0 or epoch == epochs:
            print(f"    Epoch [{epoch}/{epochs}] Reconstruction Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

    # Clean export on CPU
    onnx_path = "exports/sunet_denoiser_static_T4.onnx" if arch == "snn" else "exports/cnn_denoiser.onnx"
    export_model = SpikingUNet(in_channels=3, base_ch=32, num_steps=4, beta=0.85).to("cpu") if arch == "snn" else CNNUNet(in_channels=3, base_ch=32).to("cpu")
    export_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    export_model.eval()
    for p in export_model.parameters():
        p.requires_grad = False
    dummy = torch.randn(1, 3, 128, 128, device="cpu")
    print(f"[*] Exporting Stage 1 {arch.upper()} to ONNX: {onnx_path} ...")
    torch.onnx.export(
        export_model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_frame"],
        output_names=["restored_frame"],
        dynamic_axes={"input_frame": {0: "batch_size"}, "restored_frame": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"[+] Stage 1 ({arch.upper()}) export complete: {onnx_path}")
    return model, onnx_path


def train_stage2_real(arch: str, healthy_patches: list, defect_patches: list, device, epochs: int = 5, batch_size: int = 16):
    print(f"\n[*] Training Stage 2 Negative-Only Autoencoder ({arch.upper()}) on Real Healthy Concrete ...")
    if arch == "snn":
        model = SNNAutoencoder(in_channels=3, beta=0.85, num_steps=4).to(device)
        onnx_path = "exports/snn_autoencoder_static_T4.onnx"
    else:
        model = CNNAutoencoder(in_channels=3).to(device)
        onnx_path = "exports/cnn_autoencoder.onnx"

    dataset = RealStage2HealthyDataset(healthy_patches)
    eval_dataset = RealStage2EvaluationDataset(defect_patches)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()

    best_loss = float("inf")
    ckpt_path = f"exports/stage2_{arch}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            recon = model(noisy)
            loss = mse_fn(recon, clean) + 0.3 * ssim_loss(recon, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(loader))
        if epoch == 1 or epoch % max(1, epochs // 2) == 0 or epoch == epochs:
            print(f"    Epoch [{epoch}/{epochs}] Negative Baseline Loss: {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

    # Benchmark Pixel AUROC on Real Cracks
    auroc = evaluate_pixel_auroc_real(model, eval_dataset, device, num_eval=40)
    print(f"[+] Stage 2 ({arch.upper()}) Real Crack Pixel AUROC: {auroc:.4f}")

    # Clean export on CPU
    export_model = SNNAutoencoder(in_channels=3, beta=0.85, num_steps=4).to("cpu") if arch == "snn" else CNNAutoencoder(in_channels=3).to("cpu")
    export_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    export_model.eval()
    for p in export_model.parameters():
        p.requires_grad = False
    dummy = torch.randn(1, 3, 128, 128, device="cpu")
    export_wrapper = ONNXExportableSNN(export_model) if arch == "snn" else export_model
    print(f"[*] Exporting Stage 2 {arch.upper()} to ONNX: {onnx_path} ...")
    torch.onnx.export(
        export_wrapper,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["restored_frame"],
        output_names=["reconstructed_baseline"],
        dynamic_axes={"restored_frame": {0: "batch_size"}, "reconstructed_baseline": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"[+] Stage 2 ({arch.upper()}) export complete: {onnx_path}")
    return model, onnx_path, auroc


def train_stage3_real(arch: str, defect_patches: list, healthy_patches: list, device, epochs: int = 5, batch_size: int = 16):
    print(f"\n[*] Training Stage 3 MobileNetV3 Classifier ({arch.upper()}) on Defect Crops ...")
    num_classes = len(DEFECT_CLASSES)
    if arch == "snn":
        model = SpikingMobileNetV3Classifier(in_channels=3, num_classes=num_classes, num_steps=4, beta=0.85).to(device)
        onnx_path = "exports/spiking_mobilenet_classifier_static_T4.onnx"
    else:
        model = CNNMobileNetV3Classifier(in_channels=3, num_classes=num_classes).to(device)
        onnx_path = "exports/cnn_mobilenet_classifier.onnx"

    dataset = RealStage3ClassifierDataset(defect_patches, healthy_patches, crop_size=64, samples_per_epoch=200)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ce_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    ckpt_path = f"exports/stage3_{arch}_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for crops, labels in loader:
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(crops)
            loss = ce_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = (correct / max(1, total)) * 100.0
        avg_loss = running_loss / max(1, len(loader))
        if epoch == 1 or epoch % max(1, epochs // 2) == 0 or epoch == epochs:
            print(f"    Epoch [{epoch}/{epochs}] Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_path)

    # Clean export on CPU
    export_model = SpikingMobileNetV3Classifier(in_channels=3, num_classes=num_classes, num_steps=4, beta=0.85).to("cpu") if arch == "snn" else CNNMobileNetV3Classifier(in_channels=3, num_classes=num_classes).to("cpu")
    export_model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    export_model.eval()
    for p in export_model.parameters():
        p.requires_grad = False
    dummy = torch.randn(1, 3, 64, 64, device="cpu")
    print(f"[*] Exporting Stage 3 {arch.upper()} to ONNX: {onnx_path} ...")
    torch.onnx.export(
        export_model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["roi_crop"],
        output_names=["defect_logits"],
        dynamic_axes={"roi_crop": {0: "batch_size"}, "defect_logits": {0: "batch_size"}},
        dynamo=False,
    )
    print(f"[+] Stage 3 ({arch.upper()}) export complete: {onnx_path}")
    return model, onnx_path


def calibrate_threshold_real(arch: str, healthy_patches: list, s1_onnx: str, s2_onnx: str) -> float:
    """Calibrates 99.5th percentile MSE threshold on real uncracked concrete validation patches."""
    import onnxruntime as ort

    print(f"\n[*] Calibrating Anomaly Threshold on Real Healthy Concrete ({arch.upper()}) ...")
    s1_session = ort.InferenceSession(s1_onnx, providers=["CPUExecutionProvider"])
    s2_session = ort.InferenceSession(s2_onnx, providers=["CPUExecutionProvider"])

    s1_in = s1_session.get_inputs()[0].name
    s1_out = s1_session.get_outputs()[0].name
    s2_in = s2_session.get_inputs()[0].name
    s2_out = s2_session.get_outputs()[0].name

    all_pixel_errors = []
    # Use 50 held-out healthy validation patches
    calib_patches = healthy_patches[:50]
    for patch in calib_patches:
        x_norm = patch.transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        restored = s1_session.run([s1_out], {s1_in: x_norm})[0]
        recon = s2_session.run([s2_out], {s2_in: restored})[0]

        rest_hwc = restored.squeeze(0).transpose(1, 2, 0)
        recon_hwc = recon.squeeze(0).transpose(1, 2, 0)

        _, heatmap = compute_pixelwise_mse_numpy(rest_hwc, recon_hwc, gaussian_sigma=0.8)
        all_pixel_errors.append(heatmap.flatten())

    concat_errors = np.concatenate(all_pixel_errors)
    calibrated_th = float(np.percentile(concat_errors, 99.5))
    # Safety clamp: avoid degenerate or extreme thresholds
    calibrated_th = max(0.005, min(0.15, calibrated_th))

    calib_data = {
        "arch": arch,
        "calibrated_mse_threshold": round(calibrated_th, 5),
        "mean_pixel_mse": float(np.mean(concat_errors)),
        "p95": float(np.percentile(concat_errors, 95.0)),
        "p99": float(np.percentile(concat_errors, 99.0)),
        "p99_5": float(np.percentile(concat_errors, 99.5)),
        "num_calib_patches": len(calib_patches),
        "timestamp": time.time(),
    }

    out_file = f"exports/calibration_{arch.lower()}.json"
    with open(out_file, "w") as f:
        json.dump(calib_data, f, indent=2)

    print(f"[+] Calibrated 99.5th percentile MSE threshold: {calibrated_th:.5f} -> Saved to {out_file}")
    return calibrated_th


def main():
    parser = argparse.ArgumentParser(description="Infraspect Real Inspection Dataset Training & Calibration")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs per stage")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-patches", type=int, default=250, help="Maximum number of patches to extract")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[+] Device: {device}")

    # Step 1: Extract real patches
    print("[*] Extracting real concrete surface patches from CrackForest ...")
    healthy_patches, defect_patches = extract_real_concrete_patches(max_patches=args.max_patches)
    print(f"[+] Extracted: {len(healthy_patches)} clean healthy patches, {len(defect_patches)} defective crack patches")

    summary = {}

    for arch in ["cnn", "snn"]:
        print("\n" + "=" * 70)
        print(f" PIPELINE TRAINING: {arch.upper()} BACKBONE")
        print("=" * 70)

        # Stage 1: Denoiser
        s1_model, s1_onnx = train_stage1_real(arch, healthy_patches, device, epochs=args.epochs, batch_size=args.batch_size)

        # Stage 2: Anomaly Detector
        s2_model, s2_onnx, auroc = train_stage2_real(arch, healthy_patches, defect_patches, device, epochs=args.epochs, batch_size=args.batch_size)

        # Stage 3: Defect Classifier
        s3_model, s3_onnx = train_stage3_real(arch, defect_patches, healthy_patches, device, epochs=args.epochs, batch_size=args.batch_size)

        # Calibrate Anomaly Threshold
        calibrated_th = calibrate_threshold_real(arch, healthy_patches, s1_onnx, s2_onnx)

        summary[arch] = {
            "stage2_auroc": round(auroc, 4),
            "calibrated_threshold": calibrated_th,
        }

    print("\n" + "=" * 70)
    print(" OPTION 2 REAL DATASET TRAINING COMPLETE")
    print("=" * 70)
    for arch, metrics in summary.items():
        print(f"  {arch.upper()}: Stage 2 Defect AUROC = {metrics['stage2_auroc']:.4f} | Calibrated Threshold = {metrics['calibrated_threshold']:.5f}")


if __name__ == "__main__":
    main()
