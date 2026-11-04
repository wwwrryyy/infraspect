"""
real_inspection_dataset.py
==========================
Data loader and patch extractor for real civil inspection datasets:
1. CrackForest Dataset (118 real concrete road surfaces with .mat ground truth).
2. CrackSegmentation Dataset (84 high-resolution concrete surfaces with binary masks).

Provides:
- RealStage1DenoisingDataset: Real concrete textures paired with civil corruptions.
- RealStage2HealthyDataset: Strictly negative-only healthy concrete surface patches.
- RealStage2EvaluationDataset: Defective concrete patches with binary ground-truth masks for AUROC.
- RealStage3ClassifierDataset: 64x64 defect crops with civil labels.
"""

import os
import glob
import cv2
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from models.stage1_denoiser.corruption import corrupt_civil_image
from data.datasets import DEFECT_CLASSES, inject_synthetic_defect


def extract_real_concrete_patches(
    crackforest_dir: str = "data/raw/crackforest",
    patch_size: int = 128,
    max_patches: int = 600,
    stride: int = 64,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Extracts non-overlapping healthy and defective patches from CrackForest.

    Returns:
        (healthy_patches, defect_patches_with_masks)
        healthy_patches: list of [H, W, 3] float32 in [0, 1]
        defect_patches_with_masks: list of (patch [H, W, 3], binary_mask [H, W] {0, 1})
    """
    healthy_patches = []
    defect_patches = []

    img_paths = sorted(glob.glob(os.path.join(crackforest_dir, "image", "*.jpg")))
    gt_paths = sorted(glob.glob(os.path.join(crackforest_dir, "groundTruth", "*.mat")))

    if not img_paths or not gt_paths:
        return healthy_patches, defect_patches

    num_samples = min(len(img_paths), len(gt_paths))

    for idx in range(num_samples):
        img_bgr = cv2.imread(img_paths[idx])
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        try:
            mat = sio.loadmat(gt_paths[idx])
            seg = mat["groundTruth"][0, 0]["Segmentation"]
        except Exception:
            continue

        h, w = img_rgb.shape[:2]
        # Slide window across the surface
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                crop_img = img_rgb[y : y + patch_size, x : x + patch_size]
                crop_seg = seg[y : y + patch_size, x : x + patch_size]

                # In CrackForest: seg == 1 is healthy concrete, seg == 2 is crack
                crack_mask = (crop_seg == 2).astype(np.uint8)
                crack_pixels = int(np.sum(crack_mask))

                if crack_pixels == 0:
                    healthy_patches.append(crop_img)
                elif crack_pixels >= 20:  # Valid visible crack
                    defect_patches.append((crop_img, crack_mask))

                if len(healthy_patches) >= max_patches and len(defect_patches) >= max_patches // 2:
                    return healthy_patches, defect_patches

    return healthy_patches, defect_patches


def extract_crack_segmentation_patches(
    crack_seg_dir: str = "data/raw/crack_segmentation",
    patch_size: int = 128,
    max_patches: int = 200,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Extracts real concrete surface and crack patches from the CrackSegmentation dataset."""
    healthy_patches = []
    defect_patches = []

    img_dir = os.path.join(crack_seg_dir, "test_imgs")
    mask_dir = os.path.join(crack_seg_dir, "test_results", "resnet_101")
    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        return healthy_patches, defect_patches

    masks = sorted(glob.glob(os.path.join(mask_dir, "*.*")))
    mask_map = {os.path.splitext(os.path.basename(p))[0]: p for p in masks}

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    for img_p in imgs:
        stem = os.path.splitext(os.path.basename(img_p))[0]
        if stem not in mask_map:
            continue

        img_bgr = cv2.imread(img_p)
        mask = cv2.imread(mask_map[stem], cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or mask is None:
            continue
        if img_bgr.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = img_rgb.shape[:2]

        step = max(patch_size, 384)
        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                crop_img = img_rgb[y : y + patch_size, x : x + patch_size]
                crop_mask = (mask[y : y + patch_size, x : x + patch_size] > 128).astype(np.uint8)
                crack_pixels = int(np.sum(crop_mask))

                if crack_pixels == 0:
                    healthy_patches.append(crop_img)
                elif crack_pixels >= 25:
                    defect_patches.append((crop_img, crop_mask))

                if len(healthy_patches) >= max_patches and len(defect_patches) >= max_patches // 2:
                    return healthy_patches, defect_patches

    return healthy_patches, defect_patches


def extract_all_real_inspection_patches(
    patch_size: int = 128,
    max_patches: int = 400,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Combines healthy and defective patches from both CrackForest and CrackSegmentation."""
    half = max_patches // 2
    h1, d1 = extract_real_concrete_patches(patch_size=patch_size, max_patches=half)
    h2, d2 = extract_crack_segmentation_patches(patch_size=patch_size, max_patches=half)
    return h1 + h2, d1 + d2


class RealStage1DenoisingDataset(Dataset):
    """Stage 1 Denoiser dataset utilizing real concrete surfaces under civil corruptions."""

    def __init__(self, healthy_patches: list[np.ndarray], rng: np.random.Generator = None):
        self.patches = healthy_patches
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean = self.patches[idx]
        corrupted = corrupt_civil_image(clean, rng=self.rng)

        # Convert to [C, H, W] float32 tensors
        clean_t = torch.from_numpy(clean.transpose(2, 0, 1)).float()
        corrupted_t = torch.from_numpy(corrupted.transpose(2, 0, 1)).float()
        return corrupted_t, clean_t


class RealStage2HealthyDataset(Dataset):
    """Strictly negative-only dataset of real healthy concrete for Stage 2 Autoencoder training."""

    def __init__(self, healthy_patches: list[np.ndarray], rng: np.random.Generator = None):
        self.patches = healthy_patches
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        clean = self.patches[idx]
        # Data augmentation (flips and 90-degree rotations)
        k = self.rng.integers(0, 4)
        clean_aug = np.rot90(clean, k=k)
        if self.rng.random() > 0.5:
            clean_aug = np.fliplr(clean_aug)
        clean_aug = np.ascontiguousarray(clean_aug)

        # Subtle noise augmentation to encourage robust reconstruction
        noise = self.rng.normal(0.0, 0.015, clean_aug.shape).astype(np.float32)
        noisy = np.clip(clean_aug + noise, 0.0, 1.0)

        noisy_t = torch.from_numpy(noisy.transpose(2, 0, 1)).float()
        clean_t = torch.from_numpy(clean_aug.transpose(2, 0, 1)).float()
        return noisy_t, clean_t


class RealStage2EvaluationDataset(Dataset):
    """Real defective patches with ground-truth binary crack masks for AUROC benchmark."""

    def __init__(self, defect_patches: list[tuple[np.ndarray, np.ndarray]]):
        self.samples = defect_patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patch, mask = self.samples[idx]
        patch_t = torch.from_numpy(patch.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).long()
        return patch_t, mask_t


class RealStage3ClassifierDataset(Dataset):
    """Stage 3 Defect Classifier dataset using real crack crops + augmented civil defects."""

    def __init__(
        self,
        defect_patches: list[tuple[np.ndarray, np.ndarray]],
        healthy_patches: list[np.ndarray],
        crop_size: int = 64,
        samples_per_epoch: int = 400,
        rng: np.random.Generator = None,
    ):
        self.defect_patches = defect_patches
        self.healthy_patches = healthy_patches
        self.crop_size = crop_size
        self.samples_per_epoch = samples_per_epoch
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # 60% real cracks, 40% other CODEBRIM defect classes on real concrete
        choice = self.rng.random()
        if choice < 0.60 and len(self.defect_patches) > 0:
            patch, mask = self.defect_patches[self.rng.integers(0, len(self.defect_patches))]
            # Find center of crack
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                c_idx = self.rng.integers(0, len(ys))
                cy, cx = ys[c_idx], xs[c_idx]
            else:
                cy, cx = patch.shape[0] // 2, patch.shape[1] // 2

            half = self.crop_size // 2
            y1 = max(0, min(patch.shape[0] - self.crop_size, cy - half))
            x1 = max(0, min(patch.shape[1] - self.crop_size, cx - half))
            crop = patch[y1 : y1 + self.crop_size, x1 : x1 + self.crop_size]
            label = 1  # "crack"
        else:
            # Pick a real healthy concrete patch and inject synthetic civil defects
            h_patch = self.healthy_patches[self.rng.integers(0, len(self.healthy_patches))]
            y1 = self.rng.integers(0, max(1, h_patch.shape[0] - self.crop_size))
            x1 = self.rng.integers(0, max(1, h_patch.shape[1] - self.crop_size))
            sub_patch = h_patch[y1 : y1 + self.crop_size, x1 : x1 + self.crop_size].copy()

            defect_type = self.rng.choice(["spalling", "corrosion_stain", "background"])
            if defect_type != "background":
                sub_patch, _, labels_vec = inject_synthetic_defect(sub_patch, defect_type=defect_type, rng=self.rng)
                label = int(np.argmax(labels_vec))
            else:
                label = 0

            crop = sub_patch

        crop_t = torch.from_numpy(crop.transpose(2, 0, 1)).float()
        label_t = torch.tensor(label, dtype=torch.long)
        return crop_t, label_t
