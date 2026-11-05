"""
datasets.py
===========
Civil infrastructure inspection datasets:
- Procedural realistic concrete texture & synthetic defect generator (hairline cracks, spalling, rust stains, efflorescence).
- Stage 1 Denoising Dataset (paired corrupted/clean).
- Stage 2 Negative-Only Anomaly Dataset (strictly negative healthy concrete, with safety checks).
- Stage 3 Multi-Label Defect Classification Dataset (6-class CODEBRIM taxonomy).
"""

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from models.stage1_denoiser.corruption import corrupt_civil_image, load_and_normalize

# CODEBRIM 6-class civil defect taxonomy
DEFECT_CLASSES = [
    "background",        # 0: Healthy concrete
    "crack",             # 1: Hairline / structural crack
    "spalling",          # 2: Concrete spalling / delamination
    "efflorescence",     # 3: White salt leaching / lime streaks
    "exposed_bars",      # 4: Exposed rebar
    "corrosion_stain",   # 5: Red/orange iron oxide staining
]


# ---------------------------------------------------------------------------
# Procedural Realistic Surface & Defect Synthesis
# ---------------------------------------------------------------------------

def generate_concrete_patch(
    height: int = 256,
    width: int = 256,
    in_channels: int = 3,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Generates a realistic multi-scale concrete texture patch in [0, 1] float32."""
    if rng is None:
        rng = np.random.default_rng()

    # Base tonal intensity (gray to warm concrete)
    base_lum = rng.uniform(0.55, 0.72)
    # Multi-frequency low-frequency surface gradient (forms, weathering)
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    f1 = rng.uniform(1.0, 3.0)
    f2 = rng.uniform(2.0, 5.0)
    grad = 0.08 * np.sin(f1 * x_grid / width * np.pi) * np.cos(f2 * y_grid / height * np.pi)

    # Aggregate fine flecks (micro-porosity)
    noise_fine = rng.normal(0.0, 0.035, (height, width))
    # Medium texture
    noise_med = cv2.GaussianBlur(rng.normal(0.0, 0.06, (height, width)), (5, 5), 1.5)

    lum = np.clip(base_lum + grad + noise_fine + noise_med, 0.1, 0.95).astype(np.float32)

    if in_channels == 1:
        return lum
    else:
        # Subtle tint variation (concrete warmth/coolness)
        r_mult = rng.uniform(0.98, 1.03)
        g_mult = rng.uniform(0.97, 1.01)
        b_mult = rng.uniform(0.95, 1.00)
        rgb = np.stack([lum * r_mult, lum * g_mult, lum * b_mult], axis=-1)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def inject_synthetic_defect(
    patch: np.ndarray,
    defect_type: str = "crack",
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Injects a procedural defect into a clean concrete patch.

    Returns:
        (corrupted_patch, pixel_mask, multi_label_one_hot)
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = patch.shape[:2]
    result = patch.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    labels = [0] * len(DEFECT_CLASSES)

    if defect_type == "crack":
        # Multi-segment hairline crack (1-3 px width)
        labels[1] = 1
        num_points = rng.integers(5, 12)
        start_x = rng.integers(int(w * 0.1), int(w * 0.9))
        start_y = rng.integers(int(h * 0.1), int(h * 0.9))
        pts = [(start_x, start_y)]

        curr_x, curr_y = start_x, start_y
        angle = rng.uniform(0, 2 * np.pi)
        step_len = rng.uniform(10, 25)

        for _ in range(num_points):
            angle += rng.uniform(-0.5, 0.5)
            curr_x = int(np.clip(curr_x + step_len * np.cos(angle), 2, w - 3))
            curr_y = int(np.clip(curr_y + step_len * np.sin(angle), 2, h - 3))
            pts.append((curr_x, curr_y))

        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        thickness = int(rng.choice([1, 2]))
        cv2.polylines(mask, [pts_arr], isClosed=False, color=255, thickness=thickness)

        # Dark crack groove with slight edge softening
        crack_mask = (mask > 0)
        darkness = rng.uniform(0.15, 0.35)
        if patch.ndim == 3:
            crack_mask_3d = crack_mask[:, :, np.newaxis]
            result = np.where(crack_mask_3d, result * darkness, result)
        else:
            result = np.where(crack_mask, result * darkness, result)

    elif defect_type == "spalling":
        labels[2] = 1
        # Irregular chipped concrete crater
        cx = int(rng.integers(int(w * 0.25), int(w * 0.75)))
        cy = int(rng.integers(int(h * 0.25), int(h * 0.75)))
        axes = (int(rng.integers(15, 40)), int(rng.integers(15, 40)))
        angle = rng.uniform(0, 180)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

        # Roughen spall boundary
        noise_mat = rng.uniform(0, 255, (h, w)).astype(np.uint8)
        mask = cv2.bitwise_and(mask, mask, mask=(noise_mat > 50).astype(np.uint8))

        spall_mask = (mask > 0)
        if patch.ndim == 3:
            spall_mask_3d = spall_mask[:, :, np.newaxis]
            spall_color = np.array([0.35, 0.32, 0.30], dtype=np.float32)
            result = np.where(spall_mask_3d, spall_color + rng.normal(0, 0.04, result.shape), result)
        else:
            result = np.where(spall_mask, 0.32 + rng.normal(0, 0.04, result.shape), result)

    elif defect_type == "efflorescence":
        labels[3] = 1
        # White calcium carbonate salt leaching / lime streaks
        cx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
        cy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
        axes = (int(rng.integers(15, 45)), int(rng.integers(8, 25)))
        angle = rng.uniform(0, 180)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

        eff_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (9, 9), 3.0)
        salt_noise = rng.normal(0.0, 0.05, (h, w)).astype(np.float32)
        salt_layer = np.clip(eff_mask * 0.40 + salt_noise * eff_mask, 0.0, 0.55)

        if patch.ndim == 3:
            salt_3d = salt_layer[:, :, np.newaxis]
            result = np.clip(result + salt_3d, 0.0, 1.0)
        else:
            result = np.clip(result + salt_layer, 0.0, 1.0)

    elif defect_type == "exposed_bars":
        labels[4] = 1
        # Spalled concrete cavity exposing embedded linear rebar
        cx = int(rng.integers(int(w * 0.3), int(w * 0.7)))
        cy = int(rng.integers(int(h * 0.3), int(h * 0.7)))
        axes = (int(rng.integers(20, 40)), int(rng.integers(20, 40)))
        angle = rng.uniform(0, 180)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 255, -1)

        spall_mask = (mask > 0)
        if patch.ndim == 3:
            cavity_color = np.array([0.28, 0.26, 0.24], dtype=np.float32)
            result = np.where(spall_mask[:, :, np.newaxis], cavity_color, result)
        else:
            result = np.where(spall_mask, 0.26, result)

        # Draw exposed linear rebar rod through the cavity
        bar_mask = np.zeros((h, w), dtype=np.uint8)
        bar_angle = rng.uniform(-45, 45)
        rad = np.deg2rad(bar_angle)
        x1 = int(cx - 35 * np.cos(rad))
        y1 = int(cy - 35 * np.sin(rad))
        x2 = int(cx + 35 * np.cos(rad))
        y2 = int(cy + 35 * np.sin(rad))
        bar_thick = int(rng.choice([3, 4]))
        cv2.line(bar_mask, (x1, y1), (x2, y2), 255, thickness=bar_thick)
        bar_active = (bar_mask > 0) & spall_mask

        if patch.ndim == 3:
            rebar_color = np.array([0.55, 0.28, 0.14], dtype=np.float32)  # Oxidized rusty rebar
            result[bar_active] = rebar_color
        else:
            result[bar_active] = 0.35

    elif defect_type == "corrosion_stain":
        labels[5] = 1
        # Diffuse amber/rust bloom
        cx = int(rng.integers(int(w * 0.2), int(w * 0.8)))
        cy = int(rng.integers(int(h * 0.2), int(h * 0.8)))
        sigma = rng.uniform(15, 45)
        y_g, x_g = np.mgrid[0:h, 0:w]
        dist_sq = (x_g - cx) ** 2 + (y_g - cy) ** 2
        bloom = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
        mask = (bloom > 0.3).astype(np.uint8) * 255

        if patch.ndim == 3:
            # Add warm rust color (boost R, suppress B)
            rust_layer = np.stack([bloom * 0.45, bloom * 0.18, -bloom * 0.25], axis=-1)
            result = np.clip(result + rust_layer, 0.0, 1.0)
        else:
            result = np.clip(result - bloom * 0.2, 0.0, 1.0)

    else:
        # Background healthy
        labels[0] = 1

    result = np.clip(result, 0.0, 1.0).astype(np.float32)
    return result, mask, labels


# ---------------------------------------------------------------------------
# PyTorch Dataset Classes
# ---------------------------------------------------------------------------

class Stage1DenoiseDataset(Dataset):
    """Stage 1 Denoiser Dataset: pairs clean concrete with civil optical corruptions."""

    def __init__(
        self,
        image_dir: str = None,
        num_synthetic: int = 400,
        height: int = 128,
        width: int = 128,
        in_channels: int = 3,
        seed: int = 42,
    ):
        # Enforce divisible by 4
        self.h = (height // 4) * 4
        self.w = (width // 4) * 4
        self.in_channels = in_channels
        self.rng = np.random.default_rng(seed)

        if image_dir and os.path.isdir(image_dir):
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            self.paths = []
            for ext in exts:
                self.paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
            self.synthetic = len(self.paths) == 0
        else:
            self.paths = []
            self.synthetic = True

        self.num_synthetic = num_synthetic

    def __len__(self):
        return len(self.paths) if not self.synthetic else self.num_synthetic

    def __getitem__(self, idx):
        if self.synthetic:
            clean = generate_concrete_patch(
                height=self.h, width=self.w, in_channels=self.in_channels, rng=self.rng
            )
        else:
            clean = load_and_normalize(
                self.paths[idx],
                grayscale=(self.in_channels == 1),
                target_size=(self.w, self.h),
            )

        corrupted = corrupt_civil_image(clean, rng=self.rng)

        # Convert to [C, H, W] tensor
        if clean.ndim == 2:
            clean_t = torch.from_numpy(clean[np.newaxis, :, :])
            corrupted_t = torch.from_numpy(corrupted[np.newaxis, :, :])
        else:
            clean_t = torch.from_numpy(clean.transpose(2, 0, 1))
            corrupted_t = torch.from_numpy(corrupted.transpose(2, 0, 1))

        return corrupted_t, clean_t


class Stage2HealthyDataset(Dataset):
    """Strictly negative-only dataset for Stage 2 Autoencoder anomaly detection.

    Guarantees zero defective samples in training data.
    """

    def __init__(
        self,
        image_dir: str = None,
        num_synthetic: int = 400,
        height: int = 128,
        width: int = 128,
        in_channels: int = 3,
        seed: int = 42,
    ):
        self.h = (height // 4) * 4
        self.w = (width // 4) * 4
        self.in_channels = in_channels
        self.rng = np.random.default_rng(seed)

        if image_dir and os.path.isdir(image_dir):
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            all_files = []
            for ext in exts:
                all_files.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))

            # Safety check: strictly reject any cracked/defective subpaths
            prohibited_keywords = ["crack", "defect", "damage", "spall", "corrosion", "/c/", "_c_"]
            clean_files = []
            for p in all_files:
                p_lower = p.lower()
                if any(kw in p_lower for kw in prohibited_keywords):
                    raise ValueError(
                        f"Stage 2 Negative-Only violation! Found defective sample in training path: {p}"
                    )
                clean_files.append(p)

            self.paths = clean_files
            self.synthetic = len(self.paths) == 0
        else:
            self.paths = []
            self.synthetic = True

        self.num_synthetic = num_synthetic

    def __len__(self):
        return len(self.paths) if not self.synthetic else self.num_synthetic

    def __getitem__(self, idx):
        if self.synthetic:
            clean = generate_concrete_patch(
                height=self.h, width=self.w, in_channels=self.in_channels, rng=self.rng
            )
        else:
            clean = load_and_normalize(
                self.paths[idx],
                grayscale=(self.in_channels == 1),
                target_size=(self.w, self.h),
            )

        # Convert to [C, H, W]
        if clean.ndim == 2:
            clean_t = torch.from_numpy(clean[np.newaxis, :, :])
        else:
            clean_t = torch.from_numpy(clean.transpose(2, 0, 1))

        # Input to autoencoder can have mild sensor noise
        noise = torch.randn_like(clean_t) * 0.03
        noisy_t = torch.clamp(clean_t + noise, 0.0, 1.0)
        return noisy_t, clean_t


class Stage3DefectDataset(Dataset):
    """Stage 3 Multi-Label Defect Classification Dataset on 64x64 crops."""

    def __init__(
        self,
        image_dir: str = None,
        num_synthetic: int = 400,
        in_channels: int = 3,
        seed: int = 42,
    ):
        self.in_channels = in_channels
        self.rng = np.random.default_rng(seed)
        self.crop_size = 64

        if image_dir and os.path.isdir(image_dir):
            exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
            self.paths = []
            for ext in exts:
                self.paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))
            self.synthetic = len(self.paths) == 0
        else:
            self.paths = []
            self.synthetic = True

        self.num_synthetic = num_synthetic

    def __len__(self):
        return len(self.paths) if not self.synthetic else self.num_synthetic

    def __getitem__(self, idx):
        if self.synthetic:
            # 6-class CODEBRIM distribution: background, crack, spalling, efflorescence, exposed_bars, corrosion_stain
            defect_type = self.rng.choice(
                ["background", "crack", "spalling", "efflorescence", "exposed_bars", "corrosion_stain"],
                p=[0.15, 0.30, 0.15, 0.15, 0.10, 0.15],
            )
            base = generate_concrete_patch(
                height=self.crop_size,
                width=self.crop_size,
                in_channels=self.in_channels,
                rng=self.rng,
            )
            crop, _, label_vec = inject_synthetic_defect(
                base, defect_type=defect_type, rng=self.rng
            )
            labels = torch.tensor(label_vec, dtype=torch.float32)
        else:
            crop = load_and_normalize(
                self.paths[idx],
                grayscale=(self.in_channels == 1),
                target_size=(self.crop_size, self.crop_size),
            )
            # Default single label inference from filename / folder structure
            label_vec = [0.0] * len(DEFECT_CLASSES)
            p_lower = self.paths[idx].lower()
            matched = False
            for c_idx, name in enumerate(DEFECT_CLASSES):
                if name != "background" and name in p_lower:
                    label_vec[c_idx] = 1.0
                    matched = True
            if not matched:
                label_vec[0] = 1.0
            labels = torch.tensor(label_vec, dtype=torch.float32)

        if crop.ndim == 2:
            crop_t = torch.from_numpy(crop[np.newaxis, :, :])
        else:
            crop_t = torch.from_numpy(crop.transpose(2, 0, 1))

        return crop_t, labels
