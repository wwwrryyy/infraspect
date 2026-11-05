"""
test_p3_data.py
===============
Unit tests for Phase P3:
- Optical civil corruptions (illumination, droplets, motion blur, glare).
- Sizing guard (H % 4 == 0, W % 4 == 0).
- Negative-only safety check on Stage 2 dataset.
- Stage 3 6-class multi-label taxonomy.
"""

import numpy as np
import pytest
import torch
import os
import tempfile

from models.stage1_denoiser.corruption import (
    apply_illumination_gradient,
    apply_water_droplets,
    apply_vibration_motion_blur,
    apply_specular_glare,
    corrupt_civil_image,
)
from data.datasets import (
    generate_concrete_patch,
    inject_synthetic_defect,
    Stage1DenoiseDataset,
    Stage2HealthyDataset,
    Stage3DefectDataset,
    DEFECT_CLASSES,
)


def test_civil_corruptions_shape_and_bounds():
    rng = np.random.default_rng(123)
    # Test both 2D grayscale and 3D RGB
    for shape in [(128, 128), (128, 128, 3)]:
        clean = rng.uniform(0.2, 0.8, shape).astype(np.float32)

        ill = apply_illumination_gradient(clean, rng=rng)
        assert ill.shape == shape
        assert ill.min() >= 0.0 and ill.max() <= 1.0

        drop = apply_water_droplets(clean, rng=rng, num_droplets=3)
        assert drop.shape == shape
        assert drop.min() >= 0.0 and drop.max() <= 1.0

        blur = apply_vibration_motion_blur(clean, rng=rng, kernel_size=5)
        assert blur.shape == shape
        assert blur.min() >= 0.0 and blur.max() <= 1.0

        glare = apply_specular_glare(clean, rng=rng)
        assert glare.shape == shape
        assert glare.min() >= 0.0 and glare.max() <= 1.0

        corrupted = corrupt_civil_image(clean, rng=rng)
        assert corrupted.shape == shape
        assert corrupted.min() >= 0.0 and corrupted.max() <= 1.0


def test_corruptions_determinism():
    clean = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    res1 = corrupt_civil_image(clean, rng=rng1)
    res2 = corrupt_civil_image(clean, rng=rng2)
    assert np.allclose(res1, res2)


def test_concrete_generator_and_defects():
    rng = np.random.default_rng(42)
    patch = generate_concrete_patch(height=128, width=128, in_channels=3, rng=rng)
    assert patch.shape == (128, 128, 3)
    assert patch.dtype == np.float32
    assert patch.min() >= 0.0 and patch.max() <= 1.0

    # Inject crack
    c_patch, c_mask, c_labels = inject_synthetic_defect(patch, "crack", rng=rng)
    assert c_patch.shape == (128, 128, 3)
    assert c_mask.shape == (128, 128)
    assert np.sum(c_mask > 0) > 0  # Crack pixels present
    assert c_labels[1] == 1        # Crack label active

    # Inject spalling
    s_patch, s_mask, s_labels = inject_synthetic_defect(patch, "spalling", rng=rng)
    assert np.sum(s_mask > 0) > 0
    assert s_labels[2] == 1

    # Inject efflorescence (class 3)
    e_patch, e_mask, e_labels = inject_synthetic_defect(patch, "efflorescence", rng=rng)
    assert np.sum(e_mask > 0) > 0
    assert e_labels[3] == 1
    assert e_patch.min() >= 0.0 and e_patch.max() <= 1.0

    # Inject exposed rebar (class 4)
    b_patch, b_mask, b_labels = inject_synthetic_defect(patch, "exposed_bars", rng=rng)
    assert np.sum(b_mask > 0) > 0
    assert b_labels[4] == 1
    assert b_patch.min() >= 0.0 and b_patch.max() <= 1.0

    # Inject corrosion
    r_patch, r_mask, r_labels = inject_synthetic_defect(patch, "corrosion_stain", rng=rng)
    assert r_labels[5] == 1


def test_stage1_denoise_dataset_divisible_by_4():
    # Pass non-multiple of 4 (e.g. 129x127) -> should automatically snap to 128x124
    ds = Stage1DenoiseDataset(height=129, width=127, in_channels=3, num_synthetic=10)
    assert ds.h % 4 == 0 and ds.w % 4 == 0
    corrupted, clean = ds[0]
    assert corrupted.shape == (3, ds.h, ds.w)
    assert clean.shape == (3, ds.h, ds.w)
    assert isinstance(corrupted, torch.Tensor)


def test_stage2_healthy_safety_guard():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create a fake defective sample directory
        bad_dir = os.path.join(tmp_dir, "cracked_walls")
        os.makedirs(bad_dir)
        with open(os.path.join(bad_dir, "sample_crack_01.jpg"), "w") as f:
            f.write("fake")

        # Initializing Stage2HealthyDataset on directory containing defective keyword must raise ValueError
        with pytest.raises(ValueError) as excinfo:
            Stage2HealthyDataset(image_dir=tmp_dir)
        assert "Stage 2 Negative-Only violation" in str(excinfo.value)


def test_stage3_multi_label_dataset():
    ds = Stage3DefectDataset(num_synthetic=20, in_channels=3)
    crop, labels = ds[0]
    assert crop.shape == (3, 64, 64)
    assert labels.shape == (len(DEFECT_CLASSES),)
    assert labels.dtype == torch.float32
    assert torch.sum(labels) >= 1.0  # At least one active label


def test_legacy_speckle_determinism():
    from models.stage1_denoiser.corruption import corrupt_image

    img = np.random.default_rng(0).uniform(0.1, 0.9, (32, 32)).astype(np.float32)
    # Fixed seed must reproduce identical corruption regardless of global np.random state
    np.random.seed(1)
    a = corrupt_image(img, rng=np.random.default_rng(42))
    np.random.seed(2)
    b = corrupt_image(img, rng=np.random.default_rng(42))
    assert np.array_equal(a, b)
