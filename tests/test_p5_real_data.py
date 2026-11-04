"""
test_p5_real_data.py
====================
Unit tests verifying Option 2 Real Inspection Dataset pipeline, AUROC calculation,
and trained ONNX models on real concrete crack imagery.
"""

import os
import json
import numpy as np
import pytest
import onnxruntime as ort

from data.real_inspection_dataset import (
    extract_real_concrete_patches,
    extract_crack_segmentation_patches,
    extract_all_real_inspection_patches,
    RealStage1DenoisingDataset,
    RealStage2HealthyDataset,
    RealStage2EvaluationDataset,
    RealStage3ClassifierDataset,
)
from train_real_inspection import roc_auc_numpy


def test_roc_auc_numpy_synthetic():
    # Perfect separation
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    auc = roc_auc_numpy(y_true, y_score)
    assert auc == 1.0

    # Inverted separation
    y_score_inv = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    auc_inv = roc_auc_numpy(y_true, y_score_inv)
    assert auc_inv == 0.0


def test_real_patch_extraction():
    healthy_patches, defect_patches = extract_real_concrete_patches(
        crackforest_dir="data/raw/crackforest",
        patch_size=128,
        max_patches=20,
        stride=128,
    )

    assert len(healthy_patches) > 0, "Expected at least 1 healthy patch from CrackForest"
    assert len(defect_patches) > 0, "Expected at least 1 defect patch from CrackForest"

    # Verify healthy patch shape and range
    h_patch = healthy_patches[0]
    assert h_patch.shape == (128, 128, 3)
    assert 0.0 <= np.min(h_patch) and np.max(h_patch) <= 1.0

    # Verify defect patch and mask
    d_patch, d_mask = defect_patches[0]
    assert d_patch.shape == (128, 128, 3)
    assert d_mask.shape == (128, 128)
    assert np.all(np.isin(d_mask, [0, 1]))


def test_crack_segmentation_patch_extraction():
    healthy_patches, defect_patches = extract_crack_segmentation_patches(
        crack_seg_dir="data/raw/crack_segmentation",
        patch_size=128,
        max_patches=20,
    )
    assert len(healthy_patches) > 0, "Expected at least 1 healthy patch from CrackSegmentation"
    assert len(defect_patches) > 0, "Expected at least 1 defect patch from CrackSegmentation"


def test_combined_real_patch_extraction():
    healthy_patches, defect_patches = extract_all_real_inspection_patches(
        patch_size=128,
        max_patches=30,
    )
    assert len(healthy_patches) >= 10
    assert len(defect_patches) >= 5


def test_real_datasets_tensors():
    healthy_patches, defect_patches = extract_real_concrete_patches(max_patches=10)

    ds1 = RealStage1DenoisingDataset(healthy_patches)
    corr, clean = ds1[0]
    assert corr.shape == (3, 128, 128)
    assert clean.shape == (3, 128, 128)

    ds2 = RealStage2HealthyDataset(healthy_patches)
    noisy, target = ds2[0]
    assert noisy.shape == (3, 128, 128)
    assert target.shape == (3, 128, 128)

    ds2_eval = RealStage2EvaluationDataset(defect_patches)
    eval_p, eval_m = ds2_eval[0]
    assert eval_p.shape == (3, 128, 128)
    assert eval_m.shape == (128, 128)

    ds3 = RealStage3ClassifierDataset(defect_patches, healthy_patches, crop_size=64, samples_per_epoch=20)
    crop, label = ds3[0]
    assert crop.shape == (3, 64, 64)
    assert 0 <= label.item() < 6


def test_calibrated_threshold_files():
    for arch in ["cnn", "snn"]:
        calib_path = f"exports/calibration_{arch}.json"
        assert os.path.isfile(calib_path), f"Missing calibration file: {calib_path}"
        with open(calib_path) as f:
            data = json.load(f)
        assert "calibrated_mse_threshold" in data
        assert data["calibrated_mse_threshold"] > 0.0


def test_real_inference_benchmark():
    from deploy_inference import PipelineRunner

    healthy_patches, _ = extract_real_concrete_patches(max_patches=5)
    real_patch_u8 = (healthy_patches[0] * 255.0).astype(np.uint8)

    for arch in ["cnn", "snn"]:
        runner = PipelineRunner(arch=arch, exports_dir="exports")
        timing, defects, vis_map, restored = runner.run_frame(real_patch_u8)
        assert "fps" in timing
        assert "total_ms" in timing
        assert timing["total_ms"] > 0.0
        assert restored.shape == real_patch_u8.shape
        assert vis_map.shape == (real_patch_u8.shape[0], real_patch_u8.shape[1], 3)
