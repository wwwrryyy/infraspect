"""
test_p2_logic.py
================
Unit tests for Phase P2:
- Hairline crack survival (2-pixel crack must survive filtering and yield >= 1 box).
- Isolated noise rejection (single-pixel speckle must yield 0 boxes).
- Aspect-padded 64x64 ROI crop extraction.
- NumPy reference anomaly calculus.
- Civil inspection telemetry SQLite buffering & sync.
"""

import numpy as np
import cv2
import tempfile
import os

from src_python.tactical_logic import TacticalDecisionLogic, compute_pixelwise_mse_numpy
from src_python.sharp_resilience import SHARPOfflineResilienceManager


def test_hairline_crack_survival():
    logic = TacticalDecisionLogic(mse_threshold=0.06, min_crack_length=10)
    heatmap = np.zeros((128, 128), dtype=np.float32)

    # Draw a 2-pixel wide diagonal hairline crack of length ~30 px
    cv2.line(heatmap, (20, 20), (50, 45), 0.25, thickness=2)

    defects, mask = logic.detect_defects(heatmap, gsd_mm_per_px=0.05)
    # Hairline crack must NOT be erased
    assert len(defects) >= 1
    assert defects[0]["score"] >= 0.06
    # Estimated physical width should be ~0.1 mm (CS1)
    assert defects[0]["est_width_mm"] is not None
    assert "CS1" in defects[0]["condition_state"]


def test_isolated_speckle_noise_rejection():
    logic = TacticalDecisionLogic(mse_threshold=0.06, min_area=15, min_crack_length=10)
    heatmap = np.zeros((128, 128), dtype=np.float32)

    # Scatter isolated 1-pixel hot pixels (high MSE, but 0 structural connectivity)
    rng = np.random.default_rng(42)
    hot_x = rng.integers(10, 118, size=25)
    hot_y = rng.integers(10, 118, size=25)
    heatmap[hot_y, hot_x] = 0.50

    defects, mask = logic.detect_defects(heatmap)
    # Isolated speckle noise must be completely filtered out
    assert len(defects) == 0
    assert np.sum(mask > 0) == 0


def test_extract_roi_crops_aspect_padded():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[10:50, 10:20] = [200, 100, 50]  # Narrow tall rectangle (40x10)

    boxes = [[10, 10, 10, 40]]
    crops = TacticalDecisionLogic.extract_roi_crops(img, boxes, crop_size=64)

    assert len(crops) == 1
    assert crops[0].shape == (64, 64, 3)
    assert crops[0].dtype == np.uint8


def test_compute_pixelwise_mse_numpy():
    frame_a = np.ones((64, 64, 3), dtype=np.float32) * 0.8
    frame_b = np.ones((64, 64, 3), dtype=np.float32) * 0.6

    mean_mse, heatmap = compute_pixelwise_mse_numpy(frame_a, frame_b, gaussian_sigma=0.0)
    # (0.8 - 0.6)^2 = 0.04
    assert np.isclose(mean_mse, 0.04, atol=1e-5)
    assert heatmap.shape == (64, 64)
    assert np.allclose(heatmap, 0.04, atol=1e-5)


def test_civil_inspection_telemetry_and_sync():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        mgr = SHARPOfflineResilienceManager(db_path=db_path)
        mgr.auto_sync_running = False

        record = {
            "id": 101,
            "defect_class": "crack",
            "standard": "AASHTO_MBEI_1130",
            "condition_state": "CS2 (Fair / 0.3-1.25mm)",
            "confidence": 0.92,
            "score": 0.085,
            "est_width_mm": 0.45,
            "bbox": [15, 25, 40, 12],
            "pose": {"x": 1.2, "y": 3.4, "z": 0.5},
        }

        mgr.log_inspection_offline(record)
        assert mgr.get_pending_sync_count() == 1

        recent = mgr.get_recent_inspections(limit=5)
        assert len(recent) == 1
        assert recent[0]["defect_class"] == "crack"
        assert recent[0]["condition_state"] == "CS2 (Fair / 0.3-1.25mm)"
        assert recent[0]["est_width_mm"] == 0.45
        assert recent[0]["pose"]["x"] == 1.2
        assert not recent[0]["synced"]

        # Test sync callback
        sync_called = []
        def my_sync(records):
            sync_called.append(len(records))
            return True

        mgr.sync_callback = my_sync
        mgr.is_connected = True
        mgr._synchronize_logs()

        assert len(sync_called) == 1
        assert mgr.get_pending_sync_count() == 0
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)
