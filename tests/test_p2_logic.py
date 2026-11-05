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


def test_hairline_1px_crack_detection():
    # 1-pixel wide cracks have cv2.contourArea == 0.0 but length/pixel_count >= 10
    logic = TacticalDecisionLogic(mse_threshold=0.06, min_crack_length=10)
    heatmap = np.zeros((128, 128), dtype=np.float32)

    # Draw a 1-pixel wide diagonal hairline crack
    cv2.line(heatmap, (20, 20), (50, 45), 0.30, thickness=1)

    defects, mask = logic.detect_defects(heatmap, gsd_mm_per_px=0.05)
    assert len(defects) >= 1
    assert defects[0]["score"] >= 0.06
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


def test_resolution_scaling_gsd_invariance():
    from deploy_inference import PipelineRunner

    # Compare 128x128 baseline against 256x256 scaled camera frame of the same physical structure
    runner_128 = PipelineRunner(arch="cnn", gsd_mm_per_px=0.10)
    runner_256 = PipelineRunner(arch="cnn", gsd_mm_per_px=0.05)

    img_128 = np.ones((128, 128, 3), dtype=np.uint8) * 128
    cv2.line(img_128, (30, 30), (70, 70), (0, 0, 0), thickness=2)

    img_256 = np.ones((256, 256, 3), dtype=np.uint8) * 128
    cv2.line(img_256, (60, 60), (140, 140), (0, 0, 0), thickness=4)

    timing_256, defects_256, vis_map, restored = runner_256.run_frame(img_256)
    timing_128, defects_128, _, _ = runner_128.run_frame(img_128)

    # In 256x256, visual heatmap and restored frame should match input resolution
    assert vis_map.shape[:2] == (256, 256)
    assert restored.shape[:2] == (256, 256)
    assert timing_256["fps"] > 0

    assert len(defects_128) > 0
    assert len(defects_256) > 0

    # Physical crack width and AASHTO condition state must match across resolutions
    d128 = defects_128[0]
    d256 = defects_256[0]
    assert d128["condition_state"] == d256["condition_state"]
    assert abs(d128["est_width_mm"] - d256["est_width_mm"]) < 1.0

    # Rescaled bounding box must stay within 256x256 bounds
    x, y, w, h = d256["bbox"]
    assert 0 <= x < 256 and 0 <= y < 256
    assert x + w <= 256 and y + h <= 256


def test_batch_inspections_logging(tmp_path):
    db_file = tmp_path / "batch_test.db"
    mgr = SHARPOfflineResilienceManager(db_path=str(db_file))
    mgr.auto_sync_running = False

    records = [
        {
            "id": i + 1,
            "defect_class": "crack" if i % 2 == 0 else "spalling",
            "standard": "AASHTO_MBEI_1130",
            "condition_state": "CS2",
            "confidence": 0.88,
            "score": 0.05,
            "est_width_mm": 0.5,
            "bbox": [10, 10, 20, 20],
            "pose": {"frame_idx": i, "x": i * 0.1, "y": 0.0, "z": 1.5},
        }
        for i in range(10)
    ]

    # Batch logging must insert all records in one atomic transaction
    mgr.log_inspections_batch(records)
    assert mgr.get_pending_sync_count() == 10

    recent = mgr.get_recent_inspections(limit=15)
    assert len(recent) == 10
    assert recent[0]["pose"]["frame_idx"] == 9



def test_anisotropic_scale_crack_width_is_orientation_aware():
    logic = TacticalDecisionLogic(mse_threshold=0.06, min_crack_length=10)

    horiz = np.zeros((128, 128), dtype=np.float32)
    cv2.line(horiz, (10, 64), (110, 64), 0.5, thickness=3)
    vert = np.zeros((128, 128), dtype=np.float32)
    cv2.line(vert, (64, 10), (64, 110), 0.5, thickness=3)

    # Wide source frame: x is stretched 4x relative to y during the square resize
    h_def, _ = logic.detect_defects(horiz, gsd_mm_per_px=0.1, scale_xy=(4.0, 1.0))
    v_def, _ = logic.detect_defects(vert, gsd_mm_per_px=0.1, scale_xy=(4.0, 1.0))
    h_iso, _ = logic.detect_defects(horiz, gsd_mm_per_px=0.1)

    # Horizontal crack width is measured along y (unscaled); vertical along x (4x)
    assert abs(h_def[0]["est_width_mm"] - h_iso[0]["est_width_mm"]) < 0.05
    assert v_def[0]["est_width_mm"] > 3.0 * h_def[0]["est_width_mm"]


def test_scoped_sync_preserves_concurrent_inserts(tmp_path):
    db_file = tmp_path / "sync_test.db"
    holder = {}

    def callback(rows):
        # Simulate a record arriving from the main thread while the sync is in flight
        holder["mgr"].log_inspection_offline({"id": 99, "defect_class": "crack", "bbox": [1, 2, 3, 4]})
        return True

    mgr = SHARPOfflineResilienceManager(db_path=str(db_file), sync_callback=callback)
    holder["mgr"] = mgr
    mgr.stop()

    mgr.log_inspection_offline({"id": 1, "defect_class": "crack", "bbox": [1, 2, 3, 4]})
    mgr._synchronize_logs()

    # The late insert must remain pending for the next cycle
    assert mgr.get_pending_sync_count() == 1
    assert mgr.get_recent_inspections(limit=1)[0]["synced"] is False


def test_extract_roi_crops_guarantees_length_invariant_and_clamps_boundary_boxes():
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    boxes = [
        [0, 0, 10, 10],            # Top-left corner
        [120, 120, 20, 20],        # Bottom-right overflowing border
        [150, 150, 10, 10],        # Coordinates beyond image bounds
        [50, 50, 0, 0],            # Zero dimension
        [64, 64, 20, 20],          # Normal interior box
    ]

    crops = TacticalDecisionLogic.extract_roi_crops(img, boxes, crop_size=64)
    # Crucial invariant: exactly 1 crop per input box, preventing downstream IndexError
    assert len(crops) == len(boxes)
    for c in crops:
        assert c.shape == (64, 64, 3)
        assert c.dtype == np.uint8
