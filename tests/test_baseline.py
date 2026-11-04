"""
test_baseline.py
================
Baseline unit tests pinning pre-existing Python components before refactoring:
- TacticalDecisionLogic (IoU, NMS)
- SHARPOfflineResilienceManager (SQLite schema, offline threat buffering)
- corruption.py (Rayleigh, Gamma, corrupt_image bounds and shapes)
"""

import os
import tempfile
import numpy as np
import pytest

from src_python.tactical_logic import TacticalDecisionLogic
from src_python.sharp_resilience import SHARPOfflineResilienceManager
from models.stage1_denoiser.corruption import (
    apply_rayleigh_speckle,
    apply_gamma_speckle,
    corrupt_image,
)


def test_iou_computation():
    logic = TacticalDecisionLogic()
    # Box: [x, y, w, h]
    box1 = [10, 10, 20, 20]
    box2 = [10, 10, 20, 20]
    # Identical boxes should have IoU == 1.0
    assert pytest.approx(logic.compute_iou(box1, box2), rel=1e-3) == 1.0

    # Non-overlapping boxes should have IoU == 0.0
    box3 = [50, 50, 10, 10]
    assert logic.compute_iou(box1, box3) == 0.0

    # Half overlap: box1 area=400, box4 area=400, intersection=200, union=600 -> 1/3
    box4 = [20, 10, 20, 20]
    assert pytest.approx(logic.compute_iou(box1, box4), rel=1e-3) == 1.0 / 3.0


def test_nms_suppression():
    logic = TacticalDecisionLogic(nms_iou_threshold=0.3)
    boxes = [
        [10, 10, 20, 20],
        [12, 12, 20, 20],  # Heavy overlap with box 0
        [100, 100, 30, 30],  # Disjoint box
    ]
    scores = [0.9, 0.7, 0.85]

    keep_boxes, keep_scores = logic.apply_nms(boxes, scores)
    # The overlapping box with lower score (0.7) should be suppressed
    assert len(keep_boxes) == 2
    assert keep_scores[0] == 0.9
    assert keep_scores[1] == 0.85


def test_sharp_resilience_offline_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        mgr = SHARPOfflineResilienceManager(db_path=db_path)
        mgr.auto_sync_running = False  # disable background monitor thread for deterministic test

        assert mgr.get_pending_sync_count() == 0

        threat = {
            "id": 1,
            "type": "TEST_ALERT",
            "confidence": 0.95,
            "score": 0.12,
            "bbox": [10, 20, 30, 40],
        }
        mgr.log_threat_offline(threat)

        assert mgr.get_pending_sync_count() == 1
        logs = mgr.get_recent_logs(limit=5)
        assert len(logs) == 1
        assert logs[0]["type"] == "TEST_ALERT"
        assert logs[0]["bbox"] == [10, 20, 30, 40]
        assert not logs[0]["synced"]
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


def test_speckle_corruptions():
    rng = np.random.default_rng(42)
    clean_2d = rng.uniform(0.1, 0.9, (64, 64)).astype(np.float32)

    rayleigh = apply_rayleigh_speckle(clean_2d, scale=0.3)
    assert rayleigh.shape == (64, 64)
    assert rayleigh.dtype == np.float32
    assert rayleigh.min() >= 0.0 and rayleigh.max() <= 1.0

    gamma = apply_gamma_speckle(clean_2d, num_looks=3.0)
    assert gamma.shape == (64, 64)
    assert gamma.dtype == np.float32
    assert gamma.min() >= 0.0 and gamma.max() <= 1.0

    corrupted = corrupt_image(clean_2d, rng=rng)
    assert corrupted.shape == (64, 64)
    assert corrupted.dtype == np.float32
    assert corrupted.min() >= 0.0 and corrupted.max() <= 1.0
