"""
test_gui.py
===========
Unit tests verifying LiveInspectionHUD rendering and state transitions.
"""

import numpy as np
import pytest

from gui_dashboard import LiveInspectionHUD


def test_hud_rendering_headless():
    hud = LiveInspectionHUD(arch="cnn", source="synthetic")

    # Generate test frame
    frame = hud.get_frame()
    assert frame.shape == (256, 256, 3)

    timing, defects, vis_map, restored = hud.runner.run_frame(frame)
    assert "fps" in timing
    assert "total_ms" in timing
    assert restored.shape == frame.shape

    # Render 1280x720 canvas
    canvas = hud.render_hud_dashboard(
        frame_rgb=frame,
        restored_rgb=restored,
        heatmap_jet=vis_map,
        defects=defects,
        timing=timing,
    )

    assert canvas.shape == (720, 1280, 3)
    assert canvas.dtype == np.uint8


def test_hud_arch_toggle():
    hud = LiveInspectionHUD(arch="cnn", source="synthetic")
    assert hud.arch == "cnn"

    hud.toggle_arch()
    assert hud.arch == "snn"

    hud.toggle_arch()
    assert hud.arch == "cnn"


def test_hud_run_headless_loop():
    hud = LiveInspectionHUD(arch="cnn", source="synthetic")
    hud.run(max_frames=3, headless=True)
    assert hud.frame_idx == 3


def test_dearpygui_headless_dashboard():
    import argparse
    args = argparse.Namespace(
        gui="dpg",
        arch="cnn",
        source="synthetic",
        exports_dir="exports",
        gsd_mm_per_px=0.08,
        db_path="civil_inspection_telemetry.db",
        headless=True,
        frames=3,
    )
    from gui_dashboard import run_dearpygui_dashboard
    run_dearpygui_dashboard(args)
