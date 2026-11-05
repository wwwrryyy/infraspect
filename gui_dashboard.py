"""
gui_dashboard.py
================
Real-Time Interactive Civil Infrastructure Inspection GUI & Telemetry Dashboard.

Supports two display frontends:
1. OpenCV HUD Mode (--gui cv, default): Multi-panel 2x2 heads-up display with real-time controls,
   color-coded AASHTO severity overlays, live latency gauges, and instant backbone switching.
2. DearPyGui Mode (--gui dpg): Desktop GUI with dedicated control panels, texture viewports,
   and telemetry logs.

Controls:
  [C] / [Tab] : Toggle Backbone (SNN T=4 <-> Continuous CNN T=1)
  [+] / [-]   : Increase / Decrease Anomaly Detection Sensitivity (MSE Threshold)
  [G]         : Toggle GSD Physical Crack Width (mm) Estimation
  [S]         : Capture Frame & Defect Snapshot
  [Space]     : Pause / Resume Stream
  [Q] / [Esc] : Exit
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Environment guards
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src_python"))

from deploy_inference import PipelineRunner
from data.datasets import generate_concrete_patch, inject_synthetic_defect
from models.stage1_denoiser.corruption import corrupt_civil_image
from src_python.sharp_resilience import SHARPOfflineResilienceManager


class LiveInspectionHUD:
    """High-performance 2x2 Multi-View Inspection HUD rendered via OpenCV."""

    def __init__(
        self,
        arch: str = "cnn",
        source: str = "synthetic",
        exports_dir: str = "exports",
        gsd_mm_per_px: float = 0.08,
        db_path: str = "civil_inspection_telemetry.db",
    ):
        self.arch = arch.lower()
        self.source = source
        self.gsd_mm_per_px = gsd_mm_per_px
        self.exports_dir = exports_dir
        self.db_path = db_path

        # Active runner
        self.runner = PipelineRunner(arch=self.arch, exports_dir=self.exports_dir, gsd_mm_per_px=self.gsd_mm_per_px)
        self.resilience_mgr = SHARPOfflineResilienceManager(db_path=self.db_path)
        self.resilience_mgr.auto_sync_running = False

        # State
        self.is_paused = False
        self.frame_idx = 0
        self.rng = np.random.default_rng(101)
        self.history_fps = []

        # Open video source if specified
        self.use_cpp_stream = False
        if self.source in ["camera", "0"] or (self.source != "synthetic" and os.path.isfile(self.source)):
            if self.runner.cpp_engine is not None:
                self.runner.cpp_engine.start_stream(self.source)
                self.use_cpp_stream = True
                self.cap = None
            elif self.source in ["camera", "0"]:
                self.cap = cv2.VideoCapture(0)
            else:
                self.cap = cv2.VideoCapture(self.source)
        else:
            self.cap = None

    def toggle_arch(self):
        new_arch = "snn" if self.arch == "cnn" else "cnn"
        print(f"[*] Switching Pipeline Backbone: {self.arch.upper()} -> {new_arch.upper()} ...")
        self.arch = new_arch
        self.runner = PipelineRunner(arch=self.arch, exports_dir=self.exports_dir, gsd_mm_per_px=self.gsd_mm_per_px)

    def get_frame(self) -> np.ndarray:
        if self.use_cpp_stream and self.runner.cpp_engine is not None:
            has_frame, frame_rgb = self.runner.cpp_engine.grab_frame()
            if has_frame and frame_rgb is not None:
                return frame_rgb

        if self.cap is not None:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procedural live concrete inspection feed
        patch = generate_concrete_patch(height=256, width=256, in_channels=3, rng=self.rng)
        # Inject defect every 2nd or 3rd frame
        if self.frame_idx % 2 == 0:
            defect = "crack" if self.frame_idx % 4 == 0 else "spalling"
            patch, _, _ = inject_synthetic_defect(patch, defect_type=defect, rng=self.rng)
        corrupted = corrupt_civil_image(patch, rng=self.rng)
        return (corrupted * 255.0).astype(np.uint8)

    def draw_defect_overlays(self, frame_bgr: np.ndarray, defects: list[dict]) -> np.ndarray:
        vis = frame_bgr.copy()
        for d in defects:
            x, y, w, h = d["bbox"]
            cs = d.get("condition_state", "")
            cls_name = d.get("defect_class", d.get("type", "DEFECT"))

            # Color coding: Green (CS1), Yellow (CS2), Orange (CS3), Red (CS4)
            if "CS1" in cs:
                box_color = (0, 220, 0)      # Green
            elif "CS2" in cs:
                box_color = (0, 220, 220)    # Yellow
            elif "CS3" in cs:
                box_color = (0, 140, 255)    # Orange
            else:
                box_color = (0, 0, 255)      # Red / Alert

            cv2.rectangle(vis, (x, y), (x + w, y + h), box_color, 2)

            # Label text
            width_str = f" | {d['est_width_mm']:.2f}mm" if d.get("est_width_mm") else ""
            label = f"{cls_name} ({cs[:3]}{width_str})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)

            cv2.rectangle(vis, (x, max(0, y - th - 6)), (x + tw + 6, max(0, y)), box_color, -1)
            cv2.putText(vis, label, (x + 3, max(10, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return vis

    def record_defects(self, defects: list[dict]) -> None:
        """Injects standardized pose telemetry and logs batch to resilience DB."""
        now_ts = time.time()
        for d in defects:
            d["timestamp"] = now_ts
            d["pose"] = {
                "frame_idx": self.frame_idx,
                "x": self.frame_idx * 0.05,
                "y": 0.0,
                "z": 1.5,
            }
        self.resilience_mgr.log_inspections_batch(defects)

    def render_hud_dashboard(
        self,
        frame_rgb: np.ndarray,
        restored_rgb: np.ndarray,
        heatmap_jet: np.ndarray,
        defects: list[dict],
        timing: dict,
    ) -> np.ndarray:
        """Composes 4 panels into a high-res 1280x720 inspection heads-up display."""
        hud_w, hud_h = 1280, 720
        canvas = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)

        # Panel dimensions (580x310 each)
        pw, ph = 580, 310

        # Panel 1: Live Input + Overlays
        p1 = cv2.resize(self.draw_defect_overlays(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), defects), (pw, ph))
        # Panel 2: Stage 1 Restored Frame
        p2 = cv2.resize(cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2BGR), (pw, ph))
        # Panel 3: Stage 2 Anomaly Heatmap
        p3 = cv2.resize(heatmap_jet, (pw, ph))
        # Panel 4: Telemetry Analytics HUD
        p4 = np.zeros((ph, pw, 3), dtype=np.uint8)
        p4[:] = (24, 24, 28)  # Dark slate background

        # Populate Analytics in Panel 4
        fps = timing["fps"]
        self.history_fps.append(fps)
        if len(self.history_fps) > 30:
            self.history_fps.pop(0)
        avg_fps = float(np.mean(self.history_fps))

        cv2.putText(p4, "EDGE TELEMETRY & DIAGNOSTICS", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.line(p4, (15, 35), (pw - 15, 35), (80, 80, 90), 1)

        # Backbone indicator
        arch_tag = f"BACKBONE: {self.arch.upper()} (" + ("T=4 Unrolled" if self.arch == "snn" else "T=1 Single-Pass") + ")"
        tag_color = (120, 230, 100) if self.arch == "cnn" else (100, 180, 255)
        cv2.putText(p4, arch_tag, (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, tag_color, 2)

        # Latency breakdown
        cv2.putText(p4, f"Total Latency:   {timing['total_ms']:6.2f} ms", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(p4, f"  Stage 1 (Restor): {timing['s1_ms']:6.2f} ms", (15, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1)
        cv2.putText(p4, f"  Stage 2 (Anomaly): {timing['s2_ms']:6.2f} ms", (15, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1)
        cv2.putText(p4, f"  C++ MSE Calculus: {timing['calc_ms']:6.2f} ms", (15, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1)
        cv2.putText(p4, f"  Stage 3 Classifier: {timing['s3_ms']:6.2f} ms", (15, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 170), 1)

        # FPS meter
        cv2.putText(p4, f"Sustained FPS:   {avg_fps:6.1f} FPS", (15, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        prd_status = "PRD: MET (>=10 FPS)" if avg_fps >= 10.0 else "PRD: SUB-10 FPS"
        cv2.putText(p4, prd_status, (15, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 0), 1)

        # SNN vs CNN Comparative Baseline Profiles
        cv2.putText(p4, "COMPARATIVE PROFILES (SNN vs CNN):", (15, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 210, 220), 1)
        snn_marker = ">> " if self.arch == "snn" else "   "
        cnn_marker = ">> " if self.arch == "cnn" else "   "
        cv2.putText(p4, f"{snn_marker}SNN (T=4): ~14.0 ms | 71.3 FPS | 0.85 beta", (15, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100, 180, 255) if self.arch == "snn" else (130, 130, 140), 1)
        cv2.putText(p4, f"{cnn_marker}CNN (T=1):  ~3.9 ms | 254.9 FPS | Single-Pass", (15, 276), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 230, 100) if self.arch == "cnn" else (130, 130, 140), 1)

        # Telemetry & Sync State
        cv2.putText(p4, f"SHARP Buffer: {self.resilience_mgr.get_pending_sync_count()} alerts (SQLite)", (15, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Title headers for panels
        def add_panel_header(panel, title):
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 26), (20, 20, 24), -1)
            cv2.putText(panel, title, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 240), 1, cv2.LINE_AA)
            return panel

        p1 = add_panel_header(p1, "1. LIVE INSPECTION & AASHTO OVERLAYS")
        p2 = add_panel_header(p2, "2. STAGE 1 OPTICAL DENOISER / RESTORATION")
        p3 = add_panel_header(p3, "3. STAGE 2 ANOMALY MSE HEATMAP (C++ ENGINE)")
        p4 = add_panel_header(p4, "4. REAL-TIME TELEMETRY & EDGE HEALTH")

        # Position panels onto 1280x720 canvas
        canvas[30:340, 40:620] = p1
        canvas[30:340, 660:1240] = p2
        canvas[370:680, 40:620] = p3
        canvas[370:680, 660:1240] = p4

        # Top Bar & Help
        cv2.rectangle(canvas, (0, 0), (hud_w, 28), (14, 14, 18), -1)
        title_str = "INFRASPECT: Autonomous Civil Infrastructure & Structural Health Monitoring"
        cv2.putText(canvas, title_str, (40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        help_str = "[C/Tab]: Switch Backbone | [+/-]: Threshold | [G]: GSD mm | [Space]: Pause | [Q]: Quit"
        cv2.putText(canvas, help_str, (hud_w - 680, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 190), 1)

        return canvas

    def run(self, max_frames: int | None = None, headless: bool = False):
        window_name = "Infraspect - Civil Infrastructure Inspection HUD"
        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        print(f"[+] Launching Live HUD [{self.arch.upper()}] ...")
        print("    Keys: [C] Toggle SNN/CNN | [+/-] Sensitivity | [G] Toggle GSD | [Q] Quit")

        while True:
            if max_frames is not None and self.frame_idx >= max_frames:
                break

            if not self.is_paused:
                frame_rgb = self.get_frame()
                self.frame_idx += 1

                timing, defects, vis_map, restored_hwc = self.runner.run_frame(frame_rgb)

                # Log defects with standardized pose schema in a single batch transaction
                self.record_defects(defects)

                hud_canvas = self.render_hud_dashboard(
                    frame_rgb=frame_rgb,
                    restored_rgb=restored_hwc,
                    heatmap_jet=vis_map,
                    defects=defects,
                    timing=timing,
                )

                if headless:
                    print(f"    Frame {self.frame_idx}: {timing['fps']:.1f} FPS, {timing['total_ms']:.2f} ms, {len(defects)} defect(s)")

            if not headless:
                cv2.imshow(window_name, hud_canvas)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q"), 27]:
                    break
                elif key in [ord("c"), ord("C"), 9]:  # Tab or C
                    self.toggle_arch()
                elif key in [ord("+"), ord("=")]:
                    # Increasing sensitivity requires lower anomaly threshold
                    new_th = max(0.001, round(self.runner.get_mse_threshold() - 0.005, 4))
                    self.runner.set_mse_threshold(new_th)
                    print(f"[+] Sensitivity INCREASED: MSE Threshold = {new_th}")
                elif key in [ord("-"), ord("_")]:
                    # Decreasing sensitivity requires higher anomaly threshold
                    new_th = round(self.runner.get_mse_threshold() + 0.005, 4)
                    self.runner.set_mse_threshold(new_th)
                    print(f"[-] Sensitivity DECREASED: MSE Threshold = {new_th}")
                elif key in [ord("g"), ord("G")]:
                    self.gsd_mm_per_px = None if self.gsd_mm_per_px is not None else 0.08
                    self.runner.gsd_mm_per_px = self.gsd_mm_per_px
                    print(f"[*] GSD mm Estimation: {'ON (0.08 mm/px)' if self.gsd_mm_per_px else 'OFF (Pixel Mode)'}")
                elif key == ord(" "):
                    self.is_paused = not self.is_paused
                    print(f"[*] {'PAUSED' if self.is_paused else 'RESUMED'}")
                elif key in [ord("s"), ord("S")]:
                    snap_name = f"snapshot_frame_{self.frame_idx}_{self.arch}.png"
                    cv2.imwrite(snap_name, hud_canvas)
                    print(f"[+] Snapshot saved: {snap_name}")

        if not headless:
            cv2.destroyAllWindows()
        if self.use_cpp_stream and self.runner.cpp_engine is not None:
            self.runner.cpp_engine.stop_stream()
        if self.cap is not None:
            self.cap.release()
        print("[+] HUD closed cleanly.")


# ---------------------------------------------------------------------------
# DearPyGui Native Desktop Mode
# ---------------------------------------------------------------------------

def run_dearpygui_dashboard(args):
    """DearPyGui GPU-accelerated desktop dashboard."""
    hud = LiveInspectionHUD(
        arch=args.arch,
        source=args.source,
        exports_dir=args.exports_dir,
        gsd_mm_per_px=args.gsd_mm_per_px,
        db_path=args.db_path,
    )

    if args.headless:
        print("[*] Running DearPyGui dashboard in headless mode...")
        hud.run(max_frames=args.frames or 5, headless=True)
        return

    import dearpygui.dearpygui as dpg

    dpg.create_context()
    dpg.create_viewport(title="Infraspect Civil Inspection Dashboard (DearPyGui)", width=1280, height=760)

    tex_w, tex_h = 256, 256
    dummy_data = np.zeros((tex_h, tex_w, 4), dtype=np.float32).flatten()
    with dpg.texture_registry(show=False):
        dpg.add_raw_texture(tex_w, tex_h, dummy_data, format=dpg.mvFormat_Float_rgba, tag="tex_raw")
        dpg.add_raw_texture(tex_w, tex_h, dummy_data, format=dpg.mvFormat_Float_rgba, tag="tex_restored")
        dpg.add_raw_texture(tex_w, tex_h, dummy_data, format=dpg.mvFormat_Float_rgba, tag="tex_heatmap")

    def on_slider_change(sender, app_data):
        hud.runner.set_mse_threshold(float(app_data))

    def on_toggle_arch():
        hud.toggle_arch()
        dpg.set_value("lbl_arch", f"Backbone: {hud.arch.upper()}")

    def on_snapshot():
        snap_name = f"snapshot_frame_{hud.frame_idx}_{hud.arch}.png"
        print(f"[+] Snapshot requested: {snap_name}")

    with dpg.window(label="Control Panel & Telemetry", width=360, height=720, pos=(900, 10), no_close=True):
        dpg.add_text("INFRASPECT CONTROL HUD", color=(100, 200, 255))
        dpg.add_separator()

        dpg.add_text(f"Backbone: {hud.arch.upper()}", tag="lbl_arch", color=(120, 240, 120))
        dpg.add_button(label="Switch Backbone (SNN <-> CNN)", callback=on_toggle_arch)
        dpg.add_button(label="Capture Snapshot", callback=on_snapshot)
        dpg.add_separator()

        dpg.add_text("FPS: 0.0", tag="lbl_fps", color=(0, 255, 200))
        dpg.add_text("Latency: 0.0 ms", tag="lbl_latency")
        dpg.add_separator()

        dpg.add_text("Sensitivity (MSE Threshold):")
        dpg.add_slider_float(
            label="Threshold",
            default_value=hud.runner.get_mse_threshold(),
            min_value=0.001,
            max_value=0.20,
            tag="slider_th",
            callback=on_slider_change,
        )
        dpg.add_separator()

        dpg.add_text("Recent Detections:")
        with dpg.table(header_row=True, tag="table_defects", row_background=True, borders_innerH=True, borders_outerH=True):
            dpg.add_table_column(label="Class")
            dpg.add_table_column(label="State")
            dpg.add_table_column(label="Width")

    with dpg.window(label="Multi-View Inspection Viewport", width=880, height=720, pos=(10, 10), no_close=True):
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("1. Raw Inspection Feed (With Overlays)")
                dpg.add_image("tex_raw", width=420, height=310)
            with dpg.group():
                dpg.add_text("2. Stage 1 Restored Frame")
                dpg.add_image("tex_restored", width=420, height=310)

        dpg.add_separator()
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("3. Stage 2 Anomaly Heatmap (C++)")
                dpg.add_image("tex_heatmap", width=420, height=310)

    dpg.setup_dearpygui()
    dpg.show_viewport()

    def to_rgba(img_bgr):
        h, w = img_bgr.shape[:2]
        if (w, h) != (tex_w, tex_h):
            img_bgr = cv2.resize(img_bgr, (tex_w, tex_h))
        rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGBA).astype(np.float32) / 255.0
        return rgba.flatten()

    print("[+] DearPyGui Viewport active.")

    while dpg.is_dearpygui_running():
        frame_rgb = hud.get_frame()
        hud.frame_idx += 1

        timing, defects, vis_map, restored_hwc = hud.runner.run_frame(frame_rgb)

        # Log defects to SQLite in single batch transaction
        hud.record_defects(defects)

        # Populate DearPyGui defect table with active detections
        if dpg.does_item_exist("table_defects"):
            children = dpg.get_item_children("table_defects", 1) or []
            for child in children:
                dpg.delete_item(child)

            for d in defects[:5]:
                cls_name = d.get("defect_class", d.get("type", "defect")).upper()
                cs_state = d.get("condition_state", "")[:3] if d.get("condition_state") else "N/A"
                width_str = f"{d['est_width_mm']:.2f} mm" if d.get("est_width_mm") is not None else f"{d.get('est_width_px', 0.0):.1f} px"
                with dpg.table_row(parent="table_defects"):
                    dpg.add_text(cls_name)
                    dpg.add_text(cs_state)
                    dpg.add_text(width_str)

        overlay_bgr = hud.draw_defect_overlays(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), defects)

        dpg.set_value("tex_raw", to_rgba(overlay_bgr))
        dpg.set_value("tex_restored", to_rgba(cv2.cvtColor(restored_hwc, cv2.COLOR_RGB2BGR)))
        dpg.set_value("tex_heatmap", to_rgba(vis_map))

        dpg.set_value("lbl_fps", f"FPS: {timing['fps']:.1f}")
        dpg.set_value("lbl_latency", f"Latency: {timing['total_ms']:.2f} ms")

        dpg.render_dearpygui_frame()

    dpg.destroy_context()
    print("[+] DearPyGui closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Infraspect Real-Time Interactive Inspection GUI")
    parser.add_argument("--gui", type=str, default="cv", choices=["cv", "dpg"], help="GUI frontend: 'cv' (OpenCV HUD) or 'dpg' (DearPyGui)")
    parser.add_argument("--arch", type=str, default="cnn", choices=["snn", "cnn"], help="Initial backbone architecture")
    parser.add_argument("--source", type=str, default="synthetic", help="Video source: 'synthetic', 'camera', or video file path")
    parser.add_argument("--exports-dir", type=str, default="exports")
    parser.add_argument("--gsd-mm-per-px", type=float, default=0.08)
    parser.add_argument("--db-path", type=str, default="civil_inspection_telemetry.db")
    parser.add_argument("--headless", action="store_true", help="Run headlessly without opening display window")
    parser.add_argument("--frames", type=int, default=None, help="Maximum number of frames to run")
    args = parser.parse_args()

    if args.gui == "dpg":
        run_dearpygui_dashboard(args)
    else:
        hud = LiveInspectionHUD(
            arch=args.arch,
            source=args.source,
            exports_dir=args.exports_dir,
            gsd_mm_per_px=args.gsd_mm_per_px,
            db_path=args.db_path,
        )
        hud.run(max_frames=args.frames, headless=args.headless)


if __name__ == "__main__":
    main()
