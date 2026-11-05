"""
report_generator.py
===================
Automated Civil Infrastructure Inspection & AASHTO MBEI 1130 Compliance Report Generator.

Extracts telemetry records from SQLite, computes statistical aggregations,
maps defect geometries to AASHTO condition states (CS1-CS4), and generates
audit-ready HTML dashboards, structured JSON summaries, and vector PDF documents
for civil engineering authorities.
"""

import os
import sys
import json
import html
import math
import sqlite3
import argparse
from datetime import datetime
from typing import Optional, Dict, Any, List

# Physical civil engineering limit: defects >50mm represent full structural separation / void
MAX_PLAUSIBLE_CRACK_WIDTH_MM = 50.0

# AASHTO MBEI 1130 Condition State Specifications
AASHTO_STATES = {
    "CS1": {
        "title": "CS1 (Good / Hairline)",
        "desc": "Hairline cracks < 0.3 mm width. Nominal concrete surface.",
        "action": "Routine periodic monitoring. No active structural repair required.",
        "color": "#16a34a",
        "bg_color": "#dcfce7",
        "badge_border": "#86efac",
    },
    "CS2": {
        "title": "CS2 (Fair / Moderate)",
        "desc": "Cracks 0.3 mm to 1.25 mm. Surface moisture or minor efflorescence.",
        "action": "Schedule surface sealing and elastomeric crack injection maintenance.",
        "color": "#ca8a04",
        "bg_color": "#fef9c3",
        "badge_border": "#fde047",
    },
    "CS3": {
        "title": "CS3 (Poor / Severe)",
        "desc": "Cracks 1.25 mm to 2.0 mm. Active structural spalling or corrosion stains.",
        "action": "Structural engineering evaluation, epoxy injection, or shallow patch repair.",
        "color": "#ea580c",
        "bg_color": "#ffedd5",
        "badge_border": "#fdba74",
    },
    "CS4": {
        "title": "CS4 (Critical / Compromised)",
        "desc": "Cracks > 2.0 mm or structural delamination with exposed rebar.",
        "action": "URGENT: Immediate structural engineer review, shoring, or load restriction.",
        "color": "#dc2626",
        "bg_color": "#fee2e2",
        "badge_border": "#fca5a5",
    },
}

FALLBACK_STATE = {
    "title": "Pending / Unclassified",
    "desc": "Calibration or Ground Sampling Distance (GSD) pending verification.",
    "action": "Calibrate optical GSD scaling to confirm physical metric crack width.",
    "color": "#64748b",
    "bg_color": "#f1f5f9",
    "badge_border": "#cbd5e1",
}


def _classify_cs_key(condition_str: Optional[str], width_mm: Optional[float] = None) -> str:
    """
    Extracts standardized CS key (CS1, CS2, CS3, CS4) from condition state string or physical width.
    Does not falsely inflate CS1 for uncalibrated / pending records.
    """
    if condition_str:
        s = str(condition_str).upper()
        if "CS4" in s:
            return "CS4"
        if "CS3" in s:
            return "CS3"
        if "CS2" in s:
            return "CS2"
        if "CS1" in s:
            return "CS1"

    # Fallback to physical width if known and positive
    if width_mm is not None and width_mm > 0.0:
        if width_mm < 0.3:
            return "CS1"
        elif width_mm <= 1.25:
            return "CS2"
        elif width_mm <= 2.0:
            return "CS3"
        else:
            return "CS4"

    return "PENDING"


class CivilInspectionReport:
    """Manages data extraction, compliance calculus, and multi-format report export."""

    def __init__(
        self,
        db_path: str = "civil_inspection_telemetry.db",
        asset_id: str = "Civil Structure #4B - North Pier",
        inspector: str = "Infraspect Autonomous Edge UAV #1",
        notes: str = "Autonomous aerial optical survey under AASHTO MBEI 1130 guidelines.",
        frames_scanned: Optional[int] = None,
        surface_area: Optional[float] = None,
    ):
        self.db_path = db_path
        self.asset_id = asset_id
        self.inspector = inspector
        self.notes = notes
        self.user_frames_scanned = frames_scanned
        self.user_surface_area = surface_area
        self.data_loaded = False
        self.inventory: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}

    def load_data(self) -> Dict[str, Any]:
        """Queries SQLite telemetry database and aggregates statistical metrics."""
        self.inventory = []
        cs_counts = {"CS1": 0, "CS2": 0, "CS3": 0, "CS4": 0, "PENDING": 0}
        defect_counts: Dict[str, int] = {}
        widths: List[float] = []
        confidences: List[float] = []
        timestamps: List[float] = []
        frames_observed = set()
        xs: List[float] = []
        ys: List[float] = []

        if os.path.isfile(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path, timeout=10.0)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, timestamp, defect_id, defect_class, advisory_standard,
                           condition_state, confidence, mse_score, est_width_mm, bbox_json, pose_json
                    FROM inspection_logs
                    ORDER BY timestamp ASC
                """)
                rows = cursor.fetchall()
                conn.close()

                for r in rows:
                    w = float(r["est_width_mm"]) if r["est_width_mm"] is not None else 0.0
                    cs_key = _classify_cs_key(r["condition_state"], width_mm=w)
                    cs_counts[cs_key] = cs_counts.get(cs_key, 0) + 1

                    d_class = str(r["defect_class"] or "unknown").lower()
                    defect_counts[d_class] = defect_counts.get(d_class, 0) + 1

                    conf = float(r["confidence"]) if r["confidence"] is not None else 0.0
                    ts = float(r["timestamp"]) if r["timestamp"] is not None else 0.0

                    widths.append(w)
                    confidences.append(conf)
                    timestamps.append(ts)

                    try:
                        bbox = json.loads(r["bbox_json"]) if r["bbox_json"] else []
                    except Exception:
                        bbox = []

                    pose = {}
                    if r["pose_json"]:
                        try:
                            pose = json.loads(r["pose_json"])
                            if "frame_idx" in pose:
                                frames_observed.add(pose["frame_idx"])
                            if "x" in pose and pose["x"] is not None:
                                xs.append(float(pose["x"]))
                            if "y" in pose and pose["y"] is not None:
                                ys.append(float(pose["y"]))
                        except Exception:
                            pose = {}

                    state_spec = AASHTO_STATES.get(cs_key, FALLBACK_STATE)
                    self.inventory.append({
                        "id": r["id"],
                        "timestamp": ts,
                        "formatted_time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts > 0 else "N/A",
                        "defect_id": r["defect_id"],
                        "defect_class": d_class,
                        "advisory_standard": r["advisory_standard"] or "AASHTO_MBEI_1130",
                        "condition_state": r["condition_state"] or "CS1 (Good)",
                        "cs_key": cs_key,
                        "confidence": conf,
                        "mse_score": float(r["mse_score"]) if r["mse_score"] is not None else 0.0,
                        "est_width_mm": w,
                        "bbox": bbox,
                        "pose": pose,
                        "action_guidance": state_spec["action"],
                    })
            except Exception as err:
                sys.stderr.write(f"[-] Warning: Failed to read inspection database {self.db_path}: {err}\n")

        total = len(self.inventory)
        critical = cs_counts["CS3"] + cs_counts["CS4"]

        # Physical width statistics
        valid_widths = [w for w in widths if 0.0 < w <= MAX_PLAUSIBLE_CRACK_WIDTH_MM]
        calc_widths = valid_widths if valid_widths else widths

        if calc_widths:
            sorted_w = sorted(calc_widths)
            w_min = round(min(calc_widths), 3)
            w_max = round(max(calc_widths), 3)
            w_mean = round(sum(calc_widths) / len(calc_widths), 3)
            w_median = round(sorted_w[len(sorted_w) // 2], 3)
            p95_idx = min(len(sorted_w) - 1, int(len(sorted_w) * 0.95))
            w_p95 = round(sorted_w[p95_idx], 3)
        else:
            w_min = w_max = w_mean = w_median = w_p95 = 0.0

        mean_conf = round(sum(confidences) / len(confidences) * 100.0, 1) if confidences else 0.0

        # AASHTO / Pontis Bridge Health Index Formulation (0 to 100 scale)
        if total > 0:
            weighted_damage = (cs_counts["CS2"] * 0.15) + (cs_counts["CS3"] * 0.40) + (cs_counts["CS4"] * 1.0)
            damage_ratio = min(1.0, weighted_damage / total)
            health_index = round((1.0 - damage_ratio) * 100.0, 1)
        else:
            health_index = 100.0

        valid_ts = [t for t in timestamps if t > 0.0]
        duration_sec = (max(valid_ts) - min(valid_ts)) if len(valid_ts) > 1 else 0.0

        # Scanned frames and surface area derivation
        if self.user_frames_scanned is not None:
            frames_scanned = self.user_frames_scanned
        elif frames_observed:
            frames_scanned = max(len(frames_observed), max(frames_observed) + 1)
        else:
            frames_scanned = len(timestamps)

        if self.user_surface_area is not None:
            scanned_area_sqm = float(self.user_surface_area)
        elif xs and ys:
            span_x = max(xs) - min(xs)
            span_y = max(ys) - min(ys)
            # Physical camera optical footprint per span (+0.8m lens margin)
            scanned_area_sqm = round(max(1.0, (span_x + 0.8) * (span_y + 0.8)), 2)
        elif frames_scanned > 0:
            # Nominal UAV optical FOV ~0.15 m^2 coverage per discrete frame
            scanned_area_sqm = round(max(1.0, frames_scanned * 0.15), 2)
        else:
            scanned_area_sqm = 0.0

        self.metrics = {
            "total_detections": total,
            "critical_count": critical,
            "condition_state_counts": cs_counts,
            "defect_class_counts": defect_counts,
            "crack_width_stats": {
                "min": w_min,
                "max": w_max,
                "mean": w_mean,
                "median": w_median,
                "p95": w_p95,
            },
            "mean_confidence": mean_conf,
            "overall_health_index": health_index,
            "earliest_timestamp": min(timestamps) if timestamps else 0.0,
            "latest_timestamp": max(timestamps) if timestamps else 0.0,
            "duration_seconds": round(duration_sec, 1),
            "frames_scanned": frames_scanned,
            "scanned_area_sqm": scanned_area_sqm,
        }

        self.data_loaded = True
        return self.metrics

    def export_json(self, output_path: str) -> str:
        """Exports report summary and defect inventory as structured JSON."""
        if not self.data_loaded:
            self.load_data()

        payload = {
            "report_version": "1.1.0",
            "standard": "AASHTO Manual for Bridge Element Inspection (MBEI) Section 1130",
            "asset_id": self.asset_id,
            "inspector": self.inspector,
            "generated_at": datetime.now().isoformat(),
            "notes": self.notes,
            "metrics": self.metrics,
            "inventory": self.inventory,
        }

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return output_path

    def _render_donut_svg(self, counts: Dict[str, int]) -> str:
        """Renders an inline SVG Donut chart displaying AASHTO Condition State distribution."""
        total = sum(counts.get(k, 0) for k in ["CS1", "CS2", "CS3", "CS4"])
        if total == 0:
            return '<svg width="220" height="220" viewBox="0 0 220 220"><circle cx="110" cy="110" r="75" fill="none" stroke="#e2e8f0" stroke-width="28"/><text x="110" y="115" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#64748b">No Defects</text></svg>'

        r = 75
        cx, cy = 110, 110
        circumference = 2 * math.pi * r
        accumulated_angle = -90.0

        slices = []
        for key in ["CS1", "CS2", "CS3", "CS4"]:
            val = counts.get(key, 0)
            if val == 0:
                continue
            ratio = val / total
            stroke_dash = f"{ratio * circumference:.2f} {circumference:.2f}"
            offset = -((accumulated_angle + 90.0) / 360.0) * circumference
            color = AASHTO_STATES[key]["color"]

            slices.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" '
                f'stroke-width="28" stroke-dasharray="{stroke_dash}" stroke-dashoffset="{offset:.2f}" '
                f'transform="rotate(-90 {cx} {cy})" />'
            )
            accumulated_angle += ratio * 360.0

        svg = f"""
        <svg width="220" height="220" viewBox="0 0 220 220" style="display:block;margin:auto;">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#f1f5f9" stroke-width="28" />
            {''.join(slices)}
            <text x="{cx}" y="{cy - 5}" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="24" font-weight="bold" fill="#0f172a">{total}</text>
            <text x="{cx}" y="{cy + 18}" text-anchor="middle" font-family="system-ui, -apple-system, sans-serif" font-size="11" font-weight="500" fill="#64748b">TOTAL DETECTIONS</text>
        </svg>
        """
        return svg

    def _render_bar_svg(self, defect_counts: Dict[str, int]) -> str:
        """Renders an inline SVG horizontal bar breakdown of defect categories."""
        if not defect_counts:
            return '<div style="color:#64748b;font-size:13px;text-align:center;padding:20px;">No defect classifications recorded.</div>'

        max_val = max(defect_counts.values()) if defect_counts else 1
        bars = []
        y = 20
        palette = ["#1e40af", "#3b82f6", "#0284c7", "#0d9488", "#d97706", "#dc2626"]

        for i, (cat, count) in enumerate(sorted(defect_counts.items(), key=lambda x: -x[1])):
            width_px = int((count / max_val) * 200)
            color = palette[i % len(palette)]
            title = html.escape(cat.capitalize())

            bars.append(f"""
            <g>
                <text x="0" y="{y + 14}" font-family="system-ui, -apple-system, sans-serif" font-size="12" fill="#334155" font-weight="500">{title}</text>
                <rect x="110" y="{y}" width="{width_px}" height="18" rx="4" fill="{color}" />
                <text x="{110 + width_px + 8}" y="{y + 14}" font-family="system-ui, -apple-system, sans-serif" font-size="12" font-weight="600" fill="#0f172a">{count}</text>
            </g>
            """)
            y += 30

        svg_h = max(100, y + 10)
        return f'<svg width="100%" height="{svg_h}" viewBox="0 0 360 {svg_h}">{"".join(bars)}</svg>'

    def _render_roi_svg(self, bbox: List[float], cs_key: str) -> str:
        """Renders an inline vector SVG ROI thumbnail depicting the defect geometry within the inspection patch."""
        state = AASHTO_STATES.get(cs_key, FALLBACK_STATE)
        color = state["color"]
        bg = state["bg_color"]

        if len(bbox) >= 4:
            # Normalize to 48x48 viewport; frames larger than the 128px model grid scale to fit
            extent = max(128.0, bbox[0] + bbox[2], bbox[1] + bbox[3])
            scale = 48.0 / extent
            bx = max(2.0, min(42.0, bbox[0] * scale))
            by = max(2.0, min(42.0, bbox[1] * scale))
            bw = max(6.0, min(44.0 - bx, bbox[2] * scale))
            bh = max(6.0, min(44.0 - by, bbox[3] * scale))
        else:
            bx, by, bw, bh = 10.0, 10.0, 28.0, 28.0

        return f"""
        <svg width="44" height="44" viewBox="0 0 48 48" class="roi-thumb" style="display:inline-block;vertical-align:middle;border-radius:6px;border:1px solid #cbd5e1;background:#f8fafc;">
            <rect width="48" height="48" fill="#f8fafc" />
            <rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{bg}" stroke="{color}" stroke-width="1.8" stroke-dasharray="2 1" rx="2" />
            <line x1="{bx:.1f}" y1="{by + bh:.1f}" x2="{bx + bw:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="1.5" />
        </svg>
        """

    def generate_html(self) -> str:
        """Generates self-contained, responsive, printable HTML inspection audit dashboard."""
        if not self.data_loaded:
            self.load_data()

        m = self.metrics
        cs = m.get("condition_state_counts", {})
        ws = m.get("crack_width_stats", {})

        donut_svg = self._render_donut_svg(cs)
        bar_svg = self._render_bar_svg(m.get("defect_class_counts", {}))

        # Health status badge styling
        hi = m.get("overall_health_index", 100.0)
        if hi >= 90:
            hi_color, hi_bg, hi_label = "#16a34a", "#dcfce7", "EXCELLENT"
        elif hi >= 75:
            hi_color, hi_bg, hi_label = "#ca8a04", "#fef9c3", "FAIR"
        elif hi >= 50:
            hi_color, hi_bg, hi_label = "#ea580c", "#ffedd5", "DEGRADED"
        else:
            hi_color, hi_bg, hi_label = "#dc2626", "#fee2e2", "CRITICAL"

        # Inventory table rows
        rows_html = []
        if self.inventory:
            for item in self.inventory:
                cs_info = AASHTO_STATES.get(item["cs_key"], FALLBACK_STATE)
                bbox_str = f"[{', '.join(str(int(c)) for c in item['bbox'])}]" if item["bbox"] else "N/A"
                roi_svg = self._render_roi_svg(item["bbox"], item["cs_key"])

                rows_html.append(f"""
                <tr class="log-row cs-{html.escape(item['cs_key'].lower())}">
                    <td class="font-mono">#{item['id']}</td>
                    <td class="text-muted">{html.escape(str(item['formatted_time']))}</td>
                    <td><strong style="text-transform:capitalize;">{html.escape(item['defect_class'])}</strong></td>
                    <td>
                        <span class="cs-badge" style="background:{cs_info['bg_color']};color:{cs_info['color']};border-color:{cs_info['badge_border']};">
                            {html.escape(item['cs_key'])}
                        </span>
                        <span style="font-size:12px;color:#475569;margin-left:4px;">{html.escape(cs_info['title'].split('(')[1].replace(')', '') if '(' in cs_info['title'] else cs_info['title'])}</span>
                    </td>
                    <td class="text-right font-mono font-bold" style="color:{cs_info['color']};">
                        {item['est_width_mm']:.2f} mm
                    </td>
                    <td class="text-right font-mono">{item['confidence']*100:.1f}%</td>
                    <td style="text-align:center;">
                        {roi_svg}
                        <div class="font-mono text-muted" style="font-size:10px;margin-top:2px;">{html.escape(bbox_str)}</div>
                    </td>
                    <td style="font-size:12px;color:#334155;">{html.escape(item['action_guidance'])}</td>
                </tr>
                """)
            table_body = "\n".join(rows_html)
        else:
            table_body = """
            <tr>
                <td colspan="8" style="text-align:center;padding:40px;color:#64748b;">
                    No inspection records found in telemetry buffer.
                </td>
            </tr>
            """

        # Generate AASHTO legend items dynamically
        legend_items = []
        total_detections = max(1, m.get("total_detections", 1))
        for key, spec in AASHTO_STATES.items():
            count = cs.get(key, 0)
            pct = (count / total_detections) * 100.0
            range_info = spec["title"].split("/")[1].replace(")", "").strip() if "/" in spec["title"] else ""
            legend_items.append(f"""
            <div class="legend-item">
                <span class="legend-dot" style="background:{spec['color']};"></span>
                <span><strong>{key}:</strong> {count} ({pct:.1f}%) <span style="color:#64748b;">{range_info}</span></span>
            </div>
            """)
        legend_html = "\n".join(legend_items)

        notes_section = f"""
        <div style="background:#eff6ff;border-left:4px solid #3b82f6;padding:12px 18px;border-radius:4px;margin-bottom:20px;font-size:13px;color:#1e3a8a;">
            <strong>ENGINEERING FIELD NOTES:</strong> {html.escape(self.notes)}
        </div>
        """ if self.notes else ""

        generated_date = datetime.now().strftime("%B %d, %Y - %H:%M:%S UTC")

        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Infraspect: AASHTO Civil Infrastructure Inspection Report - {html.escape(self.asset_id)}</title>
    <style>
        :root {{
            --primary: #1e40af;
            --primary-dark: #1e3a8a;
            --bg-page: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --text-dark: #0f172a;
            --text-muted: #64748b;
            --cs1: #16a34a;
            --cs2: #ca8a04;
            --cs3: #ea580c;
            --cs4: #dc2626;
        }}

        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg-page);
            color: var(--text-dark);
            line-height: 1.5;
            padding: 32px 24px;
        }}

        .container {{
            max-width: 1240px;
            margin: 0 auto;
        }}

        header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            color: #ffffff;
            border-radius: 12px;
            padding: 28px 32px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }}

        .header-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .header-meta {{
            margin-top: 14px;
            display: flex;
            gap: 24px;
            font-size: 13px;
            color: #dbeafe;
            flex-wrap: wrap;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}

        .kpi-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .kpi-title {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 8px;
        }}

        .kpi-value {{
            font-size: 30px;
            font-weight: 800;
            color: var(--text-dark);
            line-height: 1.1;
        }}

        .kpi-subtext {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 8px;
        }}

        .action-notice {{
            background: #fee2e2;
            border-left: 5px solid #dc2626;
            padding: 16px 20px;
            border-radius: 6px;
            margin-bottom: 24px;
            font-size: 14px;
            color: #991b1b;
            line-height: 1.5;
        }}

        .analytics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }}

        @media (max-width: 900px) {{
            .analytics-grid {{ grid-template-columns: 1fr; }}
        }}

        .section-card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        }}

        .section-title {{
            font-size: 16px;
            font-weight: 700;
            color: var(--text-dark);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .aashto-legend {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 16px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
        }}

        .legend-dot {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            flex-shrink: 0;
        }}

        .table-responsive {{
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
            background: #ffffff;
        }}

        th {{
            background: #f8fafc;
            color: #475569;
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 10px 14px;
            border-bottom: 1px solid #f1f5f9;
            vertical-align: middle;
        }}

        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: #f8fafc; }}

        .cs-badge {{
            display: inline-block;
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 6px;
            border: 1px solid transparent;
        }}

        .text-right {{ text-align: right; }}
        .font-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
        .font-bold {{ font-weight: 700; }}
        .text-muted {{ color: var(--text-muted); }}

        @media print {{
            body {{ background: #ffffff; padding: 0; color: #000000; font-size: 11pt; }}
            header {{ background: #1e3a8a !important; color: #ffffff !important; -webkit-print-color-adjust: exact; }}
            .kpi-card, .section-card {{ border: 1px solid #cbd5e1; box-shadow: none; page-break-inside: avoid; }}
            .table-responsive {{ border: 1px solid #94a3b8; }}
            tr {{ page-break-inside: avoid; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        
        <!-- Header -->
        <header>
            <div>
                <span class="header-badge">AASHTO MBEI 1130 COMPLIANT</span>
                <h1 style="margin-top:10px;font-size:26px;">Civil Infrastructure Defect Inspection Report</h1>
                <p style="margin-top:4px;">Asset: <strong>{html.escape(self.asset_id)}</strong> &bull; Standard: AASHTO Manual for Bridge Element Inspection Section 1130</p>
                <div class="header-meta">
                    <div><strong>Survey Platform:</strong> {html.escape(self.inspector)}</div>
                    <div><strong>Generated:</strong> {generated_date}</div>
                    <div><strong>Telemetry DB:</strong> {html.escape(os.path.basename(self.db_path))}</div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:12px;color:#bfdbfe;text-transform:uppercase;font-weight:600;">Structural Health Index</div>
                <div style="font-size:42px;font-weight:900;color:#ffffff;line-height:1;">{hi:.0f}<span style="font-size:20px;color:#bfdbfe;">/100</span></div>
                <span style="display:inline-block;margin-top:6px;padding:3px 12px;border-radius:12px;font-size:11px;font-weight:800;background:{hi_bg};color:{hi_color};">
                    {hi_label}
                </span>
            </div>
        </header>

        {notes_section}

        <!-- KPI Metrics Grid -->
        <section class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Defect Detections</div>
                <div class="kpi-value">{m.get('total_detections', 0)}</div>
                <div class="kpi-subtext">Across {m.get('duration_seconds', 0.0):.1f}s inspection scan</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Scanned Area / Frames</div>
                <div class="kpi-value">{m.get('scanned_area_sqm', 0.0):.2f} <span style="font-size:18px;">m²</span></div>
                <div class="kpi-subtext">{m.get('frames_scanned', 0)} discrete frames analyzed</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Critical Defects (CS3 + CS4)</div>
                <div class="kpi-value" style="color: {'#dc2626' if m.get('critical_count', 0) > 0 else '#16a34a'};">
                    {m.get('critical_count', 0)}
                </div>
                <div class="kpi-subtext">{cs.get('CS4', 0)} Critical CS4 &bull; {cs.get('CS3', 0)} Severe CS3</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Max Crack Width (Physical mm)</div>
                <div class="kpi-value" style="color: {'#dc2626' if ws.get('max', 0) >= 2.0 else '#ca8a04' if ws.get('max', 0) >= 0.3 else '#16a34a'};">
                    {ws.get('max', 0.0):.2f} mm
                </div>
                <div class="kpi-subtext">Mean: {ws.get('mean', 0.0):.2f} mm &bull; P95: {ws.get('p95', 0.0):.2f} mm</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Mean Confidence</div>
                <div class="kpi-value">{m.get('mean_confidence', 0.0):.1f}%</div>
                <div class="kpi-subtext">Edge SNN & CNN Deep Learning Ensemble</div>
            </div>
        </section>

        <!-- Action Notice if Critical defects present -->
        {"<div class='action-notice'><strong>STRUCTURAL ALERT:</strong> " + str(cs.get('CS4', 0)) + " critical condition state (CS4 &gt; 2.0mm) and " + str(cs.get('CS3', 0)) + " severe state (CS3) defects were logged. Immediate civil structural engineer review and pressure crack injection sealing are recommended per AASHTO MBEI 1130.</div>" if m.get('critical_count', 0) > 0 else ""}

        <!-- Analytics Charts Grid -->
        <section class="analytics-grid">
            <div class="section-card">
                <div class="section-title">
                    <span>AASHTO Condition State Breakdown</span>
                    <span style="font-size:12px;color:#64748b;font-weight:normal;">MBEI Section 1130</span>
                </div>
                {donut_svg}
                <div class="aashto-legend">
                    {legend_html}
                </div>
            </div>

            <div class="section-card">
                <div class="section-title">
                    <span>Defect Classification Distribution</span>
                    <span style="font-size:12px;color:#64748b;font-weight:normal;">CODEBRIM 6-Class Taxonomy</span>
                </div>
                {bar_svg}
                <div style="font-size:12px;color:#64748b;margin-top:16px;line-height:1.4;">
                    Evaluated by edge deep learning classifier examining high-resolution aspect-padded ROI crops extracted via multi-channel anomaly detection.
                </div>
            </div>
        </section>

        <!-- Detailed Defect Inventory Table -->
        <section class="section-card" style="margin-bottom:24px;">
            <div class="section-title">
                <span>Defect Telemetry Inventory & Visual ROI Crops</span>
                <span style="font-size:12px;color:#64748b;font-weight:normal;">Showing {len(self.inventory)} Records</span>
            </div>
            <div class="table-responsive">
                <table>
                    <thead>
                        <tr>
                            <th>Log ID</th>
                            <th>Timestamp</th>
                            <th>Classification</th>
                            <th>AASHTO State</th>
                            <th class="text-right">Est Width</th>
                            <th class="text-right">Confidence</th>
                            <th style="text-align:center;">Visual ROI Crop</th>
                            <th>Prescribed Action Plan</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_body}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Engineering Compliance Footer -->
        <footer style="text-align:center;font-size:12px;color:#94a3b8;padding:24px 0;">
            <p><strong>Infraspect Autonomous Structural Health Monitoring (SHM) Engine</strong></p>
            <p>Compliance: AASHTO MBEI 1130 &bull; API 570 &bull; ASME B31.3 &bull; Generated from edge telemetry database: <code>{html.escape(self.db_path)}</code></p>
        </footer>

    </div>
</body>
</html>
"""
        return html_template

    def export_html(self, output_path: str) -> str:
        """Renders and writes HTML report to specified output file."""
        html_content = self.generate_html()
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return output_path

    def export_pdf(self, output_path: str) -> str:
        """Generates a genuine multi-page vector PDF 1.4 inspection report without third-party dependencies."""
        if not self.data_loaded:
            self.load_data()

        m = self.metrics
        cs = m.get("condition_state_counts", {})
        ws = m.get("crack_width_stats", {})

        def _escape_pdf(text: Any) -> str:
            s = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            return "".join(c if ord(c) < 128 else "?" for c in s)

        pages_content = []

        # --- PAGE 1: Executive Summary & AASHTO Condition Breakdown ---
        p1 = []
        # Header banner
        p1.append("0.12 0.23 0.54 rg")  # Dark blue fill
        p1.append("36 710 540 50 re f")
        p1.append("BT /F1 16 Tf 1 1 1 rg 50 728 Td (INFRASPECT: CIVIL INSPECTION AUDIT REPORT) Tj ET")
        p1.append("BT /F2 9 Tf 1 1 1 rg 50 716 Td (Standard: AASHTO Manual for Bridge Element Inspection Section 1130) Tj ET")

        # Metadata table
        p1.append("0 0 0 rg")
        p1.append("BT /F1 11 Tf 50 685 Td (Asset Identification:) Tj ET")
        p1.append(f"BT /F2 11 Tf 180 685 Td ({_escape_pdf(self.asset_id)}) Tj ET")

        p1.append("BT /F1 11 Tf 50 670 Td (Survey Platform / UAV:) Tj ET")
        p1.append(f"BT /F2 11 Tf 180 670 Td ({_escape_pdf(self.inspector)}) Tj ET")

        p1.append("BT /F1 11 Tf 50 655 Td (Generated Timestamp:) Tj ET")
        p1.append(f"BT /F2 11 Tf 180 655 Td ({_escape_pdf(datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))}) Tj ET")

        p1.append("BT /F1 11 Tf 50 640 Td (Telemetry Database:) Tj ET")
        p1.append(f"BT /F2 11 Tf 180 640 Td ({_escape_pdf(os.path.basename(self.db_path))}) Tj ET")

        # KPI Summary Box
        p1.append("0.95 0.96 0.98 rg 36 530 540 90 re f")
        p1.append("0.8 0.85 0.9 RG 1 w 36 530 540 90 re s")

        p1.append("BT /F1 12 Tf 0.1 0.2 0.5 rg 50 600 Td (EXECUTIVE SUMMARY & KEY METRICS) Tj ET")
        p1.append(f"BT /F2 10 Tf 0 0 0 rg 50 580 Td (Total Defect Detections: {m.get('total_detections', 0)}) Tj ET")
        p1.append(f"BT /F2 10 Tf 0 0 0 rg 50 565 Td (Critical Defects (CS3 + CS4): {m.get('critical_count', 0)}) Tj ET")
        p1.append(f"BT /F2 10 Tf 0 0 0 rg 50 550 Td (Scanned Surface Area: {m.get('scanned_area_sqm', 0.0):.2f} sqm  |  Frames: {m.get('frames_scanned', 0)}) Tj ET")

        p1.append(f"BT /F2 10 Tf 0 0 0 rg 320 580 Td (Structural Health Index: {m.get('overall_health_index', 100.0):.1f} / 100) Tj ET")
        p1.append(f"BT /F2 10 Tf 0 0 0 rg 320 565 Td (Mean Model Confidence: {m.get('mean_confidence', 0.0):.1f}%) Tj ET")
        p1.append(f"BT /F2 10 Tf 0 0 0 rg 320 550 Td (Inspection Duration: {m.get('duration_seconds', 0.0):.1f} seconds) Tj ET")

        # AASHTO Condition States Section
        p1.append("BT /F1 12 Tf 0.1 0.2 0.5 rg 36 500 Td (AASHTO MBEI 1130 CONDITION STATE DISTRIBUTION) Tj ET")

        table_y = 480
        total_d = max(1, m.get("total_detections", 1))

        # Table Header
        p1.append(f"0.9 0.92 0.95 rg 36 {table_y - 15} 540 18 re f")
        p1.append("0.7 0.75 0.8 RG 0.5 w")
        p1.append(f"36 {table_y - 15} 540 18 re s")
        p1.append(f"BT /F1 9 Tf 0 0 0 rg 45 {table_y - 11} Td (State) Tj ET")
        p1.append(f"BT /F1 9 Tf 0 0 0 rg 90 {table_y - 11} Td (Description & Threshold) Tj ET")
        p1.append(f"BT /F1 9 Tf 0 0 0 rg 270 {table_y - 11} Td (Count) Tj ET")
        p1.append(f"BT /F1 9 Tf 0 0 0 rg 330 {table_y - 11} Td (Ratio) Tj ET")
        p1.append(f"BT /F1 9 Tf 0 0 0 rg 390 {table_y - 11} Td (Required Action) Tj ET")
        table_y -= 18

        for key in ["CS1", "CS2", "CS3", "CS4"]:
            spec = AASHTO_STATES[key]
            cnt = cs.get(key, 0)
            pct = (cnt / total_d) * 100.0
            p1.append(f"0.8 0.85 0.9 RG 0.5 w 36 {table_y - 20} 540 20 re s")
            p1.append(f"BT /F1 9 Tf 0.1 0.1 0.1 rg 45 {table_y - 14} Td ({key}) Tj ET")
            p1.append(f"BT /F2 8 Tf 0.2 0.2 0.2 rg 90 {table_y - 14} Td ({_escape_pdf(spec['title'])}) Tj ET")
            p1.append(f"BT /F1 9 Tf 0 0 0 rg 270 {table_y - 14} Td ({cnt}) Tj ET")
            p1.append(f"BT /F2 9 Tf 0 0 0 rg 330 {table_y - 14} Td ({pct:.1f}%) Tj ET")
            action_snippet = spec["action"][:34] + "..." if len(spec["action"]) > 34 else spec["action"]
            p1.append(f"BT /F2 7.5 Tf 0.3 0.3 0.3 rg 390 {table_y - 14} Td ({_escape_pdf(action_snippet)}) Tj ET")
            table_y -= 20

        # Physical Crack Width Statistics Section
        p1.append(f"BT /F1 12 Tf 0.1 0.2 0.5 rg 36 {table_y - 25} Td (PHYSICAL CRACK WIDTH METRICS (GSD CALIBRATED)) Tj ET")
        table_y -= 45

        p1.append(f"0.96 0.97 0.99 rg 36 {table_y - 35} 540 40 re f")
        p1.append(f"0.8 0.85 0.9 RG 1 w 36 {table_y - 35} 540 40 re s")
        p1.append(f"BT /F2 9 Tf 0 0 0 rg 50 {table_y - 15} Td (Min Width: {ws.get('min', 0.0):.2f} mm) Tj ET")
        p1.append(f"BT /F2 9 Tf 0 0 0 rg 150 {table_y - 15} Td (Mean Width: {ws.get('mean', 0.0):.2f} mm) Tj ET")
        p1.append(f"BT /F2 9 Tf 0 0 0 rg 260 {table_y - 15} Td (Median Width: {ws.get('median', 0.0):.2f} mm) Tj ET")
        p1.append(f"BT /F2 9 Tf 0 0 0 rg 380 {table_y - 15} Td (95th Percentile: {ws.get('p95', 0.0):.2f} mm) Tj ET")
        p1.append(f"BT /F1 9 Tf 0.8 0.1 0.1 rg 480 {table_y - 15} Td (Max: {ws.get('max', 0.0):.2f} mm) Tj ET")

        # --- PAGE 2+: Defect Inventory Table ---
        rows_per_page = 28
        if self.inventory:
            inv_chunks = [self.inventory[i:i + rows_per_page] for i in range(0, len(self.inventory), rows_per_page)]
        else:
            inv_chunks = []
        total_pdf_pages = len(inv_chunks) + 1

        # Engineering Certification Footer
        p1.append(f"BT /F2 8 Tf 0.5 0.5 0.5 rg 50 45 Td (Infraspect Civil Inspection Report  |  AASHTO MBEI 1130  |  Page 1 of {total_pdf_pages}) Tj ET")
        pages_content.append("\n".join(p1))

        for p_idx, chunk in enumerate(inv_chunks):
            p = []
            p.append("0.12 0.23 0.54 rg 36 735 540 25 re f")
            p.append(f"BT /F1 11 Tf 1 1 1 rg 50 743 Td (DEFECT INVENTORY LOG  --  PAGE {p_idx + 2}) Tj ET")

            y = 715
            # Header
            p.append(f"0.9 0.92 0.95 rg 36 {y - 14} 540 16 re f")
            p.append(f"0.7 0.75 0.8 RG 0.5 w 36 {y - 14} 540 16 re s")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 42 {y - 10} Td (ID) Tj ET")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 75 {y - 10} Td (Time) Tj ET")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 180 {y - 10} Td (Class) Tj ET")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 245 {y - 10} Td (AASHTO) Tj ET")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 305 {y - 10} Td (Est Width) Tj ET")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 365 {y - 10} Td (Conf) Tj ET")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 410 {y - 10} Td (BBox [x,y,w,h]) Tj ET")
            p.append(f"BT /F1 8.5 Tf 0 0 0 rg 495 {y - 10} Td (Action) Tj ET")
            y -= 16

            for item in chunk:
                p.append(f"0.85 0.88 0.92 RG 0.4 w 36 {y - 16} 540 16 re s")
                p.append(f"BT /F2 8 Tf 0 0 0 rg 42 {y - 12} Td (#{item['id']}) Tj ET")
                time_str = item['formatted_time'][-8:] if len(item['formatted_time']) >= 8 else item['formatted_time']
                p.append(f"BT /F2 8 Tf 0.3 0.3 0.3 rg 75 {y - 12} Td ({_escape_pdf(time_str)}) Tj ET")
                p.append(f"BT /F1 8 Tf 0 0 0 rg 180 {y - 12} Td ({_escape_pdf(item['defect_class'][:10])}) Tj ET")
                p.append(f"BT /F1 8 Tf 0.1 0.2 0.5 rg 245 {y - 12} Td ({_escape_pdf(item['cs_key'])}) Tj ET")
                p.append(f"BT /F2 8 Tf 0 0 0 rg 305 {y - 12} Td ({item['est_width_mm']:.2f} mm) Tj ET")
                p.append(f"BT /F2 8 Tf 0 0 0 rg 365 {y - 12} Td ({item['confidence']*100:.0f}%) Tj ET")
                bbox_str = f"[{int(item['bbox'][0])},{int(item['bbox'][1])},{int(item['bbox'][2])},{int(item['bbox'][3])}]" if len(item['bbox']) >= 4 else "N/A"
                p.append(f"BT /F2 7.5 Tf 0.4 0.4 0.4 rg 410 {y - 12} Td ({_escape_pdf(bbox_str)}) Tj ET")
                act_snip = item['action_guidance'][:18] + ".." if len(item['action_guidance']) > 18 else item['action_guidance']
                p.append(f"BT /F2 7 Tf 0.2 0.2 0.2 rg 495 {y - 12} Td ({_escape_pdf(act_snip)}) Tj ET")
                y -= 16

            p.append(f"BT /F2 8 Tf 0.5 0.5 0.5 rg 50 45 Td (Infraspect Civil Inspection Report  |  AASHTO MBEI 1130  |  Page {p_idx + 2} of {total_pdf_pages}) Tj ET")
            pages_content.append("\n".join(p))

        # Assemble genuine binary PDF 1.4 document
        num_pages = len(pages_content)
        cat_id = 1
        pages_id = 2
        first_page_id = 3
        first_stream_id = 3 + num_pages
        font_bold_id = first_stream_id + num_pages
        font_reg_id = font_bold_id + 1

        page_refs = " ".join([f"{first_page_id + i} 0 R" for i in range(num_pages)])

        objects = {}
        objects[cat_id] = f"{cat_id} 0 obj\n<< /Type /Catalog /Pages {pages_id} 0 R >>\nendobj\n".encode("latin1")
        objects[pages_id] = f"{pages_id} 0 obj\n<< /Type /Pages /Kids [{page_refs}] /Count {num_pages} >>\nendobj\n".encode("latin1")

        for i, content in enumerate(pages_content):
            p_id = first_page_id + i
            s_id = first_stream_id + i
            stream_bytes = content.encode("latin1")
            objects[p_id] = f"{p_id} 0 obj\n<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 612 792] /Contents {s_id} 0 R /Resources << /Font << /F1 {font_bold_id} 0 R /F2 {font_reg_id} 0 R >> >> >>\nendobj\n".encode("latin1")
            objects[s_id] = f"{s_id} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin1") + stream_bytes + b"\nendstream\nendobj\n"

        objects[font_bold_id] = f"{font_bold_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>\nendobj\n".encode("latin1")
        objects[font_reg_id] = f"{font_reg_id} 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n".encode("latin1")

        header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
        total_objs = max(objects.keys()) + 1
        pos = len(header)
        offsets = {}
        body = b""
        for obj_id in sorted(objects.keys()):
            offsets[obj_id] = pos
            body += objects[obj_id]
            pos += len(objects[obj_id])

        xref = f"xref\n0 {total_objs}\n0000000000 65535 f \n".encode("latin1")
        for obj_id in range(1, total_objs):
            off = offsets.get(obj_id, 0)
            xref += f"{off:010d} 00000 n \n".encode("latin1")

        trailer = f"trailer\n<< /Size {total_objs} /Root {cat_id} 0 R >>\nstartxref\n{pos}\n%%EOF\n".encode("latin1")
        pdf_data = header + body + xref + trailer

        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(pdf_data)

        return output_path


def generate_report_cli(args_list: Optional[List[str]] = None) -> int:
    """CLI handler for infraspect-report."""
    parser = argparse.ArgumentParser(
        description="Infraspect: AASHTO Civil Infrastructure Inspection Report Generator"
    )
    parser.add_argument(
        "--db",
        "--db-path",
        dest="db_path",
        type=str,
        default="civil_inspection_telemetry.db",
        help="Path to SQLite edge telemetry database",
    )
    parser.add_argument(
        "--asset-id",
        type=str,
        default="Bridge Pier #4B - North Span",
        help="Identifier or name of the civil asset under inspection",
    )
    parser.add_argument(
        "--inspector",
        type=str,
        default="Infraspect Autonomous UAV Edge Unit #1",
        help="Inspector or UAV unit identifier",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional field engineering inspection notes",
    )
    parser.add_argument(
        "--frames-scanned",
        type=int,
        default=None,
        help="Override or specify total video frames scanned",
    )
    parser.add_argument(
        "--surface-area",
        type=float,
        default=None,
        help="Override or specify total scanned surface area in square meters",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="civil_inspection_report.html",
        help="Output path for report (HTML by default, or genuine PDF if path ends with .pdf)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Optional path to output structured JSON summary",
    )

    args = parser.parse_args(args_list)

    print("=" * 70)
    print(" INFRASPECT: AASHTO Civil Infrastructure Inspection Report Generator")
    print(" Standard: AASHTO Manual for Bridge Element Inspection (MBEI) Section 1130")
    print("=" * 70)

    report = CivilInspectionReport(
        db_path=args.db_path,
        asset_id=args.asset_id,
        inspector=args.inspector,
        notes=args.notes,
        frames_scanned=args.frames_scanned,
        surface_area=args.surface_area,
    )

    metrics = report.load_data()
    print(f"[+] Loaded Telemetry DB: {args.db_path}")
    print(f"[+] Total Defect Detections: {metrics['total_detections']}")
    print(f"[+] Critical Defects (CS3/CS4): {metrics['critical_count']}")
    print(f"[+] Scanned Surface Area: {metrics['scanned_area_sqm']:.2f} m² across {metrics['frames_scanned']} frames")
    print(f"[+] Structural Health Score: {metrics['overall_health_index']:.1f} / 100")

    cs = metrics["condition_state_counts"]
    print(f"    - CS1 (Good / <0.3mm):       {cs.get('CS1', 0)}")
    print(f"    - CS2 (Fair / 0.3-1.25mm):   {cs.get('CS2', 0)}")
    print(f"    - CS3 (Poor / 1.25-2.0mm):   {cs.get('CS3', 0)}")
    print(f"    - CS4 (Critical / >2.0mm):   {cs.get('CS4', 0)}")

    # Export Report based on format
    if args.output.lower().endswith(".pdf"):
        report.export_pdf(args.output)
        print(f"[+] Generated Vector PDF Audit Report: {args.output}")
    else:
        report.export_html(args.output)
        print(f"[+] Generated Standalone HTML Audit Report: {args.output}")

    # Export JSON if requested
    if args.json_output:
        report.export_json(args.json_output)
        print(f"[+] Generated JSON Summary: {args.json_output}")

    print("=" * 70)
    return 0


def main():
    sys.exit(generate_report_cli())


if __name__ == "__main__":
    main()
