"""
tests/test_report.py
====================
Unit tests verifying the AASHTO Civil Infrastructure Inspection Report Generator.
"""

import os
import json
import sqlite3
import pytest
from src_python.report_generator import CivilInspectionReport, generate_report_cli, _classify_cs_key


@pytest.fixture
def sample_inspection_db(tmp_path):
    """Creates a temporary SQLite database with known civil inspection telemetry."""
    db_file = tmp_path / "test_telemetry.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE inspection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            defect_id INTEGER,
            defect_class TEXT,
            advisory_standard TEXT,
            condition_state TEXT,
            confidence REAL,
            mse_score REAL,
            est_width_mm REAL,
            bbox_json TEXT,
            pose_json TEXT,
            synced INTEGER DEFAULT 0
        )
    """)

    # Insert sample records across CS1-CS4 and pending
    records = [
        # CS1 (Hairline <0.3mm)
        (1700000001.0, 1, "crack", "AASHTO_MBEI_1130", "CS1 (Good / Hairline <0.3mm)", 0.92, 0.0031, 0.18, "[10, 10, 40, 40]", '{"frame_idx": 0, "x": 0.0, "y": 0.0, "z": 1.5}', 1),
        # CS2 (Fair 0.3-1.25mm)
        (1700000002.0, 2, "crack", "AASHTO_MBEI_1130", "CS2 (Fair / 0.3-1.25mm)", 0.88, 0.0052, 0.65, "[20, 20, 50, 50]", '{"frame_idx": 1, "x": 0.5, "y": 0.2, "z": 1.5}', 1),
        (1700000003.0, 3, "spalling", "AASHTO_MBEI_1130", "CS2 (Fair / 0.3-1.25mm)", 0.85, 0.0061, 0.90, "[15, 15, 60, 60]", '{"frame_idx": 2, "x": 1.0, "y": 0.4, "z": 1.5}', 1),
        # CS3 (Poor 1.25-2.0mm)
        (1700000004.0, 4, "crack", "AASHTO_MBEI_1130", "CS3 (Poor / 1.25-2.0mm)", 0.95, 0.0094, 1.62, "[30, 30, 80, 80]", '{"frame_idx": 3, "x": 1.5, "y": 0.6, "z": 1.5}', 0),
        # CS4 (Critical >2.0mm)
        (1700000005.0, 5, "crack", "AASHTO_MBEI_1130", "CS4 (Severe / Structural Review >2.0mm)", 0.97, 0.0152, 3.45, "[5, 5, 95, 95]", '{"frame_idx": 4, "x": 2.0, "y": 0.8, "z": 1.5}', 0),
    ]

    cursor.executemany("""
        INSERT INTO inspection_logs (
            timestamp, defect_id, defect_class, advisory_standard,
            condition_state, confidence, mse_score, est_width_mm,
            bbox_json, pose_json, synced
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()
    return str(db_file)


def test_classify_cs_key():
    """Verifies that _classify_cs_key correctly classifies states without falsely inflating CS1."""
    assert _classify_cs_key("CS1 (Good / Hairline <0.3mm)") == "CS1"
    assert _classify_cs_key("CS2 (Fair / 0.3-1.25mm)") == "CS2"
    assert _classify_cs_key("CS3 (Poor / 1.25-2.0mm)") == "CS3"
    assert _classify_cs_key("CS4 (Severe / Structural Review >2.0mm)") == "CS4"
    # Fallback to width
    assert _classify_cs_key(None, width_mm=0.2) == "CS1"
    assert _classify_cs_key(None, width_mm=0.8) == "CS2"
    assert _classify_cs_key(None, width_mm=1.5) == "CS3"
    assert _classify_cs_key(None, width_mm=2.5) == "CS4"
    # Uncalibrated pending does not default to CS1
    assert _classify_cs_key("CS_PENDING_GSD", width_mm=None) == "PENDING"
    assert _classify_cs_key(None, width_mm=None) == "PENDING"


def test_empty_database_graceful_handling(tmp_path):
    empty_db = tmp_path / "empty.db"
    report = CivilInspectionReport(db_path=str(empty_db))
    metrics = report.load_data()

    assert metrics["total_detections"] == 0
    assert metrics["overall_health_index"] == 100.0
    assert metrics["crack_width_stats"]["max"] == 0.0
    assert metrics["frames_scanned"] == 0

    html = report.generate_html()
    assert "No inspection records found" in html
    assert "AASHTO MBEI 1130" in html


def test_telemetry_aggregation(sample_inspection_db):
    report = CivilInspectionReport(db_path=sample_inspection_db, asset_id="Test Overpass #12")
    metrics = report.load_data()

    assert metrics["total_detections"] == 5
    assert metrics["critical_count"] == 2  # CS3 + CS4
    assert metrics["condition_state_counts"]["CS1"] == 1
    assert metrics["condition_state_counts"]["CS2"] == 2
    assert metrics["condition_state_counts"]["CS3"] == 1
    assert metrics["condition_state_counts"]["CS4"] == 1

    assert metrics["defect_class_counts"]["crack"] == 4
    assert metrics["defect_class_counts"]["spalling"] == 1

    width_stats = metrics["crack_width_stats"]
    assert width_stats["min"] == 0.18
    assert width_stats["max"] == 3.45
    assert width_stats["mean"] > 1.0

    assert metrics["frames_scanned"] == 5
    assert metrics["scanned_area_sqm"] > 1.0


def test_html_report_rendering(sample_inspection_db, tmp_path):
    report = CivilInspectionReport(
        db_path=sample_inspection_db,
        asset_id="Interstate 95 Span 4",
        notes="Pre-winter structural audit",
    )
    html_out = tmp_path / "inspection_report.html"
    report.export_html(str(html_out))

    assert html_out.is_file()
    content = html_out.read_text()

    # Verify critical components
    assert "Interstate 95 Span 4" in content
    assert "AASHTO MBEI 1130" in content
    assert "Condition State Breakdown" in content
    assert "CS4" in content
    assert "Critical" in content
    assert "3.45 mm" in content
    assert "<svg" in content  # Embedded offline SVG charts & ROI thumbnails
    assert "roi-thumb" in content  # Visual ROI crop thumbnails
    assert "Pre-winter structural audit" in content  # Field notes
    assert "@media print" in content  # Printable stylesheet


def test_pdf_report_generation(sample_inspection_db, tmp_path):
    report = CivilInspectionReport(db_path=sample_inspection_db, asset_id="Bridge Pier 4B")
    pdf_out = tmp_path / "inspection_report.pdf"
    report.export_pdf(str(pdf_out))

    assert pdf_out.is_file()
    assert pdf_out.stat().st_size > 500

    pdf_bytes = pdf_out.read_bytes()
    assert pdf_bytes.startswith(b"%PDF-1.4")
    assert b"%%EOF" in pdf_bytes
    assert b"INFRASPECT: CIVIL INSPECTION AUDIT REPORT" in pdf_bytes


def test_json_report_export(sample_inspection_db, tmp_path):
    report = CivilInspectionReport(db_path=sample_inspection_db)
    json_out = tmp_path / "inspection_report.json"
    report.export_json(str(json_out))

    assert json_out.is_file()
    data = json.loads(json_out.read_text())

    assert data["asset_id"] is not None
    assert data["metrics"]["total_detections"] == 5
    assert data["metrics"]["critical_count"] == 2
    assert data["metrics"]["frames_scanned"] == 5
    assert len(data["inventory"]) == 5


def test_cli_execution_with_aliases(sample_inspection_db, tmp_path):
    html_file = tmp_path / "cli_report.html"
    pdf_file = tmp_path / "cli_report.pdf"
    json_file = tmp_path / "cli_report.json"

    # Test HTML export using --db alias
    ret_html = generate_report_cli([
        "--db", sample_inspection_db,
        "--asset-id", "Highway 101 Culvert",
        "--output", str(html_file),
        "--json-output", str(json_file),
    ])
    assert ret_html == 0
    assert html_file.is_file()
    assert json_file.is_file()
    assert "Highway 101 Culvert" in html_file.read_text()

    # Test PDF export using --output report.pdf
    ret_pdf = generate_report_cli([
        "--db", sample_inspection_db,
        "--asset-id", "Highway 101 Culvert",
        "--output", str(pdf_file),
    ])
    assert ret_pdf == 0
    assert pdf_file.is_file()
    assert pdf_file.read_bytes().startswith(b"%PDF-1.4")
