"""
sharp_resilience.py
===================
SHARP Paradigm - Edge Fault Tolerance & Offline Telemetry Resilience Manager.

Guarantees 100% local edge operation during GPS-denied inspection in tunnels,
bridge culverts, and underground infrastructure. Buffers telemetry into SQLite
and supports pluggable background synchronization upon link restoration.
"""

import sqlite3
import os
import time
import json
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger("infraspect.sharp_resilience")


class SHARPOfflineResilienceManager:
    """Local SQLite metadata buffer and background synchronization manager."""

    def __init__(
        self,
        db_path: str = "civil_inspection_telemetry.db",
        sync_callback: Optional[Callable[[list[dict]], bool]] = None,
    ):
        self.db_path = db_path
        self.is_connected = False
        self.auto_sync_running = False
        self.sync_callback = sync_callback
        self._init_db()
        self._start_network_monitor()

    def _init_db(self):
        """Initializes SQLite schema for both civil inspection records and legacy alerts."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.OperationalError:
            pass
        cursor = conn.cursor()

        # Civil Infrastructure Inspection Log Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspection_logs (
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

        # Legacy threat table for backward compatibility
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threat_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                threat_id INTEGER,
                threat_type TEXT,
                confidence REAL,
                mse_score REAL,
                bbox_json TEXT,
                synced INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def log_inspection_offline(self, defect_record: dict):
        """Logs a single civil defect observation locally."""
        self.log_inspections_batch([defect_record])

    def log_inspections_batch(self, defect_records: list[dict]):
        """Atomically logs multiple civil defect observations in a single transaction."""
        if not defect_records:
            return
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        now = time.time()
        rows = [
            (
                d.get("timestamp", now),
                d.get("id", i + 1),
                d.get("defect_class", d.get("type", "UNKNOWN")),
                d.get("standard", "AASHTO_MBEI_1130"),
                d.get("condition_state", "CS_PENDING"),
                d.get("confidence", 0.0),
                d.get("mse_score", d.get("score", 0.0)),
                d.get("est_width_mm"),
                json.dumps(d.get("bbox", [])),
                json.dumps(d.get("pose", {})),
            )
            for i, d in enumerate(defect_records)
        ]
        cursor.executemany("""
            INSERT INTO inspection_logs (
                timestamp, defect_id, defect_class, advisory_standard,
                condition_state, confidence, mse_score, est_width_mm,
                bbox_json, pose_json, synced
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, rows)
        conn.commit()
        conn.close()

    def log_threat_offline(self, threat_dict: dict):
        """Legacy alert logging method."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO threat_logs (timestamp, threat_id, threat_type, confidence, mse_score, bbox_json, synced)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (
            time.time(),
            threat_dict.get('id', 1),
            threat_dict.get('type', 'UNKNOWN'),
            threat_dict.get('confidence', 0.0),
            threat_dict.get('score', 0.0),
            json.dumps(threat_dict.get('bbox', []))
        ))
        conn.commit()
        conn.close()

    def get_pending_sync_count(self) -> int:
        """Returns total un-synchronized records across inspection and threat logs."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM inspection_logs WHERE synced = 0")
        insp_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM threat_logs WHERE synced = 0")
        threat_count = cursor.fetchone()[0]
        conn.close()
        return insp_count + threat_count

    def get_recent_inspections(self, limit: int = 10) -> list[dict]:
        """Fetches the most recent civil defect inspection records."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, defect_class, advisory_standard, condition_state,
                   confidence, mse_score, est_width_mm, bbox_json, pose_json, synced
            FROM inspection_logs ORDER BY id DESC LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()

        records = []
        for r in rows:
            records.append({
                "log_id": r[0],
                "timestamp": time.strftime("%H:%M:%S", time.localtime(r[1])),
                "defect_class": r[2],
                "standard": r[3],
                "condition_state": r[4],
                "confidence": r[5],
                "mse_score": r[6],
                "est_width_mm": r[7],
                "bbox": json.loads(r[8]),
                "pose": json.loads(r[9]),
                "synced": bool(r[10]),
            })
        return records

    def get_recent_logs(self, limit: int = 10) -> list[dict]:
        """Fetches legacy threat logs."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, timestamp, threat_type, confidence, mse_score, bbox_json, synced "
            "FROM threat_logs ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        logs = []
        for r in rows:
            logs.append({
                'log_id': r[0],
                'timestamp': time.strftime("%H:%M:%S", time.localtime(r[1])),
                'type': r[2],
                'confidence': r[3],
                'mse_score': r[4],
                'bbox': json.loads(r[5]),
                'synced': bool(r[6])
            })
        return logs

    def _start_network_monitor(self):
        self.auto_sync_running = True
        self.monitor_thread = threading.Thread(target=self._network_loop, daemon=True)
        self.monitor_thread.start()

    def _network_loop(self):
        while self.auto_sync_running:
            time.sleep(3.0)
            try:
                pending = self.get_pending_sync_count()
                if pending > 0 and self.is_connected:
                    self._synchronize_logs()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as err:
                logger.debug("Database unavailable or busy during background sync check: %s", err)

    def _synchronize_logs(self):
        """Synchronizes local offline logs via pluggable sync callback stub."""
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        cursor = conn.cursor()

        # If a custom sync callback is attached, invoke it with un-synced items
        if self.sync_callback:
            cursor.execute("SELECT * FROM inspection_logs WHERE synced = 0")
            un_synced = cursor.fetchall()
            if un_synced:
                success = self.sync_callback(un_synced)
                if not success:
                    conn.close()
                    return

        cursor.execute("UPDATE inspection_logs SET synced = 1 WHERE synced = 0")
        cursor.execute("UPDATE threat_logs SET synced = 1 WHERE synced = 0")
        conn.commit()
        conn.close()

    def toggle_simulated_connection(self) -> bool:
        self.is_connected = not self.is_connected
        return self.is_connected
