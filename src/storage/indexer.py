"""
storage/indexer.py
Frame-by-frame SQLite indexer. Stores every processed frame with
timestamp, location, description, detected objects, and alert flags.
Supports querying by time range, location, or object keyword.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional

DB_PATH = os.environ.get("DB_PATH", "drone_security.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS frames (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id    INTEGER UNIQUE,
            time        TEXT,
            location    TEXT,
            description TEXT,
            objects     TEXT,   -- JSON list of detected objects
            alert_flag  INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id    INTEGER,
            rule_id     TEXT,
            severity    TEXT,
            message     TEXT,
            time        TEXT,
            location    TEXT,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (frame_id) REFERENCES frames(frame_id)
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS event_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            time        TEXT,
            location    TEXT,
            event_type  TEXT,
            description TEXT,
            objects     TEXT,
            created_at  TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Initialized database.")


def index_frame(frame_id: int, time: str, location: str,
                description: str, objects: list, alert_flag: bool = False):
    """Insert or replace a processed frame into the index."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO frames (frame_id, time, location, description, objects, alert_flag)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (frame_id, time, location, description, json.dumps(objects), int(alert_flag)))
    conn.commit()
    conn.close()


def log_alert(frame_id: int, rule_id: str, severity: str,
              message: str, time: str, location: str):
    """Store a triggered alert."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO alerts (frame_id, rule_id, severity, message, time, location)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (frame_id, rule_id, severity, message, time, location))
    conn.commit()
    conn.close()


def log_event(time: str, location: str, event_type: str,
              description: str, objects: list):
    """Log a detected security event."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO event_log (time, location, event_type, description, objects)
        VALUES (?, ?, ?, ?, ?)
    """, (time, location, event_type, description, json.dumps(objects)))
    conn.commit()
    conn.close()


# ── Query helpers ──────────────────────────────────────────────────────────────

def query_frames_by_time(start: str, end: str) -> list:
    """Return frames within a time range (HH:MM format)."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM frames WHERE time >= ? AND time <= ?
        ORDER BY time ASC
    """, (start, end))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def query_frames_by_object(keyword: str) -> list:
    """Return frames whose description or objects contain the keyword."""
    conn = get_connection()
    c = conn.cursor()
    keyword_like = f"%{keyword.lower()}%"
    c.execute("""
        SELECT * FROM frames
        WHERE LOWER(description) LIKE ?
           OR LOWER(objects)     LIKE ?
        ORDER BY time ASC
    """, (keyword_like, keyword_like))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def query_frames_by_location(location: str) -> list:
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM frames WHERE LOWER(location) LIKE ?
        ORDER BY time ASC
    """, (f"%{location.lower()}%",))
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_alerts() -> list:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY created_at DESC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_frames() -> list:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM frames ORDER BY time ASC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_all_events() -> list:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM event_log ORDER BY time ASC")
    rows = [dict(r) for r in c.fetchall()]
    conn.close()
    return rows


def get_summary_stats() -> dict:
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) as total FROM frames")
    total_frames = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM alerts")
    total_alerts = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM alerts WHERE severity='CRITICAL'")
    critical = c.fetchone()["total"]
    c.execute("SELECT COUNT(*) as total FROM alerts WHERE severity='HIGH'")
    high = c.fetchone()["total"]
    conn.close()
    return {
        "total_frames": total_frames,
        "total_alerts": total_alerts,
        "critical_alerts": critical,
        "high_alerts": high,
    }
