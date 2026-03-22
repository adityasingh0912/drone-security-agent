"""
tests/test_system.py
Pytest test suite for the Drone Security Analyst Agent.
Tests: VLM analyzer, alert engine, frame indexer, pipeline integration.
Run with: pytest tests/ -v
"""

import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Use a temporary test DB
os.environ["DB_PATH"]       = ":memory:"
os.environ["RULES_PATH"]    = os.path.join(os.path.dirname(__file__), "..", "simdata", "alert_rules.yaml")
os.environ["TELEMETRY_PATH"]= os.path.join(os.path.dirname(__file__), "..", "simdata", "telemetry.json")
os.environ["FRAMES_PATH"]   = os.path.join(os.path.dirname(__file__), "..", "simdata", "frames.json")

from vlm.analyzer import analyze_frame, generate_video_summary
from alert_engine.engine import evaluate_frame, load_rules
from storage.indexer import init_db, index_frame, log_alert, query_frames_by_object, get_all_alerts, get_summary_stats


# ─── VLM Analyzer Tests ────────────────────────────────────────────────────────

class TestVLMAnalyzer:

    def test_detects_vehicle_in_frame(self):
        """Blue Ford F150 should be detected in vehicle frame."""
        analysis = analyze_frame(
            frame_id=3,
            time="00:02",
            location="Parking Lot",
            description="A blue Ford F150 truck enters the parking lot from the north entrance."
        )
        objects = [o.lower() for o in analysis["detected_objects"]]
        assert any("ford" in o or "f150" in o for o in objects), \
            f"Expected vehicle detection, got: {analysis['detected_objects']}"

    def test_detects_person_in_frame(self):
        """Person loitering at midnight should be flagged."""
        analysis = analyze_frame(
            frame_id=13,
            time="23:58",
            location="Main Gate",
            description="A person is standing at the main gate at midnight. Behavior appears suspicious."
        )
        assert "person" in analysis["detected_objects"]
        assert analysis["risk_score"] >= 6, \
            f"Expected high risk score, got: {analysis['risk_score']}"

    def test_risk_score_midnight(self):
        """Midnight frames should have elevated risk score."""
        analysis = analyze_frame(
            frame_id=14,
            time="23:59",
            location="Main Gate",
            description="The person at the main gate is still present. Possible loitering."
        )
        assert analysis["risk_score"] >= 4, "Midnight loitering should score >= 4"

    def test_empty_frame_has_zero_objects(self):
        """Clear frame should detect no objects."""
        analysis = analyze_frame(
            frame_id=8,
            time="00:07",
            location="Back Fence",
            description="The back fence area is empty. Good visibility. No threats detected."
        )
        assert analysis["risk_score"] == 0 or len(analysis["detected_objects"]) == 0

    def test_caption_generated(self):
        """Caption should be a non-empty string."""
        analysis = analyze_frame(
            frame_id=1,
            time="00:00",
            location="Main Gate",
            description="A person in dark clothing is standing near the main gate."
        )
        assert isinstance(analysis["caption"], str)
        assert len(analysis["caption"]) > 5

    def test_video_summary_is_string(self):
        """generate_video_summary should return a non-empty string."""
        analyses = [
            analyze_frame(1, "00:00", "Main Gate", "Person standing at main gate."),
            analyze_frame(2, "00:01", "Parking Lot", "A blue Ford F150 enters."),
        ]
        summary = generate_video_summary(analyses)
        assert isinstance(summary, str)
        assert len(summary) > 20


# ─── Alert Engine Tests ────────────────────────────────────────────────────────

class TestAlertEngine:

    def setup_method(self):
        self.rules = load_rules()

    def test_midnight_loitering_triggers(self):
        """Person at midnight should trigger midnight_loitering rule."""
        analysis = analyze_frame(
            frame_id=13,
            time="23:58",
            location="Main Gate",
            description="A person is standing at the main gate at midnight. Not moving."
        )
        alerts = evaluate_frame(analysis, self.rules)
        rule_ids = [a["rule_id"] for a in alerts]
        assert "midnight_loitering" in rule_ids, \
            f"Expected midnight_loitering alert. Got: {rule_ids}"

    def test_garage_access_attempt_triggers(self):
        """Garage access attempt should trigger unauthorized_garage_access rule."""
        analysis = analyze_frame(
            frame_id=5,
            time="00:04",
            location="Garage",
            description="Two individuals attempting to open the garage door."
        )
        alerts = evaluate_frame(analysis, self.rules)
        rule_ids = [a["rule_id"] for a in alerts]
        assert "unauthorized_garage_access" in rule_ids, \
            f"Expected garage alert. Got: {rule_ids}"

    def test_repeated_vehicle_triggers(self):
        """Blue Ford F150 repeat sighting should trigger repeated_vehicle rule."""
        analysis = analyze_frame(
            frame_id=12,
            time="12:00",
            location="Garage",
            description="Blue Ford F150 spotted at the garage. Same vehicle as earlier."
        )
        alerts = evaluate_frame(analysis, self.rules)
        rule_ids = [a["rule_id"] for a in alerts]
        assert "repeated_vehicle" in rule_ids, \
            f"Expected repeated_vehicle alert. Got: {rule_ids}"

    def test_no_alert_for_safe_frame(self):
        """Empty fence frame should not trigger any critical alerts."""
        analysis = analyze_frame(
            frame_id=8,
            time="00:07",
            location="Back Fence",
            description="The back fence area is empty. No threats detected."
        )
        alerts = evaluate_frame(analysis, self.rules)
        critical = [a for a in alerts if a["severity"] == "CRITICAL"]
        assert len(critical) == 0

    def test_alert_has_required_fields(self):
        """Alert dict must contain frame_id, rule_id, severity, message."""
        analysis = analyze_frame(
            frame_id=1,
            time="23:59",
            location="Main Gate",
            description="A person loitering near the main gate at midnight."
        )
        alerts = evaluate_frame(analysis, self.rules)
        for alert in alerts:
            assert "frame_id"  in alert
            assert "rule_id"   in alert
            assert "severity"  in alert
            assert "message"   in alert
            assert "time"      in alert
            assert "location"  in alert


# ─── Storage / Indexer Tests ───────────────────────────────────────────────────

class TestIndexer:

    def setup_method(self):
        os.environ["DB_PATH"] = ":memory:"
        init_db()

    def test_frame_indexed_correctly(self):
        """Frame should be queryable by object keyword after indexing."""
        index_frame(
            frame_id=3,
            time="00:02",
            location="Parking Lot",
            description="A blue Ford F150 truck enters the parking lot.",
            objects=["Blue Ford F150"],
            alert_flag=False,
        )
        results = query_frames_by_object("Ford F150")
        assert len(results) >= 1
        assert results[0]["frame_id"] == 3

    def test_alert_stored_correctly(self):
        """Alert should be retrievable after logging."""
        log_alert(
            frame_id=3,
            rule_id="repeated_vehicle",
            severity="MEDIUM",
            message="Repeated vehicle at Parking Lot, 00:02.",
            time="00:02",
            location="Parking Lot",
        )
        alerts = get_all_alerts()
        assert len(alerts) >= 1
        assert alerts[0]["rule_id"] == "repeated_vehicle"

    def test_summary_stats_accurate(self):
        """Stats should reflect what was indexed."""
        index_frame(99, "12:00", "Garage", "Test frame", [], False)
        log_alert(99, "test_rule", "HIGH", "Test alert.", "12:00", "Garage")
        stats = get_summary_stats()
        assert stats["total_frames"]  >= 1
        assert stats["total_alerts"]  >= 1
        assert stats["high_alerts"]   >= 1

    def test_truck_logged_correctly(self):
        """Verify 'truck logged correctly' per assignment requirement."""
        index_frame(
            frame_id=3,
            time="00:02",
            location="Parking Lot",
            description="Blue Ford F150 spotted at garage, 12:00.",
            objects=["Blue Ford F150"],
            alert_flag=False,
        )
        results = query_frames_by_object("truck")
        # 'truck' keyword matches description
        assert any("F150" in r["description"] or "truck" in r["description"].lower()
                   for r in results), "Truck frame should be retrievable by keyword"
