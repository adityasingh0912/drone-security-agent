"""
alert_engine/engine.py
Rule-based alert engine.
Loads rules from alert_rules.yaml and checks each VLM analysis
against all rules. Returns a list of triggered alerts.
"""

import yaml
import os
from typing import Optional

RULES_PATH = os.environ.get("RULES_PATH", "simdata/alert_rules.yaml")


def load_rules() -> list:
    with open(RULES_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config.get("rules", [])


def _parse_hour(time_str: str) -> int:
    try:
        return int(time_str.split(":")[0])
    except Exception:
        return -1


def _check_time_range(time_str: str, time_range: list) -> bool:
    """
    Check if time_str falls in [start, end].
    Handles midnight-crossing ranges like 23:00 -> 05:00.
    """
    start_h = _parse_hour(time_range[0])
    end_h   = _parse_hour(time_range[1])
    hour    = _parse_hour(time_str)

    if start_h <= end_h:
        return start_h <= hour <= end_h
    else:
        # Wraps midnight
        return hour >= start_h or hour <= end_h


def evaluate_frame(analysis: dict, rules: Optional[list] = None) -> list:
    """
    Evaluate a single VLM analysis dict against all rules.
    Returns a list of triggered alert dicts.
    """
    if rules is None:
        rules = load_rules()

    triggered = []
    desc  = analysis["raw_description"].lower()
    loc   = analysis["location"]
    time  = analysis["time"]
    frame = analysis["frame_id"]

    for rule in rules:
        condition = rule.get("condition", {})
        matched   = False

        # Time range check
        time_range = condition.get("time_range")
        if time_range:
            if not _check_time_range(time, time_range):
                continue  # rule requires a specific time window; skip if outside

        # Keyword check (any keyword must appear in description)
        keywords = condition.get("keywords", [])
        if keywords:
            if not any(kw.lower() in desc for kw in keywords):
                continue

        # Location filter (optional)
        locations = condition.get("locations", [])
        if locations:
            if not any(l.lower() in loc.lower() for l in locations):
                continue

        # All conditions passed
        message = rule["alert_template"].format(location=loc, time=time)
        triggered.append({
            "frame_id": frame,
            "rule_id":  rule["id"],
            "name":     rule["name"],
            "severity": rule["severity"],
            "message":  message,
            "time":     time,
            "location": loc,
        })

    return triggered


def evaluate_all_frames(analyses: list) -> list:
    """Run alert evaluation across all frame analyses."""
    rules = load_rules()
    all_alerts = []
    for analysis in analyses:
        alerts = evaluate_frame(analysis, rules)
        all_alerts.extend(alerts)
    return all_alerts
