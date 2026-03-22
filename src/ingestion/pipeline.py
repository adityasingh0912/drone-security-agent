"""
ingestion/pipeline.py
Main ingestion pipeline.
1. Load simulated telemetry + frame data
2. Merge by timestamp
3. Run VLM analyzer on each frame
4. Run alert engine on each analysis
5. Index everything into SQLite
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from storage.indexer import init_db, index_frame, log_alert, log_event, get_summary_stats
from vlm.analyzer import analyze_frame, generate_video_summary
from alert_engine.engine import evaluate_frame, load_rules

TELEMETRY_PATH = os.environ.get("TELEMETRY_PATH", "simdata/telemetry.json")
FRAMES_PATH    = os.environ.get("FRAMES_PATH",    "simdata/frames.json")


def load_data():
    with open(TELEMETRY_PATH) as f:
        telemetry = json.load(f)
    with open(FRAMES_PATH) as f:
        frames = json.load(f)
    # Build telemetry lookup by time
    telem_map = {t["time"]: t for t in telemetry}
    return frames, telem_map


def run_pipeline(verbose: bool = True) -> dict:
    """
    Full ingestion pipeline. Returns summary dict.
    """
    init_db()
    frames, telem_map = load_data()
    rules = load_rules()

    all_analyses = []
    all_alerts   = []

    if verbose:
        print("\n" + "="*60)
        print("   DRONE SECURITY ANALYST — INGESTION PIPELINE")
        print("="*60)

    for frame in frames:
        fid  = frame["frame_id"]
        time = frame["time"]
        loc  = frame["location"]
        desc = frame["description"]

        # Merge telemetry context
        telem = telem_map.get(time, {})
        alt   = telem.get("altitude_m", "N/A")

        if verbose:
            print(f"\n[Frame {fid:02d}] {time} | {loc} | Alt: {alt}m")
            print(f"  Description : {desc}")

        # VLM analysis
        analysis = analyze_frame(fid, time, loc, desc)
        all_analyses.append(analysis)

        if verbose:
            print(f"  Objects     : {analysis['detected_objects']}")
            print(f"  Caption     : {analysis['caption']}")
            print(f"  Risk Score  : {analysis['risk_score']}/10")

        # Log event
        log_event(
            time=time,
            location=loc,
            event_type="frame_analysis",
            description=analysis["caption"],
            objects=analysis["detected_objects"],
        )

        # Alert engine
        alerts = evaluate_frame(analysis, rules)
        has_alert = len(alerts) > 0

        # Index frame
        index_frame(
            frame_id=fid,
            time=time,
            location=loc,
            description=desc,
            objects=analysis["detected_objects"],
            alert_flag=has_alert,
        )

        # Log alerts
        for alert in alerts:
            log_alert(
                frame_id=fid,
                rule_id=alert["rule_id"],
                severity=alert["severity"],
                message=alert["message"],
                time=time,
                location=loc,
            )
            all_alerts.append(alert)
            if verbose:
                print(f"  🚨 ALERT [{alert['severity']}]: {alert['message']}")

    # Video summary (bonus)
    video_summary = generate_video_summary(all_analyses)
    stats = get_summary_stats()

    if verbose:
        print("\n" + "="*60)
        print("   PIPELINE COMPLETE")
        print("="*60)
        print(f"  Frames processed : {stats['total_frames']}")
        print(f"  Alerts triggered : {stats['total_alerts']}")
        print(f"  Critical         : {stats['critical_alerts']}")
        print(f"  High             : {stats['high_alerts']}")
        print(f"\n  VIDEO SUMMARY: {video_summary}")
        print("="*60)

    return {
        "stats":         stats,
        "video_summary": video_summary,
        "analyses":      all_analyses,
        "alerts":        all_alerts,
    }


if __name__ == "__main__":
    run_pipeline(verbose=True)
