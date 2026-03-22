"""
real_pipeline.py
Full pipeline for REAL video analysis.
1. Extract frames from security_footage.mp4
2. Run BLIP VLM on each frame
3. Run alert engine
4. Index into SQLite
5. Print summary

Run from project root:
    python real_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

os.environ.setdefault("DB_PATH",        "drone_security_real.db")
os.environ.setdefault("RULES_PATH",     "simdata/alert_rules.yaml")
os.environ.setdefault("TELEMETRY_PATH", "simdata/telemetry.json")
os.environ.setdefault("FRAMES_PATH",    "simdata/frames.json")

from extract_frames import extract_frames
from src.vlm.real_analyzer import analyze_real_frame, generate_video_summary
from src.alert_engine.engine import evaluate_frame, load_rules
from src.storage.indexer import (
    init_db, index_frame, log_alert, log_event, get_summary_stats
)


def run_real_pipeline(video_path: str = "security_footage.mp4",
                      frames_dir: str = "frames",
                      verbose: bool = True) -> dict:

    print("\n" + "="*60)
    print("   DRONE SECURITY ANALYST — REAL VIDEO PIPELINE")
    print("="*60)

    # Step 1: Extract frames
    print(f"\n[STEP 1] Extracting frames from {video_path}...")
    frame_infos = extract_frames(video_path, frames_dir)
    if not frame_infos:
        print("[ERROR] No frames extracted. Check video file path.")
        return {}

    # Step 2: Init DB
    init_db()
    rules        = load_rules()
    all_analyses = []
    all_alerts   = []

    # Step 3: Analyze each frame with BLIP
    print(f"\n[STEP 2] Running BLIP VLM on {len(frame_infos)} frames...")
    from src.vlm.real_analyzer import analyze_real_frame

    for frame_info in frame_infos:
        analysis = analyze_real_frame(frame_info)
        all_analyses.append(analysis)

        if verbose:
            print(f"\n  [Frame {analysis['frame_id']:02d}] {analysis['time']} | {analysis['location']}")
            print(f"    Caption : {analysis['caption']}")
            print(f"    Objects : {analysis['detected_objects']}")
            print(f"    Risk    : {analysis['risk_score']}/10")

        # Log event
        log_event(
            time=analysis["time"],
            location=analysis["location"],
            event_type="real_frame_analysis",
            description=analysis["caption"],
            objects=analysis["detected_objects"],
        )

        # Alert engine
        alerts    = evaluate_frame(analysis, rules)
        has_alert = len(alerts) > 0

        # Index frame
        index_frame(
            frame_id=analysis["frame_id"],
            time=analysis["time"],
            location=analysis["location"],
            description=analysis["caption"],
            objects=analysis["detected_objects"],
            alert_flag=has_alert,
        )

        # Log alerts
        for alert in alerts:
            log_alert(
                frame_id=analysis["frame_id"],
                rule_id=alert["rule_id"],
                severity=alert["severity"],
                message=alert["message"],
                time=analysis["time"],
                location=analysis["location"],
            )
            all_alerts.append(alert)
            if verbose:
                print(f"    🚨 ALERT [{alert['severity']}]: {alert['message']}")

    # Step 4: Summary
    summary = generate_video_summary(all_analyses)
    stats   = get_summary_stats()

    print("\n" + "="*60)
    print("   REAL VIDEO PIPELINE COMPLETE")
    print("="*60)
    print(f"  Frames processed : {stats['total_frames']}")
    print(f"  Alerts triggered : {stats['total_alerts']}")
    print(f"  Critical         : {stats['critical_alerts']}")
    print(f"  High             : {stats['high_alerts']}")
    print(f"\n  VIDEO SUMMARY: {summary}")
    print("="*60)

    return {
        "stats":         stats,
        "video_summary": summary,
        "analyses":      all_analyses,
        "alerts":        all_alerts,
    }


if __name__ == "__main__":
    run_real_pipeline()