"""
main.py
FastAPI server — Drone Security Analyst Agent
Endpoints:
  GET  /                    → HTML dashboard
  POST /run-pipeline        → Run REAL video pipeline (BLIP on actual frames)
  POST /run-sim-pipeline    → Run simulated text pipeline (fallback)
  GET  /frames              → All indexed frames
  GET  /frames/search       → Search by object keyword
  GET  /frames/time         → Search by time range
  GET  /frames/location     → Search by location
  GET  /alerts              → All triggered alerts
  GET  /stats               → Summary stats
  POST /agent/query         → Ask the LangChain agent
  GET  /video-summary       → One-sentence video summary
  GET  /health              → Health check
"""

import os
import sys

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# Set default env paths relative to project root
os.environ.setdefault("DB_PATH",        os.path.join(BASE_DIR, "drone_security.db"))
os.environ.setdefault("RULES_PATH",     os.path.join(BASE_DIR, "simdata", "alert_rules.yaml"))
os.environ.setdefault("TELEMETRY_PATH", os.path.join(BASE_DIR, "simdata", "telemetry.json"))
os.environ.setdefault("FRAMES_PATH",    os.path.join(BASE_DIR, "simdata", "frames.json"))

FRAMES_DIR  = os.path.join(BASE_DIR, "frames")
VIDEO_PATH  = os.path.join(BASE_DIR, "security_footage.mp4")

from storage.indexer import (
    get_all_frames, get_all_alerts, get_all_events,
    query_frames_by_object, query_frames_by_time,
    query_frames_by_location, get_summary_stats, init_db,
)
from agent.security_agent import run_agent

app = FastAPI(
    title="Drone Security Analyst Agent",
    description="AI-powered drone surveillance — FlytBase Assignment",
    version="2.0.0",
)

_pipeline_result = {}


class AgentQuery(BaseModel):
    query: str


# ── Dashboard ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_path = os.path.join(BASE_DIR, "static", "templates", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


# ── REAL video pipeline ───────────────────────────────────────────────────────

@app.post("/run-pipeline")
def trigger_pipeline():
    """
    Run the REAL pipeline:
    1. Extract frames from security_footage.mp4 using OpenCV
    2. Run BLIP VLM on each real frame
    3. Run alert engine
    4. Index into SQLite
    """
    global _pipeline_result

    # Check video exists
    if not os.path.exists(VIDEO_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Video not found at {VIDEO_PATH}. Please add security_footage.mp4 to the project root."
        )

    try:
        # Import here to avoid loading BLIP at startup
        sys.path.insert(0, BASE_DIR)
        from extract_frames import extract_frames
        from vlm.real_analyzer import analyze_real_frame, generate_video_summary, analyze_all_frames, load_cache, save_cache
        from alert_engine.engine import evaluate_frame, load_rules
        from storage.indexer import index_frame, log_alert, log_event

        init_db()

        # Step 1: Extract frames
        print("[Pipeline] Extracting frames from video...")
        frame_infos = extract_frames(VIDEO_PATH, FRAMES_DIR)
        if not frame_infos:
            raise Exception("No frames extracted from video.")

        # Step 2: Load rules
        rules        = load_rules()
        all_analyses = []
        all_alerts   = []

        # Step 3: Analyze with BLIP (uses cache if same video)
        print(f"[Pipeline] Running BLIP on {len(frame_infos)} frames (using cache if available)...")
        cached = load_cache(VIDEO_PATH)
        if cached:
            all_analyses = cached
            print(f"[Pipeline] Cache hit — loaded {len(all_analyses)} analyses instantly ✅")
        else:
            for frame_info in frame_infos:
                analysis = analyze_real_frame(frame_info)
                all_analyses.append(analysis)
            save_cache(VIDEO_PATH, all_analyses)
            print(f"[Pipeline] BLIP done — cached for next run ✅")

            log_event(
                time=analysis["time"],
                location=analysis["location"],
                event_type="real_frame_blip",
                description=analysis["caption"],
                objects=analysis["detected_objects"],
            )

            alerts    = evaluate_frame(analysis, rules)
            has_alert = len(alerts) > 0

            index_frame(
                frame_id=analysis["frame_id"],
                time=analysis["time"],
                location=analysis["location"],
                description=analysis["caption"],
                objects=analysis["detected_objects"],
                alert_flag=has_alert,
            )

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

        # Step 4: Summary
        summary = generate_video_summary(all_analyses)
        stats   = get_summary_stats()

        _pipeline_result = {
            "stats":         stats,
            "video_summary": summary,
            "analyses":      all_analyses,
            "alerts":        all_alerts,
            "source":        "real_video",
        }

        print(f"[Pipeline] Done. Frames: {stats['total_frames']} | Alerts: {stats['total_alerts']}")

        return {
            "status":        "success",
            "source":        "real_video_blip",
            "stats":         stats,
            "video_summary": summary,
            "alerts_count":  len(all_alerts),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Simulated pipeline (fallback) ─────────────────────────────────────────────

@app.post("/run-sim-pipeline")
def trigger_sim_pipeline():
    """Run simulated text pipeline (no video needed, for testing)."""
    global _pipeline_result
    try:
        from ingestion.pipeline import run_pipeline
        result          = run_pipeline(verbose=False)
        _pipeline_result = result
        return {
            "status":        "success",
            "source":        "simulated_text",
            "stats":         result["stats"],
            "video_summary": result["video_summary"],
            "alerts_count":  len(result["alerts"]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Frame endpoints ───────────────────────────────────────────────────────────

@app.get("/frames")
def get_frames():
    init_db()
    return get_all_frames()

@app.get("/frames/search")
def search_frames(keyword: str = Query(...)):
    init_db()
    results = query_frames_by_object(keyword)
    return {"keyword": keyword, "count": len(results), "frames": results}

@app.get("/frames/time")
def frames_by_time(start: str = Query("00:00"), end: str = Query("23:59")):
    init_db()
    results = query_frames_by_time(start, end)
    return {"start": start, "end": end, "count": len(results), "frames": results}

@app.get("/frames/location")
def frames_by_location(location: str = Query(...)):
    init_db()
    results = query_frames_by_location(location)
    return {"location": location, "count": len(results), "frames": results}


# ── Alert + stats ─────────────────────────────────────────────────────────────

@app.get("/alerts")
def get_alerts():
    init_db()
    alerts = get_all_alerts()
    return {"count": len(alerts), "alerts": alerts}

@app.get("/stats")
def get_stats():
    init_db()
    return get_summary_stats()


# ── Video summary ─────────────────────────────────────────────────────────────

@app.get("/video-summary")
def video_summary():
    global _pipeline_result
    if not _pipeline_result:
        return {"summary": "Pipeline not yet run. Click Run Pipeline first."}
    return {"summary": _pipeline_result.get("video_summary", "No summary available.")}


# ── Agent ─────────────────────────────────────────────────────────────────────

@app.post("/agent/query")
def agent_query(body: AgentQuery):
    init_db()
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = run_agent(body.query)
    return result


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    video_exists = os.path.exists(VIDEO_PATH)
    frames_exist = os.path.exists(FRAMES_DIR) and len(os.listdir(FRAMES_DIR)) > 0 if os.path.exists(FRAMES_DIR) else False
    return {
        "status":        "ok",
        "service":       "Drone Security Analyst Agent v2",
        "video_found":   video_exists,
        "frames_ready":  frames_exist,
        "db_path":       os.environ.get("DB_PATH"),
    }