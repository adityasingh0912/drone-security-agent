# 🛸 Drone Security Analyst Agent

> AI-powered drone surveillance system for automated property monitoring.
> **FlytBase AI Engineer Assignment** — Built with BLIP VLM, Groq qwen3-32b, FastAPI, SQLite, and OpenCV.

---

## 📋 Feature Spec

**Value to property owners:** Enhances physical security by automating 24/7 drone surveillance — processing real video footage, detecting vehicles, persons, and suspicious behaviors using a Vision Language Model, and enabling natural language querying via an AI agent.

**Key requirements:**
1. **Real video frame analysis** — OpenCV extracts frames from actual footage. BLIP VLM runs dual-pass captioning + 7 VQA safety checks per frame to detect objects and safety events.
2. **Rule-based alert generation** — Predefined YAML rules trigger alerts for crowd gathering, road obstructions, loitering, person-on-ground emergencies, motorcycles, and vehicle activity.
3. **Queryable frame index** — All frames stored in SQLite with BLIP captions, detected objects, VQA flags, and timestamps — queryable by time range, object keyword, or location.

---

## 🏗️ Architecture

```
security_footage.mp4        ← Real input video
frames/                     ← Extracted JPG frames (1 per second)
frames_cache.json           ← BLIP result cache (skips reprocessing same video)
simdata/
  telemetry.json            ← Simulated drone GPS + altitude data
  alert_rules.yaml          ← Rule definitions (YAML, 9 alert types)

src/
  vlm/
    real_analyzer.py        ← BLIP VLM: dual-pass caption + 7 VQA checks per frame
    analyzer.py             ← Mock VLM fallback (keyword-based, no model needed)
  agent/security_agent.py   ← Groq qwen3-32b agent with 5 tool-calling tools
  alert_engine/engine.py    ← Rule checker against alert_rules.yaml
  storage/indexer.py        ← SQLite frame index + query helpers
  ingestion/pipeline.py     ← Simulated text pipeline (fallback/testing)

extract_frames.py           ← OpenCV frame extractor
real_pipeline.py            ← Full real video pipeline (BLIP + alerts + indexing)
main.py                     ← FastAPI server + HTML dashboard
static/templates/index.html ← Dark-themed dashboard UI
tests/test_system.py        ← pytest test suite (14 test cases)
```

**Data flow:**
```
security_footage.mp4
        ↓
OpenCV frame extractor (1 FPS)
        ↓
BLIP VLM — dual-pass caption + VQA safety checks
        ↓
Alert engine (YAML rules) → SQLite index
        ↓
FastAPI server ← Groq qwen3-32b agent (5 tools)
        ↓
HTML Dashboard
```

---

## ⚙️ Setup & Running

### Prerequisites
- Python 3.10–3.12 recommended (3.13 works with `--only-binary=:all:`)
- A video file named `security_footage.mp4` in the project root
- (Optional) Free Groq API key from [console.groq.com](https://console.groq.com)

### Option A — Local

```bash
# 1. Clone the repo
git clone https://github.com/adityasingh0912/drone-security-agent
cd drone-security-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt --only-binary=:all:

# 4. Add your Groq API key (optional but recommended)
echo GROQ_API_KEY=your_key_here > .env

# 5. Extract frames from video
python extract_frames.py

# 6. Run full BLIP pipeline (first run ~3-5 min, cached after)
python real_pipeline.py

# 7. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 8. Open dashboard
# http://localhost:8000
```

### Option B — Docker

```bash
docker-compose up --build
# Open: http://localhost:8000
```

---

## 🚀 Usage

### Dashboard
- Open `http://localhost:8000`
- Click **▶ Run Pipeline** — BLIP analyzes all frames, alerts trigger, results cached
- Search frames by keyword, ask the agent questions

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/run-pipeline` | Run real BLIP pipeline on video |
| POST | `/run-sim-pipeline` | Run simulated text pipeline (no video needed) |
| GET | `/frames` | All indexed frames |
| GET | `/frames/search?keyword=person` | Search by object keyword |
| GET | `/frames/time?start=00:00&end=00:10` | Search by time range |
| GET | `/frames/location?location=Street` | Search by location |
| GET | `/alerts` | All triggered alerts |
| GET | `/stats` | Summary statistics |
| GET | `/video-summary` | One-sentence AI summary (bonus) |
| POST | `/agent/query` | Ask the Groq security agent |
| GET | `/health` | Health check + video/cache status |

### Agent Example Queries
```
"What activity was detected on the street?"
"Were any motorcycles or vehicles detected?"
"What are the highest severity alerts?"
"Show me what happened between 00:00 and 00:10"
"Give me a full security report for today"
```

---

## 🤖 VLM — BLIP with VQA

Each frame analyzed two ways:

**Dual-pass captioning** (anti-hallucination):
- Pass 1: Free caption
- Pass 2: Conditional prompt anchored to `"a security camera shows..."`
- Higher keyword-scoring caption is selected

**7 VQA safety checks per frame:**

| Check | Question | Alert Flag |
|-------|----------|------------|
| people_crossing | Are there people crossing the street? | PEDESTRIANS_CROSSING |
| vehicle_on_road | Is there a car or vehicle on the road? | VEHICLE_ON_ROAD |
| crowd_present | Is there a large crowd? | CROWD_DETECTED |
| person_on_ground | Is anyone lying on the ground? | PERSON_ON_GROUND (CRITICAL) |
| motorcycle | Is there a motorcycle or scooter? | MOTORCYCLE_DETECTED |
| road_blocked | Is there a vehicle blocking the road? | ROAD_BLOCKED |
| tuk_tuk | Is there a tuk-tuk or rickshaw? | TUK_TUK_DETECTED |

---

## 🚨 Alert Rules (alert_rules.yaml)

| Rule | Severity | Trigger |
|------|----------|---------|
| person_on_ground | CRITICAL | Person lying on ground |
| road_blocked | HIGH | Vehicle obstructing traffic |
| crowd_gathering | HIGH | Large crowd detected |
| running_detected | HIGH | Person running |
| suspicious_loitering | HIGH | Person stationary |
| vehicle_activity | MEDIUM | Vehicle in monitored zone |
| motorcycle_detected | MEDIUM | Motorcycle present |
| person_activity | LOW | General person activity |
| street_activity | LOW | General street activity |

---

## 🧪 Test Cases

```bash
pytest tests/ -v
```

14 test cases covering VLM analyzer, alert engine, and storage indexer.
All tests use an in-memory SQLite DB to avoid side effects.

---

## 🤖 AI Tools Used

| Tool | Contribution |
|------|-------------|
| **Claude (Anthropic)** | Project skeleton, FastAPI endpoints, dashboard, SQLite schema, alert engine, test suite, VQA implementation, caching system |
| **Groq qwen3-32b** | Live security analyst agent — tool calling, multi-step reasoning |
| **BLIP (Salesforce)** | VLM for real frame captioning and VQA safety checks |
| **Gemini (Google)** | Suggested domain-specific alerts for urban street scenes — I evaluated and implemented feasible ones via VQA |

---

## 🎯 Design Decisions

| Decision | Rationale |
|----------|-----------|
| BLIP over CLIP | BLIP generates captions + supports VQA. CLIP only does similarity matching |
| BLIP over BLIP2/LLaVA | BLIP-base runs on CPU. BLIP2/LLaVA require GPU |
| Direct Groq over LangChain agents | LangChain agents broke on Python 3.13. Direct Groq client is more stable |
| SQLite over ChromaDB | Sufficient for prototype. ChromaDB adds semantic search — natural next step |
| YAML alert rules | Extensible without code changes |
| Frame caching | BLIP takes 3-5s/frame on CPU. Cache skips reprocessing on same video |
| Dual-pass captioning | Reduces hallucination by selecting highest-scoring caption |

---

## 💡 What Could Be Better With More Time

- Real-time RTSP streaming with Redis queue
- YOLOv8 for fast bounding-box detection alongside BLIP
- BLIP2 / LLaVA on GPU for better caption quality
- ChromaDB for semantic frame search
- Multi-drone support with per-drone pipelines
- WebSocket real-time alert push to dashboard
- Helmet / PPE detection via fine-tuned YOLOv8
