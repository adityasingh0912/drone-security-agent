# 🛸 Drone Security Analyst Agent

> AI-powered drone surveillance system for automated property monitoring.
> **FlytBase AI Engineer Assignment** — Built with LangChain, FastAPI, SQLite, and a VLM mock analyzer.

---

## 📋 Feature Spec

**Value to property owners:** Enhances physical security by automating 24/7 drone surveillance — detecting vehicles, persons, and suspicious behaviors without human oversight.

**Key requirements:**
1. **Real-time frame analysis** — Every video frame is analyzed by a VLM to extract objects, actions, and risk scores.
2. **Rule-based alert generation** — Predefined YAML rules trigger alerts for midnight loitering, unauthorized access attempts, and repeated vehicle entries.
3. **Queryable frame index** — All frames are stored in SQLite, queryable by time, location, or detected object.

---

## 🏗️ Architecture

```
simdata/
  telemetry.json      ← Simulated drone GPS + altitude data
  frames.json         ← Simulated video frame descriptions
  alert_rules.yaml    ← Rule definitions (YAML)

src/
  ingestion/pipeline.py    ← Merges telemetry + frames, runs full pipeline
  vlm/analyzer.py          ← Mock VLM: extracts objects/actions, generates captions
  alert_engine/engine.py   ← Rule checker against alert_rules.yaml
  storage/indexer.py       ← SQLite frame index + query helpers
  agent/security_agent.py  ← LangChain ReAct agent with 5 tools

main.py               ← FastAPI server (all endpoints)
static/templates/index.html  ← HTML dashboard UI
tests/test_system.py  ← pytest test suite (12 test cases)
```

**Data flow:**
```
Frames + Telemetry → Pipeline → VLM Analyzer → Alert Engine → SQLite Index
                                                               ↓
FastAPI ←────────────────────────────────────────── LangChain Agent
   ↓
HTML Dashboard
```

---

## ⚙️ Setup & Running

### Option A — Local (recommended for demo)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/drone-security-agent
cd drone-security-agent

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Set API key for real LLM — skip for offline mock mode
export GROQ_API_KEY=your_groq_key_here   # Free at console.groq.com

# 5. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# 6. Open dashboard
# http://localhost:8000
```

### Option B — Docker

```bash
# Build and run
docker-compose up --build

# Open dashboard
# http://localhost:8000
```

---

## 🚀 Usage

### 1. Run the pipeline (web UI)
- Open `http://localhost:8000`
- Click **▶ Run Pipeline**
- Frames are analyzed, alerts triggered, and everything indexed

### 2. Run the pipeline (CLI)
```bash
cd src
python -m ingestion.pipeline
```

### 3. Run tests
```bash
pytest tests/ -v
```

### 4. API endpoints

| Method | Endpoint             | Description                          |
|--------|----------------------|--------------------------------------|
| POST   | `/run-pipeline`      | Run ingestion pipeline               |
| GET    | `/frames`            | All indexed frames                   |
| GET    | `/frames/search?keyword=truck` | Search frames by object   |
| GET    | `/frames/time?start=23:00&end=23:59` | Search by time range |
| GET    | `/frames/location?location=Main Gate` | Search by location  |
| GET    | `/alerts`            | All triggered alerts                 |
| GET    | `/stats`             | Summary statistics                   |
| GET    | `/video-summary`     | One-sentence video summary (bonus)   |
| POST   | `/agent/query`       | Ask the LangChain agent a question   |

### 5. Agent example queries
```
"What happened at midnight?"
"Show all truck events"
"Were there any critical alerts today?"
"What objects were detected at the Garage?"
"Summarize today's security events"
```

---

## 🧪 Test Cases

```bash
pytest tests/ -v
```

| Test | What it verifies |
|------|-----------------|
| `test_detects_vehicle_in_frame`     | Blue Ford F150 correctly detected |
| `test_detects_person_in_frame`      | Person at midnight flagged |
| `test_risk_score_midnight`          | Midnight elevates risk score |
| `test_empty_frame_has_zero_objects` | Clear frame = no false positives |
| `test_caption_generated`            | VLM produces non-empty caption |
| `test_video_summary_is_string`      | Summary generation works |
| `test_midnight_loitering_triggers`  | Alert rule fires at 23:xx |
| `test_garage_access_attempt_triggers` | CRITICAL alert on garage attempt |
| `test_repeated_vehicle_triggers`    | MEDIUM alert on repeat F150 |
| `test_no_alert_for_safe_frame`      | No false CRITICAL on empty frame |
| `test_alert_has_required_fields`    | Alert schema is complete |
| `test_frame_indexed_correctly`      | Truck logged + queryable |
| `test_alert_stored_correctly`       | Alert retrievable from DB |
| `test_summary_stats_accurate`       | Stats reflect indexed data |

---

## 🤖 AI Tools Used

| Tool | How it helped |
|------|--------------|
| **Claude (Anthropic)** | Generated the LangChain agent skeleton, VLM analyzer logic, alert engine rule checker, and full HTML dashboard. I customized the agent prompt, added the mock LLM fallback, and tuned alert rules. |
| **Claude** | Drafted architecture decisions, suggested SQLite schema, helped write pytest test cases |

---

## 🎯 Design Decisions

- **Mock VLM over real BLIP2** — Running BLIP2 requires GPU. For CPU-only prototype, a keyword-based mock produces identical structured output for demo purposes. Swap `analyze_frame()` in `vlm/analyzer.py` for real inference.
- **SQLite over ChromaDB** — Simpler, zero dependencies, fully queryable. ChromaDB would be better for semantic search at scale.
- **LangChain ReAct agent** — Chosen over simple prompt because it can chain multiple tool calls (e.g., search frames → get alerts → summarize), which demonstrates real reasoning.
- **Groq free tier (LLaMA 3)** — Fast, free, no GPU required. Falls back to MockSecurityLLM if no API key.
- **YAML rules** — Easier to extend without touching code. Security teams can add new rules without a redeploy.

---

## 💡 What Could Be Better (if more time)

- Integrate real BLIP2 / LLaVA on HuggingFace for actual image frame analysis
- Add WebSocket for real-time frame streaming to the dashboard
- ChromaDB for semantic frame search ("show me all nighttime outdoor scenes")
- Multi-drone support with per-drone telemetry streams
- Email/SMS alert delivery via Twilio or SendGrid
- More sophisticated anomaly detection with time-series models
