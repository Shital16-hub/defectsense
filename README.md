# DefectSense

**Manufacturing Defect Root-Cause Intelligence — Hybrid ML + GenAI Multi-Agent System**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-orange)](https://langchain-ai.github.io/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq_llama3-purple)](https://groq.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00)](https://tensorflow.org)
[![Tests](https://img.shields.io/badge/tests-179_passing-brightgreen)](#testing)

Manufacturing plants lose **$260 billion per year** to unplanned downtime (Deloitte, 2024).
Existing condition-monitoring systems tell you *when* a machine failed — not *why*.
DefectSense closes that gap: a reasoning AI that ingests live sensor readings, detects
gradual deterioration before it becomes a failure, and delivers a root cause with ranked
maintenance actions and an explanation a factory-floor engineer can act on immediately.

---

## What This System Does

DefectSense is a real-time anomaly detection and root-cause reasoning platform built for
manufacturing environments. It continuously ingests sensor readings — voltage, rotation
speed, pressure, and vibration — from industrial machines and runs them through an LSTM
Autoencoder trained on 737,786 clean normal sequences from the Microsoft Azure PdM
dataset. When reconstruction error exceeds the learned threshold, an anomaly is flagged.

A LangGraph multi-agent pipeline then activates: a retrieval agent pulls the most
relevant historical maintenance records from Qdrant using semantic search, a ReAct
reasoning agent synthesises those records with current sensor trends and per-machine
memory to produce a structured root-cause report, and an alert generator formats the
result into plain language a maintenance engineer can act on without reading a data
science output. Every alert waits for human approval before being filed — or
auto-approves after 15 minutes if no operator is available. Approved incidents are
automatically indexed back into the RAG knowledge base, so the system gets more accurate
with every resolved failure it processes.

---

## Architecture

```
Sensor Stream (Azure PdM / Live Data)
         │
         ▼  POST /api/sensors/ingest
  ┌─────────────┐
  │  FastAPI     │  :8080
  │  :8080       │
  └──────┬──────┘
         │
         ▼
  AnomalyDetectorAgent
  ├── Redis buffer  (210 readings per machine, LPUSH + LTRIM)
  ├── LSTM Autoencoder  (seq_len=198, MSE reconstruction error)
  └── Failure type classifier  (BWF / RSF / PBF / EVF, z-score rules)
         │
    is_anomaly?
    ┌────┴────┐
   No        Yes
    │         │
  return    LangGraph Orchestrator
            │
            ├── ContextRetrieverAgent
            │    ├── Qdrant semantic search (maintenance_logs collection)
            │    └── Redis sensor trend summary
            │
            ├── A-MEM update (Zettelkasten note, anomaly context)
            │
            ├── RootCauseReasonerAgent  (llama-3.3-70b via Groq)
            │    ├── Reads: RAG incidents + sensor trends + Letta profile + A-MEM notes
            │    ├── Executes: THINK → ACT → OBSERVE → THINK → CONCLUDE
            │    └── Returns: root_cause, confidence, severity, evidence, actions
            │
            ├── AlertGeneratorAgent  (llama-3.1-8b via Groq)
            │    └── Saves alert to MongoDB with approved=None (PENDING)
            │
            ├── [HITL interrupt_before=apply_approval]
            │    ├── conf >= 0.95  →  auto-approve immediately
            │    └── conf <  0.95  →  human approves via Gradio dashboard / REST API
            │
            └── post_resolution_indexer
                 └── POST /api/maintenance-logs/add  →  Qdrant + MongoDB

─── MLOps ──────────────────────────────────────────────────────────────────────
  PostgreSQL (Supabase)          876,100 Azure PdM rows, training source of truth
  ml/models/                     Trained artefacts (gitignored, generated locally)
  MLflow SQLite                  challenger → champion alias lifecycle
  Evidently AI (03:00 UTC)       Nightly KS-test: live Redis window vs PG reference

─── Agent Memory ────────────────────────────────────────────────────────────────
  A-MEM (MongoDB + embeddings)   Zettelkasten notes, auto-linked by cosine similarity
  Letta (MongoDB)                Per-machine core memory + archival pattern ring
```

---

## Key Features

### 1. LSTM Autoencoder Anomaly Detection

The LSTM Autoencoder learns normal operating patterns across a 198-reading (hour) window
and flags gradual sensor drift — the slow deterioration that kills machines over days
rather than the sudden spikes that rule-based threshold systems are built for.

Reconstruction error (MSE) is compared against a threshold of 0.011913 (mean + 3 std
over 737,786 clean normal validation sequences). Readings where reconstruction error
exceeds this threshold are classified as anomalous. The model achieves **AUC 0.846**
on real pre-failure sequences from the Azure PdM dataset.

The sequence length of 198 was not chosen arbitrarily — it was computed automatically
by analysing the distribution of 761 recorded failure events to find the median time
window containing detectable precursor signals.

### 2. Failure Type Classification

When an anomaly is detected, sensor z-scores are used to classify the failure type
before the LLM pipeline runs, preserving accuracy through the pipeline:

| Type | Meaning | Trigger |
|---|---|---|
| **BWF** | Bearing Wear Failure | vibration z-score > 2.0 |
| **RSF** | Rotor Speed Failure | rotation z-score < −2.0 |
| **PBF** | Pressure Blockage Failure | pressure z-score > 2.0 |
| **EVF** | Electrical Overload Failure | voltage z-score > 2.0 |

Rules are evaluated in priority order (BWF first — most common failure type in the Azure
PdM dataset). A secondary magnitude comparison handles anomalies with multiple elevated
sensors. The classified failure type is then passed directly to the RAG search and LLM
prompt — the orchestrator never re-runs detection, so z-scores cannot shift from a
second Redis write.

### 3. ReAct Root Cause Reasoning

The LangGraph reasoning agent (`llama-3.3-70b-versatile` on Groq) executes a structured
THINK → ACT → OBSERVE → THINK → CONCLUDE loop with the following context injected into
every prompt:

- Sensor deviations (z-scores with CRITICAL / HIGH flags for |z| ≥ 3.0 / 2.0)
- Sensor trend summary over the last 10 Redis readings
- Top 3 similar past incidents from Qdrant (semantic search by failure type + deviations)
- Letta core memory (machine profile + last 5 anomaly pattern observations)
- A-MEM notes (linked Zettelkasten recall from past similar reasoning sessions)
- Archival memory (previous root-cause reports for this specific machine)

The response is parsed into a structured `RootCauseReport` with `root_cause`,
`confidence`, `severity`, `evidence`, `recommended_actions`, and the full reasoning
trace — visible in the dashboard so engineers can verify the agent's logic, not just
its conclusion.

### 4. A-MEM Agentic Memory (Zettelkasten)

Every reasoning session creates a memory note stored in MongoDB with a 384-dimensional
embedding from `sentence-transformers/all-MiniLM-L6-v2`. When a note is created, it
auto-links to existing notes with cosine similarity > 0.75 (up to 3 links), building a
self-organising knowledge graph. When searching, linked neighbours of top hits are
surfaced with a 0.85 similarity discount (Zettelkasten recall). The system becomes
measurably more accurate with each resolved incident without any retraining.

### 5. Human-in-the-Loop Approval

LangGraph's `interrupt_before` API pauses the pipeline after every alert is generated
and saved to MongoDB as `approved=None`. Operators approve or reject via the Gradio
dashboard or the REST API (`POST /api/alerts/{id}/approve`). Alerts with confidence
≥ 0.95 auto-approve immediately; all others wait up to 15 minutes before a background
task escalates them. Every decision is logged with a timestamp and operator ID.

When an alert is approved, the orchestrator resumes the paused LangGraph thread:
the `post_resolution_indexer` node builds a `MaintenanceLog` from the alert's root
cause and sensor context, then POSTs it to `/api/maintenance-logs/add`, which embeds
and upserts it into Qdrant. The RAG knowledge base grows automatically with every
resolved incident.

### 6. Evidently AI Drift Monitoring

A nightly APScheduler job (03:00 UTC) discovers active machine IDs from Redis key
patterns, fetches the last 100 readings per machine, and runs Evidently's
`DataDriftPreset` — a KS-test per feature comparing the live distribution against the
PostgreSQL reference (737,786 clean normal samples). If more than 50% of the four
sensor features drift (p < 0.05), `is_drifted=True` is recorded in MongoDB's
`drift_reports` collection and the training script is triggered automatically as a
background subprocess, subject to a 24-hour cooldown to prevent repeated launches when
drift persists across consecutive checks.

### 7. MLflow Model Registry (Challenger/Champion)

Every model version is registered in MLflow with a SQLite backend under two aliases:
`challenger` (just trained, under evaluation) and `champion` (production-ready). The
`promote_to_production.py` CLI handles promotion in one command, automatically removes
the alias from the previous holder, and supports one-command rollback. AUC scores from
post-training evaluation are attached to every registered version — `--list` shows a
ranked comparison table.

---

## ML Model Performance

Trained on the **Microsoft Azure PdM dataset** — 100 machines, 1 year of hourly
readings, 761 failure events across 4 failure types.

| Model | AUC | Detection Rate | Threshold |
|---|---|---|---|
| **LSTM Autoencoder** | **0.846** | **67%** | **0.011913** |

**AUC interpretation:** The model correctly ranks 85 out of 100 randomly drawn
failure/normal sequence pairs. The sequence length of 198 readings was computed
automatically from the failure event distribution — not chosen arbitrarily — which
is why the model achieves this AUC on real industrial data.

**Training set:** 737,786 clean normal sequences (rows where no failure occurs within
the computed precursor window). Failure-adjacent rows are excluded from training but
used to calibrate the anomaly threshold and compute detection rate.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.135, Pydantic v2, Uvicorn |
| **Orchestration** | LangGraph 1.1 (HITL state machine with `interrupt_before`) |
| **LLM** | Groq API — llama-3.3-70b (ReAct reasoning) + llama-3.1-8b (alert generation) |
| **ML** | TensorFlow 2.21 (LSTM Autoencoder, sole anomaly detector) |
| **RAG** | Qdrant Cloud or local Docker + sentence-transformers/all-MiniLM-L6-v2 |
| **Agent Memory** | A-MEM custom (Zettelkasten, MongoDB) + Letta (MemGPT-style per-machine profiles) |
| **Databases** | MongoDB Atlas (motor async) · Redis local or Upstash (pub/sub + reading cache) |
| **Model Storage** | Local — `ml/models/` directory (generated by training script, gitignored) |
| **Training DB** | PostgreSQL via Supabase (SQLAlchemy, SSL) — 876,100 rows |
| **Drift Monitor** | Evidently AI 0.7.x (DataDriftPreset, nightly APScheduler, 24h retraining cooldown) |
| **Model Registry** | MLflow 3.x (SQLite backend, challenger/champion aliases) |
| **Frontend** | Gradio 6.9, Plotly 6.6 |
| **Observability** | MLflow live prediction tracking |
| **Dataset** | Microsoft Azure PdM (876,100 rows, 100 machines, 4 failure types) |

---

## Quick Start

> Models are never committed to the repository. Run the training script once to
> generate them locally in `ml/models/`.

### Prerequisites

- Python 3.11+
- Redis (local: `redis-server` or Docker, or Upstash)
- API keys for:
  - `GROQ_API_KEY` — [console.groq.com](https://console.groq.com)
  - `QDRANT_URL` + `QDRANT_API_KEY` — [cloud.qdrant.io](https://cloud.qdrant.io) or local Docker (`docker run -p 6333:6333 qdrant/qdrant`)
  - `MONGODB_URL` — [mongodb.com/atlas](https://mongodb.com/atlas) (free tier works)
  - `REDIS_URL` — local `redis://localhost:6379/0` or [upstash.com](https://upstash.com)
  - `POSTGRES_URL` — [supabase.com](https://supabase.com) (free tier: 500 MB)

### Step 1 — Clone and install

```bash
git clone https://github.com/Shital16-hub/defectsense.git
cd defectsense
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2 — Configure environment

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

Key variables:

```env
# LLM
GROQ_API_KEY=gsk_...

# Databases
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=...
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net
REDIS_URL=redis://localhost:6379/0

# Training data
POSTGRES_URL=postgresql://postgres.xxx:password@aws-0-region.pooler.supabase.com:6543/postgres

# MLflow (local SQLite — no server needed)
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db

# App base URL (used by post_resolution_indexer to index resolved incidents)
APP_BASE_URL=http://localhost:8080
```

### Step 3 — Download the Azure PdM dataset

Download from Kaggle:
[arnabbiswas1/microsoft-azure-predictive-maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

Place the five CSV files in `data/azure_pdm/`:

```
data/azure_pdm/
├── PdM_telemetry.csv    # 876,100 hourly sensor readings
├── PdM_failures.csv     # 761 failure events
├── PdM_machines.csv     # machine metadata
├── PdM_maint.csv        # maintenance records
└── PdM_errors.csv       # error logs
```

### Step 4 — Load training data into PostgreSQL

```bash
python data/load_azure_to_postgres.py
```

This loads all 876,100 rows into the `azure_sensor_readings` table, computes the
`is_clean_normal` flag by excluding rows within the failure precursor window, and
writes `data/azure_lstm_config.json` with the computed sequence length.

### Step 5 — Train the LSTM Autoencoder

```bash
python ml/train_autoencoder_azure.py
```

The script: loads 737,786 clean normal rows from PostgreSQL → fits the scaler →
builds per-machine sequences → trains the LSTM Autoencoder → computes the anomaly
threshold (mean + 3 std on validation reconstruction errors) → evaluates on real
pre-failure sequences → logs to MLflow → saves to `ml/models/` → registers as
`challenger` in MLflow.

Expected summary output:

```
  AUC                  : 0.8460  (vs AI4I baseline 0.455)
  Detection rate       : 67.0%
  Threshold            : 0.011913
  Model saved          : ml/models/azure_lstm_autoencoder.keras
  MLflow registry      : defectsense_lstm_autoencoder_azure v1 [challenger]
```

### Step 6 — Promote model to production

```bash
# List all registered versions with AUC scores
python ml/promote_to_production.py --list

# Promote to champion
python ml/promote_to_production.py --model lstm --version 1

# Roll back if needed
python ml/promote_to_production.py --model lstm --rollback
```

### Step 7 — Index RAG knowledge base

```bash
python data/generate_logs.py              # generate maintenance logs from Azure PdM records
python data/index_maintenance_logs.py     # embed and upload to Qdrant
```

### Step 8 — Start the system

```bash
# Terminal 1 — FastAPI backend
uvicorn app.main:app --port 8080 --reload

# Terminal 2 — Gradio dashboard
python frontend/app.py

# Terminal 3 — stream simulator (replays Azure PdM telemetry at configurable speed)
python data/stream_simulator.py
```

### Step 9 — Open the dashboard

```
http://localhost:7860
```

Four tabs: **Live Monitor** (real-time sensor charts + failure probability gauge) ·
**Alerts** (pending/approved table with approve/reject) · **Root Cause** (full ReAct
reasoning trace) · **System Health** (service status + pipeline stats + ML info).

---

## Project Structure

```
defectsense/
├── app/
│   ├── agents/
│   │   ├── anomaly_detector.py       # Redis buffer → LSTM inference → store anomaly
│   │   ├── context_retriever.py      # Qdrant RAG + Redis sensor trend summary
│   │   ├── root_cause_reasoner.py    # ReAct reasoning agent (llama-3.3-70b)
│   │   ├── alert_generator.py        # Plain-language alert + MongoDB persistence
│   │   └── orchestrator.py           # LangGraph HITL pipeline (18 stages)
│   ├── api/routes/
│   │   ├── sensors.py                # POST /ingest (30s per-machine cooldown), GET /history
│   │   ├── alerts.py                 # CRUD + approve/reject (resumes LangGraph thread)
│   │   ├── dashboard.py              # Machine health summary
│   │   ├── evaluation.py             # Eval results + drift endpoints + manual triggers
│   │   └── maintenance_logs.py       # RAG knowledge base — add / bulk-add / list / count
│   ├── api/websocket.py              # WebSocket hub (anomaly stream + heartbeat)
│   ├── models/
│   │   ├── sensor.py                 # SensorReading (volt, rotate, pressure, vibration)
│   │   ├── anomaly.py                # AnomalyResult (score, failure_type, sensor_deltas)
│   │   ├── alert.py                  # MaintenanceAlert, RootCauseReport, Severity
│   │   └── maintenance.py            # MaintenanceLog (RAG document schema)
│   ├── services/
│   │   ├── ml_service.py             # LSTM loading, predict_anomaly, MLflow tracking
│   │   ├── redis_service.py          # Store/fetch readings (MAX_READINGS=210), pub/sub
│   │   ├── qdrant_service.py         # Embed + upsert + search maintenance logs
│   │   ├── amem_service.py           # Zettelkasten memory (MongoDB + sentence-transformers)
│   │   ├── letta_service.py          # Per-machine core + archival memory (MongoDB)
│   │   ├── drift_monitoring_service.py  # Evidently AI drift reports + retraining trigger
│   │   ├── evaluation_service.py     # RAG precision + LLM-judge scoring
│   │   ├── postgres_service.py       # SQLAlchemy access to azure_sensor_readings
│   │   └── blob_storage_service.py   # Stub (is_available=False) — interface preserved
│   └── main.py                       # FastAPI lifespan, APScheduler, WebSocket, routers
├── ml/
│   ├── models/                       # Generated artefacts (gitignored)
│   │   ├── azure_lstm_autoencoder.keras
│   │   ├── azure_sensor_scaler.pkl
│   │   └── azure_anomaly_threshold.pkl
│   ├── train_autoencoder_azure.py    # Full training pipeline (13 steps, MLflow logged)
│   ├── model_registry_service.py     # MLflow 3.x aliases wrapper (challenger/champion)
│   ├── promote_to_production.py      # CLI: --list / --model lstm --version N / --rollback
│   └── evaluate_azure_lstm.py        # Standalone AUC evaluation against held-out failures
├── frontend/app.py                   # Gradio dashboard (4 tabs, Plotly charts)
├── data/
│   ├── azure_pdm/                    # Azure PdM CSV files (download from Kaggle)
│   ├── azure_lstm_config.json        # sequence_length=198 (auto-computed by loader)
│   ├── load_azure_to_postgres.py     # Kaggle CSVs → PostgreSQL (876,100 rows)
│   ├── generate_logs.py              # Synthetic maintenance logs for RAG seeding
│   ├── index_maintenance_logs.py     # Embed logs → Qdrant maintenance_logs collection
│   └── stream_simulator.py           # Replays Azure PdM telemetry at configurable speed
├── evaluation/
│   └── run_evaluation.py             # RAG + LLM-judge evaluation runner
└── tests/                            # 179 pytest tests across 10 modules
```

---

## MLOps and Data Infrastructure

### Model Artifact Storage — Local (`ml/models/`)

Models are stored in `ml/models/` (gitignored) and generated by running the training
script. The `BlobStorageService` class is kept as an interface-compatible stub
(`is_available=False`) so health checks, `MLService`, and tests work without
modification. To restore cloud storage, replace the stub in
`app/services/blob_storage_service.py` — no other files need to change because the
interface is preserved.

Keeping models out of the repository and always trainable from source data is a
deliberate MLOps discipline: it prevents model/code version drift.

### Training Data — PostgreSQL on Supabase

The 876,100 Azure PdM sensor readings are stored in a Supabase PostgreSQL instance in
the `azure_sensor_readings` table. The training script queries clean normal rows
(`is_clean_normal = TRUE`) for LSTM training, and failure-adjacent rows for threshold
calibration and AUC computation. SQLAlchemy with `sslmode=require` is used for the
Supabase session pooler. If PostgreSQL is unavailable, the training script exits early
— training data is too large to bundle as a CSV fallback. The drift monitoring service
also queries PostgreSQL for the reference distribution; it falls back to the local
`data/azure_pdm/PdM_telemetry.csv` if the database is unreachable.

### Drift Monitoring — Evidently AI

An APScheduler cron job fires at 03:00 UTC. It discovers active machine IDs from Redis
key patterns, fetches the last 100 readings per machine as the current window, and runs
Evidently's `DataDriftPreset` — a KS-test per feature comparing the live distribution
against 737,786 PostgreSQL reference samples. If more than 50% of the four sensor
features drift (p < 0.05), the report is saved to MongoDB's `drift_reports` collection
and `ml/train_autoencoder_azure.py` is launched as a background subprocess.

A **24-hour cooldown** prevents hammering the training script when drift persists across
consecutive nightly checks — the `_last_retraining_triggered` timestamp is checked
before each launch and the trigger is skipped if fewer than 24 hours have elapsed.

### Model Registry — MLflow (Challenger/Champion)

Models follow a two-stage lifecycle using MLflow 3.x aliases. A freshly trained model
receives the `challenger` alias automatically. After reviewing the AUC score (visible in
`--list`), the operator promotes it to `champion`. The previous champion has its alias
removed first — there is always at most one champion. Rollback moves `champion` to
`current_version - 1` after verifying that version still exists in the registry.

One implementation detail: `search_model_versions()` does not populate `mv.aliases` with
the SQLite backend. The `ModelRegistryService` works around this by building an explicit
alias map via `get_model_version_by_alias()` for each known alias.

---

## Engineering Decisions

**Why LSTM Autoencoder only? (No Isolation Forest or ensemble)**

Azure PdM failures are gradual — vibration drifts from 1.0 to 1.5 std above baseline
over hours before a bearing fails; rotation speed slowly decays below rated RPM before
an RSF event. These are temporal patterns with z-score deviations in the 0.5–1.5 std
range that accumulate across 198 readings. Isolation Forest is a point anomaly detector:
it operates on a single feature vector with no memory of the preceding sequence. It
cannot detect the gradual drift that characterises real industrial failures. The LSTM
Autoencoder's inductive bias — learning to reconstruct normal sequences and failing on
anomalous ones — is the right tool for this failure mode. One well-chosen model
outperforms an ensemble of wrong ones.

**Why the Azure PdM dataset?**

876,100 real sensor readings from 100 industrial machines over one year, with 761
recorded failure events and known failure types. This is real operational data with a
realistic failure rate (0.087%) — not a synthetic benchmark. The dataset has enough
clean normal rows (737,786 after precursor window exclusion) to train a sequence model
properly, and enough failure events to compute a meaningful AUC against real pre-failure
sequences. Synthetic datasets produce models that perform well on themselves and fail in
production; this dataset is a genuine proxy for industrial deployment conditions.

**Why sequence length 198?**

The sequence length was not chosen by intuition — it was computed automatically by
analysing the distribution of 761 failure events in the dataset to find the median
time window (in hourly readings) within which detectable precursor signals appear.
`data/azure_lstm_config.json` stores this value and all training and inference code
reads from it. If the dataset changes or more failure events are analysed, the length
updates automatically without touching any model code.

**Why Groq instead of OpenAI?**

Groq's `llama-3.3-70b-versatile` delivers sub-2-second reasoning responses on
manufacturing anomaly prompts. For a real-time monitoring system where a factory-floor
engineer needs an answer before they walk to the machine, inference latency is a product
requirement, not a nice-to-have. OpenAI's GPT-4o at typical API latency would add 8–15
seconds to every alert — unacceptable for an incident response workflow.

**Why LangGraph for orchestration?**

Human-in-the-loop interrupts are a first-class LangGraph concept. The `interrupt_before`
API integrates naturally with the async FastAPI backend: the pipeline pauses, persists
its full state in a MemorySaver checkpoint, waits for human input via REST API, and
resumes — all without polling loops or manual state management. Writing this workflow
with plain async tasks would require rebuilding LangGraph's checkpoint and resume
machinery from scratch.

**Why A-MEM + Letta as custom implementations?**

The official libraries were not stable enough for production use at project start, and
both had dependency conflicts with LangGraph and motor. Custom implementations gave full
control over memory format, embedding storage, Zettelkasten linking logic, and the
per-machine ring buffer. The interfaces are designed to be drop-in replaceable — if
stability improves, substituting the official library requires changing one import.

**Why PostgreSQL for training data and MongoDB for operational data?**

Training data is tabular, fixed-schema, and benefits from SQL joins, window functions,
and typed queries (`WHERE is_clean_normal = TRUE`). Operational data — alerts,
reasoning traces, A-MEM notes, drift reports — is document-structured, schema-flexible,
and written by different agents in different formats. Using the right database for the
access pattern is a cleaner architectural choice than forcing everything into one store.

**Why Evidently AI for drift detection?**

Evidently provides production-quality statistical drift tests (KS-test per feature,
dataset-level 50% drift threshold) with a clean Python API and structured output that
stores directly in MongoDB. The `DataDriftPreset` covers all four sensor features in
one call, returning per-feature p-values and a dataset-level drift flag. Custom
implementations of the same tests would require more code, more maintenance, and less
statistical rigour.

**Why MLflow challenger/champion aliases instead of stages?**

MLflow 3.x deprecated `Production`, `Staging`, and `Archived` stages in favour of
free-form aliases. The `challenger`/`champion` pattern maps naturally to the actual
workflow: a model is a challenger until explicitly validated by a human reviewing the
AUC comparison table, then promoted. Aliases also avoid the constraint of a single
stage per version — useful when a version needs both `champion` and `long-term-stable`
designations simultaneously.

---

## Pipeline Audit — All 18 Stages Verified

| Stage | Status |
|---|---|
| 1. Ingest → Redis buffer (210 slots) → LSTM at 198 readings | WORKING |
| 2. Failure type classifier (BWF/RSF/PBF/EVF z-score rules) | WORKING |
| 3. LangGraph state: machine_id, anomaly_score, failure_type, sensor_deltas, timestamp | WORKING |
| 4. ContextRetriever → Qdrant semantic search by failure type | WORKING |
| 5. RootCauseReasoner → Groq LLM (full context, structured output) | WORKING |
| 6. AlertGenerator → MongoDB with approved=None (PENDING) | WORKING |
| 7. A-MEM add_memory — correct (content, keywords) signature | WORKING |
| 8. A-MEM search_memory — called before LLM, injected into prompt | WORKING |
| 9. Letta update_machine_profile — after every reasoning session | WORKING |
| 10. Letta machine profile read — injected into LLM prompt | WORKING |
| 11. Alert visible in dashboard (machine_id, severity, failure_type, root_cause) | WORKING |
| 12. Human approves → POST /api/alerts/{id}/approve | WORKING |
| 13. LangGraph resumes from interrupt_before checkpoint | WORKING |
| 14. post_resolution_indexer receives machine_id + session_id (re-injected by resume) | WORKING |
| 15. Approved alert indexed into Qdrant (degrades gracefully if Qdrant unavailable) | WORKING |
| 16. Evidently drift computation (KS-test, snapshot.dict() structure verified) | WORKING |
| 17. Drift → retraining subprocess with 24-hour cooldown | WORKING |
| 18. Retrained model registered in MLflow with challenger alias | WORKING |

---

## Testing

179 tests across 10 modules — all pass with no external credentials required for the
unit suite. Integration tests marked `@pytest.mark.integration` require real
credentials in `.env`.

| Module | Tests | What it covers |
|---|---|---|
| `test_ml_models.py` | 15 | Scaler, LSTM Autoencoder, MLService buffering, failure type classifier |
| `test_agents.py` | 20 | AnomalyDetector, AlertGenerator, Orchestrator nodes, HITL post-indexer |
| `test_api.py` | 21 | All REST endpoints, validation, error handling |
| `test_maintenance_logs.py` | 24 | Add (Qdrant optional), bulk-add, list, count, MongoDB/Qdrant integration |
| `test_evaluation.py` | 21 | RAG eval, LLM-judge, nightly scheduler |
| `test_blob_storage.py` | 14 | Stub interface, MLService graceful degradation |
| `test_postgres_service.py` | 11 | Init, data access, graceful degradation |
| `test_drift_monitoring.py` | 18 | Reference load, Evidently report, Redis window, retraining cooldown, API |
| `test_model_registry.py` | 26 | Aliases, promote, rollback, compare, AUC resolution |
| `test_integration_new_features.py` | 9 | End-to-end pipeline, graceful degradation, HITL flow |

```bash
# Run all non-integration tests (no credentials needed)
pytest tests/ -v -m "not integration"

# Run with verbose output
pytest tests/ -v -m "not integration" --tb=short

# Run real integration tests (requires full .env)
pytest tests/ -v -m integration

# Run a specific module
pytest tests/test_agents.py -v
```

---

## API Reference

All endpoints are documented at `http://localhost:8080/docs` (Swagger UI).

**Sensor Ingestion**
```
POST /api/sensors/ingest               Ingest one SensorReading; returns AnomalyResult
                                       Pipeline cooldown: 30s per machine to prevent LLM rate limits
GET  /api/sensors/{machine_id}/history Last N readings (default 50, max 200)
```

**Alerts**
```
GET  /api/alerts                       List alerts (filter: machine_id, status, limit)
GET  /api/alerts/{alert_id}            Full alert with reasoning trace and sensor data
POST /api/alerts/{alert_id}/approve    Approve pending alert (resumes LangGraph thread)
POST /api/alerts/{alert_id}/reject     Reject with mandatory reason (resumes LangGraph thread)
GET  /api/alerts/stats                 Aggregate counts by severity × status
```

**Dashboard**
```
GET  /api/dashboard/stats              Active machines, anomalies last 24h, pending alerts
GET  /api/dashboard/machines           Per-machine health: status, failure_probability, last anomaly
```

**Evaluation and Drift**
```
GET  /api/evaluation/latest            Most recent RAG precision + LLM-judge scores
GET  /api/evaluation/history           Last 30 evaluation results
GET  /api/evaluation/run               Trigger evaluation manually (background task)
GET  /api/evaluation/drift/latest      Most recent Evidently drift report
GET  /api/evaluation/drift/history     Last N drift reports
GET  /api/evaluation/drift/run         Trigger nightly drift check manually
```

**Maintenance Logs (RAG Knowledge Base)**
```
POST /api/maintenance-logs/add         Embed + upsert one log; Qdrant optional (degrades gracefully)
POST /api/maintenance-logs/bulk-add    Embed + upsert up to 100 logs (Qdrant required)
GET  /api/maintenance-logs             List logs (filter by failure_type, machine_id, pagination)
GET  /api/maintenance-logs/count       Counts in MongoDB vs Qdrant (sync verification)
```

**WebSocket**
```
WS /ws/sensors    Real-time AnomalyResult stream; heartbeat every 5s; ping/pong
```

**Health**
```
GET /health       All service states: ml_ready, redis_connected, mongo_connected,
                  qdrant_connected, rag_ready, amem_ready, letta_ready,
                  orchestrator_ready, postgres_ready, drift_monitor_ready
```

---

## Dataset

**Microsoft Azure Predictive Maintenance**
Kaggle: [arnabbiswas1/microsoft-azure-predictive-maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

| Property | Value |
|---|---|
| Total rows | 876,100 hourly sensor readings |
| Machines | 100 |
| Duration | 1 year |
| Failure events | 761 |
| Failure rate | 0.087% |
| Clean normal rows | 737,786 (after precursor window exclusion) |
| Sensors | volt (V), rotate (RPM), pressure, vibration |
| Failure types | EVF · RSF · PBF · BWF |
| Sequence length | 198 hours (auto-computed from failure distribution) |

**Data flow through the system:**
```
Kaggle download  →  data/azure_pdm/  (5 CSV files, ~80 MB)
         ↓
data/load_azure_to_postgres.py
         ↓
PostgreSQL azure_sensor_readings  (876,100 rows, is_clean_normal flag)
         ↓
ml/train_autoencoder_azure.py  reads 737,786 clean rows
         ↓
ml/models/azure_lstm_autoencoder.keras  (stored locally, gitignored)
         ↓
MLflow registry: defectsense_lstm_autoencoder_azure [champion]
         ↓
MLService loads on startup → AnomalyDetectorAgent
         ↓
Live sensor stream → POST /api/sensors/ingest
         ↓
Evidently AI: live Redis window vs PostgreSQL reference (nightly, 03:00 UTC)
```

---

*Built by Shital Nandre as a portfolio project targeting industrial AI and MLOps roles.*

*Stack: FastAPI · LangGraph · Groq · TensorFlow · Qdrant · MongoDB · Redis · PostgreSQL · MLflow · Evidently · Gradio*
