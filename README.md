# 🏭 DefectSense

**Manufacturing Defect Root-Cause Intelligence — Hybrid ML + GenAI Multi-Agent System**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-orange)](https://langchain-ai.github.io/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq_llama3-purple)](https://groq.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00)](https://tensorflow.org)
[![Tests](https://img.shields.io/badge/tests-182_passed-brightgreen)](#testing)

Manufacturing plants lose **$260 billion per year** to unplanned downtime (Deloitte, 2024).
Existing condition-monitoring systems tell you *when* a machine failed — not *why*.
DefectSense closes that gap with a reasoning AI that delivers a root cause, ranked
maintenance actions, and an explanation a factory-floor engineer can actually act on.

---

## What This System Does

DefectSense is a real-time anomaly detection and root-cause reasoning platform built
for manufacturing environments. It continuously ingests sensor readings from CNC machines
— air temperature, process temperature, rotational speed, torque, and tool wear — and
runs them through a hybrid ML pipeline that combines an LSTM Autoencoder for temporal
pattern detection with an Isolation Forest for single-point outlier detection. When
both models agree an anomaly is occurring, a LangGraph multi-agent pipeline activates:
a retrieval agent pulls relevant maintenance records from a vector store, a ReAct reasoning
agent synthesises those records with the sensor data and machine history to produce a
structured root-cause report, and an alert generator formats that into plain language
a maintenance engineer can act on immediately. Every incident is stored in an agentic
memory system that improves future diagnoses, and every alert waits for human approval
before being actioned — or auto-approves after 15 minutes if the operator is unavailable.
Unlike rule-based threshold systems that generate constant false positives, DefectSense
learns the specific failure patterns of each machine and explains them, cutting mean
time to repair while keeping human judgment in the loop for critical decisions.

---

## Architecture

```mermaid
graph TD
    A[Sensor Stream<br/>AI4I 2020 / Live Data] -->|POST /api/sensors/ingest| B[FastAPI :8080]
    B --> C[AnomalyDetectorAgent]
    C -->|LSTM + IForest ensemble| D{Anomaly?}
    D -->|No| E[Redis cache only]
    D -->|Yes| F[LangGraph Orchestrator]

    F --> G[ContextRetrieverAgent<br/>Qdrant RAG]
    F --> H[A-MEM Update<br/>Zettelkasten memory]
    G --> I[RootCauseReasonerAgent<br/>ReAct · llama-3.3-70b]
    H --> I
    I --> J[AlertGeneratorAgent<br/>llama-3.1-8b]
    J --> K[(MongoDB<br/>alerts + drift_reports)]
    J -->|alerts:new| L[Redis pub/sub]

    K --> M[HITL Gate<br/>interrupt_before=apply_approval]
    M -->|conf >= 0.95| N[Auto-approve]
    M -->|conf < 0.95| O[Human via<br/>Gradio Dashboard]
    N --> P[apply_approval]
    O --> P

    L --> Q[WebSocket ws/alerts]
    Q --> R[Gradio Dashboard :7860]

    subgraph MLOps
        S[(Azure Blob Storage<br/>versioned model artefacts)]
        T[(PostgreSQL / Supabase<br/>10,000 training rows)]
        U[MLflow Registry<br/>challenger → champion]
        V[Evidently AI<br/>nightly drift checks]
    end

    subgraph AgentMemory
        W[A-MEM Zettelkasten<br/>MongoDB + embeddings]
        X[Letta<br/>per-machine profiles]
    end

    C <-->|load model| S
    T -->|training data| C
    U -->|alias management| S
    V -->|compare live vs baseline| T

    I <-->|recall + update| W
    I <-->|core + archival| X
```

---

## Key Features

**1. Hybrid ML Anomaly Detection**
Two models run in parallel on every sensor reading. The LSTM Autoencoder learns normal
operating sequences over a 30-reading window and flags gradual drift — the kind of slow
degradation that kills machines over days. The Isolation Forest catches single-point
outliers — sudden temperature spikes or torque surges. High-confidence alerts require
both to agree; low-confidence alerts fire when either detects something unusual. This
ensemble approach eliminates the false-positive flood that plagues single-model systems.

**2. ReAct Root Cause Reasoning**
When an anomaly is detected, a LangGraph agent running Groq's `llama-3.3-70b-versatile`
executes a THINK → ACT → OBSERVE → CONCLUDE loop. It queries the RAG knowledge base,
analyses sensor trends, and synthesises machine history into a structured `RootCauseReport`
with a ranked list of maintenance actions and a confidence score. The full reasoning trace
is visible in the dashboard so engineers can verify the agent's logic — not just its
conclusion. Each reasoning step completes in under 2 seconds on Groq's inference
infrastructure.

**3. A-MEM Agentic Memory (Zettelkasten)**
Every reasoning session creates a memory note stored in MongoDB with a vector embedding.
Notes auto-link to semantically similar past incidents (cosine similarity > 0.75), building
a self-organising knowledge graph. The next time a similar failure pattern appears, the
reasoner pulls linked notes as context — so the system gets measurably smarter with every
incident it processes, without any retraining.

**4. Human-in-the-Loop Approval**
LangGraph's `interrupt_before` API pauses the pipeline after every alert is generated.
Operators approve or reject via the Gradio dashboard or the REST API. Alerts with
confidence ≥ 0.95 auto-approve immediately; all others wait up to 15 minutes before a
background task escalates them automatically. Every approval decision is logged with
a timestamp and operator ID, providing a complete audit trail.

**5. Azure Blob Model Storage**
After every training run, both ML models upload to Azure Blob Storage under two names:
`*_latest.pkl` (always the most recent) and `*_YYYYMMDD.pkl` (date-stamped archive).
On startup, the `MLService` checks for local files first; if missing, it downloads from
blob automatically. This means a fresh deployment on any machine has the correct models
within seconds — no manual file transfers, no model files in the repository.

**6. PostgreSQL Training Data**
All 10,000 sensor readings from the AI4I 2020 dataset live in a Supabase PostgreSQL
database. Both training scripts query PostgreSQL as their primary data source via
SQLAlchemy (with SSL required), falling back to the CSV if the database is unavailable.
The `feature_queries.sql` file contains window functions for rolling-average feature
engineering at scale — useful for extending the model to larger production datasets
without changing the training code.

**7. Evidently AI Drift Monitoring**
A nightly APScheduler job (03:00 UTC) fetches the last 100 readings per machine from
Redis and runs Evidently's `DataDriftPreset` against the PostgreSQL reference distribution.
If more than 50% of features have drifted (p < 0.05), a warning is logged and the result
is stored in MongoDB's `drift_reports` collection. The `/api/evaluation/drift/latest`
endpoint exposes the most recent check; the dashboard shows `drift_detected` status in
real time. This gives the operations team an early signal that retraining is needed
before model performance degrades visibly.

**8. MLflow Model Registry (Challenger/Champion)**
Every model version is registered in MLflow with a SQLite backend under two aliases:
`challenger` (just trained, under evaluation) and `champion` (production-ready). The
`promote_to_production.py` CLI handles promotion in one command, automatically removes
the alias from the previous holder, and supports one-command rollback to the prior
version. AUC scores from post-training evaluation are attached to every registered
version, so `--list` shows a ranked comparison table. The champion model is what the
running `MLService` loads.

---

## ML Model Performance

Evaluated on the AI4I 2020 dataset — **2,000-row held-out test set** (last 20% of 10,000
rows), containing 39 failure events (1.9% failure rate).

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Isolation Forest | 0.196 | 0.231 | 0.212 | **0.929** |
| LSTM Autoencoder | 0.000 | 0.000 | 0.000 | 0.455 |
| **Ensemble** | **0.200** | **0.237** | **0.217** | **0.905** |

**Why AUC, not precision/recall?**
AUC (Area Under the ROC Curve) measures ranking quality — how reliably the model assigns
higher anomaly scores to actual failures than to normal readings. An AUC of **0.929**
means the Isolation Forest correctly ranks 93 out of 100 randomly drawn failure/normal
pairs, regardless of threshold. On a dataset where only 1.9% of readings are failures,
precision and recall are dominated by the threshold choice, not model quality. A lower
threshold would boost recall dramatically (catching more failures earlier) at the cost
of more false alarms — that trade-off is an operational decision, not a model limitation.
The LSTM Autoencoder's low AUC reflects training on mixed data (including failure
sequences), which degrades reconstruction-error reliability in isolation; it contributes
meaningfully to ensemble confidence scoring without driving the primary detection signal.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.135, Pydantic v2, Uvicorn |
| **Orchestration** | LangGraph 1.1 (HITL state machine) |
| **LLM** | Groq API — llama-3.3-70b (reasoning) + llama-3.1-8b (alerts) |
| **ML** | TensorFlow 2.21 (LSTM Autoencoder) + scikit-learn (Isolation Forest) |
| **RAG** | Qdrant Cloud (vector store) + sentence-transformers/all-MiniLM-L6-v2 |
| **Agent Memory** | A-MEM custom (Zettelkasten, MongoDB) + Letta (MemGPT-style) |
| **Databases** | MongoDB Atlas (motor async) · Redis/Upstash (pub/sub + cache) |
| **Cloud Storage** | Azure Blob Storage (versioned model artefacts) |
| **Training DB** | PostgreSQL via Supabase (SQLAlchemy, SSL) |
| **Drift Monitor** | Evidently AI (DataDriftPreset, nightly APScheduler) |
| **Model Registry** | MLflow 3.x (SQLite backend, challenger/champion aliases) |
| **Frontend** | Gradio 6.9, Plotly 6.6 |
| **Observability** | MLflow (live prediction tracking) · LangSmith (LLM traces) |
| **Dataset** | AI4I 2020 Predictive Maintenance (10,000 rows, 5 failure types) |

---

## Quick Start

> Models are never committed to the repository — they are always trained from scratch
> and stored in Azure Blob Storage + MLflow SQLite registry.

### Prerequisites

- Python 3.11+
- API keys / credentials for:
  - `GROQ_API_KEY` — [console.groq.com](https://console.groq.com)
  - `QDRANT_URL` + `QDRANT_API_KEY` — [cloud.qdrant.io](https://cloud.qdrant.io) or local Docker
  - `MONGODB_URL` — [mongodb.com/atlas](https://mongodb.com/atlas) (free tier works)
  - `REDIS_URL` — [upstash.com](https://upstash.com) or local Redis
  - `POSTGRES_URL` — [supabase.com](https://supabase.com) (free tier: 500 MB)
  - `AZURE_STORAGE_CONNECTION_STRING` — Azure Portal → Storage Account → Access Keys
  - `AZURE_STORAGE_CONTAINER` — container name (must exist before training)

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

Key variables to set (full list in `.env.example`):

```env
# LLM
GROQ_API_KEY=gsk_...

# Databases
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=...
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net
REDIS_URL=rediss://...upstash.io:6380

# Cloud infrastructure
POSTGRES_URL=postgresql://postgres.xxx:password@aws-0-region.pooler.supabase.com:6543/postgres
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_STORAGE_CONTAINER=defectsense-models

# MLflow
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
```

### Step 3 — Load training data

```bash
python data/download_data.py           # download AI4I 2020 CSV
python data/load_to_postgres.py        # load 10,000 rows into PostgreSQL
```

### Step 4 — Train models

Both scripts automatically: train the model → evaluate on held-out test set →
log metrics to MLflow → save locally → upload to Azure Blob → register in MLflow
as `challenger`.

```bash
python ml/train_autoencoder.py
python ml/train_isolation_forest.py
```

Expected output (Isolation Forest):

```
Registry:  defectsense_isolation_forest v1
Alias:     challenger
AUC:       0.9289
Precision: 0.1963
Recall:    0.2308
F1:        0.2121
```

### Step 5 — Promote models to production

```bash
# See all registered versions with AUC scores
python ml/promote_to_production.py --list

# Promote both models to champion
python ml/promote_to_production.py --model lstm    --version 1
python ml/promote_to_production.py --model iforest --version 1

# Roll back if needed
python ml/promote_to_production.py --model iforest --rollback
```

### Step 6 — Index RAG knowledge base

```bash
python data/generate_logs.py              # generate synthetic maintenance logs
python data/index_maintenance_logs.py     # embed and upload to Qdrant
```

### Step 7 — Start the system

```bash
# Terminal 1 — FastAPI backend
uvicorn app.main:app --port 8080 --reload

# Terminal 2 — Gradio dashboard
python frontend/app.py

# Terminal 3 — stream simulator (replays AI4I dataset at configurable speed)
python data/stream_simulator.py
```

### Step 8 — Open the dashboard

```
http://localhost:7860
```

Four tabs: **Live Monitor** (real-time sensor stream) · **Root Cause** (reasoning
traces + alerts) · **Model Registry** (challenger/champion versions) · **Evaluation**
(RAG metrics + drift reports).

---

## Project Structure

```
defectsense/
├── app/
│   ├── agents/
│   │   ├── anomaly_detector.py       # LSTM + IForest ensemble agent
│   │   ├── context_retriever.py      # Qdrant RAG agent
│   │   ├── root_cause_reasoner.py    # ReAct reasoning agent (llama-3.3-70b)
│   │   ├── alert_generator.py        # Alert formatting agent (llama-3.1-8b)
│   │   └── orchestrator.py           # LangGraph HITL state machine
│   ├── api/routes/
│   │   ├── sensors.py                # POST /ingest, GET /history
│   │   ├── alerts.py                 # CRUD + approve/reject
│   │   ├── dashboard.py              # stats, machine health
│   │   ├── evaluation.py             # eval results + drift endpoints
│   │   └── maintenance_logs.py       # RAG knowledge base management
│   ├── api/websocket.py              # Redis pub/sub WebSocket hub
│   ├── models/                       # Pydantic v2 schemas
│   ├── services/
│   │   ├── ml_service.py             # model loading, predict, blob fallback
│   │   ├── blob_storage_service.py   # Azure Blob upload/download/list
│   │   ├── postgres_service.py       # SQLAlchemy training data access
│   │   ├── drift_monitoring_service.py  # Evidently AI drift checks
│   │   ├── evaluation_service.py     # RAG + LLM-judge evaluation
│   │   ├── redis_service.py
│   │   ├── qdrant_service.py
│   │   ├── amem_service.py           # Zettelkasten memory
│   │   └── letta_service.py          # per-machine profile memory
│   └── main.py                       # FastAPI lifespan, APScheduler, routers
├── ml/
│   ├── models/                       # trained artefacts (gitignored)
│   ├── model_registry_service.py     # MLflow 3.x aliases API wrapper
│   ├── promote_to_production.py      # CLI: list / promote / rollback
│   ├── train_autoencoder.py
│   └── train_isolation_forest.py
├── frontend/app.py                   # Gradio 4-tab dashboard
├── data/
│   ├── ai4i_2020.csv                 # AI4I 2020 dataset
│   ├── download_data.py
│   ├── load_to_postgres.py           # CSV → PostgreSQL loader
│   ├── feature_queries.sql           # SQL window functions for features
│   ├── generate_logs.py
│   ├── index_maintenance_logs.py
│   └── stream_simulator.py
├── evaluation/
│   ├── run_evaluation.py
│   └── ml_benchmark.json
└── tests/                            # 182 pytest tests
```

---

## MLOps and Data Infrastructure

This is the section interviewers most often ask about. Here is how each layer works.

### Model Artifact Storage — Azure Blob Storage

Every time a model is trained, the script calls `BlobStorageService.upload_model()` twice
— once as `*_latest.pkl` (always the current version) and once as `*_YYYYMMDD.pkl`
(date-stamped for rollback). On `MLService.load()`, if the local `.pkl` file does not
exist, the service automatically downloads from blob before attempting to load. This
enables clean deployments: clone the repo, set environment variables, start the server —
the correct model arrives via blob without any manual intervention. The service degrades
gracefully if `AZURE_STORAGE_CONNECTION_STRING` is not set, falling back to local files
only; the app still starts and the `/health` endpoint reports `blob_storage_ready: false`
so the degraded state is immediately visible.

### Training Data — PostgreSQL on Supabase

The AI4I 2020 dataset's 10,000 sensor readings are stored in a Supabase PostgreSQL
instance. Both training scripts call `PostgresService.get_normal_samples()` and
`get_failure_samples()` via SQLAlchemy with `sslmode=require` for the Supabase session
pooler. `feature_queries.sql` contains window functions for rolling averages and
lag features — the kind of SQL feature engineering that scales to millions of rows
without loading everything into Python memory. If PostgreSQL is unavailable (network
issue, quota exceeded), both scripts fall back to the local CSV automatically, so
training is never blocked by database outages. The `is_connected` property and graceful
`init()` pattern mean the app starts cleanly even without a database connection.

### Drift Monitoring — Evidently AI

An APScheduler cron job fires at 03:00 UTC every night. It discovers active machine IDs
from Redis key patterns (`sensor:*:readings`), fetches the last 100 readings per machine,
and runs Evidently's `DataDriftPreset` — comparing the live distribution against the
PostgreSQL reference distribution of normal samples. Evidently computes a KS-test
p-value per feature; if more than 50% of the five sensor features drift (p < 0.05),
`is_drifted` is set to `True` and the report is saved to MongoDB's `drift_reports`
collection. The operations team can query `/api/evaluation/drift/latest` or `/drift/history`
to track distribution health over time. When drift is detected, the natural next step
is to retrain with the new distribution as reference — the training scripts are already
wired to do this from PostgreSQL.

### Model Registry — MLflow (Challenger/Champion)

Models follow a two-stage lifecycle using MLflow 3.x aliases: a freshly trained model
receives the `challenger` alias automatically at the end of the training script. After
manual review of the AUC and precision/recall trade-off (visible in `--list`), the
operator runs `promote_to_production.py --model iforest --version N` to move the
`champion` alias. The previous champion has its alias removed first, so there is always
at most one champion per model. Rollback is equally simple: `--rollback` moves `champion`
to `current_version - 1` after verifying that version still exists in the registry.
One subtle implementation detail: `search_model_versions()` does not populate `mv.aliases`
with the SQLite backend — the service builds an explicit alias map via
`get_model_version_by_alias()` for each known alias to work around this.

---

## Engineering Decisions

**Why Groq instead of OpenAI?**
Groq's llama-3.3-70b delivers sub-2-second reasoning responses on manufacturing anomaly
prompts. For a real-time monitoring system where a factory-floor engineer needs an answer
now, not in 15 seconds, inference latency is a product requirement, not a nice-to-have.

**Why LSTM + Isolation Forest ensemble instead of one model?**
Isolation Forest detects single-point outliers well (sudden sensor spike) but misses
gradual drift. LSTM catches temporal patterns across a 30-reading window but requires
history to be meaningful. The ensemble covers both failure modes: IForest provides
immediate detection, LSTM improves confidence when history is available.

**Why LangGraph for orchestration?**
Human-in-the-loop interrupts are a first-class LangGraph concept. The `interrupt_before`
API integrates naturally with the async FastAPI backend, allowing the pipeline to pause,
persist state, wait for human input, and resume — all without polling loops or manual
state management.

**Why A-MEM + Letta (custom implementations)?**
The official libraries were not stable enough for production use at project start. Custom
implementations gave full control over memory format, embedding storage, and Zettelkasten
linking logic — and avoided dependency conflicts between LangGraph, motor, and external
agent SDKs.

**Why Azure Blob for model storage?**
Azure Blob provides immutable, versioned artefact storage with a straightforward Python
SDK. The `BlobServiceClient` API is simple enough to wrap completely in a 160-line
service class with full graceful degradation. Keeping models out of the repository
(never committed) and always trained from source data is also a deliberate MLOps
discipline — it prevents model/code version drift.

**Why PostgreSQL over MongoDB for training data?**
Training data is tabular, with known schema, and benefits from SQL joins and window
functions for feature engineering. MongoDB is used for unstructured documents (alerts,
memory notes, drift reports) where schema flexibility matters. Using the right database
for the access pattern — relational for training, document for operational data — is
a cleaner architectural choice than forcing everything into one store.

**Why Evidently AI for drift detection?**
Evidently provides a production-quality statistical drift test (KS-test + chi-squared)
with a clean Python API and structured output that can be stored in MongoDB as-is.
The `DataDriftPreset` covers all five sensor features in a single call, returning
per-feature p-values and a dataset-level drift flag. Custom implementations of the
same tests would require more code with less statistical rigour.

**Why MLflow challenger/champion over stages?**
MLflow 3.x deprecated `Production`, `Staging`, and `Archived` stages in favour of
free-form aliases. The `challenger`/`champion` alias pattern maps more naturally to the
actual workflow: a model is a challenger until explicitly validated, then becomes the
champion. Aliases also allow multiple concurrent aliases on a single version (e.g.
`champion` and `stable`) without the constraints of the old single-stage model.

---

## Testing

182 tests across 10 test modules — all pass, none skipped (unless real credentials
are unavailable for `@pytest.mark.integration` tests).

| Test Module | Tests | Coverage |
|---|---|---|
| `test_ml_models.py` | 20 | Scaler, IForest, LSTM, MLService |
| `test_agents.py` | 20 | AnomalyDetector, AlertGenerator, Orchestrator, HITL |
| `test_api.py` | 21 | All REST endpoints, validation, error handling |
| `test_maintenance_logs.py` | 24 | Bulk add, list, count, Qdrant/MongoDB integration |
| `test_evaluation.py` | 21 | RAG eval, LLM-judge, nightly scheduler |
| `test_blob_storage.py` | 14 | Upload, download, list, exists, MLService fallback |
| `test_postgres_service.py` | 11 | Init, data access, graceful degradation, CSV fallback |
| `test_drift_monitoring.py` | 15 | Reference load, Evidently report, Redis window, API |
| `test_model_registry.py` | 26 | Aliases, promote, rollback, compare, AUC resolution |
| `test_integration_new_features.py` | 14 | End-to-end + 2 real-credential integration tests |

```bash
# Run all tests (skips integration if credentials not in .env)
pytest tests/ -v

# Run only mocked unit tests (no credentials needed)
pytest tests/ -v -m "not integration"

# Run real integration tests (requires AZURE + POSTGRES in .env)
pytest tests/ -v -m integration

# Run full pipeline smoke test
python data/test_pipeline.py
```

---

## API Endpoints

All endpoints are documented at `http://localhost:8080/docs` (Swagger UI) once the
server is running.

**Sensor Ingestion**
```
POST /api/sensors/ingest               Ingest a sensor reading; returns AnomalyResult
GET  /api/sensors/{machine_id}/history Last N readings for a machine
```

**Alerts**
```
GET  /api/alerts                       List alerts (filterable by machine, severity)
GET  /api/alerts/{alert_id}            Get single alert with full reasoning trace
POST /api/alerts/{alert_id}/approve    Approve a pending alert
POST /api/alerts/{alert_id}/reject     Reject a pending alert
GET  /api/alerts/stats                 Aggregate counts by severity and status
```

**Dashboard**
```
GET  /api/dashboard/stats              Anomalies last 24h, pending alerts, drift status
GET  /api/dashboard/machines           Per-machine health summary
```

**Evaluation & Drift**
```
GET  /api/evaluation/latest            Most recent RAG + LLM-judge scores
GET  /api/evaluation/history           Last 30 evaluation results
GET  /api/evaluation/run               Trigger evaluation manually (background task)
GET  /api/evaluation/drift/latest      Most recent Evidently drift report
GET  /api/evaluation/drift/history     Last N drift reports
GET  /api/evaluation/drift/run         Trigger drift check manually
```

**Maintenance Logs (RAG Knowledge Base)**
```
POST /api/maintenance-logs/add         Add a single maintenance log + embed in Qdrant
POST /api/maintenance-logs/bulk-add    Add up to 100 logs in one request
GET  /api/maintenance-logs             List logs (filter by failure type)
GET  /api/maintenance-logs/count       Counts in MongoDB vs Qdrant (sync check)
```

**WebSocket**
```
WS /ws/alerts     Real-time alert stream (Redis pub/sub)
WS /ws/sensors    Real-time sensor reading stream
```

**Health**
```
GET /health       Service status: ml_ready, redis, mongo, qdrant, blob_storage,
                  postgres, drift_monitor, orchestrator — all fields always present
```

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository)

- 10,000 synthetic CNC machine readings · 5 failure types
- TWF (tool wear), HDF (heat dissipation), PWF (power), OSF (overstrain), RNF (random)
- 339 failure rows (3.4%) · Features: air temp, process temp, rotational speed, torque, tool wear

**Data flow:**
```
UCI download  →  data/ai4i_2020.csv
     ↓
data/load_to_postgres.py  →  PostgreSQL sensor_readings table (10,000 rows)
     ↓
ml/train_*.py reads from PostgreSQL (CSV fallback if unavailable)
     ↓
Trained model  →  ml/models/*.pkl  →  Azure Blob Storage
     ↓
MLflow registry  →  champion alias  →  MLService loads on startup
     ↓
Live sensor stream  →  POST /api/sensors/ingest  →  AnomalyDetector
     ↓
Evidently AI compares live Redis window vs PostgreSQL reference (nightly)
```

---

## Dataset Citation

> Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*.
> UCI Machine Learning Repository.
> [https://doi.org/10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)

---

*Built by Shital Nandre as a portfolio project targeting industrial AI and MLOps roles.*
*Stack: FastAPI · LangGraph · Groq · TensorFlow · Qdrant · MongoDB · Redis · Azure · PostgreSQL · MLflow · Gradio*
