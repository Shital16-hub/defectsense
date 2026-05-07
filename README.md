# 🏭 DefectSense

**Manufacturing Defect Root-Cause Intelligence — Hybrid ML + GenAI Multi-Agent System**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-orange)](https://langchain-ai.github.io/langgraph)
[![Groq](https://img.shields.io/badge/LLM-Groq_llama3-purple)](https://groq.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00)](https://tensorflow.org)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)

Manufacturing plants lose **$260 billion per year** to unplanned downtime (Deloitte, 2024).
Existing condition-monitoring systems tell you *when* a machine failed — not *why*.
DefectSense closes that gap with a reasoning AI that delivers a root cause, ranked
maintenance actions, and an explanation a factory-floor engineer can actually act on.

---

## What This System Does

DefectSense is a real-time anomaly detection and root-cause reasoning platform built
for manufacturing environments. It continuously ingests sensor readings from industrial
machines — voltage, rotational speed, pressure, and vibration — and runs them through
an LSTM Autoencoder that learns normal operating sequences and flags deviations via
reconstruction error. When an anomaly is detected, a LangGraph multi-agent pipeline
activates: a retrieval agent pulls relevant maintenance records from a vector store,
a ReAct reasoning agent synthesises those records with the sensor data and machine
history to produce a structured root-cause report, and an alert generator formats that
into plain language a maintenance engineer can act on immediately. Every incident is
stored in an agentic memory system that improves future diagnoses, and every alert
waits for human approval before being actioned — or auto-approves after 15 minutes if
the operator is unavailable. Unlike rule-based threshold systems that generate constant
false positives, DefectSense learns the specific failure patterns of each machine and
explains them, cutting mean time to repair while keeping human judgment in the loop
for critical decisions.

---

## Architecture

```mermaid
graph TD
    A[Sensor Stream<br/>Azure PdM / Live Data] -->|POST /api/sensors/ingest| B[FastAPI :8080]
    B --> C[AnomalyDetectorAgent]
    C -->|LSTM Autoencoder| D{Anomaly?}
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

**1. LSTM Autoencoder Anomaly Detection**
The LSTM Autoencoder learns normal operating sequences over a 198-reading window and
flags gradual drift — the kind of slow degradation that kills machines over days.
Reconstruction error (MSE) is compared against a statistical threshold (mean + 3 std
over validation normal sequences). When reconstruction error exceeds threshold the
reading is classified as anomalous. Trained on 876,100 Azure PdM sensor readings
across 100 machines, the model achieves AUC 0.846 on real pre-failure sequences.

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

Evaluated on the **Azure PdM dataset** — 100 machines, 1 year of hourly readings,
761 failure events across 4 component failure types.

| Model | AUC | Detection Rate |
|---|---|---|
| **LSTM Autoencoder** | **0.846** | **67%** |

**Why AUC?**
AUC (Area Under the ROC Curve) measures ranking quality — how reliably the model assigns
higher reconstruction error to pre-failure sequences than to clean normal sequences.
An AUC of **0.846** means the model correctly ranks 85 out of 100 randomly drawn
failure/normal sequence pairs. The sequence length of 198 readings was computed
automatically from the median inter-failure interval across 761 failure events,
ensuring the model sees the most informative time window before each failure.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.135, Pydantic v2, Uvicorn |
| **Orchestration** | LangGraph 1.1 (HITL state machine) |
| **LLM** | Groq API — llama-3.3-70b (reasoning) + llama-3.1-8b (alerts) |
| **ML** | TensorFlow 2.21 (LSTM Autoencoder) |
| **RAG** | Qdrant Cloud (vector store) + sentence-transformers/all-MiniLM-L6-v2 |
| **Agent Memory** | A-MEM custom (Zettelkasten, MongoDB) + Letta (MemGPT-style) |
| **Databases** | MongoDB Atlas (motor async) · Redis/Upstash (pub/sub + cache) |
| **Model Storage** | Local — `ml/models/` directory (generated by training script) |
| **Training DB** | PostgreSQL via Supabase (SQLAlchemy, SSL) |
| **Drift Monitor** | Evidently AI (DataDriftPreset, nightly APScheduler) |
| **Model Registry** | MLflow 3.x (SQLite backend, challenger/champion aliases) |
| **Frontend** | Gradio 6.9, Plotly 6.6 |
| **Observability** | MLflow (live prediction tracking) · LangSmith (LLM traces) |
| **Dataset** | Azure PdM (876,100 rows, 100 machines, 4 failure components) |

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

> **No cloud storage credentials needed.** Models are stored locally in `ml/models/`
> and generated by running the training script.

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

Key variables to set:

```env
# LLM
GROQ_API_KEY=gsk_...

# Databases
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=...
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net
REDIS_URL=rediss://...upstash.io:6380

# Training data
POSTGRES_URL=postgresql://postgres.xxx:password@aws-0-region.pooler.supabase.com:6543/postgres

# MLflow (local SQLite — no server needed)
MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db
```

### Step 3 — Load training data

```bash
python data/load_azure_to_postgres.py  # load 876,100 Azure PdM rows into PostgreSQL
```

### Step 4 — Train model

The script automatically: trains → evaluates on pre-failure sequences →
logs metrics to MLflow → saves locally to `ml/models/` → registers in MLflow
as `challenger`.

```bash
python ml/train_autoencoder_azure.py
```

Expected output:

```
Registry:  defectsense_lstm_autoencoder v1
Alias:     challenger
AUC:       0.8460
```

### Step 5 — Promote model to production

```bash
# See all registered versions with AUC scores
python ml/promote_to_production.py --list

# Promote to champion
python ml/promote_to_production.py --model lstm --version 1

# Roll back if needed
python ml/promote_to_production.py --model lstm --rollback
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
│   │   ├── anomaly_detector.py       # LSTM Autoencoder anomaly agent
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
│   └── train_autoencoder_azure.py
├── frontend/app.py                   # Gradio 4-tab dashboard
├── data/
│   ├── azure_pdm/                    # Azure PdM dataset CSVs (876,100 rows)
│   ├── azure_lstm_config.json        # sequence_length=198 (auto-computed)
│   ├── load_azure_to_postgres.py     # Azure PdM → PostgreSQL loader
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

### Model Artifact Storage — Local (`ml/models/`)

Models are stored locally in `ml/models/` and generated by running
`ml/train_autoencoder_azure.py`. The `BlobStorageService` class is kept as an
interface-compatible stub (always returns `is_available=False`) so the rest of the
codebase — health checks, `MLService`, tests — continues to work without modification.
The stub logs a clear info message on startup and returns False for all operations.

To restore cloud storage in future, replace the stub implementation in
`app/services/blob_storage_service.py` with Azure Blob Storage, AWS S3, or
Hugging Face Hub — no other files need to change because the interface is unchanged.

### Training Data — PostgreSQL on Supabase

The Azure PdM dataset's 876,100 sensor readings are stored in a Supabase PostgreSQL
instance in the `azure_sensor_readings` table. The training script queries
clean normal rows (`is_clean_normal = TRUE`) for LSTM training, and failure-adjacent
rows for threshold calibration and AUC computation. SQLAlchemy with `sslmode=require`
is used for the Supabase session pooler. If PostgreSQL is unavailable (network issue,
quota exceeded), the training script exits early with a clear error — training data
is too large to bundle as a CSV fallback. The `is_connected` property and graceful
`init()` pattern mean the app itself starts cleanly even without a database connection.

### Drift Monitoring — Evidently AI

An APScheduler cron job fires at 03:00 UTC every night. It discovers active machine IDs
from Redis key patterns (`sensor:*:readings`), fetches the last 100 readings per machine,
and runs Evidently's `DataDriftPreset` — comparing the live distribution against the
PostgreSQL reference distribution of normal samples. Evidently computes a KS-test
p-value per feature; if more than 50% of the four sensor features (volt, rotate,
pressure, vibration) drift (p < 0.05), `is_drifted` is set to `True` and the report
is saved to MongoDB's `drift_reports` collection. The operations team can query
`/api/evaluation/drift/latest` or `/drift/history` to track distribution health over
time. When drift is detected, `train_autoencoder_azure.py` is launched automatically
as a background subprocess to retrain with the new distribution.

### Model Registry — MLflow (Challenger/Champion)

Models follow a two-stage lifecycle using MLflow 3.x aliases: a freshly trained model
receives the `challenger` alias automatically at the end of the training script. After
manual review of the AUC score (visible in `--list`), the operator runs
`promote_to_production.py --model lstm --version N` to move the `champion` alias.
The previous champion has its alias removed first, so there is always at most one
champion per model. Rollback is equally simple: `--rollback` moves `champion` to
`current_version - 1` after verifying that version still exists in the registry.
One subtle implementation detail: `search_model_versions()` does not populate `mv.aliases`
with the SQLite backend — the service builds an explicit alias map via
`get_model_version_by_alias()` for each known alias to work around this.

---

## Engineering Decisions

**Why Groq instead of OpenAI?**
Groq's llama-3.3-70b delivers sub-2-second reasoning responses on manufacturing anomaly
prompts. For a real-time monitoring system where a factory-floor engineer needs an answer
now, not in 15 seconds, inference latency is a product requirement, not a nice-to-have.

**Why LSTM Autoencoder for anomaly detection?**
LSTM captures temporal dependencies across a 198-reading window — exactly the right
inductive bias for machine failures that develop over hours, not instants. Reconstruction
error is a natural anomaly score: the model learns to reconstruct normal operating
patterns and fails to reconstruct anomalous ones. The sequence length was derived
automatically from the Azure PdM failure event distribution (median inter-failure gap)
rather than picked arbitrarily, which is why the model achieves AUC 0.846 on real
industrial data.

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

**Why local model storage instead of cloud storage?**
Models are generated by running the training script once and stored in `ml/models/`
(gitignored). This removes an external dependency at no cost for a single-deployment
portfolio project. The `BlobStorageService` interface is kept as a stub so cloud
storage can be restored by replacing one file — the `is_available` property and method
signatures are identical to the Azure implementation. Keeping models out of the
repository (gitignored) and always trainable from source data remains a deliberate
MLOps discipline — it prevents model/code version drift.

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
| `test_ml_models.py` | 15 | Scaler, LSTM Autoencoder, MLService |
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

**Microsoft Azure Predictive Maintenance Dataset**

- 876,100 hourly sensor readings · 100 machines · 1 year
- 761 failure events · 0.087% failure rate · 4 failure types (EVF · RSF · PBF · BWF)
- Features: volt, rotate (rpm), pressure, vibration
- `is_clean_normal` flag marks readings with no failure within the inter-failure window

**Data flow:**
```
Kaggle download  →  data/azure_pdm/  (5 CSV files)
     ↓
data/load_azure_to_postgres.py  →  PostgreSQL azure_sensor_readings (876,100 rows)
     ↓
ml/train_autoencoder_azure.py reads clean rows from PostgreSQL
     ↓
Trained model  →  ml/models/azure_lstm_autoencoder.keras  (stored locally in ml/models/)
     ↓
MLflow registry  →  champion alias  →  MLService loads on startup
     ↓
Live sensor stream  →  POST /api/sensors/ingest  →  AnomalyDetector
     ↓
Evidently AI compares live Redis window vs PostgreSQL reference (nightly)
```

---

## Dataset Source

Microsoft Azure Predictive Maintenance dataset available via Kaggle:
[https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

---

*Built by Shital Nandre as a portfolio project targeting industrial AI and MLOps roles.*
*Stack: FastAPI · LangGraph · Groq · TensorFlow · Qdrant · MongoDB · Redis · Azure · PostgreSQL · MLflow · Gradio*
