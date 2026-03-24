"""LangGraph agent state definitions."""
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
import operator

from app.models.sensor import SensorReading
from app.models.anomaly import AnomalyResult
from app.models.maintenance import MaintenanceLog
from app.models.alert import MaintenanceAlert, RootCauseReport


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────────
    sensor_reading: SensorReading
    session_id: str

    # ── Anomaly Detection Stage ────────────────────────────────────────────────
    anomaly_result: Optional[AnomalyResult]

    # ── RAG / Context Retrieval Stage ─────────────────────────────────────────
    similar_incidents: Annotated[list[MaintenanceLog], operator.add]
    retrieved_context: str  # raw text chunks from LlamaIndex

    # ── Root Cause Reasoning Stage ────────────────────────────────────────────
    root_cause_report: Optional[RootCauseReport]
    reasoning_steps: Annotated[list[str], operator.add]
    agent_memory_recalled: Annotated[list[str], operator.add]

    # ── Human-in-the-Loop Stage ───────────────────────────────────────────────
    pending_alert: Optional[MaintenanceAlert]
    human_approved: Optional[bool]
    human_feedback: Optional[str]

    # ── Final Output ──────────────────────────────────────────────────────────
    final_alert: Optional[MaintenanceAlert]
    error: Optional[str]

    # ── Meta / Routing ────────────────────────────────────────────────────────
    messages: Annotated[list[dict[str, Any]], operator.add]  # LangGraph message history
    next_node: Optional[str]
