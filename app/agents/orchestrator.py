"""
DefectSense LangGraph Orchestrator — Session 5.

Pipeline:
  detect_anomaly
       |
   (anomaly?)
    /       \
  NO         YES
  |           |
  END    retrieve_context   ← RAG + amem update
             |
       reason_root_cause    ← ReAct LLM
             |
         hitl_gate          ← interrupt_before="generate_alert"
             |
       generate_alert
             |
            END

LangSmith: each run tagged with machine_id + session_id.
Timeout: 15 min → auto-approve (checked by background task in main.py).
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, TypedDict

from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading
from app.models.alert import MaintenanceAlert, RootCauseReport


# ── Pipeline state ─────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # Input
    reading:    SensorReading
    session_id: str
    machine_id: str

    # Anomaly stage
    anomaly_result: Optional[AnomalyResult]
    is_anomaly:     bool

    # Context stage
    similar_incidents: list         # list[MaintenanceLog]
    sensor_context:    str

    # Reasoning stage
    root_cause_report: Optional[RootCauseReport]

    # HITL stage
    approved:          Optional[bool]
    approved_by:       str
    rejection_reason:  Optional[str]
    auto_approved:     bool

    # Output
    alert: Optional[MaintenanceAlert]
    error: Optional[str]


# ── Orchestrator ───────────────────────────────────────────────────────────────

class DefectSenseOrchestrator:
    """
    Stateful LangGraph orchestrator.

    Usage:
        orch = DefectSenseOrchestrator(...)
        orch.build()

        # Run full pipeline (interrupts at HITL if confidence < threshold)
        state = await orch.run(reading, session_id)

        # Resume after human decision
        state = await orch.resume(thread_id, approved=True, approved_by="eng")
    """

    def __init__(
        self,
        detector=None,
        context_retriever=None,
        amem=None,
        reasoner=None,
        alert_generator=None,
        groq_api_key: Optional[str] = None,
        auto_approve_threshold: float = 0.95,
        approval_timeout_minutes: int = 15,
    ) -> None:
        self._detector          = detector
        self._context_retriever = context_retriever
        self._amem              = amem
        self._reasoner          = reasoner
        self._alert_gen         = alert_generator
        self._auto_threshold    = auto_approve_threshold
        self._timeout_minutes   = approval_timeout_minutes
        self._graph: Any        = None
        self._checkpointer      = MemorySaver()

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self) -> None:
        builder = StateGraph(PipelineState)

        builder.add_node("detect_anomaly",    self._node_detect_anomaly)
        builder.add_node("retrieve_context",  self._node_retrieve_context)
        builder.add_node("reason_root_cause", self._node_reason_root_cause)
        builder.add_node("hitl_gate",         self._node_hitl_gate)
        builder.add_node("generate_alert",    self._node_generate_alert)

        builder.set_entry_point("detect_anomaly")

        builder.add_conditional_edges(
            "detect_anomaly",
            lambda s: "anomaly" if s.get("is_anomaly") else "no_anomaly",
            {"anomaly": "retrieve_context", "no_anomaly": END},
        )
        builder.add_edge("retrieve_context",  "reason_root_cause")
        builder.add_edge("reason_root_cause", "hitl_gate")
        builder.add_edge("hitl_gate",         "generate_alert")
        builder.add_edge("generate_alert",    END)

        self._graph = builder.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["generate_alert"],
        )
        logger.info("Orchestrator: LangGraph compiled (HITL interrupt before generate_alert)")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        reading: SensorReading,
        session_id: Optional[str] = None,
    ) -> dict:
        """Start a new pipeline run. Returns state dict with thread_id."""
        if self._graph is None:
            self.build()

        sid       = session_id or str(uuid.uuid4())
        thread_id = f"{reading.machine_id}:{sid}"

        config = self._make_config(thread_id, reading.machine_id, sid)

        logger.info("Orchestrator: starting | machine={} thread={}", reading.machine_id, thread_id)

        final_state = await self._graph.ainvoke(
            {
                "reading":    reading,
                "session_id": sid,
                "machine_id": reading.machine_id,
            },
            config=config,
        )

        # Auto-approve if confidence is high enough
        report: Optional[RootCauseReport] = final_state.get("root_cause_report")
        if report and final_state.get("approved") is None:
            if report.confidence >= self._auto_threshold:
                logger.info(
                    "Orchestrator: auto-approving conf={:.2f} >= {:.2f}",
                    report.confidence, self._auto_threshold,
                )
                return await self.resume(thread_id, approved=True, approved_by="auto", auto=True)
            else:
                logger.info(
                    "Orchestrator: HITL pending thread={} conf={:.2f} severity={}",
                    thread_id, report.confidence, report.severity,
                )

        return {**final_state, "thread_id": thread_id}

    async def resume(
        self,
        thread_id: str,
        approved: bool,
        approved_by: str = "human",
        rejection_reason: Optional[str] = None,
        auto: bool = False,
    ) -> dict:
        """Resume a paused pipeline after approval or rejection."""
        if self._graph is None:
            raise RuntimeError("Orchestrator not built")

        parts      = thread_id.split(":", 1)
        machine_id = parts[0]
        sid        = parts[1] if len(parts) > 1 else thread_id
        config     = self._make_config(thread_id, machine_id, sid)

        if not approved:
            update: dict = {
                "approved":         False,
                "approved_by":      approved_by,
                "rejection_reason": rejection_reason or "Rejected by operator",
                "auto_approved":    False,
            }
            await self._graph.aupdate_state(config, update, as_node="hitl_gate")
            logger.info("Orchestrator: REJECTED by {} | thread={}", approved_by, thread_id)
            state = await self._graph.aget_state(config)
            return {**state.values, "thread_id": thread_id}

        update = {"approved": True, "approved_by": approved_by, "auto_approved": auto}
        await self._graph.aupdate_state(config, update, as_node="hitl_gate")

        logger.info("Orchestrator: APPROVED by {} (auto={}) | thread={}", approved_by, auto, thread_id)

        final_state = await self._graph.ainvoke(None, config=config)
        return {**final_state, "thread_id": thread_id}

    # ── Nodes ──────────────────────────────────────────────────────────────────

    async def _node_detect_anomaly(self, state: PipelineState) -> dict:
        reading = state["reading"]
        try:
            result: AnomalyResult = await self._detector.run(reading)
            logger.debug(
                "Orchestrator[detect]: machine={} anomaly={} prob={:.3f}",
                reading.machine_id, result.is_anomaly, result.failure_probability,
            )
            return {"anomaly_result": result, "is_anomaly": result.is_anomaly}
        except Exception as exc:
            logger.error("Orchestrator[detect] error: {}", exc)
            return {"is_anomaly": False, "error": str(exc)}

    async def _node_retrieve_context(self, state: PipelineState) -> dict:
        anomaly: AnomalyResult = state["anomaly_result"]
        incidents, sensor_ctx  = [], ""

        # RAG retrieval
        if self._context_retriever is not None:
            try:
                incidents, sensor_ctx = await self._context_retriever.retrieve(anomaly)
            except Exception as exc:
                logger.warning("Orchestrator[retrieve_context] error: {}", exc)

        # A-MEM note for this anomaly
        if self._amem is not None:
            try:
                content = (
                    f"Anomaly on {state['machine_id']}: "
                    f"type={anomaly.failure_type_prediction}, "
                    f"prob={anomaly.failure_probability:.3f}"
                )
                await self._amem.add_memory(
                    content=content,
                    tags=["anomaly", anomaly.failure_type_prediction or "UNKNOWN", state["machine_id"]],
                    source="orchestrator",
                )
            except Exception as exc:
                logger.warning("Orchestrator[update_amem] error: {}", exc)

        return {"similar_incidents": incidents, "sensor_context": sensor_ctx}

    async def _node_reason_root_cause(self, state: PipelineState) -> dict:
        try:
            report: RootCauseReport = await self._reasoner.analyze(
                anomaly=state["anomaly_result"],
                similar_incidents=state.get("similar_incidents", []),
                sensor_context=state.get("sensor_context", ""),
                session_id=state.get("session_id"),
            )
            logger.info(
                "Orchestrator[reason]: conf={:.2f} severity={} | machine={}",
                report.confidence, report.severity, state["machine_id"],
            )
            return {"root_cause_report": report}
        except Exception as exc:
            logger.error("Orchestrator[reason] error: {}", exc)
            return {"error": str(exc)}

    async def _node_hitl_gate(self, state: PipelineState) -> dict:
        """Runs before the interrupt fires — just logs pending state."""
        report: Optional[RootCauseReport] = state.get("root_cause_report")
        if report:
            logger.info(
                "Orchestrator[hitl_gate]: awaiting approval | machine={} conf={:.2f} sev={}",
                state.get("machine_id"), report.confidence, report.severity,
            )
        return {}

    async def _node_generate_alert(self, state: PipelineState) -> dict:
        if state.get("approved") is False:
            logger.info("Orchestrator[generate_alert]: skipped (rejected)")
            return {}

        report: Optional[RootCauseReport] = state.get("root_cause_report")
        if report is None:
            return {"error": "No root cause report available"}

        try:
            alert: MaintenanceAlert = await self._alert_gen.generate(
                report=report,
                session_id=state.get("session_id"),
            )
            if state.get("approved") is True:
                await self._alert_gen.mark_approved(
                    alert,
                    approved_by=state.get("approved_by", "auto"),
                    auto=state.get("auto_approved", False),
                )
            logger.info(
                "Orchestrator[generate_alert]: alert {} created approved={}",
                alert.alert_id[:8], alert.approved,
            )
            return {"alert": alert}
        except Exception as exc:
            logger.error("Orchestrator[generate_alert] error: {}", exc)
            return {"error": str(exc)}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_config(self, thread_id: str, machine_id: str, session_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "tags":         [f"machine:{machine_id}", f"session:{session_id}", "defectsense"],
            "metadata":     {"machine_id": machine_id, "session_id": session_id},
        }
