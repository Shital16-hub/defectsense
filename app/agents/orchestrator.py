"""
DefectSense LangGraph Orchestrator — Session 5.

Pipeline:
  ingest_sensor
       |
  detect_anomaly
       |
   (anomaly?)
    /       \
  NO         YES
  |           |
  END     [parallel]
        retrieve_context
        update_amem
             |
       reason_root_cause
             |
    [HUMAN-IN-THE-LOOP]
    interrupt_before="generate_alert"
      /            \
  auto-approve    await human
  (conf >= 0.95)
             |
       generate_alert
             |
            END

LangSmith: each run tagged with machine_id + session_id.
Timeout: if no human decision after HUMAN_APPROVAL_TIMEOUT_MINUTES → auto-approve as CRITICAL.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Annotated, Any, Optional, TypedDict

from loguru import logger

# ── LangGraph imports ──────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading
from app.models.alert import MaintenanceAlert, RootCauseReport


# ── Pipeline state ─────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    # Input
    reading: SensorReading
    session_id: str
    machine_id: str

    # Anomaly stage
    anomaly_result: Optional[AnomalyResult]
    is_anomaly: bool

    # Context stage (parallel)
    context_retrieved: bool
    amem_updated: bool

    # Reasoning stage
    root_cause_report: Optional[RootCauseReport]

    # HITL stage
    approved: Optional[bool]           # None=pending, True=approved, False=rejected
    approved_by: str
    rejection_reason: Optional[str]
    auto_approved: bool
    approval_timeout: Optional[str]    # ISO datetime string

    # Output
    alert: Optional[MaintenanceAlert]
    error: Optional[str]


# ── Orchestrator ───────────────────────────────────────────────────────────────

class DefectSenseOrchestrator:
    """
    Stateful LangGraph orchestrator.

    Usage:
        orch = DefectSenseOrchestrator(...)
        await orch.build()

        # Run full pipeline (will interrupt at HITL node if needed)
        result = await orch.run(reading, session_id)

        # Resume after human decision
        result = await orch.resume(thread_id, approved=True, approved_by="engineer")
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
        self._detector               = detector
        self._context_retriever      = context_retriever
        self._amem                   = amem
        self._reasoner               = reasoner
        self._alert_gen              = alert_generator
        self._api_key                = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._auto_threshold         = auto_approve_threshold
        self._timeout_minutes        = approval_timeout_minutes
        self._graph: Any             = None
        self._checkpointer           = MemorySaver()

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self) -> None:
        """Compile the LangGraph state machine."""
        builder = StateGraph(PipelineState)

        # Nodes
        builder.add_node("detect_anomaly",    self._node_detect_anomaly)
        builder.add_node("retrieve_context",  self._node_retrieve_context)
        builder.add_node("update_amem",       self._node_update_amem)
        builder.add_node("reason_root_cause", self._node_reason_root_cause)
        builder.add_node("hitl_gate",         self._node_hitl_gate)
        builder.add_node("generate_alert",    self._node_generate_alert)

        # Entry
        builder.set_entry_point("detect_anomaly")

        # Routing after anomaly detection
        builder.add_conditional_edges(
            "detect_anomaly",
            self._route_after_detection,
            {"anomaly": "retrieve_context", "no_anomaly": END},
        )

        # Parallel context + amem (fan-out then fan-in)
        builder.add_edge("retrieve_context",  "reason_root_cause")
        builder.add_edge("update_amem",       "reason_root_cause")
        builder.add_edge("reason_root_cause", "hitl_gate")
        builder.add_edge("hitl_gate",         "generate_alert")
        builder.add_edge("generate_alert",    END)

        self._graph = builder.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["generate_alert"],
        )
        logger.info("Orchestrator: LangGraph compiled (HITL at generate_alert)")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        reading: SensorReading,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Start a new pipeline run.

        Returns state dict. If HITL needed, state['alert'] is None and
        state['approved'] is None — call `resume()` with the thread_id.
        """
        if self._graph is None:
            self.build()

        sid = session_id or str(uuid.uuid4())
        thread_id = f"{reading.machine_id}:{sid}"

        initial_state: PipelineState = {
            "reading":    reading,
            "session_id": sid,
            "machine_id": reading.machine_id,
        }

        config = self._make_config(thread_id, reading.machine_id, sid)

        logger.info(
            "Orchestrator: starting pipeline | machine={} thread={}",
            reading.machine_id, thread_id,
        )

        final_state = await self._graph.ainvoke(initial_state, config=config)

        # If graph paused at HITL, check for auto-approve
        if final_state.get("root_cause_report") and final_state.get("approved") is None:
            report: RootCauseReport = final_state["root_cause_report"]
            if report.confidence >= self._auto_threshold:
                logger.info(
                    "Orchestrator: auto-approving (confidence={:.2f} >= {:.2f})",
                    report.confidence, self._auto_threshold,
                )
                return await self.resume(
                    thread_id,
                    approved=True,
                    approved_by="auto",
                    auto=True,
                )
            else:
                # Set approval timeout timestamp
                timeout_at = datetime.now(tz=timezone.utc) + timedelta(
                    minutes=self._timeout_minutes
                )
                final_state["approval_timeout"] = timeout_at.isoformat()
                logger.info(
                    "Orchestrator: HITL pending for thread {} (timeout={})",
                    thread_id, timeout_at.strftime("%H:%M:%S UTC"),
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
        """
        Resume a paused pipeline after human approval/rejection.

        If rejected, returns state with approved=False and no alert generated.
        """
        if self._graph is None:
            raise RuntimeError("Orchestrator not built — call build() first")

        parts = thread_id.split(":", 1)
        machine_id = parts[0] if parts else "unknown"
        sid        = parts[1] if len(parts) > 1 else thread_id

        config = self._make_config(thread_id, machine_id, sid)

        if not approved:
            # Rejection — patch state and skip generate_alert
            update: PipelineState = {
                "approved":          False,
                "approved_by":       approved_by,
                "rejection_reason":  rejection_reason or "Rejected by operator",
                "auto_approved":     False,
            }
            await self._graph.aupdate_state(config, update)
            # Skip generate_alert by jumping to END
            await self._graph.aupdate_state(config, {}, as_node="generate_alert")
            logger.info("Orchestrator: alert REJECTED by {} | thread={}", approved_by, thread_id)
            state = await self._graph.aget_state(config)
            return {**state.values, "thread_id": thread_id}

        # Approval — inject decision and resume
        update = {
            "approved":      True,
            "approved_by":   approved_by,
            "auto_approved": auto,
        }
        await self._graph.aupdate_state(config, update, as_node="hitl_gate")

        logger.info(
            "Orchestrator: alert APPROVED by {} (auto={}) | thread={}",
            approved_by, auto, thread_id,
        )

        final_state = await self._graph.ainvoke(None, config=config)
        return {**final_state, "thread_id": thread_id}

    async def check_timeouts(self) -> list[str]:
        """
        Scan pending threads for expired approval windows.
        Auto-approve expired threads. Returns list of thread_ids acted on.
        """
        # MemorySaver doesn't expose a list-all-threads API easily,
        # so callers (e.g. a background task) must track pending thread_ids.
        # This method is a hook for that integration.
        return []

    # ── Nodes ──────────────────────────────────────────────────────────────────

    async def _node_detect_anomaly(self, state: PipelineState) -> PipelineState:
        reading = state["reading"]
        try:
            result: AnomalyResult = await self._detector.run(reading)
            logger.debug(
                "Orchestrator[detect]: machine={} anomaly={} prob={:.3f}",
                reading.machine_id, result.is_anomaly, result.failure_probability,
            )
            return {
                "anomaly_result": result,
                "is_anomaly":     result.is_anomaly,
            }
        except Exception as exc:
            logger.error("Orchestrator[detect] error: {}", exc)
            return {"is_anomaly": False, "error": str(exc)}

    async def _node_retrieve_context(self, state: PipelineState) -> PipelineState:
        if self._context_retriever is None:
            return {"context_retrieved": False}
        try:
            anomaly: AnomalyResult = state["anomaly_result"]
            # Context retriever stores results on the anomaly_result / amem directly;
            # result is embedded in the reasoner's context call
            await self._context_retriever.retrieve(anomaly)
            return {"context_retrieved": True}
        except Exception as exc:
            logger.warning("Orchestrator[retrieve_context] error: {}", exc)
            return {"context_retrieved": False}

    async def _node_update_amem(self, state: PipelineState) -> PipelineState:
        if self._amem is None:
            return {"amem_updated": False}
        try:
            anomaly: AnomalyResult = state["anomaly_result"]
            content = (
                f"Anomaly detected on {state['machine_id']}: "
                f"failure_type={anomaly.failure_type_prediction}, "
                f"probability={anomaly.failure_probability:.3f}"
            )
            await self._amem.add_memory(
                content=content,
                tags=["anomaly", anomaly.failure_type_prediction or "UNKNOWN", state["machine_id"]],
                source="orchestrator",
            )
            return {"amem_updated": True}
        except Exception as exc:
            logger.warning("Orchestrator[update_amem] error: {}", exc)
            return {"amem_updated": False}

    async def _node_reason_root_cause(self, state: PipelineState) -> PipelineState:
        try:
            anomaly: AnomalyResult = state["anomaly_result"]
            report: RootCauseReport = await self._reasoner.reason(
                anomaly_result=anomaly,
                session_id=state.get("session_id"),
            )
            return {"root_cause_report": report}
        except Exception as exc:
            logger.error("Orchestrator[reason] error: {}", exc)
            return {"error": str(exc)}

    async def _node_hitl_gate(self, state: PipelineState) -> PipelineState:
        """
        HITL gate — this node runs before the interrupt fires.
        Actual interruption is handled by LangGraph's interrupt_before mechanism.
        We just log the pending state here.
        """
        report: Optional[RootCauseReport] = state.get("root_cause_report")
        if report:
            logger.info(
                "Orchestrator[hitl_gate]: waiting for approval | "
                "machine={} confidence={:.2f} severity={}",
                state.get("machine_id"), report.confidence, report.severity,
            )
        return {}

    async def _node_generate_alert(self, state: PipelineState) -> PipelineState:
        if state.get("approved") is False:
            logger.info("Orchestrator[generate_alert]: skipped (rejected)")
            return {}

        report: Optional[RootCauseReport] = state.get("root_cause_report")
        if report is None:
            return {"error": "No root cause report to generate alert from"}

        try:
            alert: MaintenanceAlert = await self._alert_gen.generate(
                report=report,
                session_id=state.get("session_id"),
            )

            # Apply approval metadata if already decided (auto-approve path)
            if state.get("approved") is True:
                await self._alert_gen.mark_approved(
                    alert,
                    approved_by=state.get("approved_by", "auto"),
                    auto=state.get("auto_approved", False),
                )

            return {"alert": alert}
        except Exception as exc:
            logger.error("Orchestrator[generate_alert] error: {}", exc)
            return {"error": str(exc)}

    # ── Routing ────────────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_detection(state: PipelineState) -> str:
        return "anomaly" if state.get("is_anomaly") else "no_anomaly"

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_config(self, thread_id: str, machine_id: str, session_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "tags":         [f"machine:{machine_id}", f"session:{session_id}", "defectsense"],
            "metadata":     {"machine_id": machine_id, "session_id": session_id},
        }
