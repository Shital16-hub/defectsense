"""
DefectSense LangGraph Orchestrator — Session 5 / Session 8.

Pipeline:
  detect_anomaly
       |
   (anomaly?)
    /       \
  NO         YES
  |           |
  END    retrieve_context         (RAG + amem update)
             |
       reason_root_cause          (ReAct LLM)
             |
       generate_alert             (creates alert, saves to MongoDB as approved=None)
             |
    [HITL interrupt_before="apply_approval"]
             |
        apply_approval            (marks approved/rejected; updates MongoDB)
             |
    post_resolution_indexer       (if approved: auto-index log into RAG)
             |
            END

This means alerts are visible in MongoDB immediately after generation.
Humans approve/reject via POST /api/alerts/{id}/approve|reject.
Auto-approve fires if confidence >= AUTO_APPROVE_CONFIDENCE_THRESHOLD.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Optional, TypedDict

from loguru import logger

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models.anomaly import AnomalyResult
from app.models.sensor import SensorReading
from app.models.alert import MaintenanceAlert, RootCauseReport


# ── Pipeline state ─────────────────────────────────────────────────────────────

class PipelineState(TypedDict, total=False):
    reading:           SensorReading
    session_id:        str
    machine_id:        str

    anomaly_result:    Optional[AnomalyResult]
    is_anomaly:        bool

    similar_incidents: list
    sensor_context:    str

    root_cause_report: Optional[RootCauseReport]

    # Alert (created before HITL, saved as approved=None)
    alert:             Optional[MaintenanceAlert]

    # HITL decision (injected by resume())
    approved:          Optional[bool]
    approved_by:       str
    rejection_reason:  Optional[str]
    auto_approved:     bool

    # Post-resolution auto-indexing
    auto_indexed:      Optional[bool]

    error:             Optional[str]


# ── Orchestrator ───────────────────────────────────────────────────────────────

class DefectSenseOrchestrator:

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
        app_base_url: str = "http://localhost:8080",
    ) -> None:
        self._detector          = detector
        self._context_retriever = context_retriever
        self._amem              = amem
        self._reasoner          = reasoner
        self._alert_gen         = alert_generator
        self._auto_threshold    = auto_approve_threshold
        self._timeout_minutes   = approval_timeout_minutes
        self._app_base_url      = app_base_url.rstrip("/")
        self._graph: Any        = None
        self._checkpointer      = MemorySaver()

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self) -> None:
        builder = StateGraph(PipelineState)

        builder.add_node("detect_anomaly",          self._node_detect_anomaly)
        builder.add_node("retrieve_context",        self._node_retrieve_context)
        builder.add_node("reason_root_cause",       self._node_reason_root_cause)
        builder.add_node("generate_alert",          self._node_generate_alert)
        builder.add_node("apply_approval",          self._node_apply_approval)
        builder.add_node("post_resolution_indexer", self._node_post_resolution_indexer)

        builder.set_entry_point("detect_anomaly")

        builder.add_conditional_edges(
            "detect_anomaly",
            lambda s: "anomaly" if s.get("is_anomaly") else "no_anomaly",
            {"anomaly": "retrieve_context", "no_anomaly": END},
        )
        builder.add_edge("retrieve_context",        "reason_root_cause")
        builder.add_edge("reason_root_cause",       "generate_alert")
        builder.add_edge("generate_alert",          "apply_approval")
        builder.add_edge("apply_approval",          "post_resolution_indexer")
        builder.add_edge("post_resolution_indexer", END)

        self._graph = builder.compile(
            checkpointer=self._checkpointer,
            interrupt_before=["apply_approval"],   # ← alert exists, now await human
        )
        logger.info("Orchestrator: compiled (interrupt_before=apply_approval)")

    # ── Public API ─────────────────────────────────────────────────────────────

    async def run(
        self,
        reading: SensorReading,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Run pipeline for one reading.

        Alert is saved to MongoDB with approved=None before the interrupt.
        Returns state dict including thread_id.
        If confidence >= auto_threshold, immediately auto-approves.
        """
        if self._graph is None:
            self.build()

        sid       = session_id or str(uuid.uuid4())
        thread_id = f"{reading.machine_id}:{sid}"
        config    = self._make_config(thread_id, reading.machine_id, sid)

        logger.info("Orchestrator: run | machine={} thread={}", reading.machine_id, thread_id)

        final_state = await self._graph.ainvoke(
            {
                "reading":    reading,
                "session_id": sid,
                "machine_id": reading.machine_id,
            },
            config=config,
        )

        # Graph paused before apply_approval — decide auto-approve or wait
        report: Optional[RootCauseReport] = final_state.get("root_cause_report")
        if report is not None and final_state.get("approved") is None:
            if report.confidence >= self._auto_threshold:
                logger.info(
                    "Orchestrator: auto-approving conf={:.2f} >= {:.2f} | thread={}",
                    report.confidence, self._auto_threshold, thread_id,
                )
                return await self.resume(thread_id, approved=True, approved_by="auto", auto=True)
            else:
                logger.info(
                    "Orchestrator: HITL pending | conf={:.2f} sev={} thread={}",
                    report.confidence, report.severity, thread_id,
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
        """Resume a paused pipeline with a human decision."""
        if self._graph is None:
            raise RuntimeError("Orchestrator not built")

        parts      = thread_id.split(":", 1)
        machine_id = parts[0]
        sid        = parts[1] if len(parts) > 1 else thread_id
        config     = self._make_config(thread_id, machine_id, sid)

        update: dict = {
            "approved":         approved,
            "approved_by":      approved_by,
            "rejection_reason": rejection_reason,
            "auto_approved":    auto,
        }
        await self._graph.aupdate_state(config, update, as_node="apply_approval")

        logger.info(
            "Orchestrator: resume approved={} by={} auto={} | thread={}",
            approved, approved_by, auto, thread_id,
        )

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

        if self._context_retriever is not None:
            try:
                incidents, sensor_ctx = await self._context_retriever.retrieve(anomaly)
            except Exception as exc:
                logger.warning("Orchestrator[retrieve_context] error: {}", exc)

        if self._amem is not None:
            try:
                await self._amem.add_memory(
                    content=(
                        f"Anomaly on {state['machine_id']}: "
                        f"type={anomaly.failure_type_prediction}, "
                        f"prob={anomaly.failure_probability:.3f}"
                    ),
                    tags=["anomaly", anomaly.failure_type_prediction or "UNKNOWN", state["machine_id"]],
                    source="orchestrator",
                )
            except Exception as exc:
                logger.warning("Orchestrator[amem] error: {}", exc)

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
                "Orchestrator[reason]: conf={:.2f} sev={} machine={}",
                report.confidence, report.severity, state["machine_id"],
            )
            return {"root_cause_report": report}
        except Exception as exc:
            logger.error("Orchestrator[reason] error: {}", exc)
            return {"error": str(exc)}

    async def _node_generate_alert(self, state: PipelineState) -> dict:
        """Create alert and save to MongoDB as pending (approved=None)."""
        report: Optional[RootCauseReport] = state.get("root_cause_report")
        if report is None:
            logger.warning("Orchestrator[generate_alert]: no report — skipping")
            return {}
        try:
            alert: MaintenanceAlert = await self._alert_gen.generate(
                report=report,
                session_id=state.get("session_id"),
            )
            logger.info(
                "Orchestrator[generate_alert]: alert {} saved (pending) | machine={}",
                alert.alert_id[:8], state["machine_id"],
            )
            return {"alert": alert}
        except Exception as exc:
            logger.error("Orchestrator[generate_alert] error: {}", exc)
            return {"error": str(exc)}

    async def _node_apply_approval(self, state: PipelineState) -> dict:
        """Apply human approval/rejection to the already-saved alert."""
        alert: Optional[MaintenanceAlert] = state.get("alert")
        if alert is None:
            return {}

        approved = state.get("approved")
        if approved is True:
            await self._alert_gen.mark_approved(
                alert,
                approved_by=state.get("approved_by", "human"),
                auto=state.get("auto_approved", False),
            )
            logger.info("Orchestrator[apply_approval]: APPROVED | alert={}", alert.alert_id[:8])
        elif approved is False:
            await self._alert_gen.mark_rejected(
                alert,
                rejection_reason=state.get("rejection_reason") or "Rejected",
                rejected_by=state.get("approved_by", "human"),
            )
            logger.info("Orchestrator[apply_approval]: REJECTED | alert={}", alert.alert_id[:8])
        else:
            logger.debug("Orchestrator[apply_approval]: no decision yet — alert stays pending")

        return {}

    async def _node_post_resolution_indexer(self, state: PipelineState) -> dict:
        """
        Auto-index the resolved incident into the RAG knowledge base.

        Only runs when approved=True. Wraps everything in try/except so it
        can never fail the pipeline — a logging failure is not a pipeline failure.
        """
        if state.get("approved") is not True:
            logger.debug(
                "Orchestrator[post_resolution_indexer]: approved={} — skipping",
                state.get("approved"),
            )
            return {"auto_indexed": False}

        try:
            from datetime import datetime, timezone
            import httpx

            from app.models.maintenance import MaintenanceLog

            report: Optional[RootCauseReport] = state.get("root_cause_report")
            anomaly: Optional[Any]            = state.get("anomaly_result")

            # machine_id: prefer state key, fall back to anomaly_result attribute
            machine_id = (
                state.get("machine_id")
                or getattr(anomaly, "machine_id", None)
                or "UNKNOWN"
            )
            logger.info(
                "post_resolution_indexer: state.keys={} machine_id={} anomaly.machine_id={}",
                sorted(state.keys()),
                state.get("machine_id"),
                getattr(anomaly, "machine_id", None),
            )

            # Build symptoms string from sensor_deltas
            deltas = getattr(anomaly, "sensor_deltas", {}) or {}
            symptom_parts = []
            label_map = {
                "air_temperature":     "Air temperature",
                "process_temperature": "Process temperature",
                "rotational_speed":    "Rotational speed",
                "torque":              "Torque",
                "tool_wear":           "Tool wear",
            }
            for key, val in deltas.items():
                if val is None:
                    continue
                human = label_map.get(key, key.replace("_", " ").title())
                direction = "above" if val >= 0 else "below"
                symptom_parts.append(f"{human} {abs(val):.1f} std {direction} normal")
            symptoms = "; ".join(symptom_parts) if symptom_parts else (
                f"Anomaly detected on {machine_id}"
            )

            # action_taken: first 2 recommended actions
            actions = (report.recommended_actions if report else []) or []
            action_taken = "; ".join(actions[:2]) if actions else "Maintenance action required"

            # notes with confidence and severity
            confidence = report.confidence if report else 0.0
            severity   = report.severity   if report else "UNKNOWN"
            notes = (
                f"Auto-indexed from DefectSense alert. "
                f"Confidence: {confidence:.0%}. Severity: {severity}."
            )

            log = MaintenanceLog(
                machine_id=machine_id,
                date=datetime.now(tz=timezone.utc),
                failure_type=(
                    getattr(anomaly, "failure_type_prediction", None) or "UNKNOWN"
                ),
                symptoms=symptoms,
                root_cause=report.root_cause if report else "Unknown",
                action_taken=action_taken,
                resolution_time_hours=0.0,
                technician=state.get("approved_by") or "auto",
                notes=notes,
            )

            logger.info(
                "post_resolution_indexer: creating log machine_id={} failure_type={} notes={}",
                log.machine_id,
                log.failure_type,
                log.notes[:80],
            )

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self._app_base_url}/api/maintenance-logs/add",
                    json=log.model_dump(mode="json"),
                )
                resp.raise_for_status()

            logger.info(
                "Orchestrator[post_resolution_indexer]: log {} indexed | machine={} type={}",
                log.log_id[:8], log.machine_id, log.failure_type,
            )
            return {"auto_indexed": True}

        except Exception as exc:
            logger.warning(
                "Orchestrator[post_resolution_indexer]: failed (non-fatal) — {}", exc
            )
            return {"auto_indexed": False}

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _make_config(self, thread_id: str, machine_id: str, session_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "tags":         [f"machine:{machine_id}", f"session:{session_id}", "defectsense"],
            "metadata":     {"machine_id": machine_id, "session_id": session_id},
        }
