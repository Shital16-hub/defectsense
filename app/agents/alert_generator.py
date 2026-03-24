"""
Alert Generator Agent — final stage of the DefectSense pipeline.

Takes a RootCauseReport and produces a MaintenanceAlert with:
  - Plain-language explanation (no jargon — for factory floor workers)
  - Severity score (rule-based + LLM-generated text)
  - Ranked recommended actions (max 3, urgency-ordered)
  - Saves to MongoDB + publishes to Redis 'alerts:new' channel

Plain language rules:
  - NO acronyms (TWF → "tool wear", HDF → "overheating", etc.)
  - Short sentences, imperative tone
  - Worker-facing: "Call the maintenance team NOW"
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from app.models.alert import MaintenanceAlert, RootCauseReport, Severity

CHANNEL_ALERTS = "alerts:new"

# Maps failure type → plain language phrasing
_FAILURE_PLAIN = {
    "TWF": "the cutting tool is wearing out faster than expected",
    "HDF": "the machine is overheating — the cooling system may be failing",
    "PWF": "there is an electrical power problem with the motor",
    "OSF": "the machine is being overloaded — mechanical strain detected",
    "RNF": "an unusual fault has been detected with no single clear cause",
}

# Severity override rules: (condition_fn) → Severity
_SEVERITY_RULES = [
    (lambda r: r.anomaly_result.failure_probability > 0.9,              "CRITICAL"),
    (lambda r: r.anomaly_result.failure_type_prediction in ("TWF","HDF")
               and r.anomaly_result.failure_probability > 0.7,          "CRITICAL"),
    (lambda r: r.anomaly_result.failure_probability > 0.7,              "HIGH"),
    (lambda r: r.anomaly_result.failure_probability > 0.5,              "MEDIUM"),
]


class AlertGeneratorAgent:
    """Stateless — inject services; call `await generate(report)`."""

    def __init__(
        self,
        mongo_db=None,
        redis_service=None,
        groq_api_key: Optional[str] = None,
        fast_model: str = "llama-3.1-8b-instant",
    ) -> None:
        self._mongo   = mongo_db
        self._redis   = redis_service
        self._api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._model   = fast_model

    # ── Public API ─────────────────────────────────────────────────────────────

    async def generate(
        self,
        report: RootCauseReport,
        session_id: Optional[str] = None,
    ) -> MaintenanceAlert:
        """
        Generate a MaintenanceAlert from a RootCauseReport.

        Returns:
            MaintenanceAlert with plain_language_explanation, severity,
            and approved=None (pending) unless auto-approved.
        """
        # Compute severity (rule-based takes precedence over LLM severity)
        severity = self._compute_severity(report)

        # Generate plain-language explanation
        explanation = await self._generate_explanation(report, severity)

        alert = MaintenanceAlert(
            alert_id=str(uuid.uuid4()),
            session_id=session_id or report.session_id,
            machine_id=report.machine_id,
            root_cause_report=report,
            plain_language_explanation=explanation,
            approved=None,  # pending human approval
            created_at=datetime.now(tz=timezone.utc),
            auto_approved=False,
        )

        logger.info(
            "AlertGenerator: created alert {} | machine={} severity={}",
            alert.alert_id[:8],
            alert.machine_id,
            severity,
        )

        # Persist + broadcast
        await self._save_to_mongo(alert)
        await self._publish_to_redis(alert)

        return alert

    async def mark_approved(
        self,
        alert: MaintenanceAlert,
        approved_by: str = "human",
        auto: bool = False,
    ) -> MaintenanceAlert:
        """Mark an alert as approved and update MongoDB."""
        alert.approved     = True
        alert.approved_by  = approved_by
        alert.approved_at  = datetime.now(tz=timezone.utc)
        alert.auto_approved = auto
        await self._update_mongo(alert)
        logger.info(
            "AlertGenerator: alert {} APPROVED by {} (auto={})",
            alert.alert_id[:8], approved_by, auto,
        )
        return alert

    async def mark_rejected(
        self,
        alert: MaintenanceAlert,
        rejection_reason: str,
        rejected_by: str = "human",
    ) -> MaintenanceAlert:
        """Mark an alert as rejected and update MongoDB."""
        alert.approved          = False
        alert.approved_by       = rejected_by
        alert.rejection_reason  = rejection_reason
        alert.approved_at       = datetime.now(tz=timezone.utc)
        await self._update_mongo(alert)
        logger.info(
            "AlertGenerator: alert {} REJECTED — {}",
            alert.alert_id[:8], rejection_reason[:60],
        )
        return alert

    # ── Severity ───────────────────────────────────────────────────────────────

    def _compute_severity(self, report: RootCauseReport) -> Severity:
        """Rule-based severity override (ignores LLM severity for safety)."""
        for condition, sev in _SEVERITY_RULES:
            try:
                if condition(report):
                    return sev  # type: ignore[return-value]
            except Exception:
                continue
        # Fall back to LLM-assigned severity if no rule fires
        return report.severity

    # ── Plain-language generation ──────────────────────────────────────────────

    async def _generate_explanation(
        self, report: RootCauseReport, severity: Severity
    ) -> str:
        """Call Groq (fast model) to generate a worker-friendly explanation."""
        import asyncio

        ftype   = report.anomaly_result.failure_type_prediction or "UNKNOWN"
        plain   = _FAILURE_PLAIN.get(ftype, "an equipment fault has been detected")
        actions = report.recommended_actions[:3]

        prompt = f"""Write a short alert message for a factory floor worker (not a technician).
Use simple, clear language. No technical jargon. Maximum 3 sentences.

Situation: On machine {report.machine_id}, {plain}.
Root cause: {report.root_cause}
Severity: {severity}
Top action: {actions[0] if actions else 'Contact maintenance team immediately'}

Rules:
- Start with the machine ID
- Say what is wrong in plain English (no acronyms like TWF, HDF, PWF, OSF)
- Say what the worker should do RIGHT NOW
- Tone: calm but urgent for CRITICAL/HIGH, informational for MEDIUM/LOW

Respond with ONLY the 2-3 sentence alert message. No quotes, no formatting."""

        try:
            loop = asyncio.get_event_loop()

            def _call() -> str:
                from langchain_groq import ChatGroq
                from langchain_core.messages import HumanMessage
                llm = ChatGroq(model=self._model, api_key=self._api_key, temperature=0.3)
                return llm.invoke([HumanMessage(content=prompt)]).content.strip()

            return await loop.run_in_executor(None, _call)

        except Exception as exc:
            logger.warning("AlertGenerator: LLM explanation failed — {}", exc)
            return self._fallback_explanation(report.machine_id, ftype, severity, actions)

    @staticmethod
    def _fallback_explanation(
        machine_id: str, ftype: str, severity: Severity, actions: list[str]
    ) -> str:
        plain = _FAILURE_PLAIN.get(ftype, "an equipment fault has been detected")
        action = actions[0] if actions else "Contact the maintenance team immediately"
        return (
            f"Machine {machine_id} alert: {plain}. "
            f"Severity: {severity}. "
            f"{action}."
        )

    # ── Persistence ────────────────────────────────────────────────────────────

    async def _save_to_mongo(self, alert: MaintenanceAlert) -> None:
        if self._mongo is None:
            return
        try:
            doc = alert.model_dump(mode="json")
            await self._mongo["alerts"].insert_one(doc)
        except Exception as exc:
            logger.warning("AlertGenerator: MongoDB save failed — {}", exc)

    async def _update_mongo(self, alert: MaintenanceAlert) -> None:
        if self._mongo is None:
            return
        try:
            doc = alert.model_dump(mode="json")
            await self._mongo["alerts"].replace_one(
                {"alert_id": alert.alert_id}, doc, upsert=True
            )
        except Exception as exc:
            logger.warning("AlertGenerator: MongoDB update failed — {}", exc)

    async def _publish_to_redis(self, alert: MaintenanceAlert) -> None:
        if self._redis is None:
            return
        try:
            payload = alert.model_dump_json()
            await self._redis._client.publish(CHANNEL_ALERTS, payload)
        except Exception as exc:
            logger.warning("AlertGenerator: Redis publish failed — {}", exc)
