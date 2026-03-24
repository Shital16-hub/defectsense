"""
Root Cause Reasoner Agent — THE STAR FEATURE of DefectSense.

Pipeline per anomaly:
  1. Load Letta core memory (machine profile + recent patterns)
  2. Search A-MEM for similar past reasoning patterns
  3. Search Letta archival for past reports on this machine
  4. Build a ReAct reasoning prompt with ALL context
  5. Call Groq DeepSeek-R1 for multi-step reasoning
  6. Parse structured JSON output → RootCauseReport
  7. Update A-MEM with new observation (agent learns)
  8. Update Letta core memory (machine profile evolves)
  9. Save to Letta archival memory

The ReAct loop (inside the LLM prompt):
  THINK:   Analyse sensor deviations and anomaly score
  ACT:     Recall similar incidents from A-MEM + RAG context
  OBSERVE: What do past incidents tell us about this pattern?
  THINK:   Synthesise all evidence
  CONCLUDE: Root cause, confidence, severity, recommended actions

Standalone test:
    python -m app.agents.root_cause_reasoner
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from loguru import logger

from app.models.alert import RootCauseReport, Severity
from app.models.anomaly import AnomalyResult
from app.models.maintenance import MaintenanceLog

if TYPE_CHECKING:
    from app.services.amem_service import AMEMService
    from app.services.letta_service import LettaService

# ── Sensor display names ────────────────────────────────────────────────────────
SENSOR_LABELS = {
    "air_temperature":     "air temperature",
    "process_temperature": "process temperature",
    "rotational_speed":    "rotational speed",
    "torque":              "torque",
    "tool_wear":           "tool wear",
}

# Failure type → plain description for the prompt
FAILURE_DESCRIPTIONS = {
    "TWF": "Tool Wear Failure — cutting tool degraded beyond tolerance",
    "HDF": "Heat Dissipation Failure — cooling system cannot remove heat",
    "PWF": "Power Failure — electrical or motor power anomaly",
    "OSF": "Overstrain Failure — mechanical overload on drive train",
    "RNF": "Random/Unknown Failure — no dominant single cause",
}

AUTO_APPROVE_THRESHOLD = float(os.getenv("AUTO_APPROVE_CONFIDENCE_THRESHOLD", "0.95"))


class RootCauseReasonerAgent:
    """
    Stateful reasoning agent. Inject services; call `await analyze(...)`.
    """

    def __init__(
        self,
        amem: Optional["AMEMService"]  = None,
        letta: Optional["LettaService"] = None,
        groq_api_key: Optional[str]    = None,
        reasoning_model: str           = "llama-3.3-70b-versatile",
        fast_model: str                = "llama-3.1-8b-instant",
    ) -> None:
        self._amem    = amem
        self._letta   = letta
        self._api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._reasoning_model = reasoning_model
        self._fast_model      = fast_model
        self._llm             = None

    def _get_llm(self, model: str):
        """Lazy-init LangChain Groq LLM."""
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            api_key=self._api_key,
            temperature=0.1,
            max_tokens=2048,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    async def analyze(
        self,
        anomaly: AnomalyResult,
        similar_incidents: list[MaintenanceLog],
        sensor_context: str,
        session_id: Optional[str] = None,
    ) -> RootCauseReport:
        """
        Run the full ReAct reasoning pipeline for one anomaly.

        Args:
            anomaly:           AnomalyResult from the anomaly detector.
            similar_incidents: Top-3 similar past incidents from RAG (ContextRetriever).
            sensor_context:    Sensor trend summary string from ContextRetriever.
            session_id:        Optional LangGraph session ID.

        Returns:
            RootCauseReport with root_cause, confidence, evidence, actions,
            reasoning_steps, and agent_memory_used.
        """
        logger.info(
            "RootCauseReasoner: analyzing machine={} type={} score={:.3f}",
            anomaly.machine_id,
            anomaly.failure_type_prediction,
            anomaly.anomaly_score,
        )

        # ── Step 1: Load Letta core memory ────────────────────────────────────
        letta_context = await self._get_letta_context(anomaly.machine_id)

        # ── Step 2: Search A-MEM for similar reasoning patterns ───────────────
        amem_memories, amem_notes_raw = await self._search_amem(anomaly)

        # ── Step 3: Archival search ───────────────────────────────────────────
        archival = await self._search_archival(anomaly)

        # ── Step 4: Build ReAct prompt ────────────────────────────────────────
        prompt = self._build_prompt(
            anomaly, similar_incidents, sensor_context,
            letta_context, amem_memories, archival,
        )

        # ── Step 5: Call LLM ──────────────────────────────────────────────────
        raw_output = await self._call_llm(prompt, anomaly)

        # ── Step 6: Parse output → RootCauseReport ────────────────────────────
        report = self._parse_output(
            raw_output, anomaly, similar_incidents, amem_memories, session_id
        )

        # ── Step 7: Update A-MEM (agent learns) ──────────────────────────────
        new_note_id = await self._update_amem(anomaly, report)
        if new_note_id:
            report.agent_memory_used.append(f"[NEW NOTE] {new_note_id[:8]}...")

        # ── Step 8: Update Letta core memory ─────────────────────────────────
        await self._update_letta(anomaly, report)

        logger.info(
            "RootCauseReasoner: done — cause='{}' confidence={:.2f} severity={}",
            report.root_cause[:60],
            report.confidence,
            report.severity,
        )
        return report

    # ── Context assembly ───────────────────────────────────────────────────────

    async def _get_letta_context(self, machine_id: str) -> str:
        if self._letta is None or not self._letta.is_ready:
            return f"No stateful memory available for machine {machine_id}."
        try:
            return await self._letta.get_core_memory(machine_id)
        except Exception as exc:
            logger.warning("RootCauseReasoner: Letta context failed — {}", exc)
            return f"Memory unavailable for machine {machine_id}."

    async def _search_amem(
        self, anomaly: AnomalyResult
    ) -> tuple[list[str], list]:
        """Search A-MEM and return (formatted strings, raw note objects)."""
        if self._amem is None or not self._amem.is_ready:
            return [], []
        try:
            query = (
                f"{anomaly.failure_type_prediction or 'anomaly'} failure "
                f"machine {anomaly.machine_id} "
                f"score {anomaly.anomaly_score:.2f}"
            )
            results = await self._amem.search_memory(query, limit=4)
            if not results:
                return [], []
            texts = [
                f"[Memory {i+1} | similarity={s:.2f}] {note.content[:300]}"
                for i, (note, s) in enumerate(results)
            ]
            return texts, results
        except Exception as exc:
            logger.warning("RootCauseReasoner: A-MEM search failed — {}", exc)
            return [], []

    async def _search_archival(self, anomaly: AnomalyResult) -> list[str]:
        if self._letta is None or not self._letta.is_ready:
            return []
        try:
            query = f"{anomaly.failure_type_prediction} {anomaly.machine_id}"
            return await self._letta.search_archival(anomaly.machine_id, query, limit=2)
        except Exception as exc:
            logger.warning("RootCauseReasoner: archival search failed — {}", exc)
            return []

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        anomaly: AnomalyResult,
        incidents: list[MaintenanceLog],
        sensor_context: str,
        letta_context: str,
        amem_memories: list[str],
        archival: list[str],
    ) -> str:
        # Format sensor deviations
        delta_lines = []
        for sensor, z in sorted(
            anomaly.sensor_deltas.items(), key=lambda x: -abs(x[1])
        ):
            label = SENSOR_LABELS.get(sensor, sensor)
            flag  = " <<CRITICAL>>" if abs(z) >= 3.0 else (" <<HIGH>>" if abs(z) >= 2.0 else "")
            delta_lines.append(f"  - {label}: {z:+.2f} std{flag}")

        # Format RAG incidents
        incident_lines = []
        for i, log in enumerate(incidents, 1):
            incident_lines.append(
                f"  [{i}] Machine {log.machine_id} | {log.failure_type} | "
                f"Date: {str(log.date)[:10]}\n"
                f"      Symptoms   : {log.symptoms}\n"
                f"      Root cause : {log.root_cause}\n"
                f"      Action     : {log.action_taken}"
            )

        # Format A-MEM memories
        amem_section = (
            "\n".join(amem_memories)
            if amem_memories
            else "  (no relevant A-MEM notes found)"
        )

        # Format archival
        archival_section = (
            "\n".join(f"  - {a}" for a in archival)
            if archival
            else "  (no archival records for this machine)"
        )

        # Failure type hint
        ftype = anomaly.failure_type_prediction or "UNKNOWN"
        ftype_desc = FAILURE_DESCRIPTIONS.get(ftype, f"{ftype} failure")

        prompt = f"""You are DefectSense, an expert manufacturing AI agent specialising in root cause analysis.

You MUST respond with ONLY valid JSON — no markdown, no commentary, no <think> tags outside the JSON.

=== CURRENT ANOMALY ===
Machine ID       : {anomaly.machine_id}
Timestamp        : {anomaly.timestamp}
Anomaly Score    : {anomaly.anomaly_score:.3f} (threshold: 0.5)
Failure Probability: {anomaly.failure_probability:.3f}
ML Predicted Type: {ftype} — {ftype_desc}
ML Model Used    : {anomaly.ml_model_used}

Sensor Deviations (z-scores vs recent baseline):
{chr(10).join(delta_lines) if delta_lines else "  (no sensor delta data)"}

=== SENSOR TRENDS ===
{sensor_context}

=== SIMILAR PAST INCIDENTS (from maintenance history) ===
{chr(10).join(incident_lines) if incident_lines else "  (no similar incidents found)"}

=== STATEFUL MACHINE MEMORY (Letta) ===
{letta_context}

=== AGENT MEMORY NOTES (A-MEM) ===
{amem_section}

=== ARCHIVAL MEMORY (past reports for this machine) ===
{archival_section}

=== YOUR TASK ===
Perform a ReAct (Reason + Act) analysis to determine the root cause.

Respond with ONLY this JSON structure (no other text):

{{
  "reasoning_steps": [
    "THINK: [your initial analysis of the sensor data and anomaly score]",
    "ACT: [what evidence you are consulting — past incidents, A-MEM notes, sensor trends]",
    "OBSERVE: [what patterns you see across the evidence]",
    "THINK: [your synthesis — what do all signals point to?]",
    "CONCLUDE: [your final determination with confidence justification]"
  ],
  "root_cause": "[one clear sentence: what is failing and why]",
  "confidence": [float between 0.0 and 1.0],
  "severity": "[CRITICAL | HIGH | MEDIUM | LOW]",
  "evidence": [
    "[evidence point 1 — specific data supporting your conclusion]",
    "[evidence point 2]",
    "[evidence point 3]"
  ],
  "recommended_actions": [
    "[action 1 — most urgent first]",
    "[action 2]",
    "[action 3]"
  ],
  "new_memory_note": "[1-2 sentence summary of this incident for A-MEM storage]",
  "letta_profile_update": "[updated machine profile sentence, or null if no change needed]"
}}

Severity guide:
  CRITICAL: immediate shutdown required, safety risk
  HIGH    : stop within 4 hours, significant damage risk
  MEDIUM  : schedule maintenance within 24 hours
  LOW     : monitor, plan maintenance within 1 week"""

        return prompt

    # ── LLM call ───────────────────────────────────────────────────────────────

    async def _call_llm(self, prompt: str, anomaly: AnomalyResult) -> str:
        """Call Groq with the ReAct prompt. Returns raw string response."""
        import asyncio

        loop = asyncio.get_event_loop()

        def _invoke() -> str:
            llm = self._get_llm(self._reasoning_model)
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=(
                    "You are a manufacturing anomaly analysis expert. "
                    "You respond ONLY with valid JSON as instructed. "
                    "No markdown code blocks. No <think> tags. Just the JSON object."
                )),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            return response.content

        try:
            return await loop.run_in_executor(None, _invoke)
        except Exception as exc:
            logger.error("RootCauseReasoner: LLM call failed — {}", exc)
            return self._fallback_response(anomaly)

    # ── Output parser ──────────────────────────────────────────────────────────

    def _parse_output(
        self,
        raw: str,
        anomaly: AnomalyResult,
        incidents: list[MaintenanceLog],
        amem_memories: list[str],
        session_id: Optional[str],
    ) -> RootCauseReport:
        """Parse LLM JSON output into a RootCauseReport."""
        data = self._extract_json(raw)

        # Validate severity
        severity_raw = str(data.get("severity", "MEDIUM")).upper()
        severity: Severity = severity_raw if severity_raw in ("CRITICAL", "HIGH", "MEDIUM", "LOW") else "MEDIUM"  # type: ignore

        # Clamp confidence
        confidence = float(data.get("confidence", 0.6))
        confidence = max(0.0, min(1.0, confidence))

        reasoning_steps = data.get("reasoning_steps", [])
        if not isinstance(reasoning_steps, list):
            reasoning_steps = [str(reasoning_steps)]

        evidence = data.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        actions = data.get("recommended_actions", [])
        if not isinstance(actions, list):
            actions = [str(actions)]

        # Store the raw A-MEM note text for memory update later
        self._last_memory_note   = data.get("new_memory_note", "")
        self._last_profile_update = data.get("letta_profile_update")

        return RootCauseReport(
            session_id=session_id or "",
            machine_id=anomaly.machine_id,
            anomaly_result=anomaly,
            similar_incidents=incidents,
            root_cause=data.get("root_cause", "Root cause analysis incomplete"),
            confidence=confidence,
            evidence=evidence[:5],
            recommended_actions=actions[:5],
            severity=severity,
            agent_memory_used=[m[:120] for m in amem_memories[:3]],
            reasoning_steps=reasoning_steps,
        )

    def _extract_json(self, raw: str) -> dict:
        """Robustly extract the first JSON object from LLM output."""
        # Strip common LLM artefacts
        text = raw.strip()

        # Remove <think>...</think> blocks (DeepSeek R1 sometimes emits these)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip markdown code fences
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.replace("```", "").strip()

        # Find first { ... } block
        start = text.find("{")
        if start == -1:
            logger.warning("RootCauseReasoner: no JSON found in LLM output")
            return {}

        # Walk to find matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

        # Last resort: try full text
        try:
            return json.loads(text)
        except Exception:
            logger.warning(
                "RootCauseReasoner: JSON parse failed. Raw (first 300 chars): {}",
                text[:300],
            )
            return {}

    # ── Memory updates ─────────────────────────────────────────────────────────

    async def _update_amem(
        self, anomaly: AnomalyResult, report: RootCauseReport
    ) -> Optional[str]:
        """Add a new A-MEM note capturing this reasoning outcome."""
        if self._amem is None or not self._amem.is_ready:
            return None
        try:
            content = (
                self._last_memory_note
                or (
                    f"Machine {anomaly.machine_id}: {anomaly.failure_type_prediction} "
                    f"failure. Root cause: {report.root_cause}. "
                    f"Confidence: {report.confidence:.2f}. Severity: {report.severity}. "
                    f"Action: {report.recommended_actions[0] if report.recommended_actions else 'N/A'}"
                )
            )
            keywords = [
                anomaly.machine_id,
                anomaly.failure_type_prediction or "anomaly",
                report.severity,
                *[s for s in anomaly.sensor_deltas if abs(anomaly.sensor_deltas[s]) >= 2.0],
            ]
            return await self._amem.add_memory(content, keywords)
        except Exception as exc:
            logger.warning("RootCauseReasoner: A-MEM update failed — {}", exc)
            return None

    async def _update_letta(
        self, anomaly: AnomalyResult, report: RootCauseReport
    ) -> None:
        """Update Letta core memory and archival after reasoning."""
        if self._letta is None or not self._letta.is_ready:
            return
        try:
            # Update recent_patterns ring
            pattern = (
                f"{anomaly.timestamp.strftime('%Y-%m-%d')}: "
                f"{anomaly.failure_type_prediction or 'anomaly'} detected "
                f"(score={anomaly.anomaly_score:.2f}, "
                f"severity={report.severity}): {report.root_cause[:80]}"
            )
            await self._letta.add_recent_pattern(anomaly.machine_id, pattern)

            # Update machine_profile if LLM suggested a change
            if self._last_profile_update and self._last_profile_update not in (None, "null", ""):
                await self._letta.update_machine_profile(
                    anomaly.machine_id, self._last_profile_update
                )

            # Add to archival
            archival_entry = (
                f"[{anomaly.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                f"{anomaly.failure_type_prediction or 'ANOMALY'} | "
                f"Confidence: {report.confidence:.2f} | Severity: {report.severity} | "
                f"Root cause: {report.root_cause} | "
                f"Actions: {'; '.join(report.recommended_actions[:2])}"
            )
            await self._letta.add_to_archival(
                anomaly.machine_id,
                archival_entry,
                metadata={
                    "anomaly_score":    anomaly.anomaly_score,
                    "failure_type":     anomaly.failure_type_prediction,
                    "confidence":       report.confidence,
                    "severity":         report.severity,
                },
            )
        except Exception as exc:
            logger.warning("RootCauseReasoner: Letta update failed — {}", exc)

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _fallback_response(self, anomaly: AnomalyResult) -> str:
        """JSON fallback when LLM call fails."""
        ftype = anomaly.failure_type_prediction or "UNKNOWN"
        return json.dumps({
            "reasoning_steps": [
                f"THINK: LLM call failed. Using rule-based fallback.",
                f"OBSERVE: ML model predicted {ftype} with score {anomaly.anomaly_score:.2f}.",
                f"CONCLUDE: Likely {ftype} — manual inspection recommended.",
            ],
            "root_cause": f"Possible {FAILURE_DESCRIPTIONS.get(ftype, ftype)} — LLM reasoning unavailable",
            "confidence": 0.45,
            "severity": "MEDIUM",
            "evidence": [f"ML anomaly score: {anomaly.anomaly_score:.3f}", f"Predicted type: {ftype}"],
            "recommended_actions": ["Inspect machine immediately", "Contact maintenance team"],
            "new_memory_note": f"Machine {anomaly.machine_id}: {ftype} detected (fallback, LLM unavailable)",
            "letta_profile_update": None,
        })


# ── Standalone test ────────────────────────────────────────────────────────────

async def test_root_cause_agent() -> None:
    """End-to-end test without the full FastAPI stack."""
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")

    from datetime import timezone
    from app.models.anomaly import AnomalyResult
    from app.models.maintenance import MaintenanceLog

    print("=" * 60)
    print("  RootCauseReasonerAgent — Standalone Test")
    print("=" * 60)

    # ── Mock data ──────────────────────────────────────────────────────────────
    mock_anomaly = AnomalyResult(
        machine_id="M0042",
        timestamp=datetime.now(tz=timezone.utc),
        anomaly_score=0.87,
        failure_probability=0.74,
        is_anomaly=True,
        failure_type_prediction="HDF",
        sensor_deltas={
            "air_temperature":     2.1,
            "process_temperature": 3.8,
            "rotational_speed":   -0.4,
            "torque":              1.2,
            "tool_wear":           0.9,
        },
        ml_model_used="ensemble",
    )

    mock_incidents = [
        MaintenanceLog(
            machine_id="M026",
            date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            failure_type="HDF",
            symptoms="Thermal alarm triggered, machine auto-shutdown at 320K",
            root_cause="Cooling system thermostat failed closed",
            action_taken="Replaced cooling fan assembly, cleaned heat exchange fins",
            resolution_time_hours=4.5,
            technician="J. Smith",
        ),
        MaintenanceLog(
            machine_id="M017",
            date=datetime(2024, 3, 8, tzinfo=timezone.utc),
            failure_type="HDF",
            symptoms="Process temperature rising 2 degrees per hour",
            root_cause="Cooling fan blade fractured, reducing airflow by 60%",
            action_taken="Emergency fan replacement, 2h shutdown",
            resolution_time_hours=2.0,
            technician="T. Williams",
        ),
    ]

    sensor_context = (
        "Sensor trends over last 10 readings for M0042:\n"
        "  - Process temperature rose 12.4% (mean: 311.2K, latest: 314.8K)\n"
        "  - Air temperature rose 4.1% (mean: 299.1K, latest: 301.3K)\n"
        "  - Rotational speed stable (mean: 1534 RPM)\n"
        "  - Torque rose 8.2% (mean: 45.1 Nm, latest: 48.8 Nm)"
    )

    # ── Try to connect MongoDB for memory (optional) ───────────────────────────
    amem_svc  = None
    letta_svc = None

    mongo_url = os.getenv("MONGODB_URL", "")
    if mongo_url:
        try:
            import motor.motor_asyncio as motor
            from app.services.amem_service import AMEMService
            from app.services.letta_service import LettaService

            client   = motor.AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=3000)
            db       = client[os.getenv("MONGODB_DB_NAME", "defectsense")]
            await client.admin.command("ping")

            amem_svc = AMEMService(db=db)
            await amem_svc.init()
            print(f"A-MEM: ready ({await amem_svc.memory_count()} notes in memory)")

            letta_svc = LettaService(db=db)
            await letta_svc.init()
            print("Letta: ready")
        except Exception as exc:
            print(f"MongoDB unavailable ({exc}) — running without memory services")

    # ── Build and run agent ────────────────────────────────────────────────────
    agent = RootCauseReasonerAgent(amem=amem_svc, letta=letta_svc)

    print(f"\nAnalyzing: machine={mock_anomaly.machine_id} type={mock_anomaly.failure_type_prediction}")
    print(f"Model: {agent._reasoning_model}\n")

    report = await agent.analyze(
        anomaly=mock_anomaly,
        similar_incidents=mock_incidents,
        sensor_context=sensor_context,
    )

    # ── Print results ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  ROOT CAUSE REPORT")
    print("=" * 60)
    print(f"\nRoot Cause    : {report.root_cause}")
    print(f"Confidence    : {report.confidence:.2f}")
    print(f"Severity      : {report.severity}")
    print(f"\nEvidence:")
    for e in report.evidence:
        print(f"  - {e}")
    print(f"\nRecommended Actions:")
    for a in report.recommended_actions:
        print(f"  -> {a}")
    print(f"\nReasoning Steps:")
    for s in report.reasoning_steps:
        print(f"  {s}")
    print(f"\nAgent Memory Used:")
    for m in report.agent_memory_used:
        print(f"  * {m[:100]}")
    print("\n" + "=" * 60)
    print("  Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_root_cause_agent())
