"""
Evaluation Service — LLM-as-judge evaluation for RAG pipeline and root-cause reports.

Two evaluators:
  RAGEvaluationService     — context_precision, faithfulness, answer_relevancy
  LLMJudgeEvaluationService — root_cause_correctness, severity_accuracy,
                               action_quality, reasoning_quality, confidence_calibration

All results stored in MongoDB `evaluation_results` collection.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


COLL_ALERTS  = "alerts"
COLL_EVAL    = "evaluation_results"


# ── RAG Evaluation ─────────────────────────────────────────────────────────────

class RAGEvaluationService:

    def __init__(self, mongo_db, qdrant_service, groq_api_key: str) -> None:
        self._mongo_db    = mongo_db
        self._qdrant      = qdrant_service
        self._groq_key    = groq_api_key

    # ── Public ─────────────────────────────────────────────────────────────────

    async def run_evaluation(self, n_samples: int = 10) -> dict:
        """
        Evaluate the RAG pipeline on the n_samples most recent approved alerts.
        Stores result in MongoDB and returns the result dict.
        """
        eval_id = str(uuid.uuid4())
        run_at  = datetime.now(tz=timezone.utc).isoformat()

        result: dict = {
            "eval_id":      eval_id,
            "eval_type":    "rag",
            "run_at":       run_at,
            "n_samples":    n_samples,
            "scores":       {},
            "sample_scores": [],
            "status":       "failed",
            "error":        None,
        }

        try:
            samples = await self._build_eval_samples(n_samples)

            if not samples:
                result["status"] = "completed"
                result["scores"] = {
                    "context_precision": 0.0,
                    "faithfulness":      0.0,
                    "answer_relevancy":  0.0,
                    "overall":           0.0,
                }
                await self._save(result)
                return result

            cp_scores, faith_scores, ar_scores = [], [], []
            sample_rows = []

            for s in samples:
                cp   = await self._score_context_precision(s["question"], s["contexts"], s["answer"])
                faith = await self._score_faithfulness(s["answer"], s["contexts"])
                ar   = await self._score_answer_relevancy(s["question"], s["answer"])

                cp_scores.append(cp)
                faith_scores.append(faith)
                ar_scores.append(ar)

                sample_rows.append({
                    "alert_id":         s.get("alert_id", ""),
                    "machine_id":       s.get("machine_id", ""),
                    "question":         s["question"],
                    "answer":           s["answer"],
                    "contexts":         s["contexts"],
                    "context_precision": cp,
                    "faithfulness":      faith,
                    "answer_relevancy":  ar,
                })

                # Respect Groq rate limits: max 30 samples/min (3 calls per sample)
                await asyncio.sleep(2.0)

            avg_cp   = sum(cp_scores)    / len(cp_scores)
            avg_faith = sum(faith_scores) / len(faith_scores)
            avg_ar   = sum(ar_scores)    / len(ar_scores)
            overall  = (avg_cp + avg_faith + avg_ar) / 3

            result["scores"] = {
                "context_precision": round(avg_cp,   4),
                "faithfulness":      round(avg_faith, 4),
                "answer_relevancy":  round(avg_ar,   4),
                "overall":           round(overall,  4),
            }
            result["sample_scores"] = sample_rows
            result["status"]        = "completed"

        except Exception as exc:
            result["error"]  = str(exc)
            result["status"] = "failed"
            logger.warning("RAGEvaluationService.run_evaluation failed: {}", exc)

        await self._save(result)
        return result

    # ── Sample builder ─────────────────────────────────────────────────────────

    async def _build_eval_samples(self, n_samples: int) -> list[dict]:
        if self._mongo_db is None:
            return []

        try:
            cursor = (
                self._mongo_db[COLL_ALERTS]
                .find({"approved": True}, {"_id": 0})
                .sort("created_at", -1)
                .limit(n_samples)
            )
            alerts = await cursor.to_list(length=n_samples)
        except Exception as exc:
            logger.warning("RAGEvaluationService._build_eval_samples: fetch failed — {}", exc)
            return []

        samples = []
        for alert in alerts:
            try:
                rcr           = alert.get("root_cause_report") or {}
                machine_id    = alert.get("machine_id", "UNKNOWN")
                anom          = rcr.get("anomaly_result") or {}
                failure_type  = anom.get("failure_type_prediction") or "UNKNOWN"
                sensor_deltas = anom.get("sensor_deltas") or {}

                # Build human-readable question matching the RAG corpus vocabulary
                symptom_parts: list[str] = []
                HIGH_Z = 2.0
                MED_Z  = 1.0

                pt = sensor_deltas.get("process_temperature", 0.0) or 0.0
                at = sensor_deltas.get("air_temperature",     0.0) or 0.0
                rs = sensor_deltas.get("rotational_speed",    0.0) or 0.0
                tq = sensor_deltas.get("torque",              0.0) or 0.0
                tw = sensor_deltas.get("tool_wear",           0.0) or 0.0

                if abs(pt) >= HIGH_Z:
                    symptom_parts.append("process temperature significantly " + ("elevated" if pt > 0 else "below normal"))
                elif abs(pt) >= MED_Z:
                    symptom_parts.append("process temperature " + ("elevated" if pt > 0 else "below normal"))

                if abs(at) >= HIGH_Z:
                    symptom_parts.append("air temperature significantly " + ("elevated" if at > 0 else "below normal"))

                if abs(rs) >= HIGH_Z:
                    symptom_parts.append("rotational speed dropped significantly" if rs < 0 else "rotational speed elevated")
                elif abs(rs) >= MED_Z:
                    symptom_parts.append("rotational speed reduced" if rs < 0 else "rotational speed increased")

                if abs(tq) >= HIGH_Z:
                    symptom_parts.append("torque high — possible mechanical overload" if tq > 0 else "torque low")
                elif abs(tq) >= MED_Z:
                    symptom_parts.append("torque above normal" if tq > 0 else "torque below normal")

                if abs(tw) >= MED_Z:
                    symptom_parts.append("tool wear elevated")

                failure_context = {
                    "HDF": "thermal alarm conditions, possible cooling system issue",
                    "TWF": "tool wear exceeded normal tolerance",
                    "PWF": "power consumption outside normal operating range",
                    "OSF": "mechanical overload, torque spike recorded",
                    "RNF": "unexpected fault, no single dominant cause",
                }
                if failure_type in failure_context:
                    symptom_parts.append(failure_context[failure_type])

                symptoms_text = "; ".join(symptom_parts) if symptom_parts else "multiple sensor deviations detected"

                question = (
                    f"What is the root cause of the anomaly on machine {machine_id} "
                    f"showing {failure_type} failure with symptoms: {symptoms_text}?"
                )
                answer   = rcr.get("root_cause") or "Unknown root cause"

                # Build contexts from similar_incidents embedded in the report
                incidents = rcr.get("similar_incidents") or []
                contexts: list[str] = []
                for inc in incidents:
                    if isinstance(inc, dict):
                        ft  = inc.get("failure_type", "")
                        sym = inc.get("symptoms", "")
                        rc  = inc.get("root_cause", "")
                        act = inc.get("action_taken", "")
                        contexts.append(
                            f"{ft}: {sym}. Root cause: {rc}. Action: {act}."
                        )
                if not contexts:
                    contexts = ["No similar incidents available."]

                samples.append({
                    "alert_id":  alert.get("alert_id", ""),
                    "machine_id": machine_id,
                    "question":  question,
                    "answer":    answer,
                    "contexts":  contexts,
                })
            except Exception as exc:
                logger.debug("RAGEvaluationService: skipping alert — {}", exc)

        return samples

    # ── LLM-as-judge scoring ───────────────────────────────────────────────────

    async def _score_context_precision(
        self, question: str, contexts: list[str], answer: str
    ) -> float:
        context_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "You are evaluating a RAG system. Given the question and the retrieved context "
            "chunks, rate what fraction of the retrieved chunks were actually relevant and "
            "useful for answering the question.\n\n"
            f"Question: {question}\n\nContexts:\n{context_text}\n\nAnswer: {answer}\n\n"
            'Return ONLY a JSON object: {"score": <float between 0.0 and 1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._llm_score(prompt)

    async def _score_faithfulness(self, answer: str, contexts: list[str]) -> float:
        context_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = (
            "You are evaluating a RAG system. Given an answer and the context it was "
            "supposedly based on, rate how faithful the answer is to the provided context "
            "(1.0 = every claim is supported by context, 0.0 = answer makes claims not in "
            "context).\n\n"
            f"Answer: {answer}\n\nContexts:\n{context_text}\n\n"
            'Return ONLY a JSON object: {"score": <float between 0.0 and 1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._llm_score(prompt)

    async def _score_answer_relevancy(self, question: str, answer: str) -> float:
        prompt = (
            "You are evaluating a RAG system. Given a question and an answer, rate how "
            "relevant and directly responsive the answer is to the question "
            "(1.0 = directly answers the question, 0.0 = completely irrelevant).\n\n"
            f"Question: {question}\n\nAnswer: {answer}\n\n"
            'Return ONLY a JSON object: {"score": <float between 0.0 and 1.0>, '
            '"reasoning": "<one sentence>"}'
        )
        return await self._llm_score(prompt)

    async def _llm_score(self, prompt: str) -> float:
        """Run a sync Groq call in an executor and parse the JSON score.
        Retries up to 3 times with backoff on rate-limit errors.
        """
        import time

        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=self._groq_key,
                temperature=0.0,
                max_tokens=128,
            )

            def _call_with_retry():
                for attempt in range(3):
                    try:
                        resp = llm.invoke([HumanMessage(content=prompt)])
                        return resp.content
                    except Exception as exc:
                        if "rate" in str(exc).lower() or "429" in str(exc):
                            wait = (attempt + 1) * 10  # 10s, 20s, 30s
                            logger.warning(
                                "Groq rate limit hit, waiting {}s (attempt {}/3)",
                                wait, attempt + 1,
                            )
                            time.sleep(wait)
                        else:
                            raise
                return None  # all retries exhausted

            loop    = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, _call_with_retry)

            if content is None:
                logger.warning("RAGEvaluationService._llm_score: all retries exhausted")
                return 0.5

            # Extract JSON — may have markdown fences
            raw = content.strip()
            if "```" in raw:
                raw = raw.split("```")[-2].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data  = json.loads(raw)
            score = float(data["score"])
            return max(0.0, min(1.0, score))

        except Exception as exc:
            logger.warning("Evaluation scoring failed (possible rate limit) — {}", exc)
            return 0.5

    # ── Persistence ────────────────────────────────────────────────────────────

    async def _save(self, result: dict) -> None:
        if self._mongo_db is None:
            return
        try:
            await self._mongo_db[COLL_EVAL].insert_one({**result})
        except Exception as exc:
            logger.warning("RAGEvaluationService._save failed: {}", exc)


# ── LLM Judge Evaluation ───────────────────────────────────────────────────────

class LLMJudgeEvaluationService:

    def __init__(self, mongo_db, groq_api_key: str) -> None:
        self._mongo_db = mongo_db
        self._groq_key = groq_api_key

    # ── Public ─────────────────────────────────────────────────────────────────

    async def run_evaluation(self, n_samples: int = 10) -> dict:
        eval_id = str(uuid.uuid4())
        run_at  = datetime.now(tz=timezone.utc).isoformat()

        result: dict = {
            "eval_id":       eval_id,
            "eval_type":     "llm_judge",
            "run_at":        run_at,
            "n_samples":     n_samples,
            "scores":        {},
            "sample_scores": [],
            "status":        "failed",
            "error":         None,
        }

        try:
            if self._mongo_db is None:
                result["status"] = "completed"
                result["scores"] = self._empty_scores()
                await self._save(result)
                return result

            cursor = (
                self._mongo_db[COLL_ALERTS]
                .find({"approved": True}, {"_id": 0})
                .sort("created_at", -1)
                .limit(n_samples)
            )
            alerts = await cursor.to_list(length=n_samples)

            if not alerts:
                result["status"] = "completed"
                result["scores"] = self._empty_scores()
                await self._save(result)
                return result

            dimension_sums: dict[str, float] = {
                "root_cause_correctness": 0.0,
                "severity_accuracy":      0.0,
                "action_quality":         0.0,
                "reasoning_quality":      0.0,
                "confidence_calibration": 0.0,
            }
            sample_rows = []

            for alert in alerts:
                try:
                    scores = await self._score_report(alert)
                    sample_rows.append(scores)
                    for k in dimension_sums:
                        dimension_sums[k] += scores.get(k, 0.0)
                except Exception as exc:
                    logger.debug("LLMJudgeEvaluationService: skipping alert — {}", exc)

                # Respect Groq rate limits: 5 LLM calls per report
                await asyncio.sleep(3.0)

            n = max(len(sample_rows), 1)
            avg: dict[str, float] = {k: round(v / n, 4) for k, v in dimension_sums.items()}
            avg["overall"] = round(sum(avg.values()) / len(avg), 4)

            result["scores"]        = avg
            result["sample_scores"] = sample_rows
            result["status"]        = "completed"

        except Exception as exc:
            result["error"]  = str(exc)
            result["status"] = "failed"
            logger.warning("LLMJudgeEvaluationService.run_evaluation failed: {}", exc)

        await self._save(result)
        return result

    # ── Scoring ────────────────────────────────────────────────────────────────

    async def _score_report(self, alert_doc: dict) -> dict:
        rcr   = alert_doc.get("root_cause_report") or {}
        anom  = rcr.get("anomaly_result") or {}

        machine_id      = alert_doc.get("machine_id", "UNKNOWN")
        failure_type    = anom.get("failure_type_prediction", "UNKNOWN")
        anomaly_score   = anom.get("anomaly_score", 0.0)
        failure_prob    = anom.get("failure_probability", 0.0)
        sensor_deltas   = anom.get("sensor_deltas") or {}
        root_cause      = rcr.get("root_cause", "")
        confidence      = rcr.get("confidence", 0.0)
        severity        = rcr.get("severity", "UNKNOWN")
        evidence        = rcr.get("evidence", "")
        actions         = rcr.get("recommended_actions") or []
        reasoning_steps = rcr.get("reasoning_steps") or []

        deltas_str = ", ".join(
            f"{k}={v:.2f}" for k, v in sensor_deltas.items() if v is not None
        ) or "none"
        actions_str = "\n".join(f"- {a}" for a in actions) or "none"
        steps_str   = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(reasoning_steps)
        ) if reasoning_steps else "none"

        prompt = (
            "You are an expert industrial maintenance evaluator. "
            "Score the following root-cause analysis report on FIVE dimensions from 1 to 5.\n\n"
            f"Machine: {machine_id}  Failure type: {failure_type}\n"
            f"Anomaly score: {anomaly_score:.3f}  Failure probability: {failure_prob:.0%}\n"
            f"Sensor deviations (std): {deltas_str}\n\n"
            f"Root cause: {root_cause}\n"
            f"Confidence: {confidence:.0%}  Severity: {severity}\n"
            f"Evidence: {evidence}\n"
            f"Recommended actions:\n{actions_str}\n"
            f"Reasoning steps:\n{steps_str}\n\n"
            "Scoring criteria:\n"
            "  root_cause_correctness (1-5): Is the root cause specific, plausible, and consistent with sensor data?\n"
            "  severity_accuracy (1-5): Is the severity appropriate given the failure probability and sensor deviations?\n"
            "  action_quality (1-5): Are the recommended actions specific, actionable, and correctly prioritized?\n"
            "  reasoning_quality (1-5): Does the reasoning trace show logical step-by-step analysis?\n"
            "  confidence_calibration (1-5): Is the confidence score appropriate (not over/under confident)?\n\n"
            "Return ONLY valid JSON (no markdown):\n"
            '{"root_cause_correctness": <int>, "severity_accuracy": <int>, '
            '"action_quality": <int>, "reasoning_quality": <int>, '
            '"confidence_calibration": <int>, '
            '"justifications": {"root_cause_correctness": "<str>", "severity_accuracy": "<str>", '
            '"action_quality": "<str>", "reasoning_quality": "<str>", '
            '"confidence_calibration": "<str>"}}'
        )

        defaults = {
            "root_cause_correctness": 0.5,
            "severity_accuracy":      0.5,
            "action_quality":         0.5,
            "reasoning_quality":      0.5,
            "confidence_calibration": 0.5,
        }

        import time

        try:
            from langchain_groq import ChatGroq
            from langchain_core.messages import HumanMessage

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=self._groq_key,
                temperature=0.0,
                max_tokens=512,
            )

            def _call_with_retry():
                for attempt in range(3):
                    try:
                        resp = llm.invoke([HumanMessage(content=prompt)])
                        return resp.content
                    except Exception as exc:
                        if "rate" in str(exc).lower() or "429" in str(exc):
                            wait = (attempt + 1) * 10  # 10s, 20s, 30s
                            logger.warning(
                                "Groq rate limit hit, waiting {}s (attempt {}/3)",
                                wait, attempt + 1,
                            )
                            time.sleep(wait)
                        else:
                            raise
                return None  # all retries exhausted

            loop    = asyncio.get_event_loop()
            content = await loop.run_in_executor(None, _call_with_retry)

            if content is None:
                logger.warning("LLMJudgeEvaluationService._score_report: all retries exhausted")
                return {
                    "alert_id":  alert_doc.get("alert_id", ""),
                    "machine_id": machine_id,
                    **defaults,
                    "justifications": {},
                }

            raw = content.strip()
            if "```" in raw:
                raw = raw.split("```")[-2].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw)

            dims = ["root_cause_correctness", "severity_accuracy",
                    "action_quality", "reasoning_quality", "confidence_calibration"]
            scores: dict[str, Any] = {
                "alert_id":  alert_doc.get("alert_id", ""),
                "machine_id": machine_id,
                "justifications": data.get("justifications", {}),
            }
            for d in dims:
                raw_score      = int(data.get(d, 3))
                normalised     = round((raw_score - 1) / 4, 4)
                scores[d]      = max(0.0, min(1.0, normalised))

            return scores

        except Exception as exc:
            logger.warning("Evaluation scoring failed (possible rate limit) — {}", exc)
            return {
                "alert_id":  alert_doc.get("alert_id", ""),
                "machine_id": machine_id,
                **defaults,
                "justifications": {},
            }

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _empty_scores(self) -> dict:
        return {
            "root_cause_correctness": 0.0,
            "severity_accuracy":      0.0,
            "action_quality":         0.0,
            "reasoning_quality":      0.0,
            "confidence_calibration": 0.0,
            "overall":                0.0,
        }

    async def _save(self, result: dict) -> None:
        if self._mongo_db is None:
            return
        try:
            await self._mongo_db[COLL_EVAL].insert_one({**result})
        except Exception as exc:
            logger.warning("LLMJudgeEvaluationService._save failed: {}", exc)


# ── Scheduler entry-point ──────────────────────────────────────────────────────

async def run_nightly_evaluation(app) -> None:
    """Called by APScheduler at 02:00 UTC every night."""
    logger.info("Nightly evaluation: starting")
    try:
        mongo_db  = getattr(app.state, "mongo_db",  None)
        qdrant    = getattr(app.state, "qdrant",     None)
        groq_key  = os.getenv("GROQ_API_KEY", "")

        rag_svc = RAGEvaluationService(
            mongo_db=mongo_db,
            qdrant_service=qdrant,
            groq_api_key=groq_key,
        )
        llm_svc = LLMJudgeEvaluationService(
            mongo_db=mongo_db,
            groq_api_key=groq_key,
        )

        rag_result = await rag_svc.run_evaluation(n_samples=10)
        logger.info(
            "Nightly eval RAG: status={} overall={:.3f}",
            rag_result.get("status"),
            rag_result.get("scores", {}).get("overall", 0.0),
        )

        llm_result = await llm_svc.run_evaluation(n_samples=10)
        logger.info(
            "Nightly eval LLM-Judge: status={} overall={:.3f}",
            llm_result.get("status"),
            llm_result.get("scores", {}).get("overall", 0.0),
        )

    except Exception as exc:
        logger.warning("run_nightly_evaluation: failed (non-fatal) — {}", exc)
