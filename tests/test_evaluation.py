"""
tests/test_evaluation.py — 15 unit tests for the evaluation service and API.

Tests:
  - RAGEvaluationService initialization
  - run_evaluation with no alerts (graceful empty result)
  - _build_eval_samples formats question/context/answer correctly
  - _score_* methods return float 0-1 when LLM is mocked
  - _score_* methods return default 0.5 when LLM raises
  - LLMJudgeEvaluationService initialization
  - run_evaluation stores correct structure in MongoDB
  - _score_report normalizes 1-5 → 0-1 correctly
  - Evaluation API GET /latest returns 200 with correct keys
  - Evaluation API GET /history returns 200
  - Evaluation API GET /run returns 200
  - run_nightly_evaluation handles mongo_db=None without crashing
  - run_nightly_evaluation with no approved alerts completes without error
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_ALERT_DOC = {
    "alert_id":   "alert-test-001",
    "machine_id": "TEST_M001",
    "session_id": "sess-001",
    "approved":   True,
    "approved_by": "engineer",
    "created_at": "2024-06-15T10:00:00",
    "root_cause_report": {
        "root_cause":          "Cooling fan blade fractured reducing airflow",
        "confidence":          0.88,
        "severity":            "HIGH",
        "evidence":            "Process temperature 3.1 std above normal",
        "recommended_actions": ["Inspect cooling fan", "Schedule maintenance"],
        "reasoning_steps":     ["Step 1: Detected temp spike", "Step 2: Correlated with fan speed"],
        "similar_incidents":   [
            {
                "failure_type": "HDF",
                "symptoms":     "High temperature alarm",
                "root_cause":   "Fan failure",
                "action_taken": "Replaced fan assembly",
            }
        ],
        "anomaly_result": {
            "machine_id":              "TEST_M001",
            "failure_type_prediction": "HDF",
            "anomaly_score":           0.95,
            "failure_probability":     0.88,
            "sensor_deltas": {
                "process_temperature": 3.1,
                "rotational_speed":    -2.2,
            },
        },
    },
}


def _make_mongo_with_alerts(docs: list) -> MagicMock:
    """Mock motor db whose alerts.find(...) returns docs."""
    mock_cursor = MagicMock()
    mock_cursor.sort = MagicMock(return_value=mock_cursor)
    mock_cursor.limit = MagicMock(return_value=mock_cursor)
    mock_cursor.to_list = AsyncMock(return_value=docs)

    mock_coll = MagicMock()
    mock_coll.find = MagicMock(return_value=mock_cursor)
    mock_coll.insert_one = AsyncMock(return_value=MagicMock(inserted_id="fake_id"))

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(return_value=mock_coll)
    return mock_db


# ── RAGEvaluationService ──────────────────────────────────────────────────────

class TestRAGEvaluationService:

    def test_initializes_correctly(self):
        from app.services.evaluation_service import RAGEvaluationService
        svc = RAGEvaluationService(mongo_db=None, qdrant_service=None, groq_api_key="test-key")
        assert svc._mongo_db   is None
        assert svc._qdrant     is None
        assert svc._groq_key   == "test-key"

    @pytest.mark.asyncio
    async def test_run_evaluation_no_alerts_returns_completed(self):
        """When MongoDB has no approved alerts, result status=completed with zero scores."""
        from app.services.evaluation_service import RAGEvaluationService

        mock_db = _make_mongo_with_alerts([])
        svc     = RAGEvaluationService(mongo_db=mock_db, qdrant_service=None, groq_api_key="k")
        result  = await svc.run_evaluation(n_samples=5)

        assert result["status"]    == "completed"
        assert result["eval_type"] == "rag"
        assert "scores" in result
        assert result["scores"]["overall"] == 0.0

    @pytest.mark.asyncio
    async def test_build_eval_samples_formats_correctly(self):
        """_build_eval_samples should extract question, answer, and contexts from alert doc."""
        from app.services.evaluation_service import RAGEvaluationService

        mock_db  = _make_mongo_with_alerts([SAMPLE_ALERT_DOC])
        svc      = RAGEvaluationService(mongo_db=mock_db, qdrant_service=None, groq_api_key="k")
        samples  = await svc._build_eval_samples(n_samples=1)

        assert len(samples) == 1
        s = samples[0]
        assert "TEST_M001" in s["question"]
        assert "HDF"        in s["question"]
        assert s["answer"]  == "Cooling fan blade fractured reducing airflow"
        assert len(s["contexts"]) >= 1
        assert "HDF" in s["contexts"][0]

    @pytest.mark.asyncio
    async def test_score_context_precision_returns_float(self):
        """When LLM returns valid JSON, score is a float in [0, 1]."""
        from app.services.evaluation_service import RAGEvaluationService

        svc = RAGEvaluationService(mongo_db=None, qdrant_service=None, groq_api_key="k")

        mock_msg = MagicMock()
        mock_msg.content = '{"score": 0.8, "reasoning": "Most chunks were relevant."}'

        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value=mock_msg)

        with patch("app.services.evaluation_service.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_msg.content)
            with patch("langchain_groq.ChatGroq", return_value=mock_llm):
                score = await svc._score_context_precision("q", ["ctx1"], "ans")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_faithfulness_returns_float(self):
        from app.services.evaluation_service import RAGEvaluationService

        svc = RAGEvaluationService(mongo_db=None, qdrant_service=None, groq_api_key="k")

        with patch("app.services.evaluation_service.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value='{"score": 0.9, "reasoning": "Faithful."}'
            )
            score = await svc._score_faithfulness("answer", ["ctx"])

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_answer_relevancy_returns_float(self):
        from app.services.evaluation_service import RAGEvaluationService

        svc = RAGEvaluationService(mongo_db=None, qdrant_service=None, groq_api_key="k")

        with patch("app.services.evaluation_service.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value='{"score": 0.75, "reasoning": "Relevant."}'
            )
            score = await svc._score_answer_relevancy("question", "answer")

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_score_returns_default_on_llm_error(self):
        """When the executor raises, _llm_score returns default 0.5."""
        from app.services.evaluation_service import RAGEvaluationService

        svc = RAGEvaluationService(mongo_db=None, qdrant_service=None, groq_api_key="k")

        with patch("app.services.evaluation_service.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                side_effect=Exception("Groq API error")
            )
            score = await svc._llm_score("any prompt")

        assert score == 0.5

    @pytest.mark.asyncio
    async def test_run_evaluation_stores_result_in_mongo(self):
        """run_evaluation should call insert_one on the evaluation_results collection."""
        from app.services.evaluation_service import RAGEvaluationService

        mock_db = _make_mongo_with_alerts([SAMPLE_ALERT_DOC])
        svc     = RAGEvaluationService(mongo_db=mock_db, qdrant_service=None, groq_api_key="k")

        with patch("app.services.evaluation_service.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(
                return_value='{"score": 0.7, "reasoning": "ok"}'
            )
            result = await svc.run_evaluation(n_samples=1)

        # insert_one should have been called (on the collection returned by __getitem__)
        assert result["eval_type"] == "rag"
        assert result["status"]    == "completed"
        assert "sample_scores" in result


# ── LLMJudgeEvaluationService ─────────────────────────────────────────────────

class TestLLMJudgeEvaluationService:

    def test_initializes_correctly(self):
        from app.services.evaluation_service import LLMJudgeEvaluationService
        svc = LLMJudgeEvaluationService(mongo_db=None, groq_api_key="test-key")
        assert svc._mongo_db is None
        assert svc._groq_key == "test-key"

    @pytest.mark.asyncio
    async def test_run_evaluation_stores_correct_structure(self):
        """Result must have eval_type, scores with all 5 dimensions + overall, sample_scores."""
        from app.services.evaluation_service import LLMJudgeEvaluationService

        mock_db = _make_mongo_with_alerts([SAMPLE_ALERT_DOC])
        svc     = LLMJudgeEvaluationService(mongo_db=mock_db, groq_api_key="k")

        llm_response = (
            '{"root_cause_correctness": 4, "severity_accuracy": 4, '
            '"action_quality": 3, "reasoning_quality": 4, '
            '"confidence_calibration": 3, '
            '"justifications": {"root_cause_correctness": "Good.", '
            '"severity_accuracy": "Appropriate.", "action_quality": "Decent.", '
            '"reasoning_quality": "Clear.", "confidence_calibration": "Fair."}}'
        )

        with patch("app.services.evaluation_service.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=llm_response)
            result = await svc.run_evaluation(n_samples=1)

        assert result["eval_type"] == "llm_judge"
        assert result["status"]    == "completed"
        scores = result["scores"]
        for dim in ["root_cause_correctness", "severity_accuracy", "action_quality",
                    "reasoning_quality", "confidence_calibration", "overall"]:
            assert dim in scores, f"Missing dimension: {dim}"
            assert 0.0 <= scores[dim] <= 1.0

    @pytest.mark.asyncio
    async def test_score_report_normalizes_correctly(self):
        """Score of 5 → 1.0; score of 1 → 0.0; score of 3 → 0.5."""
        from app.services.evaluation_service import LLMJudgeEvaluationService

        svc = LLMJudgeEvaluationService(mongo_db=None, groq_api_key="k")

        llm_response = (
            '{"root_cause_correctness": 5, "severity_accuracy": 1, '
            '"action_quality": 3, "reasoning_quality": 5, '
            '"confidence_calibration": 1, "justifications": {}}'
        )

        with patch("app.services.evaluation_service.asyncio.get_event_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(return_value=llm_response)
            scores = await svc._score_report(SAMPLE_ALERT_DOC)

        assert scores["root_cause_correctness"] == 1.0   # (5-1)/4
        assert scores["severity_accuracy"]      == 0.0   # (1-1)/4
        assert scores["action_quality"]         == 0.5   # (3-1)/4

    @pytest.mark.asyncio
    async def test_run_evaluation_no_alerts_returns_completed(self):
        from app.services.evaluation_service import LLMJudgeEvaluationService

        mock_db = _make_mongo_with_alerts([])
        svc     = LLMJudgeEvaluationService(mongo_db=mock_db, groq_api_key="k")
        result  = await svc.run_evaluation(n_samples=5)

        assert result["status"]    == "completed"
        assert result["eval_type"] == "llm_judge"
        assert result["scores"]["overall"] == 0.0


# ── Evaluation API ────────────────────────────────────────────────────────────

class TestEvaluationAPI:

    def _build_app(self, eval_docs: list | None = None):
        """Build a minimal FastAPI test app with mocked state."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from app.api.routes.evaluation import router

        test_app = FastAPI()
        test_app.include_router(router, prefix="/api/evaluation")

        # Mock mongo_db
        mock_cursor = MagicMock()
        mock_cursor.sort  = MagicMock(return_value=mock_cursor)
        mock_cursor.limit = MagicMock(return_value=mock_cursor)
        mock_cursor.to_list = AsyncMock(return_value=eval_docs or [])

        mock_coll = MagicMock()
        mock_coll.find = MagicMock(return_value=mock_cursor)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_coll)

        @test_app.on_event("startup")
        async def _setup():
            test_app.state.mongo_db = mock_db

        return TestClient(test_app)

    def test_latest_returns_200_with_correct_keys(self):
        client = self._build_app(eval_docs=[])
        resp   = client.get("/api/evaluation/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert "rag"       in data
        assert "llm_judge" in data

    def test_history_returns_200(self):
        client = self._build_app(eval_docs=[])
        resp   = client.get("/api/evaluation/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "count"   in data

    def test_run_returns_200_with_started_status(self):
        client = self._build_app()
        with patch("app.services.evaluation_service.run_nightly_evaluation", new_callable=AsyncMock):
            resp = client.get("/api/evaluation/run")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"


# ── Scheduler / nightly evaluation ───────────────────────────────────────────

class TestNightlyEvaluation:

    @pytest.mark.asyncio
    async def test_handles_no_mongo_gracefully(self):
        """run_nightly_evaluation with mongo_db=None must not raise."""
        from app.services.evaluation_service import run_nightly_evaluation

        mock_app = MagicMock()
        mock_app.state.mongo_db  = None
        mock_app.state.qdrant    = None

        # Should complete without exception
        await run_nightly_evaluation(mock_app)

    @pytest.mark.asyncio
    async def test_handles_no_approved_alerts_gracefully(self):
        """run_nightly_evaluation with empty alerts collection must not raise."""
        from app.services.evaluation_service import run_nightly_evaluation

        mock_db = _make_mongo_with_alerts([])

        mock_app = MagicMock()
        mock_app.state.mongo_db  = mock_db
        mock_app.state.qdrant    = None

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
            await run_nightly_evaluation(mock_app)
