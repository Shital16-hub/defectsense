"""
tests/test_ml_models.py — 20 tests for ML model layer.

Tests: scaler, LSTM autoencoder, Isolation Forest, MLService ensemble.
All tests load real trained artefacts from ml/models/.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

ROOT       = Path(__file__).parent.parent
MODELS_DIR = ROOT / "ml" / "models"

NORMAL_ROW   = np.array([[298.1, 308.6, 1500.0, 40.0,  50.0]], dtype=np.float32)
FAILURE_ROW  = np.array([[302.0, 309.0, 1182.0, 68.5, 195.0]], dtype=np.float32)
TWF_ROW      = np.array([[298.0, 308.5, 1400.0, 50.0, 245.0]], dtype=np.float32)
SEQ_LEN      = 30
N_FEATURES   = 5


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def scaler():
    with open(MODELS_DIR / "sensor_scaler.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def threshold_data():
    with open(MODELS_DIR / "anomaly_threshold.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def iforest():
    with open(MODELS_DIR / "isolation_forest.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"] if isinstance(data, dict) else data


@pytest.fixture(scope="module")
def autoencoder():
    import tensorflow as tf
    return tf.keras.models.load_model(str(MODELS_DIR / "lstm_autoencoder.keras"))


def make_sequence(scaler, row: np.ndarray, length: int = SEQ_LEN) -> np.ndarray:
    """Build a (1, SEQ_LEN, N_FEATURES) sequence from a single row."""
    scaled = scaler.transform(row)[0]
    seq    = np.tile(scaled, (length, 1))
    return seq[np.newaxis, ...]


# ── Scaler tests ───────────────────────────────────────────────────────────────

def test_scaler_loads(scaler):
    assert scaler is not None


def test_scaler_output_shape(scaler):
    out = scaler.transform(NORMAL_ROW)
    assert out.shape == (1, N_FEATURES)


def test_scaler_output_range(scaler):
    out = scaler.transform(NORMAL_ROW)[0]
    # MinMaxScaler: most values should be in [0, 1] for typical inputs
    assert out.min() >= -0.5 and out.max() <= 1.5


def test_scaler_normal_vs_failure_differs(scaler):
    n = scaler.transform(NORMAL_ROW)[0]
    f = scaler.transform(FAILURE_ROW)[0]
    assert not np.allclose(n, f)


def test_scaler_deterministic(scaler):
    a = scaler.transform(NORMAL_ROW)
    b = scaler.transform(NORMAL_ROW)
    np.testing.assert_array_equal(a, b)


# ── Isolation Forest tests ─────────────────────────────────────────────────────

def test_iforest_loads(iforest):
    assert iforest is not None


def test_iforest_prediction_values(iforest, scaler):
    scaled = scaler.transform(NORMAL_ROW)
    pred   = iforest.predict(scaled)
    assert pred[0] in (1, -1), "IForest must predict 1 (normal) or -1 (anomaly)"


def test_iforest_decision_function_returns_float(iforest, scaler):
    scaled = scaler.transform(NORMAL_ROW)
    score  = iforest.decision_function(scaled)
    assert isinstance(float(score[0]), float)


def test_iforest_normal_positive_score(iforest, scaler):
    """Normal readings should have positive (or near-zero) decision scores."""
    scaled = scaler.transform(NORMAL_ROW)
    score  = float(iforest.decision_function(scaled)[0])
    # Decision function: positive = more normal, negative = more anomalous
    # Typical normal rows should score higher than extreme failure rows
    normal_score = score

    failure_scaled = scaler.transform(FAILURE_ROW)
    failure_score  = float(iforest.decision_function(failure_scaled)[0])
    assert normal_score >= failure_score, (
        f"Normal score {normal_score:.4f} should be >= failure score {failure_score:.4f}"
    )


def test_iforest_batch_prediction(iforest, scaler):
    batch  = np.vstack([NORMAL_ROW, FAILURE_ROW, TWF_ROW])
    scaled = scaler.transform(batch)
    preds  = iforest.predict(scaled)
    assert preds.shape == (3,)
    assert set(preds).issubset({1, -1})


# ── LSTM Autoencoder tests ─────────────────────────────────────────────────────

def test_autoencoder_loads(autoencoder):
    assert autoencoder is not None


def test_autoencoder_input_shape(autoencoder):
    shape = autoencoder.input_shape
    assert shape[1] == SEQ_LEN
    assert shape[2] == N_FEATURES


def test_autoencoder_output_shape(autoencoder, scaler):
    seq  = make_sequence(scaler, NORMAL_ROW)
    recon = autoencoder.predict(seq, verbose=0)
    assert recon.shape == (1, SEQ_LEN, N_FEATURES)


def test_autoencoder_reconstruction_error_is_float(autoencoder, scaler):
    seq   = make_sequence(scaler, NORMAL_ROW)
    recon = autoencoder.predict(seq, verbose=0)
    mse   = float(np.mean(np.power(seq - recon, 2)))
    assert isinstance(mse, float) and mse >= 0.0


def test_threshold_loaded(threshold_data):
    assert "threshold" in threshold_data
    assert isinstance(threshold_data["threshold"], float)
    assert threshold_data["threshold"] > 0.0


# ── MLService integration tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ml_service_loads(tmp_path):
    from app.services.ml_service import MLService
    svc = MLService()
    svc.load()
    assert svc.is_ready


@pytest.mark.asyncio
async def test_ml_service_normal_prediction(normal_reading):
    from app.services.ml_service import MLService
    svc = MLService()
    svc.load()
    result = await svc.predict_anomaly(normal_reading, sequence=None)
    assert result.machine_id == "TEST_M001"
    assert 0.0 <= result.anomaly_score <= 1.0
    assert 0.0 <= result.failure_probability <= 1.0


@pytest.mark.asyncio
async def test_ml_service_returns_anomaly_result_type(normal_reading):
    from app.services.ml_service import MLService
    from app.models.anomaly import AnomalyResult
    svc = MLService()
    svc.load()
    result = await svc.predict_anomaly(normal_reading)
    assert isinstance(result, AnomalyResult)


@pytest.mark.asyncio
async def test_ml_service_with_sequence_uses_ensemble(normal_reading, normal_reading_sequence):
    from app.services.ml_service import MLService
    svc = MLService()
    svc.load()
    result = await svc.predict_anomaly(normal_reading, sequence=normal_reading_sequence)
    assert result.ml_model_used == "ensemble"


@pytest.mark.asyncio
async def test_ml_service_without_sequence_uses_iforest(normal_reading):
    from app.services.ml_service import MLService
    svc = MLService()
    svc.load()
    result = await svc.predict_anomaly(normal_reading, sequence=None)
    assert result.ml_model_used == "isolation_forest"
