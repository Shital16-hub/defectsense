"""
tests/test_blob_storage.py — BlobStorageService + MLService blob integration tests.

All Azure SDK calls are mocked — no real Azure credentials required.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_available_service():
    """Return a BlobStorageService whose Azure client is fully mocked."""
    mock_client = MagicMock()
    with patch("app.services.blob_storage_service.BlobServiceClient") as mock_bsc:
        mock_bsc.from_connection_string.return_value = mock_client
        from app.services.blob_storage_service import BlobStorageService
        svc = BlobStorageService(
            connection_string="fake_conn_str",
            container_name="defectsense-models",
        )
    # Swap the client reference so future calls use our mock
    svc._client = mock_client
    return svc, mock_client


# ── BlobStorageService — init ──────────────────────────────────────────────────

class TestBlobStorageServiceInit:
    def test_blob_service_initializes_correctly(self):
        """Service marks itself available when credentials provided and SDK init succeeds."""
        mock_client = MagicMock()
        with patch("app.services.blob_storage_service.BlobServiceClient") as mock_bsc:
            mock_bsc.from_connection_string.return_value = mock_client
            from app.services.blob_storage_service import BlobStorageService
            svc = BlobStorageService(
                connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=abc;",
                container_name="defectsense-models",
            )

        assert svc.is_available is True
        assert svc._container_name == "defectsense-models"

    def test_blob_service_graceful_without_credentials(self):
        """Service stays unavailable and never raises when no connection string given."""
        from app.services.blob_storage_service import BlobStorageService
        svc = BlobStorageService(connection_string=None, container_name="defectsense-models")

        assert svc.is_available is False
        assert svc.upload_model("/tmp/fake.pkl", "fake.pkl") is False
        assert svc.download_model("fake.pkl", "/tmp/fake.pkl") is False
        assert svc.list_models() == []
        assert svc.model_exists("fake.pkl") is False


# ── BlobStorageService — upload ────────────────────────────────────────────────

class TestBlobStorageUpload:
    def test_upload_model_success(self, tmp_path):
        """upload_model returns True and calls upload_blob when file exists."""
        svc, mock_client = _make_available_service()

        local_file = tmp_path / "model.pkl"
        local_file.write_bytes(b"fake model data")

        mock_container = MagicMock()
        mock_client.get_container_client.return_value = mock_container

        result = svc.upload_model(local_file, "model_latest.pkl")

        assert result is True
        mock_container.upload_blob.assert_called_once()
        call_kwargs = mock_container.upload_blob.call_args
        assert call_kwargs.kwargs.get("name") == "model_latest.pkl" or \
               call_kwargs.args[0] == "model_latest.pkl"
        assert call_kwargs.kwargs.get("overwrite") is True

    def test_upload_model_failure_returns_false(self, tmp_path):
        """upload_model returns False (not raises) when Azure raises an exception."""
        svc, mock_client = _make_available_service()

        local_file = tmp_path / "model.pkl"
        local_file.write_bytes(b"data")

        mock_container = MagicMock()
        mock_container.upload_blob.side_effect = Exception("Network error")
        mock_client.get_container_client.return_value = mock_container

        result = svc.upload_model(local_file, "model_latest.pkl")

        assert result is False

    def test_upload_model_missing_local_file_returns_false(self, tmp_path):
        """upload_model returns False when local file does not exist."""
        svc, _ = _make_available_service()
        result = svc.upload_model(tmp_path / "nonexistent.pkl", "blob.pkl")
        assert result is False


# ── BlobStorageService — download ──────────────────────────────────────────────

class TestBlobStorageDownload:
    def test_download_model_success(self, tmp_path):
        """download_model returns True and writes correct bytes to local_path."""
        svc, mock_client = _make_available_service()

        blob_bytes = b"trained model weights"
        mock_stream = MagicMock()
        mock_stream.readall.return_value = blob_bytes

        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value = mock_stream

        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_client.get_container_client.return_value = mock_container

        dest = tmp_path / "downloaded_model.pkl"
        result = svc.download_model("model_latest.pkl", dest)

        assert result is True
        assert dest.read_bytes() == blob_bytes

    def test_download_model_creates_directory(self, tmp_path):
        """download_model creates parent directories automatically if missing."""
        svc, mock_client = _make_available_service()

        mock_stream = MagicMock()
        mock_stream.readall.return_value = b"data"

        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.return_value = mock_stream

        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_client.get_container_client.return_value = mock_container

        dest = tmp_path / "new_dir" / "subdir" / "model.pkl"
        assert not dest.parent.exists()

        result = svc.download_model("model_latest.pkl", dest)

        assert result is True
        assert dest.parent.exists()
        assert dest.read_bytes() == b"data"

    def test_download_model_failure_returns_false(self, tmp_path):
        """download_model returns False when Azure raises."""
        svc, mock_client = _make_available_service()

        mock_blob_client = MagicMock()
        mock_blob_client.download_blob.side_effect = Exception("Blob not found")

        mock_container = MagicMock()
        mock_container.get_blob_client.return_value = mock_blob_client
        mock_client.get_container_client.return_value = mock_container

        result = svc.download_model("missing.pkl", tmp_path / "out.pkl")
        assert result is False


# ── BlobStorageService — list & exists ────────────────────────────────────────

class TestBlobStorageListAndExists:
    def test_list_models_returns_list(self):
        """list_models returns blob names as a plain list of strings."""
        svc, mock_client = _make_available_service()

        blob_a = MagicMock(); blob_a.name = "lstm_autoencoder_latest.keras"
        blob_b = MagicMock(); blob_b.name = "isolation_forest_latest.pkl"

        mock_container = MagicMock()
        mock_container.list_blobs.return_value = [blob_a, blob_b]
        mock_client.get_container_client.return_value = mock_container

        result = svc.list_models()

        assert result == ["lstm_autoencoder_latest.keras", "isolation_forest_latest.pkl"]

    def test_model_exists_returns_bool(self):
        """model_exists returns True for present blobs, False for missing ones."""
        svc, mock_client = _make_available_service()

        mock_blob_present = MagicMock()
        mock_blob_present.get_blob_properties.return_value = {"size": 1024}

        mock_blob_absent = MagicMock()
        mock_blob_absent.get_blob_properties.side_effect = Exception("BlobNotFound")

        mock_container = MagicMock()
        mock_container.get_blob_client.side_effect = (
            lambda name: mock_blob_present if name == "exists.pkl" else mock_blob_absent
        )
        mock_client.get_container_client.return_value = mock_container

        assert svc.model_exists("exists.pkl") is True
        assert svc.model_exists("missing.pkl") is False


# ── MLService blob integration ─────────────────────────────────────────────────

class TestMLServiceBlobIntegration:
    """Tests for MLService.load() local-first + blob-fallback logic."""

    def _make_ml_service(self, blob_service=None):
        from app.services.ml_service import MLService
        return MLService(blob_service=blob_service)

    def test_ml_service_loads_locally_first(self):
        """When all local model files exist, blob.download_model is never called."""
        mock_blob = MagicMock()
        mock_blob.is_available = True

        svc = self._make_ml_service(blob_service=mock_blob)

        fake_scaler = MagicMock()
        fake_threshold = {"threshold": 0.01, "mean": 0.005, "std": 0.001}
        fake_iforest = {"model": MagicMock(), "scaler": MagicMock(), "features": []}

        with patch("app.services.ml_service.AUTOENCODER_PATH") as mp_ae, \
             patch("app.services.ml_service.SCALER_PATH") as mp_sc, \
             patch("app.services.ml_service.THRESHOLD_PATH") as mp_th, \
             patch("app.services.ml_service.IFOREST_PATH") as mp_if, \
             patch("app.services.ml_service.MLService._init_mlflow", return_value=None), \
             patch("tensorflow.keras.models.load_model", return_value=MagicMock()), \
             patch("pickle.load", side_effect=[fake_scaler, fake_threshold, fake_iforest]), \
             patch("builtins.open", mock_open()):
            mp_ae.exists.return_value = True
            mp_sc.exists.return_value = True
            mp_th.exists.return_value = True
            mp_if.exists.return_value = True
            mp_ae.__str__ = lambda self: "/fake/lstm_autoencoder.keras"

            svc.load()

        mock_blob.download_model.assert_not_called()

    def test_ml_service_attempts_blob_when_local_missing(self):
        """When all local files are missing, download_model is called once per artefact."""
        mock_blob = MagicMock()
        mock_blob.is_available = True
        mock_blob.download_model.return_value = False  # download fails → graceful

        svc = self._make_ml_service(blob_service=mock_blob)

        with patch("app.services.ml_service.AUTOENCODER_PATH") as mp_ae, \
             patch("app.services.ml_service.SCALER_PATH") as mp_sc, \
             patch("app.services.ml_service.THRESHOLD_PATH") as mp_th, \
             patch("app.services.ml_service.IFOREST_PATH") as mp_if, \
             patch("app.services.ml_service.MLService._init_mlflow", return_value=None):
            mp_ae.exists.return_value = False
            mp_sc.exists.return_value = False
            mp_th.exists.return_value = False
            mp_if.exists.return_value = False

            svc.load()

        assert mock_blob.download_model.call_count == 4
        called_blobs = {call.args[0] for call in mock_blob.download_model.call_args_list}
        assert "lstm_autoencoder_latest.keras" in called_blobs
        assert "sensor_scaler_latest.pkl" in called_blobs
        assert "anomaly_threshold_latest.pkl" in called_blobs
        assert "isolation_forest_latest.pkl" in called_blobs

    def test_ml_service_graceful_when_both_unavailable(self):
        """MLService.load() completes without raising when local files and blob are both absent."""
        svc = self._make_ml_service(blob_service=None)

        with patch("app.services.ml_service.AUTOENCODER_PATH") as mp_ae, \
             patch("app.services.ml_service.SCALER_PATH") as mp_sc, \
             patch("app.services.ml_service.THRESHOLD_PATH") as mp_th, \
             patch("app.services.ml_service.IFOREST_PATH") as mp_if, \
             patch("app.services.ml_service.MLService._init_mlflow", return_value=None):
            mp_ae.exists.return_value = False
            mp_sc.exists.return_value = False
            mp_th.exists.return_value = False
            mp_if.exists.return_value = False

            svc.load()  # must not raise

        assert svc._loaded is True
        assert svc._autoencoder is None
        assert svc._iforest is None
        assert svc.is_ready is False

    def test_ml_service_is_blob_available_property(self):
        """is_blob_available property reflects the blob service state correctly."""
        from app.services.ml_service import MLService

        assert MLService(blob_service=None).is_blob_available is False

        mock_up = MagicMock(); mock_up.is_available = True
        assert MLService(blob_service=mock_up).is_blob_available is True

        mock_down = MagicMock(); mock_down.is_available = False
        assert MLService(blob_service=mock_down).is_blob_available is False
