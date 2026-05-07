"""
tests/test_blob_storage.py — BlobStorageService stub + MLService local-only tests.

BlobStorageService is now a stub (blob storage disabled).
All methods return False/empty. MLService loads from local files only.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ── BlobStorageService — stub behaviour ───────────────────────────────────────

class TestBlobStorageStub:
    """Verify the stub interface contract — all operations are no-ops."""

    def _make_svc(self, **kwargs):
        from app.services.blob_storage_service import BlobStorageService
        return BlobStorageService(**kwargs)

    def test_stub_always_unavailable(self):
        """is_available is always False regardless of arguments."""
        svc = self._make_svc()
        assert svc.is_available is False

    def test_stub_unavailable_with_args(self):
        """Passing connection_string and container_name still results in is_available=False."""
        svc = self._make_svc(
            connection_string="fake_conn",
            container_name="defectsense-models",
        )
        assert svc.is_available is False

    def test_stub_accepts_positional_args(self):
        """Stub accepts positional args without raising."""
        from app.services.blob_storage_service import BlobStorageService
        svc = BlobStorageService("fake_conn", "some-container")
        assert svc.is_available is False

    def test_upload_returns_false(self, tmp_path):
        """upload_model always returns False."""
        svc = self._make_svc()
        local_file = tmp_path / "model.pkl"
        local_file.write_bytes(b"data")
        assert svc.upload_model(local_file, "model_latest.pkl") is False

    def test_upload_returns_false_for_missing_file(self, tmp_path):
        """upload_model returns False even for a non-existent local path."""
        svc = self._make_svc()
        assert svc.upload_model(tmp_path / "nonexistent.pkl", "blob.pkl") is False

    def test_download_returns_false(self, tmp_path):
        """download_model always returns False."""
        svc = self._make_svc()
        assert svc.download_model("model_latest.pkl", tmp_path / "out.pkl") is False

    def test_list_models_returns_empty(self):
        """list_models always returns an empty list."""
        svc = self._make_svc()
        assert svc.list_models() == []

    def test_model_exists_returns_false(self):
        """model_exists always returns False."""
        svc = self._make_svc()
        assert svc.model_exists("any_blob.pkl") is False

    def test_no_azure_import_required(self):
        """BlobStorageService imports without azure-storage-blob installed."""
        import importlib
        import app.services.blob_storage_service as mod
        importlib.reload(mod)   # re-import — must not raise
        assert mod.BlobStorageService is not None


# ── MLService — local-only loading ────────────────────────────────────────────

class TestMLServiceLocalOnly:
    """Tests for MLService.load() with blob storage disabled."""

    def _make_ml_service(self, blob_service=None):
        from app.services.ml_service import MLService
        return MLService(blob_service=blob_service)

    def test_ml_service_loads_from_local_files(self):
        """When local model files exist, service loads successfully."""
        from app.services.blob_storage_service import BlobStorageService
        blob_stub = BlobStorageService()

        svc = self._make_ml_service(blob_service=blob_stub)

        fake_scaler    = MagicMock()
        fake_threshold = {"threshold": 0.01, "mean": 0.005, "std": 0.001}

        with patch("app.services.ml_service.AUTOENCODER_PATH") as mp_ae, \
             patch("app.services.ml_service.SCALER_PATH") as mp_sc, \
             patch("app.services.ml_service.THRESHOLD_PATH") as mp_th, \
             patch("app.services.ml_service.MLService._init_mlflow", return_value=None), \
             patch("tensorflow.keras.models.load_model", return_value=MagicMock()), \
             patch("pickle.load", side_effect=[fake_scaler, fake_threshold]), \
             patch("builtins.open", mock_open()):
            mp_ae.exists.return_value = True
            mp_sc.exists.return_value = True
            mp_th.exists.return_value = True
            mp_ae.__str__ = lambda self: "/fake/azure_lstm_autoencoder.keras"

            svc.load()

        assert svc._loaded is True
        assert svc._autoencoder is not None

    def test_ml_service_no_blob_download_when_local_missing(self):
        """When local files are missing, blob download is NOT attempted (stub disabled)."""
        from app.services.blob_storage_service import BlobStorageService
        blob_stub = BlobStorageService()

        svc = self._make_ml_service(blob_service=blob_stub)

        with patch("app.services.ml_service.AUTOENCODER_PATH") as mp_ae, \
             patch("app.services.ml_service.SCALER_PATH") as mp_sc, \
             patch("app.services.ml_service.THRESHOLD_PATH") as mp_th, \
             patch("app.services.ml_service.MLService._init_mlflow", return_value=None):
            mp_ae.exists.return_value = False
            mp_sc.exists.return_value = False
            mp_th.exists.return_value = False

            svc.load()

        # Stub is_available is False → no download attempted
        assert svc._loaded is True
        assert svc._autoencoder is None
        assert svc.is_ready is False

    def test_ml_service_graceful_no_local_no_blob(self):
        """MLService.load() completes without raising when local files absent."""
        svc = self._make_ml_service(blob_service=None)

        with patch("app.services.ml_service.AUTOENCODER_PATH") as mp_ae, \
             patch("app.services.ml_service.SCALER_PATH") as mp_sc, \
             patch("app.services.ml_service.THRESHOLD_PATH") as mp_th, \
             patch("app.services.ml_service.MLService._init_mlflow", return_value=None):
            mp_ae.exists.return_value = False
            mp_sc.exists.return_value = False
            mp_th.exists.return_value = False

            svc.load()  # must not raise

        assert svc._loaded is True
        assert svc._autoencoder is None
        assert svc.is_ready is False

    def test_ml_service_is_blob_available_false_for_stub(self):
        """is_blob_available is False when BlobStorageService stub is used."""
        from app.services.ml_service import MLService
        from app.services.blob_storage_service import BlobStorageService

        svc = MLService(blob_service=BlobStorageService())
        assert svc.is_blob_available is False

    def test_ml_service_is_blob_available_false_when_none(self):
        """is_blob_available is False when no blob_service provided."""
        from app.services.ml_service import MLService

        assert MLService(blob_service=None).is_blob_available is False
