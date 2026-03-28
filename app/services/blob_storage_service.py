"""
Azure Blob Storage service for DefectSense model versioning.

Provides upload/download/list/exists operations for ML model artefacts.
Degrades gracefully if AZURE_STORAGE_CONNECTION_STRING is not configured.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

try:
    from azure.storage.blob import BlobServiceClient
except ImportError:  # pragma: no cover
    BlobServiceClient = None  # type: ignore[assignment,misc]


class BlobStorageService:
    """
    Wraps Azure Blob Storage for ML model artefact management.

    If AZURE_STORAGE_CONNECTION_STRING is not set (or initialisation fails),
    all methods return False/empty without raising — the app continues locally.
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: str = "defectsense-models",
    ) -> None:
        self._connection_string = connection_string
        self._container_name = container_name
        self._client = None          # BlobServiceClient, lazily created
        self._available = False

        if not self._connection_string:
            logger.warning(
                "BlobStorageService: AZURE_STORAGE_CONNECTION_STRING not set — "
                "blob storage disabled; all operations will be no-ops."
            )
            return

        if BlobServiceClient is None:
            logger.warning("BlobStorageService: azure-storage-blob not installed — disabled.")
            return

        try:
            self._client = BlobServiceClient.from_connection_string(self._connection_string)
            # Light connectivity check — just instantiate the container client
            self._client.get_container_client(self._container_name)
            self._available = True
            logger.info(
                "BlobStorageService: connected to container '{}'", self._container_name
            )
        except Exception as exc:
            logger.warning("BlobStorageService: failed to initialise — {}", exc)

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Upload ─────────────────────────────────────────────────────────────────

    def upload_model(self, local_path: str | Path, blob_name: str) -> bool:
        """Upload a local file to Azure Blob. Log file size. Returns True on success."""
        if not self._available:
            logger.warning(
                "BlobStorageService: upload skipped (unavailable) — {}", blob_name
            )
            return False

        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning("BlobStorageService: local file not found — {}", local_path)
            return False

        try:
            file_size = local_path.stat().st_size
            container_client = self._client.get_container_client(self._container_name)
            with open(local_path, "rb") as data:
                container_client.upload_blob(
                    name=blob_name,
                    data=data,
                    overwrite=True,
                )
            logger.info(
                "BlobStorageService: uploaded '{}' → '{}' ({:.1f} KB)",
                local_path.name,
                blob_name,
                file_size / 1024,
            )
            return True
        except Exception as exc:
            logger.error(
                "BlobStorageService: upload failed for '{}' — {}", blob_name, exc
            )
            return False

    # ── Download ───────────────────────────────────────────────────────────────

    def download_model(self, blob_name: str, local_path: str | Path) -> bool:
        """Download blob to local path. Creates parent dirs if missing. Returns True on success."""
        if not self._available:
            logger.warning(
                "BlobStorageService: download skipped (unavailable) — {}", blob_name
            )
            return False

        local_path = Path(local_path)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            container_client = self._client.get_container_client(self._container_name)
            blob_client = container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())
            file_size = local_path.stat().st_size
            logger.info(
                "BlobStorageService: downloaded '{}' → '{}' ({:.1f} KB)",
                blob_name,
                local_path,
                file_size / 1024,
            )
            return True
        except Exception as exc:
            logger.error(
                "BlobStorageService: download failed for '{}' — {}", blob_name, exc
            )
            return False

    # ── List ───────────────────────────────────────────────────────────────────

    def list_models(self) -> list[str]:
        """Return list of all blob names in the container."""
        if not self._available:
            return []

        try:
            container_client = self._client.get_container_client(self._container_name)
            return [blob.name for blob in container_client.list_blobs()]
        except Exception as exc:
            logger.error("BlobStorageService: list_models failed — {}", exc)
            return []

    # ── Exists ─────────────────────────────────────────────────────────────────

    def model_exists(self, blob_name: str) -> bool:
        """Check if a blob exists without downloading. Returns True/False."""
        if not self._available:
            return False

        try:
            container_client = self._client.get_container_client(self._container_name)
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except Exception:
            return False
