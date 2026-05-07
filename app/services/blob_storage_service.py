"""
Azure Blob Storage has been removed from this project.
Models are stored locally in ml/models/.

This class is kept as a stub to maintain interface compatibility.
To restore cloud storage, replace this implementation with
Azure Blob Storage, AWS S3, or Hugging Face Hub.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger


class BlobStorageService:
    """
    Stub — blob storage is disabled.

    All methods are no-ops that return False/empty so the rest of the
    codebase continues to work without any changes.  is_available always
    returns False, which causes MLService to skip the blob fallback path
    and log a clear error when a local model file is missing.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Accept any arguments so existing call sites don't need to change,
        # but ignore them all.
        logger.info(
            "BlobStorageService: blob storage disabled — "
            "models are stored locally in ml/models/"
        )

    @property
    def is_available(self) -> bool:
        return False

    def upload_model(self, local_path: str | Path, blob_name: str) -> bool:
        logger.info(
            "BlobStorageService: upload skipped (blob storage disabled) — "
            "model saved locally only at '{}'", local_path
        )
        return False

    def download_model(self, blob_name: str, local_path: str | Path) -> bool:
        logger.info(
            "BlobStorageService: download skipped (blob storage disabled) — '{}'",
            blob_name,
        )
        return False

    def list_models(self) -> list[str]:
        return []

    def model_exists(self, blob_name: str) -> bool:
        return False
