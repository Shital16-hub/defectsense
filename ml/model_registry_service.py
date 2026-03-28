"""
MLflow Model Registry Service for DefectSense — MLflow 3.x aliases API.

Uses aliases instead of deprecated stages:
  "champion"   → production-ready model (was "Production")
  "challenger" → candidate model       (was "Staging")
  (no alias)   → archived / untagged   (was "Archived")

Never crashes — all methods degrade gracefully.
"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger


# Internal alias mapping for the public stage-named API
_STAGE_TO_ALIAS = {
    "Production": "champion",
    "Staging":    "challenger",
}


class ModelRegistryService:
    """MLflow Model Registry client — MLflow 3.x aliases API."""

    def __init__(self, tracking_uri: str = None) -> None:
        self._tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "sqlite:///mlruns/mlflow.db"
        )
        self._client = None
        self._ready  = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def init(self) -> None:
        """Connect to MLflow. Never raises."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(self._tracking_uri)
            self._client = MlflowClient(self._tracking_uri)
            # Smoke test — returns [] if no models registered yet
            self._client.search_registered_models()
            self._ready = True
            logger.info("ModelRegistry: connected to {}", self._tracking_uri)
        except Exception as exc:
            logger.warning("ModelRegistry: connection failed — {}", exc)
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready and self._client is not None

    # ── Query ──────────────────────────────────────────────────────────────────

    def get_latest_version(
        self,
        model_name: str,
        stage: str = "Production",
    ) -> Optional[dict]:
        """
        Return the version carrying the alias mapped from `stage`, or None.

        stage="Production" → alias "champion"
        stage="Staging"    → alias "challenger"
        """
        if not self.is_ready:
            return None
        alias = _STAGE_TO_ALIAS.get(stage, stage.lower())
        try:
            mv  = self._client.get_model_version_by_alias(model_name, alias)
            run = self._safe_get_run(mv.run_id)
            return {
                "version":    int(mv.version),
                "alias":      alias,
                "run_id":     mv.run_id,
                "metrics":    run.data.metrics if run else {},
                "params":     run.data.params  if run else {},
                "created_at": str(mv.creation_timestamp),
            }
        except Exception as exc:
            # MlflowException("RESOURCE_DOES_NOT_EXIST") when alias not set
            logger.debug("ModelRegistry.get_latest_version: no '{}' alias for {} — {}", alias, model_name, exc)
            return None

    def get_all_versions(self, model_name: str) -> list[dict]:
        """Return all registered versions with their current aliases and metrics."""
        if not self.is_ready:
            return []
        try:
            versions = self._client.search_model_versions(f"name='{model_name}'")

            # Build version → aliases map by probing known aliases explicitly.
            # search_model_versions() does not populate mv.aliases in some MLflow
            # builds, so we resolve them via get_model_version_by_alias().
            alias_map: dict[str, list[str]] = {}
            for alias in ("champion", "challenger"):
                try:
                    mv_a = self._client.get_model_version_by_alias(model_name, alias)
                    alias_map.setdefault(mv_a.version, []).append(alias)
                except Exception:
                    pass

            result = []
            for mv in versions:
                run = self._safe_get_run(mv.run_id)
                # Prefer aliases from the model version object; fall back to the map
                aliases = list(mv.aliases) if mv.aliases else alias_map.get(mv.version, [])
                metrics = run.data.metrics if run else {}
                result.append({
                    "version":    int(mv.version),
                    "aliases":    aliases,
                    "alias":      aliases[0] if aliases else "",
                    "run_id":     mv.run_id,
                    "metrics":    metrics,
                    "auc":        ModelRegistryService._resolve_auc(metrics),
                    "params":     run.data.params if run else {},
                    "created_at": str(mv.creation_timestamp),
                })
            return result
        except Exception as exc:
            logger.warning("ModelRegistry.get_all_versions failed: {}", exc)
            return []

    # ── Lifecycle management ───────────────────────────────────────────────────

    def promote_to_production(self, model_name: str, version: int) -> bool:
        """
        Set 'champion' alias on the given version.
        Removes 'champion' from the current holder first (if any).
        """
        if not self.is_ready:
            return False
        try:
            # Remove champion from current holder if one exists
            try:
                current = self._client.get_model_version_by_alias(model_name, "champion")
                if int(current.version) != version:
                    self._client.delete_registered_model_alias(model_name, "champion")
                    logger.info(
                        "ModelRegistry: removed 'champion' from {} v{}",
                        model_name, current.version,
                    )
            except Exception:
                pass  # No current champion — that's fine

            # Set champion on target version
            self._client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=str(version),
            )
            logger.info("Model promoted: {} v{} → champion", model_name, version)
            return True
        except Exception as exc:
            logger.warning("ModelRegistry.promote_to_production failed: {}", exc)
            return False

    def rollback(self, model_name: str) -> bool:
        """
        Move 'champion' alias from current version to (current_version - 1).
        Returns False if current version is already v1 (no previous version).
        """
        if not self.is_ready:
            return False
        try:
            # Find current champion
            try:
                current = self._client.get_model_version_by_alias(model_name, "champion")
            except Exception:
                logger.warning("ModelRegistry.rollback: no 'champion' alias for {}", model_name)
                return False

            current_ver = int(current.version)
            if current_ver <= 1:
                logger.warning(
                    "ModelRegistry.rollback: {} v{} is already the first version — cannot roll back",
                    model_name, current_ver,
                )
                return False

            prev_ver = current_ver - 1

            # Verify previous version exists
            try:
                self._client.get_model_version(model_name, str(prev_ver))
            except Exception:
                logger.warning(
                    "ModelRegistry.rollback: {} v{} does not exist",
                    model_name, prev_ver,
                )
                return False

            # Move champion: remove from current, set on previous
            self._client.delete_registered_model_alias(model_name, "champion")
            self._client.set_registered_model_alias(
                name=model_name,
                alias="champion",
                version=str(prev_ver),
            )
            logger.info(
                "Rollback: {} v{} → v{}", model_name, current_ver, prev_ver
            )
            return True
        except Exception as exc:
            logger.warning("ModelRegistry.rollback failed: {}", exc)
            return False

    def compare_versions(
        self, model_name: str, version1: int, version2: int
    ) -> dict:
        """Compare metrics between two versions. Returns empty dict on failure."""
        if not self.is_ready:
            return {}
        try:
            def _get_version_info(version: int) -> Optional[dict]:
                mvs = self._client.search_model_versions(
                    f"name='{model_name}' AND version_number={version}"
                )
                if not mvs:
                    return None
                mv  = mvs[0]
                run = self._safe_get_run(mv.run_id)
                return {
                    "version": int(mv.version),
                    "metrics": run.data.metrics if run else {},
                }

            v1_info = _get_version_info(version1)
            v2_info = _get_version_info(version2)

            if v1_info is None or v2_info is None:
                return {}

            auc1 = v1_info["metrics"].get("auc", None)
            auc2 = v2_info["metrics"].get("auc", None)

            if auc1 is not None and auc2 is not None:
                better = version1 if float(auc1) >= float(auc2) else version2
            else:
                better = version1

            return {
                "version1":       v1_info,
                "version2":       v2_info,
                "better_version": better,
            }
        except Exception as exc:
            logger.warning("ModelRegistry.compare_versions failed: {}", exc)
            return {}

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _safe_get_run(self, run_id: str):
        """Return MLflow Run object or None on failure."""
        if not run_id or not self._client:
            return None
        try:
            return self._client.get_run(run_id)
        except Exception:
            return None

    @staticmethod
    def _resolve_auc(metrics: dict):
        """Return the first AUC value found across multiple possible key names."""
        for key in ("auc", "roc_auc", "anomaly_auc", "val_auc"):
            val = metrics.get(key)
            if val is not None:
                return val
        return None
