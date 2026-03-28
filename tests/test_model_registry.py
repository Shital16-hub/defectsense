"""
Unit tests for ModelRegistryService (MLflow 3.x aliases API).

All tests are fully offline — MlflowClient is mocked throughout.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_model_version(
    version: int,
    aliases: list[str] | None = None,
    run_id: str = "run123",
) -> MagicMock:
    mv = MagicMock()
    mv.version            = str(version)
    mv.run_id             = run_id
    mv.aliases            = aliases or []
    mv.creation_timestamp = 1700000000000
    return mv


def _make_run(metrics: dict | None = None, params: dict | None = None) -> MagicMock:
    run = MagicMock()
    run.data.metrics = metrics or {}
    run.data.params  = params  or {}
    return run


def _patched_registry(client_mock: MagicMock):
    """Return a ModelRegistryService with _client replaced by client_mock."""
    from ml.model_registry_service import ModelRegistryService
    svc = ModelRegistryService()
    svc._client = client_mock
    svc._ready  = True
    return svc


# ── Initialisation ─────────────────────────────────────────────────────────────

class TestModelRegistryInit:
    def test_registry_service_initializes(self):
        """ModelRegistryService() must not crash."""
        from ml.model_registry_service import ModelRegistryService
        svc = ModelRegistryService()
        assert svc is not None
        assert svc.is_ready is False

    def test_init_sets_ready_true(self):
        """After init() with a healthy MLflow, is_ready should be True."""
        from ml.model_registry_service import ModelRegistryService

        mock_client = MagicMock()
        mock_client.search_registered_models.return_value = []

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client), \
             patch("mlflow.set_tracking_uri"):
            svc = ModelRegistryService()
            svc.init()

        assert svc.is_ready is True

    def test_graceful_when_mlflow_unavailable(self):
        """All methods return None/False/empty when MLflow is unavailable."""
        from ml.model_registry_service import ModelRegistryService

        with patch("mlflow.tracking.MlflowClient", side_effect=Exception("no mlflow")):
            svc = ModelRegistryService()
            svc.init()

        assert svc.is_ready is False
        assert svc.get_latest_version("any_model")       is None
        assert svc.get_all_versions("any_model")         == []
        assert svc.promote_to_production("any_model", 1) is False
        assert svc.rollback("any_model")                 is False
        assert svc.compare_versions("any_model", 1, 2)  == {}


# ── Query ──────────────────────────────────────────────────────────────────────

class TestGetLatestVersion:
    def test_returns_none_when_alias_not_found(self):
        """get_model_version_by_alias raising → None, no crash."""
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")

        svc = _patched_registry(client)
        result = svc.get_latest_version("defectsense_lstm_autoencoder", stage="Production")

        assert result is None

    def test_returns_dict_when_champion_exists(self):
        """champion alias found → structured dict with alias='champion'."""
        client = MagicMock()
        mv     = _make_model_version(1, aliases=["champion"], run_id="run-abc")
        client.get_model_version_by_alias.return_value = mv
        client.get_run.return_value = _make_run(metrics={"auc": 0.93})

        svc = _patched_registry(client)
        result = svc.get_latest_version("defectsense_isolation_forest", stage="Production")

        assert result is not None
        assert result["version"] == 1
        assert result["alias"]   == "champion"
        assert result["run_id"]  == "run-abc"
        assert result["metrics"]["auc"] == 0.93

    def test_stage_production_maps_to_champion_alias(self):
        """stage='Production' must query the 'champion' alias."""
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("not found")

        svc = _patched_registry(client)
        svc.get_latest_version("defectsense_isolation_forest", stage="Production")

        client.get_model_version_by_alias.assert_called_once_with(
            "defectsense_isolation_forest", "champion"
        )

    def test_stage_staging_maps_to_challenger_alias(self):
        """stage='Staging' must query the 'challenger' alias."""
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("not found")

        svc = _patched_registry(client)
        svc.get_latest_version("defectsense_isolation_forest", stage="Staging")

        client.get_model_version_by_alias.assert_called_once_with(
            "defectsense_isolation_forest", "challenger"
        )


class TestGetAllVersions:
    def test_returns_list_of_two_versions(self):
        """Two model versions in registry → list of 2 dicts."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=["champion"],   run_id="run-1")
        v2 = _make_model_version(2, aliases=["challenger"], run_id="run-2")
        client.search_model_versions.return_value = [v1, v2]
        client.get_run.side_effect = [
            _make_run(metrics={"auc": 0.92}),
            _make_run(metrics={"auc": 0.95}),
        ]

        svc = _patched_registry(client)
        result = svc.get_all_versions("defectsense_isolation_forest")

        assert len(result) == 2
        assert result[0]["version"] == 1
        assert result[1]["version"] == 2

    def test_version_alias_field_is_first_alias(self):
        """alias field should be the first item in the aliases list."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=["champion"])
        client.search_model_versions.return_value = [v1]
        client.get_run.return_value = _make_run(metrics={"auc": 0.93})

        svc = _patched_registry(client)
        result = svc.get_all_versions("defectsense_isolation_forest")

        assert result[0]["alias"]   == "champion"
        assert result[0]["aliases"] == ["champion"]

    def test_auc_resolved_from_metrics(self):
        """get_all_versions should populate top-level 'auc' from run metrics."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=["champion"])
        client.search_model_versions.return_value = [v1]
        client.get_run.return_value = _make_run(metrics={"auc": 0.929})

        svc = _patched_registry(client)
        result = svc.get_all_versions("defectsense_isolation_forest")

        assert result[0]["auc"] == pytest.approx(0.929)

    def test_auc_falls_back_to_roc_auc_key(self):
        """If 'auc' key missing, try 'roc_auc'."""
        client = MagicMock()
        v1 = _make_model_version(1)
        client.search_model_versions.return_value = [v1]
        client.get_run.return_value = _make_run(metrics={"roc_auc": 0.85})

        svc = _patched_registry(client)
        result = svc.get_all_versions("defectsense_isolation_forest")

        assert result[0]["auc"] == pytest.approx(0.85)

    def test_auc_is_none_when_no_auc_metric(self):
        """auc field is None when no recognised AUC key is present in metrics."""
        client = MagicMock()
        v1 = _make_model_version(1)
        client.search_model_versions.return_value = [v1]
        client.get_run.return_value = _make_run(metrics={"precision": 0.5})

        svc = _patched_registry(client)
        result = svc.get_all_versions("defectsense_isolation_forest")

        assert result[0]["auc"] is None

    def test_version_with_no_alias(self):
        """Version with no aliases → alias='' and aliases=[]."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=[])
        client.search_model_versions.return_value = [v1]
        client.get_run.return_value = _make_run()

        svc = _patched_registry(client)
        result = svc.get_all_versions("defectsense_isolation_forest")

        assert result[0]["alias"]   == ""
        assert result[0]["aliases"] == []

    def test_returns_empty_list_on_exception(self):
        """MLflow error → empty list, no crash."""
        client = MagicMock()
        client.search_model_versions.side_effect = Exception("registry error")

        svc = _patched_registry(client)
        result = svc.get_all_versions("any_model")

        assert result == []


# ── Promote ────────────────────────────────────────────────────────────────────

class TestPromoteToProduction:
    def test_sets_champion_alias_on_new_version(self):
        """Promote v2 → set_registered_model_alias called with alias='champion', version='2'."""
        client = MagicMock()
        # No existing champion
        client.get_model_version_by_alias.side_effect = Exception("no champion")

        svc = _patched_registry(client)
        svc.promote_to_production("defectsense_isolation_forest", version=2)

        client.set_registered_model_alias.assert_called_once_with(
            name="defectsense_isolation_forest",
            alias="champion",
            version="2",
        )

    def test_removes_champion_from_previous_holder(self):
        """When v1 is champion, promoting v2 should delete 'champion' alias first."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=["champion"])
        client.get_model_version_by_alias.return_value = v1  # v1 is current champion

        svc = _patched_registry(client)
        svc.promote_to_production("defectsense_isolation_forest", version=2)

        client.delete_registered_model_alias.assert_called_once_with(
            "defectsense_isolation_forest", "champion"
        )

    def test_no_delete_if_promoting_current_champion(self):
        """If the target version is already champion, skip delete."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=["champion"])
        client.get_model_version_by_alias.return_value = v1

        svc = _patched_registry(client)
        svc.promote_to_production("defectsense_isolation_forest", version=1)

        client.delete_registered_model_alias.assert_not_called()

    def test_returns_true_on_success(self):
        """Successful promote → True."""
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("no champion")

        svc = _patched_registry(client)
        result = svc.promote_to_production("defectsense_isolation_forest", version=1)

        assert result is True

    def test_returns_false_on_exception(self):
        """MLflow error during set_registered_model_alias → False, no crash."""
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("no champion")
        client.set_registered_model_alias.side_effect = Exception("mlflow down")

        svc = _patched_registry(client)
        result = svc.promote_to_production("defectsense_isolation_forest", version=1)

        assert result is False


# ── Rollback ───────────────────────────────────────────────────────────────────

class TestRollback:
    def test_moves_champion_to_previous_version(self):
        """v2 champion → rollback → champion moves to v1."""
        client = MagicMock()
        v2 = _make_model_version(2, aliases=["champion"])
        client.get_model_version_by_alias.return_value = v2
        client.get_model_version.return_value = _make_model_version(1)  # v1 exists

        svc = _patched_registry(client)
        result = svc.rollback("defectsense_isolation_forest")

        assert result is True
        # delete current champion
        client.delete_registered_model_alias.assert_called_once_with(
            "defectsense_isolation_forest", "champion"
        )
        # set champion on v1
        client.set_registered_model_alias.assert_called_once_with(
            name="defectsense_isolation_forest",
            alias="champion",
            version="1",
        )

    def test_returns_false_when_already_at_v1(self):
        """If current champion is v1, rollback returns False (can't go further back)."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=["champion"])
        client.get_model_version_by_alias.return_value = v1

        svc = _patched_registry(client)
        result = svc.rollback("defectsense_isolation_forest")

        assert result is False

    def test_returns_false_when_no_champion(self):
        """No champion alias → rollback returns False gracefully."""
        client = MagicMock()
        client.get_model_version_by_alias.side_effect = Exception("no champion")

        svc = _patched_registry(client)
        result = svc.rollback("defectsense_isolation_forest")

        assert result is False

    def test_returns_false_when_previous_version_missing(self):
        """champion=v3, but v2 was deleted → rollback returns False."""
        client = MagicMock()
        v3 = _make_model_version(3, aliases=["champion"])
        client.get_model_version_by_alias.return_value = v3
        client.get_model_version.side_effect = Exception("v2 not found")

        svc = _patched_registry(client)
        result = svc.rollback("defectsense_isolation_forest")

        assert result is False


# ── Compare ────────────────────────────────────────────────────────────────────

class TestCompareVersions:
    def test_returns_comparison_dict(self):
        """compare_versions returns dict with version1, version2, better_version."""
        client = MagicMock()
        v1 = _make_model_version(1, aliases=["champion"],   run_id="run-1")
        v2 = _make_model_version(2, aliases=["challenger"], run_id="run-2")

        client.search_model_versions.side_effect = [[v1], [v2]]
        client.get_run.side_effect = [
            _make_run(metrics={"auc": 0.88}),
            _make_run(metrics={"auc": 0.93}),
        ]

        svc = _patched_registry(client)
        result = svc.compare_versions("defectsense_isolation_forest", 1, 2)

        assert "version1"       in result
        assert "version2"       in result
        assert "better_version" in result

    def test_better_version_based_on_auc(self):
        """better_version picks the one with higher AUC."""
        client = MagicMock()
        v1 = _make_model_version(1, run_id="run-1")
        v2 = _make_model_version(2, run_id="run-2")

        client.search_model_versions.side_effect = [[v1], [v2]]
        client.get_run.side_effect = [
            _make_run(metrics={"auc": 0.88}),
            _make_run(metrics={"auc": 0.95}),
        ]

        svc = _patched_registry(client)
        result = svc.compare_versions("defectsense_isolation_forest", 1, 2)

        assert result["better_version"] == 2

    def test_returns_empty_dict_when_version_missing(self):
        """If one version is not found → empty dict, no crash."""
        client = MagicMock()
        client.search_model_versions.return_value = []

        svc = _patched_registry(client)
        result = svc.compare_versions("defectsense_isolation_forest", 1, 2)

        assert result == {}
