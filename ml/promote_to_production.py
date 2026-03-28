"""
DefectSense — MLflow Model Registry CLI  (MLflow 3.x aliases API)

Usage:
    python ml/promote_to_production.py --list
    python ml/promote_to_production.py --model lstm    --version 1
    python ml/promote_to_production.py --model iforest --version 1
    python ml/promote_to_production.py --model lstm    --rollback
    python ml/promote_to_production.py --model iforest --rollback
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from ml.model_registry_service import ModelRegistryService

MODEL_MAP = {
    "lstm":    "defectsense_lstm_autoencoder",
    "iforest": "defectsense_isolation_forest",
}


def _get_registry() -> ModelRegistryService:
    registry = ModelRegistryService()
    registry.init()
    if not registry.is_ready:
        print("ERROR: Could not connect to MLflow registry.")
        print("       Make sure mlruns/ exists and MLflow is configured.")
        sys.exit(1)
    return registry


def cmd_list(registry: ModelRegistryService) -> None:
    """Print table of all versions for both models."""
    top    = "+" + "-" * 33 + "+" + "-" * 9 + "+" + "-" * 15 + "+" + "-" * 11 + "+"
    header = "| {:<31} | {:^7} | {:<13} | {:<9} |".format(
        "Model", "Version", "Alias", "AUC"
    )
    sep    = top

    rows: list[tuple[str, str, str, str]] = []
    for _key, model_name in MODEL_MAP.items():
        versions = registry.get_all_versions(model_name)
        if not versions:
            rows.append((model_name, "-", "-", "-"))
            continue
        for v in sorted(versions, key=lambda x: x["version"]):
            alias_str = v.get("alias", "") or "-"
            auc_val   = v.get("auc")
            auc_str   = f"{float(auc_val):.4f}" if auc_val is not None else "-"
            rows.append((model_name, str(v["version"]), alias_str, auc_str))

    print(top)
    print(header)
    print(sep)
    for model, version, alias, auc in rows:
        print("| {:<31} | {:^7} | {:<13} | {:<9} |".format(model, version, alias, auc))
    print(top)


def cmd_promote(registry: ModelRegistryService, model_alias: str, version: int) -> None:
    model_name = MODEL_MAP.get(model_alias)
    if not model_name:
        print(f"ERROR: Unknown model alias '{model_alias}'. Use: lstm, iforest")
        sys.exit(1)

    print(f"Promoting {model_name} v{version} -> champion...")
    ok = registry.promote_to_production(model_name, version)
    if ok:
        print(f"SUCCESS: {model_name} v{version} is now champion.")
    else:
        print(f"FAILED: Could not promote {model_name} v{version}.")
        print("        Check that this version exists (use --list).")
        sys.exit(1)


def cmd_rollback(registry: ModelRegistryService, model_alias: str) -> None:
    model_name = MODEL_MAP.get(model_alias)
    if not model_name:
        print(f"ERROR: Unknown model alias '{model_alias}'. Use: lstm, iforest")
        sys.exit(1)

    print(f"Rolling back {model_name}...")
    ok = registry.rollback(model_name)
    if ok:
        print(f"SUCCESS: {model_name} rolled back to previous version.")
    else:
        print(f"FAILED: Could not roll back {model_name}.")
        print("        Either no champion version exists, or already at v1.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DefectSense — MLflow Model Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ml/promote_to_production.py --list
  python ml/promote_to_production.py --model lstm    --version 1
  python ml/promote_to_production.py --model iforest --version 1
  python ml/promote_to_production.py --model lstm    --rollback
        """,
    )
    parser.add_argument("--list",     action="store_true", help="List all versions")
    parser.add_argument("--model",    choices=list(MODEL_MAP.keys()), help="Model: lstm or iforest")
    parser.add_argument("--version",  type=int, help="Version number to promote")
    parser.add_argument("--rollback", action="store_true", help="Roll back champion to previous version")

    args = parser.parse_args()

    if not (args.list or args.model):
        parser.print_help()
        sys.exit(0)

    registry = _get_registry()

    if args.list:
        cmd_list(registry)
        return

    if args.model:
        if args.rollback:
            cmd_rollback(registry, args.model)
        elif args.version is not None:
            cmd_promote(registry, args.model, args.version)
        else:
            print("ERROR: Specify --version N or --rollback with --model.")
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
