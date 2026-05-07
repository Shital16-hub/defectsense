"""
Verify that the Azure Predictive Maintenance dataset files are present.

Run:
    python data/download_data.py
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "azure_pdm"

REQUIRED_FILES = [
    "PdM_telemetry.csv",
    "PdM_failures.csv",
    "PdM_machines.csv",
    "PdM_errors.csv",
    "PdM_maint.csv",
]


def count_rows(path: Path) -> int:
    """Return data row count (total lines minus header)."""
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def main() -> None:
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]

    if missing:
        print("ERROR: The following Azure PdM files are missing from data/azure_pdm/:")
        for f in missing:
            print(f"  - {f}")
        print()
        print(
            "Please download the Microsoft Azure Predictive Maintenance dataset from\n"
            "  https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance\n"
            "and place all CSV files in the data/azure_pdm/ directory."
        )
        sys.exit(1)

    print("=" * 55)
    print("  Azure PdM Dataset — Files Present")
    print("=" * 55)
    print(f"  {'Filename':<28}  {'Rows':>10}")
    print(f"  {'-'*28}  {'-'*10}")
    for filename in REQUIRED_FILES:
        rows = count_rows(DATA_DIR / filename)
        print(f"  {filename:<28}  {rows:>10,}")
    print("=" * 55)
    print("\nAll required files are present.")
    sys.exit(0)


if __name__ == "__main__":
    main()
