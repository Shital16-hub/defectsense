"""
Download the AI4I 2020 Predictive Maintenance dataset from UCI.

Run:
    python data/download_data.py
"""
import sys
import urllib.request
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent
OUTPUT_PATH = DATA_DIR / "ai4i_2020.csv"

# UCI ML Repository direct link
UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
)

COLUMN_RENAMES = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "machine_type",
    "Air temperature [K]": "air_temperature",
    "Process temperature [K]": "process_temperature",
    "Rotational speed [rpm]": "rotational_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}


def download() -> pd.DataFrame:
    print(f"Downloading AI4I 2020 dataset from UCI...")
    print(f"  URL : {UCI_URL}")
    print(f"  Dest: {OUTPUT_PATH}\n")

    try:
        urllib.request.urlretrieve(UCI_URL, OUTPUT_PATH)
        print("Download complete.")
    except Exception as exc:
        print(f"Download failed: {exc}")
        print(
            "\nManual download instructions:\n"
            "  1. Visit https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset\n"
            "  2. Download the CSV and save it to:  data/ai4i_2020.csv\n"
            "  3. Re-run this script to validate."
        )
        sys.exit(1)

    return pd.read_csv(OUTPUT_PATH)


def validate_and_report(df: pd.DataFrame) -> None:
    # Normalise column names
    df = df.rename(columns=COLUMN_RENAMES)
    df.to_csv(OUTPUT_PATH, index=False)  # re-save with clean names

    print("=" * 55)
    print("  AI4I 2020 Dataset — Summary")
    print("=" * 55)
    print(f"  Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns     : {list(df.columns)}")
    print(f"  Memory      : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print()

    # Failure type distribution
    failure_cols = ["twf", "hdf", "pwf", "osf", "rnf"]
    failure_labels = {
        "twf": "Tool Wear Failure   (TWF)",
        "hdf": "Heat Dissipation    (HDF)",
        "pwf": "Power Failure       (PWF)",
        "osf": "Overstrain Failure  (OSF)",
        "rnf": "Random Failure      (RNF)",
    }
    print("  Failure Type Distribution:")
    total_failures = df["machine_failure"].sum()
    print(f"    Total failures  : {total_failures} / {len(df)} ({total_failures/len(df)*100:.1f}%)")
    for col in failure_cols:
        if col in df.columns:
            n = int(df[col].sum())
            print(f"    {failure_labels[col]}: {n}")

    print()
    print("  Machine Type Distribution:")
    if "machine_type" in df.columns:
        for t, count in df["machine_type"].value_counts().items():
            print(f"    Type {t}: {count:,}")

    print()
    print("  Sensor Statistics (normal samples only):")
    normal = df[df["machine_failure"] == 0]
    sensor_cols = ["air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    for col in sensor_cols:
        if col in normal.columns:
            print(
                f"    {col:<24}: mean={normal[col].mean():.2f}  std={normal[col].std():.2f}"
                f"  min={normal[col].min():.1f}  max={normal[col].max():.1f}"
            )
    print("=" * 55)
    print(f"\nDataset saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    df = download()
    validate_and_report(df)
