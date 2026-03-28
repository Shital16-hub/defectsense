"""
Load AI4I 2020 sensor data into PostgreSQL (Supabase).

Idempotent — safe to run multiple times.
If rows already exist, the script exits without loading.

Run:
    python data/load_to_postgres.py
"""
import sys
from pathlib import Path

# Load .env from project root before anything else
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import os
import pandas as pd
from sqlalchemy import create_engine, text

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CSV_PATH = ROOT / "data" / "ai4i_2020.csv"

# ── Column mapping from original AI4I names ────────────────────────────────────
COL_MAP = {
    "Air temperature [K]":       "air_temperature",
    "Process temperature [K]":   "process_temperature",
    "Rotational speed [rpm]":    "rotational_speed",
    "Torque [Nm]":               "torque",
    "Tool wear [min]":           "tool_wear",
    "Machine failure":           "machine_failure",
    "Type":                      "machine_type",
    "UDI":                       "machine_id",
    "Product ID":                "machine_id",
}

# Failure type columns in original dataset
FAILURE_TYPE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id                  SERIAL PRIMARY KEY,
    machine_id          VARCHAR(20),
    air_temperature     FLOAT,
    process_temperature FLOAT,
    rotational_speed    FLOAT,
    torque              FLOAT,
    tool_wear           FLOAT,
    machine_failure     INTEGER,
    failure_type        VARCHAR(10),
    machine_type        VARCHAR(5),
    created_at          TIMESTAMPTZ DEFAULT NOW()
)
"""


def derive_failure_type(row: pd.Series) -> str:
    """Map boolean failure-type columns to a single label."""
    for col in FAILURE_TYPE_COLS:
        if col in row and row[col] == 1:
            return col
    return "NONE"


def main() -> None:
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        print("ERROR: POSTGRES_URL not set in .env")
        sys.exit(1)

    if not CSV_PATH.exists():
        print(f"ERROR: {CSV_PATH} not found — run python data/download_data.py first")
        sys.exit(1)

    print("Connecting to PostgreSQL...")
    engine = create_engine(
        postgres_url,
        connect_args={"sslmode": "require"},
    )

    # ── Create table ──────────────────────────────────────────────────────────
    with engine.connect() as conn:
        conn.execute(text(CREATE_TABLE_SQL))
        conn.commit()

    # ── Idempotency check ─────────────────────────────────────────────────────
    with engine.connect() as conn:
        existing = int(conn.execute(text("SELECT COUNT(*) FROM sensor_readings")).scalar())

    if existing > 0:
        print(f"Already loaded: {existing:,} rows exist — skipping")
        engine.dispose()
        return

    # ── Load CSV ──────────────────────────────────────────────────────────────
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # Rename columns
    df = df.rename(columns={k: v for k, v in COL_MAP.items() if k in df.columns})

    # Derive failure_type from boolean columns
    if all(c in df.columns for c in FAILURE_TYPE_COLS):
        df["failure_type"] = df.apply(derive_failure_type, axis=1)
    elif "failure_type" not in df.columns:
        df["failure_type"] = "NONE"

    # Ensure machine_id exists
    if "machine_id" not in df.columns:
        df["machine_id"] = df.get("UDI", pd.RangeIndex(len(df))).astype(str)

    # Keep only the columns the table expects
    keep_cols = [
        "machine_id", "air_temperature", "process_temperature",
        "rotational_speed", "torque", "tool_wear",
        "machine_failure", "failure_type", "machine_type",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    print(f"Loading {len(df):,} rows into sensor_readings (chunksize=500)...")
    df.to_sql(
        "sensor_readings",
        engine,
        if_exists="append",
        index=False,
        chunksize=500,
        method="multi",
    )

    # ── Report ─────────────────────────────────────────────────────────────────
    normal_count   = int((df["machine_failure"] == 0).sum())
    failure_count  = int((df["machine_failure"] == 1).sum())
    failure_rate   = failure_count / len(df) * 100

    print(f"Total rows loaded: {len(df):,}")
    print(f"Normal samples:    {normal_count:,}")
    print(f"Failure samples:   {failure_count:,}")
    print(f"Failure rate:      {failure_rate:.2f}%")
    print("POSTGRESQL LOAD: COMPLETE")

    engine.dispose()


if __name__ == "__main__":
    main()
