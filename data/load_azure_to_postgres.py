"""
load_azure_to_postgres.py
─────────────────────────
Loads Azure PdM telemetry data into the PostgreSQL azure_sensor_readings table.

Steps
  1. Load PdM_telemetry.csv, PdM_failures.csv, PdM_machines.csv
  2. Calculate per-machine baselines and precursor windows → derive SEQUENCE_LENGTH
  3. Save SEQUENCE_LENGTH + metadata to data/azure_lstm_config.json
  4. Mark each telemetry row as clean_normal or not
  5. Create azure_sensor_readings table (idempotent)
  6. Load all rows (with failure labels, model/age join, clean flag)
  7. Idempotency check — skip if table already has data
  8. Print final summary

Usage
  python data/load_azure_to_postgres.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent          # project root
DATA_DIR = ROOT / "data" / "azure_pdm"
CONFIG_OUT = ROOT / "data" / "azure_lstm_config.json"

load_dotenv(ROOT / ".env")

POSTGRES_URL = os.getenv("POSTGRES_URL", "").strip()

# ── Constants ─────────────────────────────────────────────────────────────────

SENSORS       = ["volt", "rotate", "pressure", "vibration"]
Z_THRESHOLD   = 1.5
LOOKBACK_HRS  = 200      # maximum window to search for precursor signal
SAFE_GAP_HRS  = 200      # rows must be this far from any failure to count as baseline
CHUNK_SIZE    = 1000
PRINT_EVERY   = 50_000

TABLE_NAME = "azure_sensor_readings"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id              SERIAL PRIMARY KEY,
    machine_id      VARCHAR(20)  NOT NULL,
    datetime        TIMESTAMPTZ  NOT NULL,
    volt            FLOAT,
    rotate          FLOAT,
    pressure        FLOAT,
    vibration       FLOAT,
    machine_failure INTEGER      NOT NULL DEFAULT 0,
    failure_type    VARCHAR(20),
    machine_model   VARCHAR(20),
    machine_age     INTEGER,
    is_clean_normal BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_machine_id
    ON {TABLE_NAME} (machine_id);
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_datetime
    ON {TABLE_NAME} (datetime);
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_machine_datetime
    ON {TABLE_NAME} (machine_id, datetime);
"""


# ── STEP 1 — Load CSVs ────────────────────────────────────────────────────────

def load_csvs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n[STEP 1] Loading CSV files...")

    tel  = pd.read_csv(DATA_DIR / "PdM_telemetry.csv",  parse_dates=["datetime"])
    fail = pd.read_csv(DATA_DIR / "PdM_failures.csv",   parse_dates=["datetime"])
    mach = pd.read_csv(DATA_DIR / "PdM_machines.csv")

    tel  = tel.sort_values(["machineID", "datetime"]).reset_index(drop=True)
    fail = fail.sort_values(["machineID", "datetime"]).reset_index(drop=True)

    print(f"  telemetry  : {len(tel):>9,} rows  |  {tel['machineID'].nunique()} machines")
    print(f"  failures   : {len(fail):>9,} rows  |  {fail['machineID'].nunique()} machines with failures")
    print(f"  machines   : {len(mach):>9,} rows  |  models: {sorted(mach['model'].unique())}")
    return tel, fail, mach


# ── STEP 2 — Precursor window calculation ─────────────────────────────────────

def _machine_baseline(
    sub: pd.DataFrame,
    fail_times: np.ndarray,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Per-machine baseline using only rows > SAFE_GAP_HRS from every failure."""
    dt_vals = sub["datetime"].values
    clean_mask = np.ones(len(sub), dtype=bool)
    for ft in fail_times:
        diff = np.abs((dt_vals - ft).astype("timedelta64[h]").astype(float))
        clean_mask &= diff > SAFE_GAP_HRS
    clean = sub[clean_mask][SENSORS]
    if len(clean) < 10:
        return None, None
    return clean.mean(), clean.std()


def calculate_precursor_windows(
    tel: pd.DataFrame,
    fail: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    Returns
    -------
    fail_with_pw : failures DataFrame with added column `precursor_hrs`
                   (hours before failure where first sensor crossed Z_THRESHOLD;
                    NaN if no crossing detected in LOOKBACK_HRS window)
    sequence_length : median precursor_hrs rounded up to nearest integer
    """
    print("\n[STEP 2] Calculating per-machine baselines and precursor windows...")

    tel_by_machine = {
        mid: grp.reset_index(drop=True)
        for mid, grp in tel.groupby("machineID")
    }

    # Per-machine baselines
    baselines: dict[int, tuple] = {}
    n_invalid = 0
    for mid, sub in tel_by_machine.items():
        ft_arr = fail[fail["machineID"] == mid]["datetime"].values
        bm, bs = _machine_baseline(sub, ft_arr)
        baselines[mid] = (bm, bs)
        if bm is None:
            n_invalid += 1
            print(f"  WARNING: machine {mid} has insufficient clean baseline rows")

    print(f"  Baselines computed: {len(baselines) - n_invalid}/{len(baselines)} valid")

    # Per-failure precursor window
    precursor_hrs_list: list[float | None] = []

    for _, row in fail.iterrows():
        mid   = int(row["machineID"])
        ft    = row["datetime"]
        bm, bs = baselines.get(mid, (None, None))

        if bm is None:
            precursor_hrs_list.append(None)
            continue

        sub = tel_by_machine[mid]
        mask = (
            (sub["datetime"] > ft - pd.Timedelta(hours=LOOKBACK_HRS)) &
            (sub["datetime"] <= ft)
        )
        window = sub[mask].copy().sort_values("datetime")

        if len(window) == 0:
            precursor_hrs_list.append(None)
            continue

        window["_hrs_before"] = (
            (window["datetime"] - ft).dt.total_seconds() / 3600
        ).round().astype(int)

        z_exceeded = pd.Series(False, index=window.index)
        for s in SENSORS:
            std = float(bs[s]) if float(bs[s]) > 0 else 1e-9
            z = (window[s] - float(bm[s])) / std
            z_exceeded |= z.abs() > Z_THRESHOLD

        first_exceed = window[z_exceeded]
        if len(first_exceed) == 0:
            precursor_hrs_list.append(None)
        else:
            precursor_hrs_list.append(abs(int(first_exceed["_hrs_before"].min())))

    fail_pw = fail.copy()
    fail_pw["precursor_hrs"] = precursor_hrs_list

    valid_pw = [v for v in precursor_hrs_list if v is not None]
    n_total  = len(precursor_hrs_list)
    n_valid  = len(valid_pw)

    median_pw       = float(np.median(valid_pw))
    sequence_length = int(np.ceil(median_pw))

    print(f"  Failure events analysed  : {n_total}")
    print(f"  Events with detectable precursor : {n_valid} ({100*n_valid/n_total:.1f}%)")
    print(f"  Median precursor window  : {median_pw:.1f} hrs")
    print(f"  SEQUENCE_LENGTH          : {sequence_length}")

    return fail_pw, sequence_length


# ── STEP 3 — Save config JSON ─────────────────────────────────────────────────

def save_config(fail_pw: pd.DataFrame, sequence_length: int) -> None:
    print(f"\n[STEP 3] Saving LSTM config to {CONFIG_OUT.relative_to(ROOT)}...")

    valid_pw = fail_pw["precursor_hrs"].dropna()
    config = {
        "sequence_length"       : sequence_length,
        "calculated_from"       : "azure_pdm",
        "total_failure_events"  : int(len(fail_pw)),
        "median_precursor_hours": float(valid_pw.median()),
        "mean_precursor_hours"  : float(valid_pw.mean()),
        "min_precursor_hours"   : int(valid_pw.min()),
        "max_precursor_hours"   : int(valid_pw.max()),
        "z_threshold_used"      : Z_THRESHOLD,
        "lookback_hrs_used"     : LOOKBACK_HRS,
        "calculation_date"      : datetime.now(timezone.utc).isoformat(),
    }

    with open(CONFIG_OUT, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved: sequence_length={sequence_length}")


# ── STEP 4 — Mark clean normal rows ───────────────────────────────────────────

def build_full_dataframe(
    tel: pd.DataFrame,
    fail: pd.DataFrame,
    mach: pd.DataFrame,
    sequence_length: int,
) -> pd.DataFrame:
    """
    Returns the full telemetry DataFrame with added columns:
      machine_failure, failure_type, machine_model, machine_age, is_clean_normal
    """
    print(f"\n[STEP 4] Marking clean normal rows (exclusion zone = {sequence_length} hrs)...")

    # --- failure labels: machine_failure flag + failure_type ----------------
    # Some machines have multiple failure types at the same timestamp → join
    # on (machineID, datetime); keep all failure types (unlikely but safe)
    fail_agg = (
        fail.groupby(["machineID", "datetime"])["failure"]
        .apply(lambda x: ",".join(sorted(x)))
        .reset_index()
        .rename(columns={"failure": "failure_type"})
    )
    fail_agg["machine_failure"] = 1

    tel = tel.merge(
        fail_agg,
        on=["machineID", "datetime"],
        how="left",
    )
    tel["machine_failure"] = tel["machine_failure"].fillna(0).astype(int)
    tel["failure_type"]    = tel["failure_type"].fillna("").astype(str)

    # --- machine metadata ---------------------------------------------------
    tel = tel.merge(
        mach.rename(columns={"model": "machine_model", "age": "machine_age"}),
        on="machineID",
        how="left",
    )

    # --- is_clean_normal flag -----------------------------------------------
    # A row is clean-normal if it is NOT within [sequence_length] hours
    # before (or at) any failure event for that machine.
    tel["is_clean_normal"] = True

    fail_by_machine = {
        mid: grp["datetime"].values
        for mid, grp in fail.groupby("machineID")
    }

    for mid, grp_idx in tel.groupby("machineID").groups.items():
        ft_arr   = fail_by_machine.get(mid, np.array([], dtype="datetime64[ns]"))
        dt_vals  = tel.loc[grp_idx, "datetime"].values
        dirty    = np.zeros(len(grp_idx), dtype=bool)
        for ft in ft_arr:
            diff = (dt_vals - ft).astype("timedelta64[h]").astype(float)
            dirty |= (diff >= -sequence_length) & (diff <= 0)
        tel.loc[grp_idx[dirty], "is_clean_normal"] = False

    n_clean     = tel["is_clean_normal"].sum()
    n_precursor = (~tel["is_clean_normal"] & (tel["machine_failure"] == 0)).sum()
    n_failure   = (tel["machine_failure"] == 1).sum()

    print(f"  Clean normal rows  : {n_clean:>9,}  ({100*n_clean/len(tel):.1f}%)")
    print(f"  Precursor rows     : {n_precursor:>9,}  ({100*n_precursor/len(tel):.1f}%)")
    print(f"  Failure rows       : {n_failure:>9,}  ({100*n_failure/len(tel):.1f}%)")

    # --- rename for DB schema -----------------------------------------------
    tel = tel.rename(columns={"machineID": "machine_id"})
    tel["machine_id"] = tel["machine_id"].astype(str)

    db_cols = [
        "machine_id", "datetime",
        "volt", "rotate", "pressure", "vibration",
        "machine_failure", "failure_type",
        "machine_model", "machine_age",
        "is_clean_normal",
    ]
    return tel[db_cols].copy()


# ── STEP 5 — Create table ─────────────────────────────────────────────────────

def create_engine_from_url(url: str):
    return create_engine(
        url,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True,
    )


def create_table(engine) -> None:
    print(f"\n[STEP 5] Creating table {TABLE_NAME} (if not exists)...")
    with engine.begin() as conn:
        for stmt in CREATE_TABLE_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    print(f"  Table and indexes ready.")


# ── STEP 7 — Idempotency check ────────────────────────────────────────────────

def check_existing_rows(engine) -> int:
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            )
            return int(result.scalar())
    except Exception:
        return 0


# ── STEP 6 — Load rows ────────────────────────────────────────────────────────

def load_rows(df: pd.DataFrame, engine) -> None:
    total = len(df)
    print(f"\n[STEP 6] Loading {total:,} rows into {TABLE_NAME}...")
    print(f"  Chunk size: {CHUNK_SIZE} | Progress update every {PRINT_EVERY:,} rows")

    loaded = 0
    n_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
    last_print = 0

    for i in range(n_chunks):
        chunk = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        chunk.to_sql(
            TABLE_NAME,
            engine,
            if_exists="append",
            index=False,
            method="multi",
        )
        loaded += len(chunk)
        if loaded - last_print >= PRINT_EVERY or loaded == total:
            pct = 100 * loaded / total
            print(f"  Loaded {loaded:>9,} / {total:,}  ({pct:.1f}%)")
            last_print = loaded

    print(f"  Done. {loaded:,} rows loaded.")


# ── STEP 8 — Summary ──────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame, sequence_length: int) -> None:
    print("\n" + "=" * 62)
    print("FINAL SUMMARY")
    print("=" * 62)

    total       = len(df)
    clean       = int(df["is_clean_normal"].sum())
    failure     = int((df["machine_failure"] == 1).sum())
    precursor   = total - clean - failure
    sequences   = max(0, clean - sequence_length + 1)

    print(f"  Total rows loaded              : {total:>9,}")
    print(f"  Clean normal rows              : {clean:>9,}  ({100*clean/total:.1f}%)")
    print(f"  Failure rows                   : {failure:>9,}  ({100*failure/total:.1f}%)")
    print(f"  Precursor (excluded) rows      : {precursor:>9,}  ({100*precursor/total:.1f}%)")
    print(f"  Calculated SEQUENCE_LENGTH     : {sequence_length:>9,} time steps")
    print(f"  Clean sequences for LSTM       : {sequences:>9,}")
    print(f"  (= clean rows - sequence_length + 1 = {clean} - {sequence_length} + 1)")
    print("=" * 62)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not POSTGRES_URL:
        print("ERROR: POSTGRES_URL not set in .env — aborting.")
        sys.exit(1)

    # STEP 1
    tel, fail, mach = load_csvs()

    # STEP 2
    fail_pw, sequence_length = calculate_precursor_windows(tel, fail)

    # STEP 3
    save_config(fail_pw, sequence_length)

    # STEP 4
    df = build_full_dataframe(tel, fail, mach, sequence_length)

    # STEP 5
    engine = create_engine_from_url(POSTGRES_URL)
    create_table(engine)

    # STEP 7 — idempotency check (before loading)
    existing = check_existing_rows(engine)
    if existing > 0:
        print(f"\n[STEP 7] Idempotency check: {TABLE_NAME} already contains "
              f"{existing:,} rows — skipping load.")
        print_summary(df, sequence_length)
        engine.dispose()
        return

    print(f"\n[STEP 7] Idempotency check: table is empty — proceeding with load.")

    # STEP 6
    load_rows(df, engine)

    engine.dispose()

    # STEP 8
    print_summary(df, sequence_length)


if __name__ == "__main__":
    main()
