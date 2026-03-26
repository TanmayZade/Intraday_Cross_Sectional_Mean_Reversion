"""
pipeline/storage.py
====================
Partitioned Parquet storage for clean daily OHLCV data.

Layout
------
data/clean/
    year=2020/data.parquet
    year=2021/data.parquet
    year=2022/data.parquet
    ...

Design
------
- Partition by year only (daily data = small files, year partitions are fast)
- Snappy compression (fast read, moderate size — ~1MB per ticker-year)
- PyArrow schema enforced on write
- Predicate pushdown: filter by year, ticker on read
- Panel pivot helpers for research use
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

# ── Canonical schema ──────────────────────────────────────────────────────────
# "timestamp" stores both daily dates AND intraday minute timestamps.
# For daily bars:   timestamp = midnight ET of the trading date
# For minute bars:  timestamp = bar open time in ET (tz-aware)
# NOTE: "year" is only a folder partition name (year=2023/), NOT a data column.
SCHEMA = pa.schema([
    pa.field("timestamp",   pa.timestamp("us", tz="America/New_York")),
    pa.field("ticker",      pa.string()),
    pa.field("open",        pa.float64()),
    pa.field("high",        pa.float64()),
    pa.field("low",         pa.float64()),
    pa.field("close",       pa.float64()),
    pa.field("volume",      pa.float64()),
    pa.field("vwap",        pa.float64()),
    pa.field("n_trades",    pa.float64()),
    pa.field("flagged",     pa.bool_()),
    pa.field("in_universe", pa.bool_()),
])

REQUIRED_COLS = {"timestamp", "ticker", "open", "high", "low", "close", "volume"}


# ─────────────────────────────────────────────────────────────────────────────
# Writer
# ─────────────────────────────────────────────────────────────────────────────

def write_clean(
    df: pd.DataFrame,
    root: str | Path,
    compression: str = "snappy",
) -> list[Path]:
    """
    Write clean OHLCV DataFrame to partitioned Parquet store (partition by year).

    Idempotent: re-running overwrites the existing year partition.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    df = _prepare_for_write(df)
    written = []

    for year, part in df.groupby(df["timestamp"].dt.year):
        part_dir  = root / f"year={year}"
        part_dir.mkdir(parents=True, exist_ok=True)
        out_path  = part_dir / "data.parquet"

        # Drop year helper if present — stored in folder name only
        part_clean = part.drop(columns=["year"], errors="ignore")
        table = _to_arrow_table(part_clean)
        pq.write_table(table, out_path, compression=compression)
        written.append(out_path)
        log.debug("Wrote %s  (%d rows)", out_path, len(part_clean))

    log.info("Wrote %d partition files → %s", len(written), root)
    return written


def write_ticker(
    df: pd.DataFrame,
    ticker: str,
    root: str | Path,
    compression: str = "snappy",
) -> None:
    """Write a single ticker's data, merging with any existing data in the store."""
    existing = pd.DataFrame()
    try:
        existing = read_clean(root, tickers=[ticker])
    except Exception:
        pass

    df = df.copy()
    df["ticker"] = ticker
    if not existing.empty:
        combined = pd.concat([existing, df])
        combined = combined[~combined.set_index(["date", "ticker"]).index.duplicated(keep="last")]
    else:
        combined = df

    write_clean(combined, root, compression=compression)


# ─────────────────────────────────────────────────────────────────────────────
# Reader
# ─────────────────────────────────────────────────────────────────────────────

def read_clean(
    root: str | Path,
    tickers: Optional[list[str]]  = None,
    start_date: Optional[str]     = None,   # "YYYY-MM-DD"
    end_date:   Optional[str]     = None,   # "YYYY-MM-DD"
    columns: Optional[list[str]]  = None,
    universe_only: bool           = False,  # filter to in_universe == True
) -> pd.DataFrame:
    """
    Read from the clean Parquet store with optional filters.

    Returns flat DataFrame, index = date (DatetimeIndex).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Clean store not found: {root}")

    filters = _build_filters(tickers, start_date, end_date, universe_only)
    cols    = columns  # None → read all columns

    dataset = pq.ParquetDataset(root, filters=filters)
    table   = dataset.read(columns=cols)
    df      = table.to_pandas()

    if df.empty:
        return df

    # Restore DatetimeIndex
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df = df.set_index("timestamp").sort_index()
    # Fine-grained date filter (partition filter is year-level only)
    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz="America/New_York")]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date + " 23:59:59", tz="America/New_York")]

    log.info(
        "Read %d rows | %d tickers | %s → %s",
        len(df),
        df["ticker"].nunique() if "ticker" in df.columns else 0,
        df.index.min().date() if len(df) else "N/A",
        df.index.max().date() if len(df) else "N/A",
    )
    return df


def read_panels(
    root: str | Path,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
    tickers: Optional[list[str]] = None,
    start_date: Optional[str]    = None,
    end_date:   Optional[str]    = None,
    universe_only: bool          = False,
) -> dict[str, pd.DataFrame]:
    """
    Read clean store and return research-ready {field: date × ticker} panels.
    Most feature engineering functions consume this format directly.
    """
    df = read_clean(root, tickers=tickers, start_date=start_date,
                    end_date=end_date, universe_only=universe_only)
    return _to_panels(df, fields)


# ─────────────────────────────────────────────────────────────────────────────
# Panel helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_panels(
    df: pd.DataFrame,
    fields: tuple[str, ...],
) -> dict[str, pd.DataFrame]:
    panels = {}
    for f in fields:
        if f in df.columns:
            panels[f] = df.pivot_table(
                index=df.index, columns="ticker", values=f, aggfunc="last"
            )
    return panels


def pivot_field(df: pd.DataFrame, field: str) -> pd.DataFrame:
    """Pivot a flat clean DataFrame on one field: date × ticker."""
    return df.pivot_table(index=df.index, columns="ticker",
                          values=field, aggfunc="last")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_for_write(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index()

    # Normalise index column name — handles both daily ("date") and
    # intraday ("timestamp") index names coming from the fetcher.
    for alias in ["timestamp", "date", "index"]:
        if alias in df.columns and "timestamp" not in df.columns:
            df = df.rename(columns={alias: "timestamp"})
            break

    # Ensure timestamp is tz-aware (America/New_York)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("America/New_York")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

    # year is partition folder only — not stored as a data column

    # Add missing optional columns
    for col, default in [
        ("vwap",        np.nan),
        ("n_trades",    np.nan),
        ("flagged",     False),
        ("in_universe", False),
    ]:
        if col not in df.columns:
            df[col] = default

    df["flagged"]     = df["flagged"].fillna(False).astype(bool)
    df["in_universe"] = df["in_universe"].fillna(False).astype(bool)

    for col in ["open", "high", "low", "close", "volume", "vwap", "n_trades"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    return df


def _to_arrow_table(df: pd.DataFrame) -> pa.Table:
    """Convert to PyArrow table, coercing to schema."""
    # Only include columns present in schema
    schema_cols = {f.name for f in SCHEMA}
    df = df[[c for c in df.columns if c in schema_cols]]

    # Build column by column to allow partial schema
    arrays, fields = [], []
    for field in SCHEMA:
        if field.name in df.columns:
            col = df[field.name]
            if field.name == "timestamp":
                arr = pa.array(col, type=pa.timestamp("us", tz="America/New_York"))
            elif field.type == pa.bool_():
                arr = pa.array(col.fillna(False).astype(bool), type=pa.bool_())
            elif field.type == pa.string():
                arr = pa.array(col.astype(str), type=pa.string())
            else:
                arr = pa.array(col, type=field.type)
            arrays.append(arr)
            fields.append(field)

    return pa.table(arrays, schema=pa.schema(fields))


def _build_filters(
    tickers:    Optional[list[str]],
    start_date: Optional[str],
    end_date:   Optional[str],
    universe_only: bool,
) -> Optional[list]:
    filters = []
    if tickers:
        filters.append(("ticker", "in", tickers))
    # year partitioning is directory-based (year=NNNN/ folders)
    # PyArrow hive partitioning handles this automatically on read
    if universe_only:
        filters.append(("in_universe", "==", True))
    return filters or None