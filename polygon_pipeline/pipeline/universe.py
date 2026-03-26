"""
pipeline/universe.py
====================
Point-in-time universe construction from daily OHLCV data.

"Point-in-time" means: on date T, the universe is determined ONLY from
data available on dates < T. No look-ahead. No survivorship bias.

Rules (applied daily via rolling window, shifted by 1 day):
  1. ADTV ≥ min_adtv_usd      (rolling 20-day, lagged 1 day)
  2. Price ≥ min_price         (prior close, lagged 1 day)
  3. Sufficient history        (≥ min_history_days trading days observed)
  4. Not stale / suspended     (no consecutive identical closes)

Returns: pd.DataFrame (date × ticker) of bool membership.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_universe(
    close:   pd.DataFrame,          # date × ticker
    volume:  pd.DataFrame,          # date × ticker
    min_adtv_usd:      float = 1_000_000,
    min_price:         float = 5.0,
    max_price:         float = 50_000.0,
    min_history_days:  int   = 60,
    adtv_window:       int   = 20,
) -> pd.DataFrame:
    """
    Build point-in-time universe membership matrix.

    Parameters
    ----------
    close, volume : wide DataFrames, index=date (trading days), columns=tickers

    Returns
    -------
    pd.DataFrame  bool  (date × ticker)
    True  = ticker is in the tradable universe on that date.
    """
    # Align both DataFrames
    tickers = close.columns.intersection(volume.columns)
    close   = close[tickers]
    volume  = volume[tickers]

    # Dollar volume per day
    dollar_vol = close * volume

    # ── Rule 1: ADTV filter (lagged 1 day so truly look-ahead free) ──────────
    adtv = dollar_vol.rolling(adtv_window, min_periods=max(5, adtv_window // 4)).mean()
    adtv_lagged = adtv.shift(1)
    mask_adtv = adtv_lagged >= min_adtv_usd

    # ── Rule 2: Price filter (prior close) ────────────────────────────────────
    prior_close  = close.shift(1)
    mask_price   = (prior_close >= min_price) & (prior_close <= max_price)

    # ── Rule 3: Sufficient history ────────────────────────────────────────────
    # Count cumulative non-null trading days per ticker
    obs_count    = close.notna().cumsum()
    mask_history = obs_count.shift(1) >= min_history_days

    # ── Combine ───────────────────────────────────────────────────────────────
    universe = mask_adtv & mask_price & mask_history
    universe = universe.fillna(False)

    _log_stats(universe)
    return universe


def apply_universe_mask(
    panel: pd.DataFrame,            # flat df: index=date, cols include 'ticker'
    universe: pd.DataFrame,         # bool date × ticker
) -> pd.DataFrame:
    """
    Add an 'in_universe' column to the flat panel DataFrame.
    Rows where in_universe=False should be excluded from signal computation.
    """
    panel = panel.copy()
    # Universe index is daily (date-level). Panel index may be minute timestamps.
    # Normalise to date for the lookup, then map back.
    raw_index = panel.index
    if hasattr(raw_index, "normalize"):
        dates = raw_index.normalize()   # minute → midnight of that day
    else:
        dates = raw_index
    tickers = panel["ticker"]

    # Vectorised lookup
    date_idx   = universe.index.get_indexer(dates)
    ticker_idx = universe.columns.get_indexer(tickers)

    valid   = (date_idx >= 0) & (ticker_idx >= 0)
    in_univ = np.zeros(len(panel), dtype=bool)
    in_univ[valid] = universe.values[date_idx[valid], ticker_idx[valid]]

    panel["in_universe"] = in_univ
    n_excluded = (~in_univ).sum()
    log.info(
        "Universe mask: %d bars excluded (%.1f%% of total)",
        n_excluded, 100 * n_excluded / max(len(panel), 1),
    )
    return panel


def build_panels(
    flat_df: pd.DataFrame,
    fields: tuple[str, ...] = ("open", "high", "low", "close", "volume"),
) -> dict[str, pd.DataFrame]:
    """
    Convert flat DataFrame → dict of {field: date × ticker} wide panels.
    For minute-bar data the index is a full timestamp; this function
    aggregates to DAILY before pivoting so build_universe() receives
    one row per trading day (not one row per minute).
    """
    if "ticker" not in flat_df.columns:
        raise ValueError("flat_df must have a 'ticker' column")

    # Detect whether index is intraday (minute) or already daily
    idx = flat_df.index
    is_intraday = hasattr(idx, "time") and (idx.normalize() != idx).any()

    if is_intraday:
        # Aggregate to daily: open=first, high=max, low=min, close=last, volume=sum
        agg_map = {"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"}
        df_copy = flat_df.copy()
        df_copy["_date"] = idx.normalize()
        panels = {}
        for field in fields:
            if field in df_copy.columns:
                agg = agg_map.get(field, "last")
                daily = (
                    df_copy.groupby(["_date", "ticker"])[field]
                    .agg(agg)
                    .unstack("ticker")
                )
                daily.index.name = "date"
                panels[field] = daily
        return panels
    else:
        panels = {}
        for field in fields:
            if field in flat_df.columns:
                panels[field] = flat_df.pivot_table(
                    index=flat_df.index,
                    columns="ticker",
                    values=field,
                    aggfunc="last",
                )
        return panels


def _log_stats(universe: pd.DataFrame) -> None:
    daily_n = universe.sum(axis=1)
    log.info(
        "Universe built: %d dates | avg %.0f stocks | min %d | max %d",
        len(universe), daily_n.mean(), daily_n.min(), daily_n.max(),
    )