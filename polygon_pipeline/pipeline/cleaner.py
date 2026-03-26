"""
pipeline/cleaner.py
===================
Cleaning pipeline for Polygon daily OHLCV bars.

Polygon's adjusted=True handles splits automatically.
What remains after fetching:
  - Occasional zero-volume days (holidays wrongly included)
  - Thin-volume days (stock suspension, exchange closure)
  - Stale prices (vendor glitch: same OHLC across multiple days)
  - Volume spikes (index inclusion, short squeeze, corporate events)
  - OHLC self-consistency violations (rare but real data errors)
  - Unadjusted events remaining after adjustment (verify step)
  - Delisted tickers with erroneous final bars

Cleaning stages (run in this order — order matters):
  1.  Sort + dedup         by date
  2.  OHLC consistency     H ≥ max(O,C) ≥ min(O,C) ≥ L > 0
  3.  Zero / NaN prices    any OHLC ≤ 0
  4.  Minimum volume       days with < min_volume shares traded
  5.  Overnight gap check  flag | open / prior_close - 1 | > threshold
  6.  Single-day return    flag |return| > max_daily_return
  7.  Intraday range       flag (H-L)/L > max_intraday_range
  8.  Volume spikes        flag vol > N × rolling median volume
  9.  Stale price detect   flag N consecutive identical closes
  10. Adjustment verify    residual extreme returns after adjustment
  11. Quality summary      emit per-ticker report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CleaningConfig:
    # Outlier thresholds
    max_daily_return:     float = 0.50   # |close-to-close| > 50% → flag
    max_intraday_range:   float = 0.40   # (H-L)/L > 40% → remove
    max_volume_spike_mult:float = 20.0   # vol > 20× median → flag (not remove)
    volume_spike_window:  int   = 20     # rolling window for volume median
    max_overnight_gap:    float = 0.40   # |open/prior_close - 1| > 40% → flag
    min_volume:           int   = 10_000 # shares; 0 to disable

    # Stale price detection
    max_stale_days:       int   = 3      # N consecutive identical closes → flag

    # What to do with each flag type
    # 'flag'   = mark as flagged but keep
    # 'remove' = drop the row entirely
    action_return_outlier:  str = "remove"
    action_range_outlier:   str = "remove"
    action_gap:             str = "flag"
    action_volume_spike:    str = "flag"
    action_stale:           str = "flag"


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerReport:
    ticker:          str
    raw_bars:        int  = 0
    dedup_removed:   int  = 0
    ohlc_invalid:    int  = 0
    zero_price:      int  = 0
    low_volume:      int  = 0
    return_outliers: int  = 0
    range_outliers:  int  = 0
    gaps_flagged:    int  = 0
    vol_spikes:      int  = 0
    stale_flagged:   int  = 0
    final_bars:      int  = 0
    date_start:      Optional[str] = None
    date_end:        Optional[str] = None
    pct_retained:    float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ─────────────────────────────────────────────────────────────────────────────
# Main functions
# ─────────────────────────────────────────────────────────────────────────────

def clean_ticker(
    df: pd.DataFrame,
    ticker: str,
    cfg: Optional[CleaningConfig] = None,
) -> tuple[pd.DataFrame, TickerReport]:
    """
    Clean a single-ticker daily OHLCV DataFrame.

    Input  : DatetimeIndex (dates), columns: open high low close volume [vwap] [n_trades]
    Output : cleaned DataFrame, TickerReport
    """
    cfg = cfg or CleaningConfig()
    rpt = TickerReport(ticker=ticker, raw_bars=len(df))

    if df.empty:
        rpt.final_bars = 0
        return df, rpt

    df = df.copy().sort_index()

    # ── 1. Deduplication ──────────────────────────────────────────────────────
    dupes = df.index.duplicated(keep="last")
    if dupes.any():
        rpt.dedup_removed = int(dupes.sum())
        df = df[~dupes]

    # ── 2. OHLC self-consistency ──────────────────────────────────────────────
    invalid = ~_ohlc_valid(df)
    if invalid.any():
        rpt.ohlc_invalid = int(invalid.sum())
        df = df[~invalid]

    # ── 3. Zero / NaN prices ──────────────────────────────────────────────────
    zero_null = df[["open", "high", "low", "close"]].le(0).any(axis=1) | \
                df[["open", "high", "low", "close"]].isna().any(axis=1)
    if zero_null.any():
        rpt.zero_price = int(zero_null.sum())
        df = df[~zero_null]

    # ── 4. Minimum volume filter ──────────────────────────────────────────────
    if cfg.min_volume > 0 and "volume" in df.columns:
        low_vol = df["volume"] < cfg.min_volume
        if low_vol.any():
            rpt.low_volume = int(low_vol.sum())
            df = df[~low_vol]

    if df.empty:
        rpt.final_bars = 0
        return df, rpt

    # ── 5. Overnight gap flag ─────────────────────────────────────────────────
    df, rpt = _check_gaps(df, ticker, cfg, rpt)

    # ── 6. Daily return outliers ──────────────────────────────────────────────
    df, rpt = _check_return_outliers(df, ticker, cfg, rpt)

    # ── 7. Intraday range outliers ────────────────────────────────────────────
    df, rpt = _check_range_outliers(df, ticker, cfg, rpt)

    # ── 8. Volume spikes ──────────────────────────────────────────────────────
    df, rpt = _check_volume_spikes(df, ticker, cfg, rpt)

    # ── 9. Stale price detection ──────────────────────────────────────────────
    df, rpt = _check_stale_prices(df, ticker, cfg, rpt)

    # ── Finalize ──────────────────────────────────────────────────────────────
    if "flagged" not in df.columns:
        df["flagged"] = False
    df["flagged"] = df["flagged"].fillna(False)

    rpt.final_bars  = len(df)
    rpt.pct_retained = round(100 * rpt.final_bars / max(rpt.raw_bars, 1), 2)
    if not df.empty:
        rpt.date_start = str(df.index.min())
        rpt.date_end   = str(df.index.max())

    log.info(
        "%-6s  raw=%4d → clean=%4d (%.0f%%)  "
        "ohlc=%d  zero=%d  vol=%d  ret=%d  range=%d  gaps=%d  spikes=%d  stale=%d",
        ticker, rpt.raw_bars, rpt.final_bars, rpt.pct_retained,
        rpt.ohlc_invalid, rpt.zero_price, rpt.low_volume,
        rpt.return_outliers, rpt.range_outliers,
        rpt.gaps_flagged, rpt.vol_spikes, rpt.stale_flagged,
    )
    return df, rpt


def clean_panel(
    panel: pd.DataFrame,                    # flat df with 'ticker' column
    cfg: Optional[CleaningConfig] = None,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean the full panel of all tickers.

    Returns
    -------
    clean_df   : cleaned flat DataFrame  (still has 'ticker' column)
    report_df  : per-ticker cleaning report as DataFrame
    """
    cfg = cfg or CleaningConfig()
    tickers = sorted(panel["ticker"].unique())
    log.info("Cleaning %d tickers ...", len(tickers))

    def _process(t: str):
        sub = panel[panel["ticker"] == t].drop(columns="ticker")
        return clean_ticker(sub, t, cfg)

    if n_jobs == 1:
        results = [_process(t) for t in tickers]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(_process)(t) for t in tickers)

    clean_frames, reports = [], []
    for clean_df, rpt in results:
        if not clean_df.empty:
            clean_df = clean_df.copy()
            clean_df["ticker"] = rpt.ticker
            clean_frames.append(clean_df)
        reports.append(rpt.to_dict())

    combined  = pd.concat(clean_frames).sort_index() if clean_frames else pd.DataFrame()
    report_df = pd.DataFrame(reports).set_index("ticker")

    total_raw   = report_df["raw_bars"].sum()
    total_clean = report_df["final_bars"].sum()
    log.info(
        "Panel cleaning complete: %d → %d bars (%.1f%% retained)",
        total_raw, total_clean, 100 * total_clean / max(total_raw, 1),
    )
    return combined, report_df


# ─────────────────────────────────────────────────────────────────────────────
# Stage implementations
# ─────────────────────────────────────────────────────────────────────────────

def _ohlc_valid(df: pd.DataFrame) -> pd.Series:
    """True where H ≥ max(O,C), L ≤ min(O,C), H ≥ L, all positive."""
    return (
        (df["high"] >= df["low"])                            &
        (df["high"] >= df["open"])                           &
        (df["high"] >= df["close"])                          &
        (df["low"]  <= df["open"])                           &
        (df["low"]  <= df["close"])                          &
        (df["low"]  > 0)                                     &
        (df["close"] > 0)
    )


def _check_gaps(df, ticker, cfg, rpt):
    prior_close = df["close"].shift(1)
    gap = (df["open"] / prior_close.replace(0, np.nan) - 1).abs()
    flagged = gap > cfg.max_overnight_gap

    n = int(flagged.sum())
    if n:
        rpt.gaps_flagged = n
        if "flagged" not in df.columns:
            df = df.copy()
            df["flagged"] = False
        df.loc[flagged, "flagged"] = True
        log.debug("%s: %d gap bars flagged (kept)", ticker, n)

    return df, rpt


def _check_return_outliers(df, ticker, cfg, rpt):
    ret = df["close"].pct_change().abs()
    outliers = ret > cfg.max_daily_return

    n = int(outliers.sum())
    if n:
        rpt.return_outliers = n
        if cfg.action_return_outlier == "remove":
            df = df[~outliers]
        else:
            if "flagged" not in df.columns:
                df["flagged"] = False
            df.loc[outliers, "flagged"] = True
        log.debug("%s: %d return outliers (%s)", ticker, n, cfg.action_return_outlier)

    return df, rpt


def _check_range_outliers(df, ticker, cfg, rpt):
    hl_range = (df["high"] - df["low"]) / df["low"].replace(0, np.nan)
    outliers = hl_range > cfg.max_intraday_range

    n = int(outliers.sum())
    if n:
        rpt.range_outliers = n
        if cfg.action_range_outlier == "remove":
            df = df[~outliers]
        else:
            if "flagged" not in df.columns:
                df["flagged"] = False
            df.loc[outliers, "flagged"] = True
        log.debug("%s: %d range outliers (%s)", ticker, n, cfg.action_range_outlier)

    return df, rpt


def _check_volume_spikes(df, ticker, cfg, rpt):
    if "volume" not in df.columns:
        return df, rpt

    med_vol = df["volume"].rolling(
        cfg.volume_spike_window, min_periods=5, center=False
    ).median().shift(1)

    spikes = df["volume"] > cfg.max_volume_spike_mult * med_vol.replace(0, np.nan)
    n = int(spikes.sum())
    if n:
        rpt.vol_spikes = n
        # Volume spikes are informative (e.g., earnings, short squeezes) → flag only
        if "flagged" not in df.columns:
            df = df.copy()
            df["flagged"] = False
        df.loc[spikes, "flagged"] = True
        log.debug("%s: %d volume spikes flagged (kept)", ticker, n)

    return df, rpt


def _check_stale_prices(df, ticker, cfg, rpt):
    """
    Flag runs of N+ consecutive identical close prices.
    Genuine markets don't have 3+ days of exactly the same close — this
    indicates a data vendor problem or a suspended stock.
    """
    close = df["close"]
    # True where price is same as previous bar
    same = close == close.shift(1)
    # Cumulative run length
    run_id  = (same != same.shift(1)).cumsum()
    run_len = same.groupby(run_id).cumsum()

    stale = same & (run_len >= cfg.max_stale_days - 1)
    n = int(stale.sum())
    if n:
        rpt.stale_flagged = n
        if "flagged" not in df.columns:
            df = df.copy()
            df["flagged"] = False
        df.loc[stale, "flagged"] = True
        log.debug("%s: %d stale-price bars flagged", ticker, n)

    return df, rpt
