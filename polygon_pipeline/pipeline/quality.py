"""
pipeline/quality.py
====================
Quality reports, diagnostics, and data validation for the clean store.

Key outputs
-----------
  pipeline_summary.csv    — aggregate stats across all tickers
  cleaning_report.csv     — per-ticker cleaning stage counts
  coverage_report.csv     — date × ticker bar coverage
  gap_report.csv          — list of all flagged overnight gaps
  flagged_bars.csv        — all bars with flagged=True

Corwin-Schultz spread estimator (requires only daily H/L) is also here
for liquidity scoring during universe selection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Coverage analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage(
    flat_df: pd.DataFrame,
    trading_calendar: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """
    Compute per-ticker coverage fraction: trading_days_present / expected_days.

    If trading_calendar is provided, uses that as expected days.
    Otherwise uses the union of all observed dates.
    """
    if "ticker" not in flat_df.columns:
        raise ValueError("flat_df must have a 'ticker' column")

    tickers = flat_df["ticker"].unique()
    dates   = flat_df.index.unique().sort_values()

    if trading_calendar is not None:
        expected = trading_calendar[
            (trading_calendar >= dates.min()) &
            (trading_calendar <= dates.max())
        ]
    else:
        expected = dates

    coverage = {}
    for t in tickers:
        sub = flat_df[flat_df["ticker"] == t]
        present = sub.index.unique()
        n_present  = len(present)
        n_expected = len(expected)
        coverage[t] = {
            "days_present":  n_present,
            "days_expected": n_expected,
            "coverage_pct":  round(100 * n_present / max(n_expected, 1), 2),
            "first_date":    str(present.min()) if n_present else None,
            "last_date":     str(present.max()) if n_present else None,
        }

    return pd.DataFrame(coverage).T.sort_values("coverage_pct", ascending=True)


# ─────────────────────────────────────────────────────────────────────────────
# Flagged bar inventory
# ─────────────────────────────────────────────────────────────────────────────

def flagged_bar_report(flat_df: pd.DataFrame) -> pd.DataFrame:
    """Return all flagged bars with ticker and date."""
    if "flagged" not in flat_df.columns:
        return pd.DataFrame()
    flagged = flat_df[flat_df["flagged"] == True].copy()
    if flagged.empty:
        return pd.DataFrame()
    flagged = flagged.reset_index()
    if "date" not in flagged.columns:
        flagged = flagged.rename(columns={flagged.columns[0]: "date"})
    keep = [c for c in ["date","ticker","open","high","low","close","volume"] if c in flagged.columns]
    flagged = flagged[keep]
    return flagged.sort_values(["ticker", "date"])


# ─────────────────────────────────────────────────────────────────────────────
# Return distribution diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def return_diagnostics(
    close_panel: pd.DataFrame,          # date × ticker
) -> pd.DataFrame:
    """
    Per-ticker return statistics. Useful for spotting data quality issues:
    - Extremely high mean return → possible unadjusted split
    - Zero std dev → stale prices
    - Very fat tails (kurtosis) → outliers remaining
    """
    rets = close_panel.pct_change()
    stats = pd.DataFrame({
        "mean_ret_ann":    rets.mean() * 252,
        "vol_ann":         rets.std() * np.sqrt(252),
        "skew":            rets.skew(),
        "kurtosis":        rets.kurt(),
        "max_1d_ret":      rets.max(),
        "min_1d_ret":      rets.min(),
        "n_extreme_pct":   (rets.abs() > 0.20).sum(),  # returns > 20%
        "n_zero_ret":      (rets == 0).sum(),
        "n_null":          rets.isna().sum(),
    })
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Corwin-Schultz bid-ask spread estimator
# ─────────────────────────────────────────────────────────────────────────────

def corwin_schultz_spread(
    high: pd.Series,
    low:  pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Corwin & Schultz (2012) daily spread estimate from H/L prices.
    Returns rolling median spread in basis points (bps).
    Negative values (noise) are clipped to 0.

    Reference: "A Simple Way to Estimate Bid-Ask Spreads from Daily High
    and Low Prices", Journal of Finance 2012.
    """
    ln_h = np.log(high)
    ln_l = np.log(low)

    # Two-day beta and gamma
    beta  = (ln_h - ln_l)**2 + (ln_h.shift(1) - ln_l.shift(1))**2
    gamma = (
        pd.concat([high, high.shift(1)], axis=1).max(axis=1).apply(np.log) -
        pd.concat([low,  low.shift(1)],  axis=1).min(axis=1).apply(np.log)
    )**2

    k     = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread_raw = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    spread_bps = (spread_raw.clip(lower=0) * 10_000
                  ).rolling(window, min_periods=5).median()
    return spread_bps


def universe_spread_estimates(
    close_panel:  pd.DataFrame,       # date × ticker
    high_panel:   pd.DataFrame,
    low_panel:    pd.DataFrame,
    window:       int = 20,
) -> pd.DataFrame:
    """Estimate bid-ask spread for each ticker. Returns date × ticker in bps."""
    spreads = {}
    for ticker in close_panel.columns:
        try:
            spreads[ticker] = corwin_schultz_spread(
                high_panel[ticker], low_panel[ticker], window
            )
        except Exception as e:
            log.debug("Spread estimate failed for %s: %s", ticker, e)
    return pd.DataFrame(spreads)


# ─────────────────────────────────────────────────────────────────────────────
# Full report generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    clean_df:    pd.DataFrame,
    report_df:   pd.DataFrame,          # from cleaner.clean_panel()
    output_dir:  str | Path = "reports/",
    close_panel: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Generate and save the full quality report suite.

    Files written:
        pipeline_summary.csv
        cleaning_report.csv
        coverage_report.csv
        return_diagnostics.csv   (if close_panel provided)
        flagged_bars.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Cleaning report ──────────────────────────────────────────────────────
    report_df.to_csv(out / "cleaning_report.csv")

    # ── Coverage ─────────────────────────────────────────────────────────────
    cov = compute_coverage(clean_df)
    cov.to_csv(out / "coverage_report.csv")

    # ── Flagged bars ─────────────────────────────────────────────────────────
    flagged = flagged_bar_report(clean_df)
    if not flagged.empty:
        flagged.to_csv(out / "flagged_bars.csv", index=False)

    # ── Return diagnostics ───────────────────────────────────────────────────
    if close_panel is not None:
        diag = return_diagnostics(close_panel)
        diag.to_csv(out / "return_diagnostics.csv")

    # ── Pipeline summary ─────────────────────────────────────────────────────
    total_raw   = int(report_df["raw_bars"].sum())
    total_clean = int(report_df["final_bars"].sum())
    summary = {
        "n_tickers":            int(len(report_df)),
        "total_raw_bars":       total_raw,
        "total_clean_bars":     total_clean,
        "pct_retained":         round(100 * total_clean / max(total_raw, 1), 2),
        "bars_dedup_removed":   int(report_df["dedup_removed"].sum()),
        "bars_ohlc_invalid":    int(report_df["ohlc_invalid"].sum()),
        "bars_zero_price":      int(report_df["zero_price"].sum()),
        "bars_low_volume":      int(report_df["low_volume"].sum()),
        "bars_return_outlier":  int(report_df["return_outliers"].sum()),
        "bars_range_outlier":   int(report_df["range_outliers"].sum()),
        "bars_gap_flagged":     int(report_df["gaps_flagged"].sum()),
        "bars_vol_spike":       int(report_df["vol_spikes"].sum()),
        "bars_stale":           int(report_df["stale_flagged"].sum()),
        "avg_coverage_pct":     round(float(cov["coverage_pct"].mean()), 2),
        "tickers_below_80pct":  int((cov["coverage_pct"] < 80).sum()),
    }
    pd.Series(summary).to_frame("value").to_csv(out / "pipeline_summary.csv")

    log.info(
        "Quality report saved to %s | %d tickers | %.1f%% bars retained | "
        "%d flagged (not removed) | %d tickers below 80%% coverage",
        out, summary["n_tickers"], summary["pct_retained"],
        summary["bars_gap_flagged"] + summary["bars_vol_spike"] + summary["bars_stale"],
        summary["tickers_below_80pct"],
    )
    return summary
