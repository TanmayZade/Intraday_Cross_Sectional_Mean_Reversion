"""
compare_frequencies.py
======================
Compare 1-minute vs 5-minute alpha signal quality.

This script validates that upgrading from 5-min to 1-min doesn't break signal IC.
IC (Information Coefficient) should be stable across frequencies since it's rank-based.

Usage
-----
    python compare_frequencies.py
    python compare_frequencies.py --tickers AAPL MSFT --start 2024-12-20 --end 2024-12-31
    python compare_frequencies.py --save-report
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from polygon_pipeline.pipeline.storage import read_panels
from features.engine import FeatureEngine
from features.resampler import Resampler
from alpha.signal import AlphaModel, compute_ic_decay, estimate_halflife

SEP = "─" * 70
log = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper()), 
                       format=fmt, datefmt=datefmt)


def compare_frequencies(
    tickers: list[str] = None,
    start_date: str = None,
    end_date: str = None,
    save_report: bool = False,
    report_dir: str = "reports/",
) -> dict:
    """
    Load 1-min and 5-min data, compute features, and compare IC.
    
    Returns dict with:
        - ic_comparison: DataFrame comparing IC stats across frequencies
        - halflives: Dict of signal half-lives
        - summary: Text report
    """
    t0 = pd.Timestamp.now()
    
    log.info(SEP)
    log.info("  Frequency Comparison: 1-minute vs 5-minute")
    log.info("  Tickers  : %s", tickers or "all (44)")
    log.info("  Date range: %s → %s", start_date or "all", end_date or "latest")
    log.info(SEP)
    
    # ── Step 1: Load 1-minute data ─────────────────────────────────────────────
    log.info("[1/4] Loading 1-minute data ...")
    try:
        panels_1m = read_panels(
            "polygon_pipeline/data/clean_1min/",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            universe_only=False,
        )
        if not panels_1m or "close" not in panels_1m:
            log.error("No 1-minute data found. Have you fetched 1-min bars yet?")
            return {}
    except FileNotFoundError:
        log.error("1-minute store not found at polygon_pipeline/data/clean_1min/")
        log.info("  First run: python -m polygon_pipeline.pipeline.orchestrator run \\")
        log.info("    --config polygon_pipeline/configs/config_1min.yaml")
        return {}
    
    n_bars_1m = len(panels_1m["close"])
    n_tickers_1m = len(panels_1m["close"].columns)
    log.info("  Loaded: %d bars × %d tickers (1-min)", n_bars_1m, n_tickers_1m)
    
    # ── Step 2: Load 5-minute data ─────────────────────────────────────────────
    log.info("[2/4] Loading 5-minute data ...")
    try:
        panels_5m = read_panels(
            "polygon_pipeline/data/clean_5min/",
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            universe_only=False,
        )
        if not panels_5m or "close" not in panels_5m:
            log.error("No 5-minute data found. Keeping 5-min as baseline.")
            log.info("  Current data only available at: clean_1min/")
            return {}
    except FileNotFoundError:
        log.warning("5-minute store not found at polygon_pipeline/data/clean_5min/")
        log.info("  This is OK if you only have 1-minute data.")
        log.info("  To create 5-minute baseline: resample 1-min to 5-min")
        
        # Resample 1-min to 5-min for comparison
        log.info("  Creating 5-min from 1-min via resampling...")
        resampler = Resampler(panels_1m, freq="5min", session_only=True)
        panels_5m = resampler.resample()
        n_bars_5m = len(panels_5m["close"])
        log.info("  Resampled: %d bars (1-min) → %d bars (5-min)", 
                n_bars_1m, n_bars_5m)
    else:
        n_bars_5m = len(panels_5m["close"])
        log.info("  Loaded: %d bars × %d tickers (5-min)", n_bars_5m, n_tickers_1m)
    
    # ── Step 3: Compute features for both frequencies ──────────────────────────
    log.info("[3/4] Computing features (1-min and 5-min) ...")
    
    # 1-minute features (300-bar IC window ≈ 5 hours)
    eng_1m = FeatureEngine(panels_1m, atr_window=300, vol_window=900, 
                           zscore_window=900, beta_window=5850, halflife=195)
    features_1m = eng_1m.compute_all()
    
    # 5-minute features (60-bar IC window ≈ 5 hours)
    eng_5m = FeatureEngine(panels_5m, atr_window=60, vol_window=180, 
                           zscore_window=180, beta_window=1170, halflife=39)
    features_5m = eng_5m.compute_all()
    
    log.info("  1-min: %d features computed", len(features_1m))
    log.info("  5-min: %d features computed", len(features_5m))
    
    # ── Step 4: Compute alpha and compare IC ────────────────────────────────────
    log.info("[4/4] Building alpha signals and computing IC ...")
    
    close_1m = panels_1m["close"]
    close_5m = panels_5m["close"]
    
    # 1-minute alpha
    model_1m = AlphaModel(features_1m, close_1m, ic_window=300, 
                          min_ic_tstat=0.5)  # lower threshold for comparison
    alpha_1m = model_1m.composite_alpha()
    ic_1m_table = model_1m.ic_summary_table()
    
    # 5-minute alpha
    model_5m = AlphaModel(features_5m, close_5m, ic_window=60, 
                          min_ic_tstat=0.5)
    alpha_5m = model_5m.composite_alpha()
    ic_5m_table = model_5m.ic_summary_table()
    
    # ── Comparison ──────────────────────────────────────────────────────────────
    log.info("")
    log.info("IC Comparison (1-min vs 5-min):")
    log.info(SEP)
    
    # Create comparison table
    comparison_rows = []
    for feat in features_1m.keys():
        ic_1m = ic_1m_table.loc[feat, "IC_mean"] if feat in ic_1m_table.index else 0.0
        ic_5m = ic_5m_table.loc[feat, "IC_mean"] if feat in ic_5m_table.index else 0.0
        tstat_1m = ic_1m_table.loc[feat, "t_stat"] if feat in ic_1m_table.index else 0.0
        tstat_5m = ic_5m_table.loc[feat, "t_stat"] if feat in ic_5m_table.index else 0.0
        
        pct_diff = ((ic_1m - ic_5m) / abs(ic_5m) * 100) if ic_5m != 0 else 0
        
        comparison_rows.append({
            "feature": feat,
            "IC_1min": round(ic_1m, 5),
            "IC_5min": round(ic_5m, 5),
            "pct_diff": round(pct_diff, 1),
            "t_stat_1min": round(tstat_1m, 2),
            "t_stat_5min": round(tstat_5m, 2),
        })
    
    comparison_df = pd.DataFrame(comparison_rows).set_index("feature")
    
    # Print summary
    log.info("  %-25s  %+.5f  %+.5f  %+6.1f%%  %+.2f → %+.2f",
             "Feature", "IC_1min", "IC_5min", "% diff", "t-stat_1min", "t-stat_5min")
    log.info("  " + "-" * 78)
    
    for idx, row in comparison_df.iterrows():
        log.info("  %-25s  %+.5f  %+.5f  %+6.1f%%  %+.2f → %+.2f",
                 idx, row["IC_1min"], row["IC_5min"], row["pct_diff"],
                 row["t_stat_1min"], row["t_stat_5min"])
    
    # ── Summary statistics ──────────────────────────────────────────────────────
    log.info("")
    log.info("Summary Statistics:")
    log.info("-" * 70)
    
    ic_1m_mean = comparison_df["IC_1min"].mean()
    ic_5m_mean = comparison_df["IC_5min"].mean()
    ic_diff_pct = ((ic_1m_mean - ic_5m_mean) / abs(ic_5m_mean) * 100) if ic_5m_mean != 0 else 0
    
    log.info("  Mean IC (1-min):     %.5f", ic_1m_mean)
    log.info("  Mean IC (5-min):     %.5f", ic_5m_mean)
    log.info("  Difference:          %+6.1f%%", ic_diff_pct)
    log.info("")
    
    # Signal half-lives
    log.info("Signal Half-Life Comparison:")
    log.info("-" * 70)
    
    hl_1m = model_1m.composite_decay_halflife()
    hl_5m = model_5m.composite_decay_halflife()
    
    hl_1m_min = hl_1m if hl_1m != float('inf') else None
    hl_5m_min = hl_5m if hl_5m != float('inf') else None
    
    if hl_1m_min:
        log.info("  1-min composite halflife:  %.1f bars (~%.1f min)", hl_1m_min, hl_1m_min)
    else:
        log.info("  1-min composite halflife:  (signal doesn't decay to 50%)")
    
    if hl_5m_min:
        log.info("  5-min composite halflife:  %.1f bars (~%.1f min)", hl_5m_min, hl_5m_min * 5)
    else:
        log.info("  5-min composite halflife:  (signal doesn't decay to 50%)")
    
    log.info("")
    log.info("Signal Quality Assessment:")
    log.info("-" * 70)
    
    # Check if IC is stable
    if abs(ic_diff_pct) <= 5.0:
        log.info("  ✓ IC is STABLE across frequencies (within ±5%)")
        log.info("    1-minute and 5-minute strategies should be comparable.")
    elif abs(ic_diff_pct) <= 10.0:
        log.info("  ~ IC shows MODERATE variation (5-10% difference)")
        log.info("    This is expected due to frequency effects and sampling.")
    else:
        log.info("  ✗ IC differs significantly (>10%)")
        log.info("    Check data quality and ensure sufficient bars for IC estimation.")
    
    # Check active features
    active_1m = (ic_1m_table["active"] == True).sum() if not ic_1m_table.empty else 0
    active_5m = (ic_5m_table["active"] == True).sum() if not ic_5m_table.empty else 0
    
    log.info("")
    log.info("  Active features (>t-stat threshold):")
    log.info("    1-min: %d / %d features", active_1m, len(features_1m))
    log.info("    5-min: %d / %d features", active_5m, len(features_5m))
    
    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    log.info("")
    log.info(SEP)
    log.info("Comparison complete in %.1fs", elapsed)
    log.info(SEP)
    
    # ── Save report ─────────────────────────────────────────────────────────────
    if save_report:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        comp_path = Path(report_dir) / "frequency_comparison.csv"
        comparison_df.to_csv(comp_path)
        log.info("Saved: %s", comp_path)
        
        # Save IC tables
        ic_1m_path = Path(report_dir) / "ic_summary_1min.csv"
        ic_1m_table.to_csv(ic_1m_path)
        log.info("Saved: %s", ic_1m_path)
        
        ic_5m_path = Path(report_dir) / "ic_summary_5min.csv"
        ic_5m_table.to_csv(ic_5m_path)
        log.info("Saved: %s", ic_5m_path)
    
    return {
        "comparison": comparison_df,
        "ic_1min": ic_1m_table,
        "ic_5min": ic_5m_table,
        "summary": {
            "ic_1min_mean": ic_1m_mean,
            "ic_5min_mean": ic_5m_mean,
            "pct_diff": ic_diff_pct,
            "hl_1min": hl_1m_min,
            "hl_5min": hl_5m_min,
        }
    }


def _parse():
    p = argparse.ArgumentParser(
        description="Compare 1-minute vs 5-minute alpha signal quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tickers",      nargs="+", default=None,
                   help="Tickers to compare (default: all 44)")
    p.add_argument("--start",        default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end",          default=None, metavar="YYYY-MM-DD")
    p.add_argument("--save-report",  action="store_true",
                   help="Save comparison tables to reports/")
    p.add_argument("--report-dir",   default="reports/")
    p.add_argument("--log-level",    default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    setup_logging(level=args.log_level)
    
    compare_frequencies(
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        save_report=args.save_report,
        report_dir=args.report_dir,
    )
