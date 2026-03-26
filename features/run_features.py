"""
run_features.py
===============
Production feature engineering entry point.

Pipeline:
  [1] Load 1-min panels from data/clean/
  [2] Resample to 15-min (optimal for cost/signal tradeoff)
  [3] Compute 15 features across 5 categories
  [4] Save to data/features/
  [5] Run IC diagnostics → reports/

Usage
-----
  python run_features.py                        # all tickers, all dates
  python run_features.py --freq 15min           # explicit frequency
  python run_features.py --tickers AAPL MSFT    # subset
  python run_features.py --start 2024-12-01     # date range
  python run_features.py --diagnostics-only     # reload + re-diagnose
  python run_features.py --save-report          # write CSVs to reports/
  python run_features.py --log-level DEBUG      # set log level
"""
from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from polygon_pipeline.pipeline.storage import read_panels
from features.resampler import Resampler
from features.engine import FeatureEngine
from features.diagnostics import FeatureDiagnostics
from features.store       import FeatureStore

SEP = "─" * 60
log = logging.getLogger(__name__)


def setup_logging(level="INFO", log_dir="logs"):
    """
    Configure logging to output to both console and file.
    
    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_dir : str
        Directory to store log files (created if it doesn't exist)
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"features_pipeline_{timestamp}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler (StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    if root_logger.handlers:
        root_logger.handlers.clear()
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    log.info(f"Logging initialized: {log_file}")


def run(
    clean_dir       = "polygon_pipeline/data/clean/",
    feature_dir     = "data/features/",
    report_dir      = "polygon_pipeline/reports/",
    freq            = "15min",
    tickers         = None,
    start_date      = None,
    end_date        = None,
    atr_window      = 20,
    vol_window      = 60,
    volume_window   = 20,
    zscore_window   = 60,
    beta_window     = 390,
    halflife        = 13,
    diagnostics_only= False,
    save_report     = False,
):
    t0 = time.perf_counter()
    log.info(SEP)
    log.info("  Feature Engineering Pipeline")
    log.info("  Frequency    : %s (resampled from 1-min)", freq)
    log.info("  Clean store  : %s", clean_dir)
    log.info("  Feature store: %s", feature_dir)
    log.info("  Date range   : %s → %s", start_date or "all", end_date or "latest")
    log.info(SEP)

    store = FeatureStore(feature_dir)

    # ── Diagnostics-only mode ─────────────────────────────────────────────────
    if diagnostics_only:
        log.info("Diagnostics-only — loading existing features ...")
        features = store.load(tickers=tickers,
                              start_date=start_date, end_date=end_date)
        panels = read_panels(clean_dir, tickers=tickers,
                             start_date=start_date, end_date=end_date,
                             universe_only=False)
        r = Resampler(panels, freq=freq)
        panels_r = r.resample()
        _run_diagnostics(features, panels_r["close"], report_dir, save_report)
        return {"features": features}

    # ── Step 1: Load 1-min panels ─────────────────────────────────────────────
    log.info("[1/5] Loading 1-min clean panels ...")
    panels_1m = read_panels(
        clean_dir, tickers=tickers,
        start_date=start_date, end_date=end_date,
        universe_only=False,
    )
    if not panels_1m or "close" not in panels_1m:
        log.error("No data in %s — run the data pipeline first.", clean_dir)
        sys.exit(1)
    c1 = panels_1m["close"]
    log.info("  Loaded: %d 1-min bars × %d tickers  (%s → %s)",
             len(c1), len(c1.columns), c1.index.min(), c1.index.max())

    # ── Step 2: Resample to target frequency ──────────────────────────────────
    log.info("[2/5] Resampling 1-min → %s ...", freq)
    resampler = Resampler(panels_1m, freq=freq, session_only=True,
                          min_bars_pct=0.5)
    panels    = read_panels(
        clean_dir, tickers=tickers,
        start_date=start_date, end_date=end_date,
        universe_only=False,
    )
    close     = panels["close"]
    log.info("  Resampled: %d %s bars × %d tickers",
             len(close), freq, len(close.columns))

    # ── Step 3: Compute features ──────────────────────────────────────────────
    log.info("[3/5] Computing 15 alpha features ...")
    eng = FeatureEngine(
        panels,
        atr_window    = atr_window,
        vol_window    = vol_window,
        volume_window = volume_window,
        zscore_window = zscore_window,
        beta_window   = beta_window,
        halflife      = halflife,
    )
    features = eng.compute_all()

    # Print feature summary table
    log.info("")
    log.info("  %-25s  %-16s  %s", "Feature", "Shape", "Valid %")
    log.info("  " + "-" * 52)
    for name, df in features.items():
        pct = 100 * df.notna().values.mean()
        log.info("  %-25s  %-16s  %.1f%%", name, str(df.shape), pct)
    log.info("")

    # ── Step 4: Save to feature store ─────────────────────────────────────────
    log.info("[4/5] Saving %d features to %s ...", len(features), feature_dir)
    store.save(features)
    info      = store.feature_info()
    total_mb  = info["size_mb"].sum() if not info.empty else 0
    log.info("  Saved: %.1f MB total", total_mb)

    # ── Step 5: Diagnostics ───────────────────────────────────────────────────
    log.info("[5/5] Running signal quality diagnostics ...")
    _run_diagnostics(features, close, report_dir, save_report)

    elapsed = time.perf_counter() - t0
    log.info(SEP)
    log.info("  Pipeline complete in %.1fs", elapsed)
    log.info("  Next step: python run_alpha.py  (alpha signal construction)")
    log.info(SEP)
    return {"features": features, "panels": panels}


def _run_diagnostics(features, close, report_dir, save_report):
    diag   = FeatureDiagnostics(features, close)
    report = diag.full_report(forward_bars=1, ic_decay_bars=15)
    diag.print_summary(report)
    if save_report:
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        diag.save_report(report, output_dir=report_dir)
        log.info("Reports saved → %s", report_dir)
        log.info("  Check: reports/feature_ic_summary.csv  (IC t-stats per feature)")
        log.info("  Check: reports/feature_signal_stats.csv  (std, skew, turnover)")


def _parse():
    p = argparse.ArgumentParser(
        description="Feature engineering pipeline: 1-min → 15-min → 15 signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clean-dir",        default="polygon_pipeline/data/clean/")
    p.add_argument("--feature-dir",      default="data/features/")
    p.add_argument("--report-dir",       default="polygon_pipeline/reports/")
    p.add_argument("--log-dir",          default="logs",
                   help="Directory to store log files")
    p.add_argument("--freq",             default="15min",
                   choices=["5min","10min","15min","30min","60min"],
                   help="Target bar frequency (15min recommended)")
    p.add_argument("--tickers",          nargs="+", default=None)
    p.add_argument("--start",            default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end",              default=None, metavar="YYYY-MM-DD")
    p.add_argument("--atr-window",       type=int, default=20)
    p.add_argument("--vol-window",       type=int, default=60)
    p.add_argument("--volume-window",    type=int, default=20)
    p.add_argument("--zscore-window",    type=int, default=60)
    p.add_argument("--beta-window",      type=int, default=390)
    p.add_argument("--halflife",         type=int, default=13)
    p.add_argument("--diagnostics-only", action="store_true")
    p.add_argument("--save-report",      action="store_true")
    p.add_argument("--log-level",        default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    setup_logging(level=args.log_level, log_dir=args.log_dir)
    run(
        clean_dir        = args.clean_dir,
        feature_dir      = args.feature_dir,
        report_dir       = args.report_dir,
        freq             = args.freq,
        tickers          = args.tickers,
        start_date       = args.start,
        end_date         = args.end,
        atr_window       = args.atr_window,
        vol_window       = args.vol_window,
        volume_window    = args.volume_window,
        zscore_window    = args.zscore_window,
        beta_window      = args.beta_window,
        halflife         = args.halflife,
        diagnostics_only = args.diagnostics_only,
        save_report      = args.save_report,
    )
