"""
backtest_single_day.py
======================
Single-day intraday backtest harness. Useful for:
  - Quick validation (runs in <10 seconds)
  - Debug specific market conditions
  - Stress test crisis days
  - Parameter tuning feedback loop

Usage
-----
    python backtest_single_day.py --date 2024-12-15 --freq 5min --save-report
    
    Or programmatically:
    
    from backtest_single_day import backtest_single_day
    
    results = backtest_single_day(
        date="2024-12-15",
        ic_window=60,
        use_precomputed_weights=True,  # Use weights from prior 60 days
    )
    print(f"Daily return: {results['daily_pnl']:.2%}")
    print(f"Sharpe: {results['sharpe']:.2f}")
    print(f"Max DD: {results['max_dd']:.2%}")
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from polygon_pipeline.pipeline.storage import read_panels
from features.resampler import Resampler
from features.engine import FeatureEngine
from features.store import FeatureStore
from alpha.signal import AlphaModel
from alpha.portfolio import PortfolioBuilder

SEP = "─" * 60
log = logging.getLogger(__name__)


def setup_logging(level="INFO"):
    """Configure logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        "%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    if root_logger.handlers:
        root_logger.handlers.clear()
    root_logger.addHandler(console_handler)


def backtest_single_day(
    date: str,
    freq: str = "5min",
    ic_window: int = 60,
    min_ic_tstat: float = 1.0,
    gross_lev: float = 2.0,
    txn_cost_bps: float = 1.0,
    use_precomputed_weights: bool = True,
    clean_dir: str = "polygon_pipeline/data/clean/",
    feature_dir: str = "data/features/",
    report_dir: str = "polygon_pipeline/reports/",
) -> dict:
    """
    Backtest strategy on a single trading day.
    
    Runs the full pipeline (feature engineering → alpha → portfolio) for one day
    and returns performance metrics.
    
    Parameters
    ----------
    date : str
        Trading date (YYYY-MM-DD format)
    freq : str
        Bar frequency ("5min", "15min", etc.). Default "5min".
    ic_window : int
        Rolling window for IC estimation (bars). Default 60.
    min_ic_tstat : float
        Minimum IC t-stat threshold for feature significance. Default 1.0.
    gross_lev : float
        Gross leverage (2.0× = dollar-neutral). Default 2.0.
    txn_cost_bps : float
        Transaction cost in basis points per side. Default 1.0 bps.
    use_precomputed_weights : bool
        If True, use IC weights from prior 60 days (more stable).
        If False, compute weights on the day itself (less stable, but quick).
        Default True.
    clean_dir : str
        Path to cleaned Polygon data. Default "polygon_pipeline/data/clean/".
    feature_dir : str
        Path to feature store. Default "data/features/".
    report_dir : str
        Path to save reports. Default "polygon_pipeline/reports/".
    
    Returns
    -------
    dict with keys:
        "date": str — trading date
        "daily_pnl": float — daily return (as fraction, e.g., 0.005 = 0.5%)
        "sharpe": float — daily Sharpe ratio (annualized if possible)
        "max_dd": float — intraday maximum drawdown
        "n_trades": int — number of rebalances
        "avg_positions": int — average positions held
        "gross_leverage": float — average gross leverage
        "n_stocks": int — number of stocks in universe
        "total_costs": float — total transaction costs (as fraction)
        "alpha_long": float — average alpha in long positions
        "alpha_short": float — average alpha in short positions
        "correlation": float — average cross-sectional correlation
    
    Raises
    ------
    FileNotFoundError
        If data not found for the requested date
    ValueError
        If date has insufficient data (e.g., holiday, too few bars)
    """
    t0 = time.perf_counter()
    
    log.info(SEP)
    log.info("  Single-Day Backtest")
    log.info("  Date: %s", date)
    log.info("  Frequency: %s", freq)
    log.info("  IC window: %d bars", ic_window)
    log.info("  Gross leverage: %.1f×", gross_lev)
    log.info("  Transaction cost: %.1f bps", txn_cost_bps)
    log.info(SEP)
    
    # ── Step 1: Load data for single day ──────────────────────────────────
    log.info("[1/5] Loading %s data for %s ...", freq, date)
    
    try:
        panels = read_panels(
            clean_dir,
            start_date=date,
            end_date=date,
            universe_only=False,
        )
    except Exception as e:
        log.error("Failed to load data: %s", e)
        raise
    
    if not panels or "close" not in panels:
        raise ValueError(f"No data found for {date}")
    
    close = panels["close"]
    
    # Check minimum bars (need at least 60 bars = 5 hours of 5-min bars)
    min_bars = 60 if freq == "5min" else 20
    if len(close) < min_bars:
        raise ValueError(
            f"Insufficient bars: {len(close)} < {min_bars}. "
            f"Date may be holiday or have early close."
        )
    
    log.info("  Loaded: %d %s bars × %d tickers", len(close), freq, len(close.columns))
    
    # ── Step 2: Compute features ─────────────────────────────────────────
    log.info("[2/5] Computing features ...")
    
    engine = FeatureEngine(
        panels,
        atr_window=60,
        vol_window=180,
        volume_window=60,
        zscore_window=180,
        beta_window=1170,
        halflife=39,
    )
    features = engine.compute_all()
    log.info("  Computed %d features", len(features))
    
    # ── Step 3: Build alpha signal ───────────────────────────────────────
    log.info("[3/5] Building alpha signal ...")
    
    if use_precomputed_weights:
        # Load pre-computed weights from feature store (more stable)
        log.info("  Using pre-computed IC weights (from prior 60 days)")
        store = FeatureStore(feature_dir)
        
        # Load prior 60 days of features to compute weights
        prior_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        try:
            features_prior = store.load(start_date=prior_date, end_date=date)
            alpha_model = AlphaModel(features_prior, close, ic_window=ic_window, 
                                    min_ic_tstat=min_ic_tstat)
            weights = alpha_model.ic_weights()
        except Exception as e:
            log.warning("Could not load prior features, computing on the day: %s", e)
            alpha_model = AlphaModel(features, close, ic_window=ic_window,
                                    min_ic_tstat=min_ic_tstat)
            weights = alpha_model.ic_weights()
    else:
        # Compute weights on the day (less stable, but no external dependencies)
        log.info("  Computing IC weights on the day")
        alpha_model = AlphaModel(features, close, ic_window=ic_window,
                                min_ic_tstat=min_ic_tstat)
        weights = alpha_model.ic_weights()
    
    # Combine features into alpha
    alpha = alpha_model._weighted_combine(weights)
    from features.core import cs_zscore
    alpha = cs_zscore(alpha, clip=3.0)
    
    log.info("  Alpha signal: min=%.2f, mean=%.2f, max=%.2f",
            alpha.stack().min(), alpha.stack().mean(), alpha.stack().max())
    
    # ── Step 4: Simulate trading ─────────────────────────────────────────
    log.info("[4/5] Simulating intraday trading ...")
    
    # Use simple vol-scaled positioning
    realized_vol = close.pct_change().rolling(60).std()
    position_sizes = (alpha / realized_vol.clip(lower=0.01)).copy()
    
    # Dollar-neutral constraint: long = short
    long_notional = (position_sizes * close).clip(lower=0).sum(axis=1)
    short_notional = (position_sizes * close).clip(upper=0).sum(axis=1).abs()
    
    # Scale to balance long/short
    scale = (long_notional / (short_notional + 1e-10)).clip(0.5, 2.0)
    for col in position_sizes.columns:
        position_sizes[col] = position_sizes[col] * scale
    
    # Apply gross leverage cap
    gross_notional = (position_sizes * close).abs().sum(axis=1)
    lev_cap = gross_notional / gross_notional.mean()
    position_sizes = position_sizes.div(lev_cap.clip(lower=1.0), axis=0)
    
    # Compute daily PnL
    daily_returns = close.pct_change().fillna(0)
    position_pnl = (position_sizes.shift(1) * daily_returns * close).sum(axis=1)
    
    # Subtract transaction costs (rebalancing cost)
    position_changes = position_sizes.diff().abs().sum(axis=1)
    cost_per_bar = position_changes * close.mean(axis=1) * txn_cost_bps / 10000
    
    net_pnl = position_pnl - cost_per_bar
    
    # Calculate cumulative returns
    cum_pnl = net_pnl.cumsum()
    
    # ── Step 5: Compute metrics ──────────────────────────────────────────
    log.info("[5/5] Computing metrics ...")
    
    daily_return = net_pnl.sum() / 100  # Normalize to % (rough estimate)
    intraday_dd = (cum_pnl - cum_pnl.cummax()).min()
    
    daily_vol = net_pnl.std()
    sharpe = (net_pnl.mean() / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
    
    total_costs = cost_per_bar.sum() / 100  # Rough estimate
    
    avg_pos = (position_sizes.abs() > 0.01).sum(axis=1).mean()
    avg_lev = (position_sizes * close).abs().sum(axis=1).mean() / 100
    
    # Alpha stats
    alpha_long = alpha[alpha > 0].mean().mean()
    alpha_short = alpha[alpha < 0].mean().mean()
    
    # Correlation
    corr_matrix = close.pct_change().corr()
    avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
    
    elapsed = time.perf_counter() - t0
    
    log.info(SEP)
    log.info("  Results:")
    log.info("  Daily return: %.3f%%", daily_return * 100)
    log.info("  Max intraday DD: %.3f%%", intraday_dd * 100)
    log.info("  Sharpe (annualized): %.2f", sharpe)
    log.info("  Total costs: %.3f%%", total_costs * 100)
    log.info("  Avg positions: %.1f / 44", avg_pos)
    log.info("  Avg leverage: %.1f×", avg_lev)
    log.info("  Avg correlation: %.3f", avg_corr)
    log.info("  Elapsed: %.1fs", elapsed)
    log.info(SEP)
    
    return {
        "date": date,
        "daily_pnl": daily_return,
        "sharpe": sharpe,
        "max_dd": intraday_dd * 100,
        "n_trades": int((position_changes > 0.01).sum()),
        "avg_positions": int(avg_pos),
        "gross_leverage": avg_lev,
        "n_stocks": len(close.columns),
        "total_costs": total_costs,
        "alpha_long": alpha_long,
        "alpha_short": alpha_short,
        "correlation": avg_corr,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Single-day intraday backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest_single_day.py --date 2024-12-15
  python backtest_single_day.py --date 2024-12-15 --freq 15min --gross-lev 1.5
  python backtest_single_day.py --date 2024-03-08 --use-precomputed-weights
        """,
    )
    parser.add_argument("--date", type=str, required=True,
                       help="Trading date (YYYY-MM-DD)")
    parser.add_argument("--freq", type=str, default="5min",
                       help="Bar frequency (default: 5min)")
    parser.add_argument("--ic-window", type=int, default=60,
                       help="IC rolling window in bars (default: 60)")
    parser.add_argument("--gross-lev", type=float, default=2.0,
                       help="Gross leverage (default: 2.0)")
    parser.add_argument("--txn-cost-bps", type=float, default=1.0,
                       help="Transaction cost in bps (default: 1.0)")
    parser.add_argument("--use-precomputed-weights", action="store_true",
                       help="Use weights from prior 60 days (more stable)")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    setup_logging(args.log_level)
    
    try:
        results = backtest_single_day(
            date=args.date,
            freq=args.freq,
            ic_window=args.ic_window,
            gross_lev=args.gross_lev,
            txn_cost_bps=args.txn_cost_bps,
            use_precomputed_weights=args.use_precomputed_weights,
        )
        
        # Pretty print results
        print("\n" + SEP)
        print("SINGLE-DAY BACKTEST RESULTS")
        print(SEP)
        for key, val in results.items():
            if isinstance(val, float):
                if "pnl" in key or "dd" in key or "costs" in key:
                    print(f"{key:25s} {val:10.3f}%")
                elif "corr" in key:
                    print(f"{key:25s} {val:10.4f}")
                else:
                    print(f"{key:25s} {val:10.2f}")
            else:
                print(f"{key:25s} {val:10}")
        print(SEP + "\n")
        
    except Exception as e:
        log.error("Backtest failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
