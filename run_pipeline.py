"""
run_pipeline.py
===============
End-to-end pipeline: Screen Universe → Fetch Data → Clean → Features → Alpha → Backtest

Usage
-----
    python run_pipeline.py                          # full run
    python run_pipeline.py --days 30                # shorter history
    python run_pipeline.py --skip-universe          # skip screening, use seed pool
    python run_pipeline.py --tickers RELIANCE TCS   # specific tickers only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nse_pipeline.orchestrator import NSEPipeline
from features.engine import FeatureEngine
from features.store import FeatureStore
from alpha.signal import AlphaModel, compute_ic_decay, estimate_halflife
from alpha.portfolio import PortfolioBuilder
from alpha.risk_management import RiskManager

SEP = "─" * 60
log = logging.getLogger(__name__)

# NSE constants
NSE_BARS_PER_YEAR = 18_900  # 252 days × 75 bars/day


def load_config(path: str) -> dict:
    """Load config.yaml and return as dict with flat defaults."""
    p = Path(path)
    if not p.exists():
        log.warning("Config not found: %s — using hardcoded defaults", path)
        return {}
    with open(p) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def run(
    config_path: str = "config/config.yaml",
    days: int = 59,
    skip_universe: bool = False,
    tickers: list[str] = None,
    report_only: bool = False,
    # Feature params
    atr_window: int = 30,
    vol_window: int = 120,
    volume_window: int = 60,
    zscore_window: int = 120,
    beta_window: int = 750,
    # Alpha params
    ic_window: int = 120,
    min_ic_tstat: float = 0.5,
    # Portfolio params
    halflife: int = 100,
    gross_lev: float = 1.0,
    max_weight: float = 0.10,
    txn_cost_bps: float = 1.0,
    rebalance_freq: int = 75,
    weight_smooth_halflife: int = 30,
    # Capital
    capital: float = 1_000_000,
) -> dict:
    """
    Execute the full pipeline.

    Returns dict with all results.
    """
    t0 = time.perf_counter()

    log.info(SEP)
    log.info("  NSE Intraday Cross-Sectional Mean Reversion")
    log.info("  Capital: ₹%.0fL | Gross Lev: %.1f× | Txn Cost: %.1f bps",
             capital / 100_000, gross_lev, txn_cost_bps)
    log.info("  Rebalance: every %d bars (%d min) | Weight EWM smooth: %d bars",
             rebalance_freq, rebalance_freq * 5, weight_smooth_halflife)
    log.info(SEP)

    # ── Phase 1: Data Pipeline ────────────────────────────────────────────────
    log.info("\n[PHASE 1] Data Pipeline")
    log.info(SEP)

    pipeline = NSEPipeline(config_path)
    result = pipeline.run(
        days=days,
        skip_universe_screen=skip_universe,
        tickers=tickers,
    )

    panels = result["panels"]
    volatile = result["volatile"]
    nonvolatile = result["nonvolatile"]
    nifty_prices = result.get("nifty_prices", pd.Series())
    nifty_returns = result.get("nifty_returns", pd.Series())

    if not panels:
        log.error("No data — aborting")
        return {"status": "no_data"}

    close = panels["close"]
    volume = panels["volume"]

    log.info("  Data: %d bars × %d tickers", len(close), len(close.columns))
    log.info("  Volatile: %d | Non-volatile: %d",
             len(volatile), len(nonvolatile))

    # ── Phase 2: Feature Engineering ──────────────────────────────────────────
    log.info("\n[PHASE 2] Feature Engineering")
    log.info(SEP)

    engine = FeatureEngine(
        panels,
        atr_window=atr_window,
        vol_window=vol_window,
        volume_window=volume_window,
        zscore_window=zscore_window,
        beta_window=beta_window,
    )
    features = engine.compute_all()

    # Apply time-of-day weighting for maximum return
    features = engine.apply_tod_weights(features)

    log.info("  %d features computed", len(features))

    # Save features
    store = FeatureStore("data/features/")
    store.save(features)
    log.info("  Features saved to data/features/")

    # ── Phase 3: Alpha Signal ─────────────────────────────────────────────────
    log.info("\n[PHASE 3] Alpha Signal Construction")
    log.info(SEP)

    model = AlphaModel(
        features=features,
        close=close,
        ic_window=ic_window,
        min_ic_tstat=min_ic_tstat,
    )

    # IC summary
    ic_table = model.ic_summary_table()
    if not ic_table.empty:
        _print_ic_table(ic_table)
        Path("reports").mkdir(parents=True, exist_ok=True)
        ic_table.to_csv("reports/ic_summary.csv")

    # Check signal quality
    active = ic_table[ic_table["active"] == True] if not ic_table.empty else pd.DataFrame()
    log.info("  %d / %d features pass IC threshold", len(active), len(features))

    if len(active) == 0:
        log.warning("NO features pass IC threshold — signal may be weak")

    if report_only:
        return {"ic_table": ic_table, "features": features}

    # Composite alpha
    alpha = model.composite_alpha()
    log.info("  Composite alpha: %s", alpha.shape)

    # ── Phase 4: Portfolio Construction ────────────────────────────────────────
    log.info("\n[PHASE 4] Portfolio Construction")
    log.info(SEP)

    # Downsample alpha for rebalancing: take every Nth bar
    if rebalance_freq > 1:
        alpha_rebal = alpha.iloc[::rebalance_freq].copy()
    else:
        alpha_rebal = alpha

    builder = PortfolioBuilder(
        alpha=alpha_rebal,
        close=close,
        volume=volume,
        halflife=halflife,
        max_weight=max_weight,
        gross_lev=gross_lev,
        bars_per_year=NSE_BARS_PER_YEAR,
        min_adtv_usd=500_000,  # ₹5L in INR context (lower for NSE)
    )
    weights_rebal = builder.build()

    # Forward-fill to full bar frequency
    if rebalance_freq > 1:
        weights = weights_rebal.reindex(alpha.index).ffill()
    else:
        weights = weights_rebal

    # ── CRITICAL: Smooth weights with EWM to prevent excessive turnover ──────
    if weight_smooth_halflife > 0:
        log.info("  Smoothing weights (EWM halflife=%d bars) ...", weight_smooth_halflife)
        alpha_ewm = 2.0 / (weight_smooth_halflife + 1)
        weights = weights.ewm(halflife=weight_smooth_halflife, adjust=False).mean()
        # Re-normalize after smoothing so gross leverage is maintained
        gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
        weights = weights.div(gross, axis=0).fillna(0.0) * gross_lev

    # Apply risk management
    rm = RiskManager(capital=capital)
    weights = rm.apply(weights, close, nifty_returns)

    log.info("  Final weights: %s", weights.shape)

    # ── Phase 5: Performance Stats ────────────────────────────────────────────
    log.info("\n[PHASE 5] Performance Statistics")
    log.info(SEP)

    fwd_ret = close.pct_change(1, fill_method=None).shift(-1)
    stats = builder.portfolio_stats(weights, fwd_ret)

    # Transaction costs — measure turnover on the SMOOTHED weights (the real ones)
    turnover_per_bar = _compute_turnover(weights)
    ann_turnover = turnover_per_bar.mean() * NSE_BARS_PER_YEAR
    stats["rebalance_turnover"] = ann_turnover
    stats["txn_cost_bps"] = txn_cost_bps
    stats["annual_cost"] = ann_turnover * txn_cost_bps / 10000
    stats["net_return_ann"] = stats.get("gross_return_ann", 0) - stats["annual_cost"]

    _print_stats(stats)

    # Save stats
    pd.Series(stats).to_frame("value").to_csv("reports/portfolio_stats.csv")

    elapsed = time.perf_counter() - t0
    log.info(SEP)
    log.info("  Pipeline complete in %.1fs", elapsed)
    log.info(SEP)

    return {
        "ic_table": ic_table,
        "alpha": alpha,
        "weights": weights,
        "stats": stats,
        "panels": panels,
        "volatile": volatile,
        "nonvolatile": nonvolatile,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """
    Compute one-way turnover as fraction of gross NAV per bar.
    Measured on the actual (smoothed) weights that would be traded.
    """
    delta = weights.diff().abs().sum(axis=1)
    gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return (delta / gross).fillna(0.0)


def _print_ic_table(df: pd.DataFrame) -> None:
    if df.empty:
        return
    log.info("")
    log.info("  %-25s  %8s  %8s  %7s  %7s  %8s  %s",
             "Feature", "IC_mean", "IC_std", "ICIR", "t_stat", "pct_pos", "active")
    log.info("  " + "-" * 78)
    for name, row in df.iterrows():
        flag = "  ✓" if row.get("active") else "  ✗"
        log.info(
            "  %-25s  %+.5f  %.5f  %+.3f  %+.3f  %.3f  %s",
            name, row["IC_mean"], row["IC_std"],
            row["ICIR"], row["t_stat"], row["pct_positive"], flag,
        )
    log.info("")


def _print_stats(stats: dict) -> None:
    log.info("")
    log.info("  Portfolio Statistics:")
    log.info("  %-30s  %s", "Annual Return (gross)",
             f"{stats.get('gross_return_ann', 0)*100:.2f}%")
    log.info("  %-30s  %s", "Annual Return (net)",
             f"{stats.get('net_return_ann', 0)*100:.2f}%")
    log.info("  %-30s  %s", "Annual Volatility",
             f"{stats.get('gross_vol_ann', 0)*100:.2f}%")
    log.info("  %-30s  %s", "Sharpe Ratio",
             f"{stats.get('gross_sharpe', 0):.3f}")
    log.info("  %-30s  %s", "Turnover (annual)",
             f"{stats.get('rebalance_turnover', 0):.0f}×")
    log.info("  %-30s  %s", "Transaction Cost",
             f"{stats.get('annual_cost', 0)*100:.2f}%")
    log.info("  %-30s  %s", "Avg Positions",
             f"{stats.get('avg_positions', 0):.0f}")
    log.info("  %-30s  %s", "Avg Gross Leverage",
             f"{stats.get('avg_gross_lev', 0):.3f}×")


def _parse() -> argparse.Namespace:
    """Parse CLI args, then overlay config.yaml values as defaults."""
    # First pass: get config path
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="config/config.yaml")
    pre_args, _ = pre.parse_known_args()

    # Load config.yaml
    cfg = load_config(pre_args.config)
    bt = cfg.get("backtest", {})
    feat = cfg.get("features", {})
    alpha_cfg = cfg.get("alpha", {})

    # Second pass: real parser with config.yaml values as defaults
    p = argparse.ArgumentParser(
        description="NSE Intraday Mean Reversion — Full Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--days", type=int,
                   default=cfg.get("data", {}).get("intraday_max_days", 59))
    p.add_argument("--skip-universe", action="store_true")
    p.add_argument("--tickers", nargs="+", default=None)
    p.add_argument("--report-only", action="store_true")
    # Feature params — defaults from config
    p.add_argument("--atr-window", type=int, default=feat.get("atr_window", 30))
    p.add_argument("--vol-window", type=int, default=feat.get("vol_window", 120))
    # Alpha params — defaults from config
    p.add_argument("--ic-window", type=int, default=alpha_cfg.get("ic_window", 120))
    p.add_argument("--min-ic-tstat", type=float,
                   default=alpha_cfg.get("min_ic_tstat", 0.5))
    # Portfolio params — defaults from config
    p.add_argument("--halflife", type=int, default=bt.get("halflife", 100))
    p.add_argument("--gross-lev", type=float, default=bt.get("gross_lev", 1.0))
    p.add_argument("--max-weight", type=float, default=bt.get("max_weight", 0.10))
    p.add_argument("--txn-cost-bps", type=float, default=bt.get("txn_cost_bps", 1.0))
    p.add_argument("--rebalance-freq", type=int,
                   default=bt.get("rebalance_freq", 75))
    p.add_argument("--weight-smooth", type=int, default=30,
                   help="EWM halflife for weight smoothing (0 = disabled)")
    p.add_argument("--capital", type=float, default=bt.get("capital", 1_000_000))
    # Logging
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    setup_logging(args.log_level, log_file="reports/pipeline.log")

    run(
        config_path=args.config,
        days=args.days,
        skip_universe=args.skip_universe,
        tickers=args.tickers,
        report_only=args.report_only,
        atr_window=args.atr_window,
        vol_window=args.vol_window,
        ic_window=args.ic_window,
        min_ic_tstat=args.min_ic_tstat,
        halflife=args.halflife,
        gross_lev=args.gross_lev,
        max_weight=args.max_weight,
        txn_cost_bps=args.txn_cost_bps,
        rebalance_freq=args.rebalance_freq,
        weight_smooth_halflife=args.weight_smooth,
        capital=args.capital,
    )
