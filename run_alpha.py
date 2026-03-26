"""
run_alpha.py
============
Phase 2 entry point: Alpha Signal + Portfolio Construction.

Loads features from the feature store, builds the composite alpha signal,
constructs portfolio weights, and writes everything to disk with reports.

Usage
-----
    python run_alpha.py                        # full run, all tickers
    python run_alpha.py --tickers AAPL MSFT   # subset
    python run_alpha.py --start 2025-01-01    # date range
    python run_alpha.py --ic-window 60        # tune IC window
    python run_alpha.py --gross-lev 2.0       # tune leverage
    python run_alpha.py --report-only         # reload and re-report

Output files
------------
    data/alpha/composite_alpha.parquet   — alpha signal [timestamp × ticker]
    data/alpha/weights.parquet           — position weights [timestamp × ticker]
    reports/ic_summary.csv               — IC per feature (your signal quality report)
    reports/ic_decay.csv                 — IC decay curve for best feature
    reports/portfolio_stats.csv          — gross Sharpe, turnover, leverage
    reports/alpha_pipeline.log           — full execution log
"""

from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "polygon_pipeline"))

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline.storage   import read_panels
from features.resampler import Resampler
from features.store     import FeatureStore
from alpha.signal       import AlphaModel, compute_ic_decay, estimate_halflife
from alpha.portfolio    import PortfolioBuilder

SEP = "─" * 60
log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level   = getattr(logging, level.upper()),
        format  = "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers= handlers,
    )


def run(
    clean_dir     = "polygon_pipeline/data/clean_1min/",
    feature_dir   = "data/features/",
    alpha_dir     = "data/alpha/",
    report_dir    = "reports/",
    freq          = "1min",
    tickers       = None,
    start_date    = None,
    end_date      = None,
    ic_window     = 300,
    min_ic_tstat  = 1.0,
    vol_window    = 900,
    halflife      = 195,
    target_vol    = 0.15,
    max_weight    = 0.10,
    gross_lev     = 1.5,
    turnover_thr  = 0.6,
    min_adtv_usd  = 100_000,
    rebalance_freq = 1,
    txn_cost_bps  = 1.0,
    report_only   = False,
) -> dict:

    t0 = time.perf_counter()
    Path(alpha_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    log.info(SEP)
    log.info("  Alpha Signal + Portfolio Construction  (Phase 2)")
    log.info("  Feature store : %s", feature_dir)
    log.info("  Alpha output  : %s", alpha_dir)
    log.info("  Rebalance freq: every %d bars (~%d min) | Txn cost: %.1f bps", 
             rebalance_freq, rebalance_freq * 15, txn_cost_bps)
    log.info("  Date range    : %s → %s", start_date or "all", end_date or "latest")
    log.info(SEP)

    # ── Step 1: Load features from store ──────────────────────────────────────
    log.info("[1/6] Loading features ...")
    store    = FeatureStore(feature_dir)
    features = store.load(tickers=tickers,
                          start_date=start_date, end_date=end_date)
    if not features:
        log.error("No features in %s — run python run_features.py first.", feature_dir)
        sys.exit(1)
    log.info("  Loaded %d features", len(features))

    # ── Step 2: Load close + volume panels aligned to feature timestamps ─────
    log.info("[2/6] Loading price panels for vol scaling ...")
    panels = read_panels(clean_dir, tickers=tickers,
                         start_date=start_date, end_date=end_date,
                         universe_only=False)
     # Note: data in clean_dir is at 1-minute frequency
     # Optionally resample to target frequency if desired:
     # panels = Resampler(panels, freq=freq, session_only=False).resample()

    # Align close/volume to the feature timestamps (features are the authority)
    feat_idx  = next(iter(features.values())).index
    close     = panels["close"].reindex(feat_idx)
    volume    = panels["volume"].reindex(feat_idx)

    # Drop all-NaN rows
    valid     = close.notna().any(axis=1)
    close     = close[valid]
    volume    = volume[valid]

    log.info("  Close panel: %s  (aligned to feature index)", close.shape)

    # ── Step 3: Build IC summary table ────────────────────────────────────────
    log.info("[3/6] Computing IC summary ...")
    model = AlphaModel(
        features     = features,
        close        = close,
        ic_window    = ic_window,
        min_ic_tstat = min_ic_tstat,
    )
    ic_table = model.ic_summary_table()
    _print_ic_table(ic_table)
    ic_table.to_csv(Path(report_dir) / "ic_summary.csv")
    log.info("  IC summary saved → %s/ic_summary.csv", report_dir)

    # ── IC health check ───────────────────────────────────────────────────────
    active_features = ic_table[ic_table["active"] == True] if not ic_table.empty else pd.DataFrame()
    n_active = len(active_features)
    log.info("  %d / %d features pass IC t-stat threshold (%.1f)",
             n_active, len(features), min_ic_tstat)

    if n_active == 0:
        log.error(
            "NO features pass the IC t-stat threshold of %.1f.\n"
            "  This means your signal quality is below viable threshold.\n"
            "  Do not proceed to portfolio construction.\n"
            "  Action: review reports/ic_summary.csv and check:\n"
            "    1. Is the feature store populated? (run python run_features.py)\n"
            "    2. Is the date range long enough? (need 60+ bars of history)\n"
            "    3. Try lowering --min-ic-tstat to 0.5 to see if any signal exists.",
            min_ic_tstat
        )
        return {"status": "no_signal", "ic_table": ic_table}

    # ── IC decay for best feature ─────────────────────────────────────────────
    if not ic_table.empty:
        best_feat = ic_table.index[0]
        log.info("  Computing IC decay for best feature: %s", best_feat)
        decay = compute_ic_decay(features[best_feat], close, max_lead=15)
        if not decay.empty:
            hl = estimate_halflife(decay)
            log.info("  Signal half-life (best feature): %.1f bars (%.0f min)",
                     hl, hl * 15 if hl != float('inf') else 0)
            decay.to_csv(Path(report_dir) / "ic_decay.csv")

    if report_only:
        log.info("Report-only mode. Stopping after IC analysis.")
        return {"ic_table": ic_table}

    # ── Step 4: Build composite alpha ─────────────────────────────────────────
    log.info("[4/6] Building composite alpha signal ...")
    alpha = model.composite_alpha()
    _save_panel(alpha, Path(alpha_dir) / "composite_alpha.parquet")
    log.info("  Composite alpha saved: %s", alpha.shape)

    # ── Step 5: Portfolio construction ────────────────────────────────────────
    log.info("[5/6] Constructing portfolio weights ...")
    
    # Downsample alpha if rebalancing less frequently
    if rebalance_freq > 1:
        log.info("  Downsampling alpha: every %d bars (was %d bars)", 
                 rebalance_freq, len(alpha))
        alpha_rebal = alpha.iloc[::rebalance_freq].copy()
    else:
        alpha_rebal = alpha
    
    builder = PortfolioBuilder(
        alpha=alpha_rebal,
        close=close,
        volume=volume,
        halflife=195,   # 1-minute bars (5x from 39 bars at 5-min)
    )
    weights_rebal = builder.build()
    
    # Forward-fill weights back to original frequency for stats computation
    if rebalance_freq > 1:
        weights = weights_rebal.reindex(alpha.index).fillna(method='ffill')
        log.info("  Forward-filled weights to original frequency: %s", weights.shape)
    else:
        weights = weights_rebal
    
    _save_panel(weights, Path(alpha_dir) / "weights.parquet")
    log.info("  Weights saved: %s", weights.shape)

    # ── Step 6: Portfolio statistics ──────────────────────────────────────────
    log.info("[6/6] Computing portfolio statistics ...")
    fwd_ret = close.pct_change(1).shift(-1)
    stats   = builder.portfolio_stats(weights, fwd_ret)
    
    # Turnover on rebalanced weights only (actual rebalance cost)
    ann_to_rebal = builder.turnover(weights_rebal).mean() * builder.bars_per_year
    stats['rebalance_turnover'] = ann_to_rebal
    stats['txn_cost_bps'] = txn_cost_bps
    stats['annual_cost'] = (ann_to_rebal * txn_cost_bps / 10000)
    stats['net_return_ann'] = stats['gross_return_ann'] - stats['annual_cost']

    pd.Series(stats).to_frame("value").to_csv(
        Path(report_dir) / "portfolio_stats.csv"
    )
    _print_portfolio_stats(stats)

    elapsed = time.perf_counter() - t0
    log.info(SEP)
    log.info("  Phase 2 complete in %.1fs", elapsed)
    log.info("  Next: python run_backtest.py  (walk-forward validation)")
    log.info(SEP)

    return {
        "ic_table":  ic_table,
        "alpha":     alpha,
        "weights":   weights,
        "stats":     stats,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_ic_table(df: pd.DataFrame) -> None:
    if df.empty:
        log.info("  (no IC data)")
        return
    log.info("")
    log.info("  %-25s  %8s  %8s  %7s  %7s  %8s  %s",
             "Feature", "IC_mean", "IC_std", "ICIR", "t_stat", "pct_pos", "active")
    log.info("  " + "-" * 78)
    for name, row in df.iterrows():
        flag = "  ✓" if row.get("active") else "  ✗"
        log.info(
            "  %-25s  %+.5f  %.5f  %+.3f  %+.3f  %.3f  %s",
            name,
            row["IC_mean"], row["IC_std"],
            row["ICIR"],    row["t_stat"],
            row["pct_positive"], flag,
        )
    log.info("")


def _print_portfolio_stats(stats: dict) -> None:
    log.info("")
    log.info("  Portfolio Statistics:")
    log.info("  %-30s  %s (gross)", "Annual Return",        f"{stats['gross_return_ann']*100:.2f}%")
    log.info("  %-30s  %s (%.1f bps cost)", "  → Net after costs",  
             f"{stats.get('net_return_ann', stats['gross_return_ann'])*100:.2f}%",
             stats.get('txn_cost_bps', 0) * stats.get('rebalance_turnover', 1))
    log.info("  %-30s  %s", "Annual Vol",              f"{stats['gross_vol_ann']*100:.2f}%")
    log.info("  %-30s  %s", "Gross Sharpe Ratio",      f"{stats['gross_sharpe']:.3f}")
    log.info("  %-30s  %s", "Rebalance Turnover",      f"{stats.get('rebalance_turnover', stats['annual_turnover']):.0f}×")
    log.info("  %-30s  %s", "Annual Cost (realized)",  f"{stats.get('annual_cost', 0)*100:.2f}%")
    log.info("  %-30s  %s", "Avg Positions",           f"{stats['avg_positions']:.0f}")
    log.info("  %-30s  %s", "Avg Gross Leverage",      f"{stats['avg_gross_lev']:.3f}×")
    log.info("  %-30s  %s", "Avg Net Leverage",        f"{stats['avg_net_lev']:.4f}")

def _save_panel(df: pd.DataFrame, path: Path) -> None:
    """Save a wide DataFrame (timestamp × ticker) to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    long = df.stack(future_stack=True).reset_index()
    long.columns = ["timestamp", "ticker", "value"]
    long["timestamp"] = pd.to_datetime(long["timestamp"])
    if long["timestamp"].dt.tz is None:
        long["timestamp"] = long["timestamp"].dt.tz_localize("America/New_York")
    table = pa.Table.from_pandas(long, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2: Alpha Signal Construction + Portfolio Construction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--clean-dir",     default="polygon_pipeline/data/clean_1min/",
                   help="Input bars dir. Use clean_1min/ for 1-min, clean_5min/ for 5-min")
    p.add_argument("--feature-dir",   default="data/features/")
    p.add_argument("--alpha-dir",     default="data/alpha/")
    p.add_argument("--report-dir",    default="reports/")
    p.add_argument("--freq",          default="1min")
    p.add_argument("--tickers",       nargs="+", default=None)
    p.add_argument("--start",         default=None, metavar="YYYY-MM-DD")
    p.add_argument("--end",           default=None, metavar="YYYY-MM-DD")
    # Alpha model params
    p.add_argument("--ic-window",     type=int,   default=300,
                   help="Rolling bars for IC estimation (300 bars ≈ 5 hours)")
    p.add_argument("--min-ic-tstat",  type=float, default=1.0,
                   help="Min IC t-stat for a feature to be included")
    # Portfolio params
    p.add_argument("--vol-window",    type=int,   default=900)
    p.add_argument("--halflife",      type=int,   default=195)
    p.add_argument("--target-vol",    type=float, default=0.15)
    p.add_argument("--max-weight",    type=float, default=0.10)
    p.add_argument("--gross-lev",     type=float, default=1.5,
                   help="Gross leverage (1.5x is safer than 2.0x with weak signal)")
    p.add_argument("--turnover-thr",  type=float, default=0.6)
    p.add_argument("--min-adtv-usd",  type=float, default=1_000_000)
    # Rebalancing & costs
    p.add_argument("--rebalance-freq",type=int,   default=1,
                   help="Rebalance every N bars (1=each minute, 5=every 5min).")
    p.add_argument("--txn-cost-bps",  type=float, default=1.0,
                   help="Transaction cost in basis points per sided trade")
    # Modes
    p.add_argument("--report-only",   action="store_true",
                   help="Only compute and print IC table — skip portfolio build")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    setup_logging(
        args.log_level,
        log_file="reports/alpha_pipeline.log",
    )
    run(
        clean_dir      = args.clean_dir,
        feature_dir    = args.feature_dir,
        alpha_dir      = args.alpha_dir,
        report_dir     = args.report_dir,
        freq           = args.freq,
        tickers        = args.tickers,
        start_date     = args.start,
        end_date       = args.end,
        ic_window      = args.ic_window,
        min_ic_tstat   = args.min_ic_tstat,
        vol_window     = args.vol_window,
        halflife       = args.halflife,
        target_vol     = args.target_vol,
        max_weight     = args.max_weight,
        gross_lev      = args.gross_lev,
        turnover_thr   = args.turnover_thr,
        min_adtv_usd   = args.min_adtv_usd,
        rebalance_freq = args.rebalance_freq,
        txn_cost_bps   = args.txn_cost_bps,
        report_only    = args.report_only,
    )