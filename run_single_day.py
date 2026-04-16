"""
run_single_day.py
=================
Single-Day Backtester for NASDAQ Intraday Max-Profit Strategy

Simulates the full day-trading workflow:
  1. Pre-open: Score all stocks using overnight & previous-day features
  2. Opening: Confirm picks using first 15 min of trading
  3. Execute: Enter at 9:45 AM ET, manage stops/targets intraday
  4. Exit: Flatten all positions by 3:50 PM ET
  5. Report: Print detailed P&L and trade log

Usage
-----
    python run_single_day.py --date 2026-04-11
    python run_single_day.py --last 5
    python run_single_day.py --from 2026-03-01 --to 2026-04-11
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nse_pipeline.orchestrator import NSEPipeline
from alpha.stock_picker import StockPicker
from alpha.execution import IntradayExecutor

SEP = "═" * 60
THIN = "─" * 60
log = logging.getLogger(__name__)


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))
        
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def run_single_day(
    panels: dict,
    nifty_close: pd.Series,
    target_date: pd.Timestamp,
    capital: float = 1_000_000,
    n_picks: int = 80,
    stop_loss_pct: float = 0.02,
    profit_take: float = 0.015,
    trail_trigger: float = 0.025,
) -> dict:
    """
    Run the strategy for a single day.
    
    Returns dict with full day result.
    """
    target_date = pd.Timestamp(target_date).normalize()
    
    # 1. Pick stocks
    picker = StockPicker(
        panels=panels,
        nifty_close=nifty_close,
        capital=capital,
        n_picks=n_picks,
        min_score=0.3,
        min_avg_volume=200_000,
    )
    
    picks = picker.pick(target_date)
    
    if not picks:
        log.warning("  No picks for %s — skipping", target_date.date())
        return {"date": target_date, "pnl": 0, "trades": [], "status": "no_picks"}
    
    # 2. Execute intraday
    executor = IntradayExecutor(
        stop_loss_pct=stop_loss_pct,
        trailing_stop_trigger=trail_trigger,
        trailing_stop_pct=0.0075,
        profit_take_1=profit_take,
        profit_take_2=0.03,
        profit_take_3=0.045,
        exit_bar=76,
    )
    
    dates_idx = panels["close"].index.normalize()
    date_mask = dates_idx == target_date
    
    result = executor.simulate_day(
        picks=picks,
        close=panels["close"],
        high=panels["high"],
        low=panels["low"],
        date_mask=date_mask,
    )
    
    result["date"] = target_date
    result["picks"] = picks
    result["status"] = "ok"
    
    # 3. Print report
    _print_day_report(result, capital)
    
    return result


def run_backtest(
    config_path: str = "config/config.yaml",
    days: int = 59,
    target_date: str = None,
    last_n: int = None,
    from_date: str = None,
    to_date: str = None,
    capital: float = 1_000_000,
    n_picks: int = 80,
    stop_loss_pct: float = 0.02,
    profit_take: float = 0.015,
    trail_trigger: float = 0.025,
    skip_universe: bool = True,
) -> list[dict]:
    """
    Run the full backtest pipeline.
    """
    t0 = time.perf_counter()
    
    log.info(SEP)
    log.info("  NASDAQ Single-Day Max-Profit Backtester")
    log.info(SEP)
    
    # ── Fetch Data ────────────────────────────────────────────────────
    log.info("\n[DATA] Fetching market data ...")
    pipeline = NSEPipeline(config_path)
    result = pipeline.run(days=days, skip_universe_screen=skip_universe)
    
    panels = result["panels"]
    nifty_close = result.get("nifty_prices", pd.Series())
    
    if not panels:
        log.error("No data — aborting")
        return []
    
    close = panels["close"]
    dates_idx = close.index.normalize()
    unique_dates = sorted(dates_idx.unique())
    
    log.info("  Data: %d bars × %d tickers | %d trading days",
             len(close), len(close.columns), len(unique_dates))
    
    # ── Attach backtest-only file logger (after data fetch) ────────────
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    bt_log_name = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    bt_log_path = reports_dir / bt_log_name
    bt_handler = logging.FileHandler(bt_log_path, mode="w", encoding="utf-8")
    bt_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(bt_handler)
    log.info("  Backtest log: %s", bt_log_path)
    log.info(SEP)
    log.info("  BACKTEST CONFIGURATION:")
    log.info("  Capital       : $%sK", f"{capital / 1_000:.0f}")
    log.info("  Stock Picks   : %d", n_picks)
    log.info("  Stop-Loss     : %.2f%%", stop_loss_pct * 100)
    log.info("  Profit Target : %.2f%%", profit_take * 100)
    log.info("  Trail Trigger : %.2f%%", trail_trigger * 100)
    log.info(SEP)
    
    # ── Determine which dates to backtest ──────────────────────────
    # Need at least 20 days of history for features
    min_lookback = 20
    tradeable_dates = unique_dates[min_lookback:]
    
    if target_date:
        td = pd.Timestamp(target_date).normalize()
        if hasattr(tradeable_dates[0], "tz") and tradeable_dates[0].tz:
            td = td.tz_localize(tradeable_dates[0].tz)
        if td in tradeable_dates:
            test_dates = [td]
        else:
            log.error("Date %s not available. Available: %s to %s",
                     td.date(), tradeable_dates[0].date(), tradeable_dates[-1].date())
            return []
    elif last_n:
        test_dates = tradeable_dates[-last_n:]
    elif from_date and to_date:
        fd = pd.Timestamp(from_date).normalize()
        td = pd.Timestamp(to_date).normalize()
        if hasattr(tradeable_dates[0], "tz") and tradeable_dates[0].tz:
            fd = fd.tz_localize(tradeable_dates[0].tz)
            td = td.tz_localize(tradeable_dates[0].tz)
        test_dates = [d for d in tradeable_dates if fd <= d <= td]
    else:
        test_dates = tradeable_dates
    
    log.info("  Backtesting %d days: %s → %s",
             len(test_dates), test_dates[0].date(), test_dates[-1].date())
    log.info(THIN)
    
    # ── Run each day ──────────────────────────────────────────────────
    all_results = []
    
    for date in test_dates:
        day_result = run_single_day(
            panels=panels,
            nifty_close=nifty_close,
            target_date=date,
            capital=capital,
            n_picks=n_picks,
            stop_loss_pct=stop_loss_pct,
            profit_take=profit_take,
            trail_trigger=trail_trigger,
        )
        all_results.append(day_result)
    
    # ── Summary ────────────────────────────────────────────────────────
    _print_summary(all_results, capital)
    
    elapsed = time.perf_counter() - t0
    log.info("  Backtest complete in %.1fs", elapsed)
    
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _print_day_report(result: dict, capital: float) -> None:
    """Print detailed trade log for a single day."""
    trades = result.get("trades", [])
    day_pnl = result.get("day_pnl", 0)
    date = result.get("date", "")
    
    if isinstance(date, pd.Timestamp):
        date_str = date.strftime("%Y-%m-%d (%A)")
    else:
        date_str = str(date)
    
    log.info("")
    log.info(SEP)
    log.info("  Date: %s", date_str)
    log.info(SEP)
    
    if not trades:
        log.info("  No trades executed.")
        return
    
    # Trade log header
    log.info("  %-12s  %-5s  %-8s  %10s  %10s  %8s  %10s  %s",
             "Ticker", "Side", "Shares", "Entry", "Exit", "P&L%", "P&L $", "Reason")
    log.info("  " + "-" * 85)
    
    for t in trades:
        pnl_str = f"${t.pnl:+,.0f}"
        pnl_pct = f"{t.pnl_pct*100:+.2f}%"
        entry_str = f"${t.entry_price:,.2f}"
        exit_str = f"${t.exit_price:,.2f}"
        
        log.info("  %-12s  %-5s  %-8d  %10s  %10s  %8s  %10s  %s",
                t.ticker, "LONG", t.shares,
                entry_str, exit_str, pnl_pct, pnl_str, t.exit_reason)
    
    log.info("  " + "-" * 85)
    
    # Day summary
    pnl_color = "+" if day_pnl >= 0 else ""
    day_pnl_pct = result.get("day_pnl_pct", 0)
    deployed = result.get("capital_deployed", capital)
    
    log.info("  Day P&L: %s  (%s on %s deployed)",
             f"${day_pnl:+,.0f}", f"{day_pnl_pct*100:+.2f}%", f"${deployed:,.0f}")
    log.info("  Stopped: %d | Profit-taken: %d | Trailing: %d | Time-exit: %d",
             result.get("n_stopped", 0),
             result.get("n_profit_taken", 0),
             result.get("n_trailing", 0),
             result.get("n_time_exit", 0))
    log.info(SEP)


def _print_summary(results: list[dict], capital: float) -> None:
    """Print multi-day backtest summary."""
    if not results:
        return
    
    valid = [r for r in results if r.get("status") == "ok"]
    if not valid:
        log.info("  No valid trading days found.")
        return
    
    pnls = [r.get("day_pnl", 0) for r in valid]
    pnl_pcts = [r.get("day_pnl_pct", 0) for r in valid]
    
    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)
    win_days = sum(1 for p in pnls if p > 0)
    lose_days = sum(1 for p in pnls if p < 0)
    flat_days = sum(1 for p in pnls if p == 0)
    win_rate = win_days / len(pnls) * 100 if pnls else 0
    
    # Best and worst days
    best_idx = np.argmax(pnls)
    worst_idx = np.argmin(pnls)
    
    # Count stops
    total_trades = sum(len(r.get("trades", [])) for r in valid)
    total_stops = sum(r.get("n_stopped", 0) for r in valid)
    
    log.info("")
    log.info(SEP)
    log.info("  BACKTEST SUMMARY: %d trading days", len(valid))
    log.info(SEP)
    log.info("")
    log.info("  %-30s  %s", "Total P&L", f"${total_pnl:+,.0f}")
    log.info("  %-30s  %.2f%%", "Total Return on Capital",
             total_pnl / capital * 100)
    log.info("  %-30s  %s", "Average Daily P&L", f"${avg_pnl:+,.0f}")
    log.info("  %-30s  %+.3f%%", "Average Daily Return",
             np.mean(pnl_pcts) * 100)
    log.info("")
    log.info("  %-30s  %d / %d (%.1f%%)", "Win Rate",
             win_days, len(pnls), win_rate)
    log.info("  %-30s  %d", "Losing Days", lose_days)
    log.info("  %-30s  %d", "Flat Days", flat_days)
    log.info("")
    log.info("  %-30s  %s (%s)", "Best Day",
             f"${pnls[best_idx]:+,.0f}", valid[best_idx]["date"].strftime("%Y-%m-%d"))
    log.info("  %-30s  %s (%s)", "Worst Day",
             f"${pnls[worst_idx]:+,.0f}", valid[worst_idx]["date"].strftime("%Y-%m-%d"))
    log.info("")
    log.info("  %-30s  %d", "Total Trades", total_trades)
    log.info("  %-30s  %d (%.1f%%)", "Stop-Losses Hit",
             total_stops, total_stops / max(total_trades, 1) * 100)
    
    # Sharpe ratio (daily)
    if len(pnl_pcts) > 1:
        daily_sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(252)
        log.info("  %-30s  %.3f", "Annualized Sharpe", daily_sharpe)
    
    # Max drawdown
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = drawdown.min()
    log.info("  %-30s  %s", "Max Drawdown", f"${max_dd:,.0f}")
    
    log.info("")
    log.info("  Daily P&L Breakdown:")
    for r in valid:
        pnl = r.get("day_pnl", 0)
        n_trades = len(r.get("trades", []))
        marker = "🟢" if pnl > 0 else ("🔴" if pnl < 0 else "⚪")
        log.info("    %s  %s  %s  (%d trades)",
                marker, r["date"].strftime("%Y-%m-%d"), f"${pnl:+,.0f}", n_trades)
    
    log.info(SEP)


# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NASDAQ Single-Day Max-Profit Backtester",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--days", type=int, default=59,
                   help="Days of intraday history to fetch")
    p.add_argument("--date", default=None,
                   help="Specific date to backtest (YYYY-MM-DD)")
    p.add_argument("--last", type=int, default=None,
                   help="Backtest last N trading days")
    p.add_argument("--from", dest="from_date", default=None)
    p.add_argument("--to", dest="to_date", default=None)
    p.add_argument("--capital", type=float, default=1_000_000)
    p.add_argument("--picks", type=int, default=80,
                   help="Number of stocks to pick per day")
    p.add_argument("--stop-loss", type=float, default=0.02,
                   help="Hard stop-loss percentage")
    p.add_argument("--profit-take", type=float, default=0.015,
                   help="First profit target percentage")
    p.add_argument("--trail-trigger", type=float, default=0.025,
                   help="Trailing stop trigger percentage")
    p.add_argument("--skip-universe", action="store_true", default=True)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    setup_logging(args.log_level)
    
    run_backtest(
        config_path=args.config,
        days=args.days,
        target_date=args.date,
        last_n=args.last,
        from_date=args.from_date,
        to_date=args.to_date,
        capital=args.capital,
        n_picks=args.picks,
        stop_loss_pct=args.stop_loss,
        profit_take=args.profit_take,
        trail_trigger=args.trail_trigger,
        skip_universe=args.skip_universe,
    )
