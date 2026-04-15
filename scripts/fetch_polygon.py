"""
scripts/fetch_polygon.py
========================
Download 5-minute intraday OHLCV data for all SEED_POOL tickers from Polygon.io
and save as Parquet panels compatible with the pipeline.

Advantages over yfinance:
  - Much faster (batch-friendly, no rate-limit throttling)
  - Up to 2 years of intraday history (vs 59 days on yfinance)
  - Reliable timestamps with proper US/Eastern conversion

Usage
-----
    # Set your API key as environment variable
    set POLYGON_API_KEY=your_key_here

    # Full download (default: last 59 days)
    python scripts/fetch_polygon.py

    # Custom date range
    python scripts/fetch_polygon.py --from 2026-01-01 --to 2026-04-14

    # Specific tickers only
    python scripts/fetch_polygon.py --tickers AAPL MSFT TSLA

    # Custom output directory
    python scripts/fetch_polygon.py --output data/polygon/

Requirements
------------
    pip install polygon-api-client pyarrow pandas pytz
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from polygon import RESTClient
from nse_pipeline.universe import SEED_POOL
from nse_pipeline.storage import save_panels

log = logging.getLogger(__name__)
ET = pytz.timezone("US/Eastern")

# US market hours
MARKET_OPEN = pd.Timestamp("09:30:00").time()
MARKET_CLOSE = pd.Timestamp("16:00:00").time()


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("reports/polygon_fetch.log", mode="w"),
        ],
    )


def fetch_ticker_bars(
    client: RESTClient,
    ticker: str,
    from_date: str,
    to_date: str,
    multiplier: int = 5,
    timespan: str = "minute",
) -> pd.DataFrame | None:
    """
    Fetch 5-minute OHLCV bars for a single ticker from Polygon.

    Parameters
    ----------
    client : Polygon RESTClient
    ticker : stock symbol (e.g. "AAPL")
    from_date : start date "YYYY-MM-DD"
    to_date : end date "YYYY-MM-DD"
    multiplier : bar size multiplier (5 for 5-min bars)
    timespan : "minute", "hour", "day"

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume]
    Index: tz-aware DatetimeIndex in US/Eastern
    Returns None if fetch fails.
    """
    try:
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=from_date,
            to=to_date,
            limit=50000,
        )

        if not aggs:
            return None

        rows = []
        for bar in aggs:
            rows.append({
                "timestamp": bar.timestamp,  # ms since epoch
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return None

        # Convert ms timestamps → tz-aware US/Eastern
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert(ET)
        df = df.set_index("timestamp").sort_index()

        # Filter to RTH (Regular Trading Hours) only: 9:30 AM - 4:00 PM ET
        times = df.index.time
        mask = (times >= MARKET_OPEN) & (times < MARKET_CLOSE)
        df = df[mask]

        # Drop zero/negative prices
        for col in ["open", "high", "low", "close"]:
            df.loc[df[col] <= 0, col] = np.nan
        df.loc[df["volume"] < 0, "volume"] = 0

        return df if not df.empty else None

    except Exception as e:
        log.warning("  %s: error — %s", ticker, str(e)[:200])
        return None


def fetch_all_tickers(
    client: RESTClient,
    tickers: list[str],
    from_date: str,
    to_date: str,
    sleep_between: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """
    Fetch data for all tickers with progress tracking and rate limiting.

    Returns dict of {ticker: DataFrame}.
    """
    total = len(tickers)
    results = {}
    failed = []

    log.info("=" * 60)
    log.info("  Polygon.io Data Fetcher")
    log.info("  Tickers: %d | Date Range: %s → %s", total, from_date, to_date)
    log.info("=" * 60)

    t0 = time.perf_counter()

    for i, ticker in enumerate(tickers):
        pct = (i + 1) / total * 100
        log.info("[%d/%d] (%.0f%%) Fetching %s ...", i + 1, total, pct, ticker)

        df = fetch_ticker_bars(client, ticker, from_date, to_date)

        if df is not None and not df.empty:
            results[ticker] = df
            n_bars = len(df)
            n_days = df.index.normalize().nunique()
            log.info(
                "  ✓ %s: %d bars across %d days | $%.2f → $%.2f",
                ticker, n_bars, n_days,
                df["close"].iloc[0], df["close"].iloc[-1],
            )
        else:
            failed.append(ticker)
            log.warning("  ✗ %s: no data", ticker)

        # Rate limiting (Polygon allows 5 req/min on free, much more on paid)
        if i < total - 1:
            time.sleep(sleep_between)

    elapsed = time.perf_counter() - t0

    log.info("")
    log.info("─" * 60)
    log.info("  Fetch Summary")
    log.info("─" * 60)
    log.info("  Success: %d / %d tickers", len(results), total)
    log.info("  Failed:  %d tickers", len(failed))
    log.info("  Time:    %.1f seconds (%.1f sec/ticker)", elapsed, elapsed / max(total, 1))

    if failed:
        log.info("  Failed tickers: %s", failed)

    return results


def build_panels(ticker_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Convert per-ticker DataFrames into wide panels.

    Returns
    -------
    dict: {"open": DataFrame, "high": DataFrame, ...}
          Each DataFrame: [timestamp × ticker]
    """
    fields = ["open", "high", "low", "close", "volume"]
    panels = {}

    for field in fields:
        series_dict = {}
        for ticker, df in ticker_data.items():
            if field in df.columns:
                series_dict[ticker] = df[field]

        if series_dict:
            panel = pd.DataFrame(series_dict).sort_index()
            panels[field] = panel

    if panels:
        close = panels.get("close", pd.DataFrame())
        log.info(
            "  Panels built: %d bars × %d tickers | %s → %s",
            len(close), len(close.columns),
            close.index.min().strftime("%Y-%m-%d %H:%M") if len(close) > 0 else "N/A",
            close.index.max().strftime("%Y-%m-%d %H:%M") if len(close) > 0 else "N/A",
        )

    return panels


def main():
    parser = argparse.ArgumentParser(
        description="Fetch NASDAQ intraday data from Polygon.io → Parquet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--api-key", default=None,
        help="Polygon API key (or set POLYGON_API_KEY env var)"
    )
    parser.add_argument(
        "--from", dest="from_date", default=None,
        help="Start date YYYY-MM-DD (default: 59 days ago)"
    )
    parser.add_argument(
        "--to", dest="to_date", default=None,
        help="End date YYYY-MM-DD (default: today)"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Specific tickers (default: all SEED_POOL)"
    )
    parser.add_argument(
        "--output", default="data/polygon/",
        help="Output directory for Parquet files"
    )
    parser.add_argument(
        "--sleep", type=float, default=0.15,
        help="Seconds between API calls (rate limiting)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )

    args = parser.parse_args()

    Path("reports").mkdir(parents=True, exist_ok=True)
    setup_logging(args.log_level)

    # ── API Key ────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        log.error(
            "No Polygon API key provided.\n"
            "  Set environment variable:  set POLYGON_API_KEY=your_key_here\n"
            "  Or pass as argument:       --api-key your_key_here"
        )
        sys.exit(1)

    client = RESTClient(api_key=api_key)

    # ── Date Range ─────────────────────────────────────────────────────
    today = datetime.now(ET).strftime("%Y-%m-%d")
    from_date = args.from_date or (datetime.now(ET) - timedelta(days=59)).strftime("%Y-%m-%d")
    to_date = args.to_date or today

    # ── Tickers ────────────────────────────────────────────────────────
    tickers = args.tickers or list(dict.fromkeys(SEED_POOL))  # deduplicated, order preserved
    log.info("Tickers to fetch: %d (from %s)", len(tickers),
             "CLI" if args.tickers else "SEED_POOL")

    # ── Fetch ──────────────────────────────────────────────────────────
    ticker_data = fetch_all_tickers(
        client=client,
        tickers=tickers,
        from_date=from_date,
        to_date=to_date,
        sleep_between=args.sleep,
    )

    if not ticker_data:
        log.error("No data fetched — aborting")
        sys.exit(1)

    # ── Build Panels ───────────────────────────────────────────────────
    log.info("\nBuilding panels ...")
    panels = build_panels(ticker_data)

    # ── Save to Parquet ────────────────────────────────────────────────
    output_dir = args.output
    log.info("\nSaving to %s ...", output_dir)
    save_panels(panels, output_dir)

    # ── Also save to data/clean/ so the pipeline can use it directly ──
    clean_dir = "data/clean/"
    log.info("Copying to %s for pipeline compatibility ...", clean_dir)
    save_panels(panels, clean_dir)

    # ── Summary ────────────────────────────────────────────────────────
    close = panels.get("close", pd.DataFrame())
    log.info("")
    log.info("=" * 60)
    log.info("  DOWNLOAD COMPLETE")
    log.info("=" * 60)
    log.info("  Tickers:    %d", len(close.columns))
    log.info("  Bars:       %d", len(close))
    log.info("  Date range: %s → %s",
             close.index.min().strftime("%Y-%m-%d") if len(close) > 0 else "N/A",
             close.index.max().strftime("%Y-%m-%d") if len(close) > 0 else "N/A")
    log.info("  Saved to:   %s", output_dir)
    log.info("  Pipeline:   %s (ready for run_pipeline.py --skip-universe)", clean_dir)

    # Save failed tickers list
    fetched_tickers = set(close.columns.tolist())
    all_tickers = set(dict.fromkeys(SEED_POOL))
    failed = all_tickers - fetched_tickers
    if failed:
        failed_path = Path(output_dir) / "failed_tickers.txt"
        with open(failed_path, "w") as f:
            f.write("\n".join(sorted(failed)))
        log.info("  Failed tickers saved to: %s (%d tickers)", failed_path, len(failed))

    log.info("=" * 60)


if __name__ == "__main__":
    main()
