"""
pipeline/orchestrator.py
=========================
Top-level pipeline: fetch → clean → universe → store → report.

CLI usage
---------
    # Full run: fetch + clean + store
    python -m pipeline.orchestrator run \
        --config configs/config.yaml \
        --start  2020-01-01 \
        --end    2024-12-31 \
        --tickers AAPL MSFT GOOG AMZN \
        --jobs 1

    # Update only: fetch new dates for existing tickers in store
    python -m pipeline.orchestrator update \
        --config configs/config.yaml

    # Read clean data into research panels (Python API only)
    from pipeline.orchestrator import Pipeline
    pipe   = Pipeline.from_config("configs/config.yaml")
    panels = pipe.load_panels(start_date="2023-01-01")
    close  = panels["close"]   # date × ticker DataFrame

Programmatic usage
------------------
    from pipeline.orchestrator import Pipeline

    pipe = Pipeline.from_config("configs/config.yaml")

    # Fetch and store 50 tickers, 4 years
    pipe.run(
        tickers    = ["AAPL", "MSFT", "GOOG", ...],
        start_date = "2020-01-01",
        end_date   = "2024-12-31",
    )

    # Research panels — always call this after run()
    panels = pipe.load_panels()
    close, volume = panels["close"], panels["volume"]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

from .fetcher    import PolygonFetcher, FetcherConfig
from .cleaner    import CleaningConfig, clean_panel
from .universe   import build_universe, build_panels, apply_universe_mask
from .storage    import write_clean, read_clean, read_panels
from .quality    import generate_report

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Polygon
    api_key:           str   = ""
    base_url:          str   = "https://api.polygon.io"
    timespan:          str   = "day"
    multiplier:        int   = 1
    adjusted:          bool  = True
    requests_per_min:  int   = 5
    retry_attempts:    int   = 3
    retry_backoff_s:   float = 13.0

    # Universe
    min_adtv_usd:      float = 1_000_000
    min_price:         float = 5.0
    max_price:         float = 50_000.0
    min_history_days:  int   = 60
    adtv_window:       int   = 20

    # Cleaning
    max_daily_return:      float = 0.50
    max_intraday_range:    float = 0.40
    max_volume_spike_mult: float = 20.0
    min_volume:            int   = 10_000

    # Storage
    raw_dir:     str = "data/raw/"
    clean_dir:   str = "data/clean/"
    cache_dir:   str = "data/cache/"
    report_dir:  str = "reports/"
    compression: str = "snappy"

    # Dates
    start_date:  str = "2020-01-01"

    # Default tickers
    default_tickers: list = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        poly    = raw.get("polygon",  {})
        univ    = raw.get("universe", {})
        clean   = raw.get("cleaning", {})
        stor    = raw.get("storage",  {})
        dates   = raw.get("dates",    {})

        api_key = os.environ.get("POLYGON_API_KEY", "")

        return cls(
            api_key           = api_key,
            base_url          = poly.get("base_url",         "https://api.polygon.io"),
            timespan          = poly.get("timespan",         "day"),
            multiplier        = poly.get("multiplier",       1),
            adjusted          = poly.get("adjusted",         True),
            requests_per_min  = poly.get("requests_per_min", 5),
            retry_attempts    = poly.get("retry_attempts",   3),
            retry_backoff_s   = poly.get("retry_backoff_s",  13.0),

            min_adtv_usd      = univ.get("min_adtv_usd",     1_000_000),
            min_price         = univ.get("min_price",         5.0),
            max_price         = univ.get("max_price",         50_000.0),
            min_history_days  = univ.get("min_history_days",  60),
            adtv_window       = univ.get("adtv_window",       20),
            default_tickers   = univ.get("default_tickers",   []),

            max_daily_return       = clean.get("max_daily_return",      0.50),
            max_intraday_range     = clean.get("max_intraday_range",    0.40),
            max_volume_spike_mult  = clean.get("max_volume_spike_mult", 20.0),
            min_volume             = clean.get("min_volume",            10_000),

            raw_dir      = stor.get("raw_dir",    "data/raw/"),
            clean_dir    = stor.get("clean_dir",  "data/clean/"),
            cache_dir    = stor.get("cache_dir",  "data/cache/"),
            report_dir   = stor.get("report_dir", "reports/"),
            compression  = stor.get("compression","snappy"),

            start_date   = dates.get("start_date", "2020-01-01"),
        )

    def to_fetcher_config(self) -> FetcherConfig:
        return FetcherConfig(
            api_key          = self.api_key,
            base_url         = self.base_url,
            timespan         = self.timespan,
            multiplier       = self.multiplier,
            adjusted         = self.adjusted,
            requests_per_min = self.requests_per_min,
            retry_attempts   = self.retry_attempts,
            retry_backoff_s  = self.retry_backoff_s,
            cache_dir        = self.cache_dir,
        )

    def to_cleaning_config(self) -> CleaningConfig:
        return CleaningConfig(
            max_daily_return      = self.max_daily_return,
            max_intraday_range    = self.max_intraday_range,
            max_volume_spike_mult = self.max_volume_spike_mult,
            min_volume            = self.min_volume,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        if not cfg.api_key:
            raise EnvironmentError(
                "POLYGON_API_KEY environment variable not set.\n"
                "  export POLYGON_API_KEY=your_key_here"
            )
        self.fetcher = PolygonFetcher(cfg.to_fetcher_config())

    @classmethod
    def from_config(cls, path: str | Path) -> "Pipeline":
        return cls(PipelineConfig.from_yaml(path))

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        tickers:    Optional[list[str]] = None,
        start_date: Optional[str]       = None,
        end_date:   Optional[str]       = None,
        n_jobs:     int = 1,
    ) -> dict:
        """
        Full pipeline: fetch → clean → universe → store → report.

        Parameters
        ----------
        tickers    : list of ticker symbols. Defaults to config default_tickers.
        start_date : "YYYY-MM-DD". Defaults to config start_date.
        end_date   : "YYYY-MM-DD". Defaults to today.
        n_jobs     : parallel workers for cleaning stage only.

        Returns
        -------
        dict with summary statistics and output paths.
        """
        t0  = time.perf_counter()
        tickers    = tickers    or self.cfg.default_tickers or []
        start_date = start_date or self.cfg.start_date
        end_date   = end_date   or pd.Timestamp.today().strftime("%Y-%m-%d")

        if not tickers:
            raise ValueError("No tickers specified. Pass tickers= or set default_tickers in config.")

        _separator()
        log.info("Polygon OHLCV Pipeline")
        log.info("  Tickers    : %d  (%s … %s)", len(tickers), tickers[0], tickers[-1])
        log.info("  Date range : %s → %s", start_date, end_date)
        log.info("  Timespan   : %s (multiplier=%d)", self.cfg.timespan, self.cfg.multiplier)
        log.info("  Adjusted   : %s", self.cfg.adjusted)
        _separator()

        # ── Step 1: Fetch ─────────────────────────────────────────────────────
        log.info("[1/5] Fetching from Polygon.io ...")
        raw = self.fetcher.fetch_universe(tickers, start_date, end_date)
        if raw.empty:
            log.error("No data fetched — check API key and tickers")
            return {}

        # ── Step 2: Clean ─────────────────────────────────────────────────────
        log.info("[2/5] Cleaning ...")
        clean_df, report_df = clean_panel(raw, self.cfg.to_cleaning_config(), n_jobs=n_jobs)

        # ── Step 3: Universe ──────────────────────────────────────────────────
        log.info("[3/5] Building universe membership ...")
        panels  = build_panels(clean_df)
        universe = build_universe(
            close  = panels["close"],
            volume = panels["volume"],
            min_adtv_usd     = self.cfg.min_adtv_usd,
            min_price        = self.cfg.min_price,
            max_price        = self.cfg.max_price,
            min_history_days = self.cfg.min_history_days,
            adtv_window      = self.cfg.adtv_window,
        )
        clean_df = apply_universe_mask(clean_df, universe)

        # ── Step 4: Store ─────────────────────────────────────────────────────
        log.info("[4/5] Writing Parquet store ...")
        write_clean(clean_df, root=self.cfg.clean_dir, compression=self.cfg.compression)

        # ── Step 5: Report ────────────────────────────────────────────────────
        log.info("[5/5] Generating quality report ...")
        panels_clean = build_panels(clean_df)
        summary = generate_report(
            clean_df   = clean_df,
            report_df  = report_df,
            output_dir = self.cfg.report_dir,
            close_panel = panels_clean.get("close"),
        )

        elapsed = time.perf_counter() - t0
        _separator()
        log.info("Pipeline complete in %.1fs", elapsed)
        log.info("  Tickers    : %d", summary["n_tickers"])
        log.info("  Clean bars : %d  (%.1f%% retained)",
                 summary["total_clean_bars"], summary["pct_retained"])
        log.info("  Output     : %s", self.cfg.clean_dir)
        log.info("  Report     : %s", self.cfg.report_dir)
        _separator()

        return {
            "summary":    summary,
            "report_df":  report_df,
            "universe":   universe,
            "clean_path": self.cfg.clean_dir,
        }

    def update(
        self,
        tickers: Optional[list[str]] = None,
    ) -> dict:
        """
        Incremental update: fetch only dates not yet in the clean store.
        Call this daily/weekly to keep the store current.
        """
        tickers = tickers or self.cfg.default_tickers or []
        try:
            existing = read_clean(self.cfg.clean_dir, tickers=tickers)
            if existing.empty:
                last_date = self.cfg.start_date
            else:
                last_date = existing.index.max().strftime("%Y-%m-%d")
        except FileNotFoundError:
            last_date = self.cfg.start_date

        today = pd.Timestamp.today().strftime("%Y-%m-%d")
        if last_date >= today:
            log.info("Store is already up-to-date (last date: %s)", last_date)
            return {}

        log.info("Updating from %s → %s", last_date, today)
        return self.run(tickers=tickers, start_date=last_date, end_date=today)

    # ── Reader shortcuts ──────────────────────────────────────────────────────

    def load_panels(
        self,
        tickers:       Optional[list[str]] = None,
        start_date:    Optional[str]       = None,
        end_date:      Optional[str]       = None,
        universe_only: bool                = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Load research-ready {field: date × ticker} panels from clean store.

            panels = pipe.load_panels(start_date="2023-01-01")
            close  = panels["close"]    # date × ticker
            volume = panels["volume"]
        """
        return read_panels(
            self.cfg.clean_dir,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            universe_only=universe_only,
        )

    def load_flat(
        self,
        tickers:    Optional[list[str]] = None,
        start_date: Optional[str]       = None,
        end_date:   Optional[str]       = None,
    ) -> pd.DataFrame:
        """Load as flat DataFrame with ticker column."""
        return read_clean(
            self.cfg.clean_dir,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _separator():
    log.info("─" * 55)


def _setup_logging(level: str, log_file: Optional[str] = None) -> None:
    fmt     = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt,
                        datefmt=datefmt, handlers=handlers)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Polygon OHLCV Pipeline")
    sub = p.add_subparsers(dest="command")

    # ── run ──────────────────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Full fetch + clean + store pipeline")
    run_p.add_argument("--config",    default="configs/config.yaml")
    run_p.add_argument("--tickers",   nargs="+", default=None)
    run_p.add_argument("--start",     default=None,  help="YYYY-MM-DD")
    run_p.add_argument("--end",       default=None,  help="YYYY-MM-DD")
    run_p.add_argument("--jobs",      type=int, default=1)
    run_p.add_argument("--log-level", default="INFO")

    # ── update ────────────────────────────────────────────────────────────────
    upd_p = sub.add_parser("update", help="Incremental update to today")
    upd_p.add_argument("--config",    default="configs/config.yaml")
    upd_p.add_argument("--tickers",   nargs="+", default=None)
    upd_p.add_argument("--log-level", default="INFO")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if not args.command:
        print("Usage: python -m pipeline.orchestrator [run|update] --help")
        sys.exit(1)

    cfg = PipelineConfig.from_yaml(args.config)
    _setup_logging(args.log_level, cfg.report_dir + "pipeline.log")

    pipe = Pipeline(cfg)

    if args.command == "run":
        results = pipe.run(
            tickers    = args.tickers,
            start_date = args.start,
            end_date   = args.end,
            n_jobs     = args.jobs,
        )
    elif args.command == "update":
        results = pipe.update(tickers=args.tickers)

    if results:
        print("\n── Summary ──────────────────────────────")
        for k, v in results.get("summary", {}).items():
            print(f"  {k:<35} {v}")
