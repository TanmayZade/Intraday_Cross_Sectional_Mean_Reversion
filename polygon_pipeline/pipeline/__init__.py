"""
polygon_pipeline
================
Production OHLCV data pipeline for Polygon.io free-tier data.

Quick start
-----------
    import os
    os.environ["POLYGON_API_KEY"] = "your_key_here"

    from pipeline.orchestrator import Pipeline

    pipe = Pipeline.from_config("configs/config.yaml")

    # First run: fetch + clean + store
    pipe.run(
        tickers    = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"],
        start_date = "2022-01-01",
        end_date   = "2024-12-31",
    )

    # Research panels
    panels = pipe.load_panels()
    close  = panels["close"]    # pd.DataFrame  date × ticker
    volume = panels["volume"]

    # Incremental update (run daily)
    pipe.update()
"""

from .fetcher      import PolygonFetcher, FetcherConfig
from .cleaner      import clean_ticker, clean_panel, CleaningConfig
from .universe     import build_universe, build_panels, apply_universe_mask
from .storage      import write_clean, read_clean, read_panels, pivot_field
from .quality      import (generate_report, compute_coverage,
                                    corwin_schultz_spread, return_diagnostics)
from .orchestrator import Pipeline, PipelineConfig

__all__ = [
    "PolygonFetcher", "FetcherConfig",
    "clean_ticker", "clean_panel", "CleaningConfig",
    "build_universe", "build_panels", "apply_universe_mask",
    "write_clean", "read_clean", "read_panels", "pivot_field",
    "generate_report", "compute_coverage",
    "corwin_schultz_spread", "return_diagnostics",
    "Pipeline", "PipelineConfig",
]
