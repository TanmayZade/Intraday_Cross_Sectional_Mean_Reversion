"""
features/
=========
Institutional feature engineering pipeline for intraday cross-sectional mean reversion.

Modules
-------
core.py        — cross-sectional math primitives (cs_zscore, cs_rank, ATR, etc.)
resampler.py   — 1-min → 15-min aggregation with flagged-bar masking
engine.py      — 15 alpha features across 5 categories
diagnostics.py — IC analysis, decay curves, correlation, quality reports
store.py       — Parquet feature store (save/load)

Quick start
-----------
    from features.engine import load_and_compute

    panels, features = load_and_compute(
        clean_dir  = "data/clean/",
        freq       = "15min",
        start_date = "2024-12-01",
    )
    signal = features["E1_residual_return"]   # strongest signal
"""
from features.core        import cs_zscore, cs_rank, atr, atr_pct
from features.resampler   import Resampler
from features.engine      import FeatureEngine, load_and_compute
from features.diagnostics import FeatureDiagnostics
from features.store       import FeatureStore
