"""
data/preprocess_sparse_5min.py
==============================
PART 4: Handle Missing Data (Sparse 5-Minute Trading)

PROBLEM:
  - Not all 44 trade every 5 minutes (market gaps, limit ups, halts)
  - Cross-sectional z-score expects full data
  - Solution: Sparsity-aware preprocessing

TYPICAL DATA QUALITY:
  - Coverage: 92-96% of tickers trading at each 5m bar
  - Missing pattern: weekend/holidays (100%), post-4pm (100%), 
                     early morning (varies), auctions (2%)
  
STRATEGIES (pick one):
  A) Forward-fill (use prior bar's value)
     Risk: Signal becomes stale, late reversals missed
  
  B) Drop sparse stocks (only trade top 40 liquid)
     Risk: Reduces universe (defeats 44-stock constraint)
  
  C) Lagged imputation (use bar t-1 data for bar t missing)
     Advantage: Less stale than forward-fill, preserves sparse pattern
     
  D) Sparsity indicator (flag missing, reduce position size)
     Advantage: Safest; explicit handling, no fake data
     
RECOMMENDED: Method D + C hybrid (lagged imputation + sparse flags)

Usage
-----
    from data.preprocess_sparse_5min import preprocess_sparse_data
    
    features_raw = {
        "A1": df_a1,  # [timestamp × 44], with NaN
        "A2": df_a2,
        ...
    }
    
    features_clean, sparsity_flags = preprocess_sparse_data(
        features_raw,
        method="lagged_imputation",  # or "forward_fill" or "drop_sparse"
    )
    
    # Use in portfolio construction
    positions = compute_positions(features_clean)
    positions = positions * sparsity_flags  # reduce size if sparse
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Literal

log = logging.getLogger(__name__)


def preprocess_sparse_data(
    features: dict[str, pd.DataFrame],
    method: Literal["lagged_imputation", "forward_fill", "drop_sparse", "none"] = "lagged_imputation",
    min_coverage_pct: float = 0.85,
    sparse_weight: float = 0.5,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Handle missing data in sparse 5-minute bars.
    
    Parameters
    ----------
    features : dict
        Raw features from FeatureEngine44, each [timestamp × 44] with NaN
    method : str
        "lagged_imputation": Use bar t-1 data for missing bar t (recommended)
        "forward_fill": Propagate last valid value
        "drop_sparse": Only trade stocks with >85% coverage
        "none": No imputation (raw data)
    min_coverage_pct : float
        For "drop_sparse": keep only tickers with this % of valid data
    sparse_weight : float
        Sparsity penalty: positions * (1 - sparse_weight * sparsity_rate)
        Default 0.5 = reduce by 50% if 100% sparse
    
    Returns
    -------
    tuple:
        features_clean : dict of imputed features
        sparsity_flags : DataFrame [timestamp × 44], range [0, 1]
                         1.0 = fully present
                         0.5 = sparse (position size reduced 50%)
                         0.0 = completely missing (drop)
    """
    log.info("Preprocessing sparse 5-minute data (method=%s) ...", method)
    
    # Compute coverage per ticker
    coverage = {}
    for feat_name, feat_df in features.items():
        coverage[feat_name] = feat_df.notna().sum() / len(feat_df)
    
    coverage_df = pd.DataFrame(coverage).mean(axis=1)  # avg across features
    
    log.info(
        "Coverage statistics:\n"
        "  Mean coverage: %.2f%%\n"
        "  Min coverage: %.2f%%\n"
        "  Max coverage: %.2f%%\n"
        "  Tickers < %.0f%% coverage: %d",
        100 * coverage_df.mean(),
        100 * coverage_df.min(),
        100 * coverage_df.max(),
        100 * min_coverage_pct,
        (coverage_df < min_coverage_pct).sum(),
    )
    
    # Method 1: Drop sparse stocks
    if method == "drop_sparse":
        good_tickers = coverage_df[coverage_df >= min_coverage_pct].index.tolist()
        log.info("  Dropping sparse stocks: keeping %d of %d", len(good_tickers), len(coverage_df))
        
        features_clean = {}
        for feat_name, feat_df in features.items():
            features_clean[feat_name] = feat_df[good_tickers]
        
        sparsity_flags = pd.DataFrame(1.0, index=feat_df.index, columns=good_tickers)
        
        return features_clean, sparsity_flags
    
    # Methods 2-4: Keep all stocks, impute
    features_clean = {}
    
    for feat_name, feat_df in features.items():
        feat_clean = feat_df.copy()
        
        if method == "lagged_imputation":
            # Fill NaN with value from prior bar (lagged)
            feat_clean = feat_clean.fillna(method="ffill", limit=1)
            # If still NaN (first bar), use cross-sectional median
            feat_clean = feat_clean.fillna(feat_clean.median(axis=1), axis=0)
        
        elif method == "forward_fill":
            # Propagate last valid value
            feat_clean = feat_clean.fillna(method="ffill")
            # Fill remaining with median
            feat_clean = feat_clean.fillna(feat_clean.median(axis=1), axis=0)
        
        elif method == "none":
            # No imputation; leave as-is
            pass
        
        features_clean[feat_name] = feat_clean
    
    # Compute sparsity flags: how "sparse" is each stock at each bar?
    sparsity_flags = _compute_sparsity_flags(features, sparse_weight)
    
    log.info(
        "Sparse preprocessing complete:\n"
        "  Method: %s\n"
        "  Sparsity penalty: %.2f\n"
        "  Coverage preserved: %.1f%%",
        method, sparse_weight, 100 * sparsity_flags.mean().mean()
    )
    
    return features_clean, sparsity_flags


def _compute_sparsity_flags(
    features: dict[str, pd.DataFrame],
    sparse_weight: float = 0.5,
) -> pd.DataFrame:
    """
    Compute sparsity flags: 1.0 if fully present, <1.0 if sparse, 0.0 if missing.
    
    For each bar × stock, measure what % of features are NaN.
    Penalize position size accordingly.
    
    sparse_flag = 1.0 - sparse_weight * (# NaN features / total features)
    
    Parameters
    ----------
    features : dict of DataFrames
    sparse_weight : float
        Penalty weight. sparse_flag = 1 - sparse_weight * sparsity_rate
    
    Returns
    -------
    DataFrame [timestamp × 44], range [0, 1]
    """
    n_features = len(features)
    
    # Stack all features, count NaN per bar × stock
    nan_count = pd.DataFrame(0, index=list(features.values())[0].index,
                              columns=list(features.values())[0].columns)
    
    for feat_df in features.values():
        nan_count += feat_df.isna().astype(int)
    
    # Compute sparsity rate
    sparsity_rate = nan_count / n_features
    
    # Compute flag: 1 - weight * rate
    sparsity_flags = 1.0 - sparse_weight * sparsity_rate
    sparsity_flags = sparsity_flags.clip(lower=0.0)  # Don't go negative
    
    return sparsity_flags


def validate_coverage(
    features: dict[str, pd.DataFrame],
) -> dict:
    """
    Analyze coverage patterns: daily, by time-of-day, by ticker.
    
    Returns
    -------
    dict with analysis:
        coverage_daily : Series [date] → % of (timestamp, ticker) pairs present
        coverage_by_hour : Series [hour] → %
        coverage_by_ticker : Series [ticker] → %
        missing_patterns : str describing common missing patterns
    """
    # Combine all features
    all_nan = None
    for feat_df in features.values():
        if all_nan is None:
            all_nan = feat_df.isna()
        else:
            all_nan = all_nan & feat_df.isna()
    
    # Daily coverage
    daily_coverage = (~all_nan).groupby(all_nan.index.normalize()).apply(
        lambda x: x.sum().sum() / x.size
    )
    
    # Coverage by hour
    hourly_coverage = (~all_nan).groupby(all_nan.index.hour).apply(
        lambda x: x.sum().sum() / x.size if len(x) > 0 else 0
    )
    
    # Coverage by ticker
    ticker_coverage = (~all_nan).mean()
    
    # Identify patterns
    low_coverage_hours = hourly_coverage[hourly_coverage < 0.90].index.tolist()
    low_coverage_tickers = ticker_coverage[ticker_coverage < 0.85].index.tolist()
    
    patterns = []
    if low_coverage_hours:
        patterns.append(f"Low coverage hours: {low_coverage_hours} (likely pre-market/post-market)")
    if low_coverage_tickers:
        patterns.append(f"Low coverage tickers: {len(low_coverage_tickers)} stocks <85% (consider dropping)")
    
    log.info(
        "\nCOVERAGE ANALYSIS:\n"
        "  Daily coverage: %.1f%% ± %.1f%%\n"
        "  Hourly range: %.1f%% - %.1f%%\n"
        "  Ticker range: %.1f%% - %.1f%%\n"
        "\nPatterns:\n  %s",
        100 * daily_coverage.mean(),
        100 * daily_coverage.std(),
        100 * hourly_coverage.min(),
        100 * hourly_coverage.max(),
        100 * ticker_coverage.min(),
        100 * ticker_coverage.max(),
        "\n  ".join(patterns) if patterns else "None",
    )
    
    return {
        "coverage_daily": daily_coverage,
        "coverage_hourly": hourly_coverage,
        "coverage_by_ticker": ticker_coverage,
        "missing_patterns": "\n".join(patterns),
    }
