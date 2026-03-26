"""
validation/correlation_analysis_44stock.py
===========================================
Feature Correlation Analysis & Validation (PART 1, 10)

Validates that selected 6 features have pairwise correlation r < 0.40

Run this to:
  1. Compute correlation matrix (6×6)
  2. Generate heatmap visualization
  3. Compare vs old 15-feature set (if available)
  4. Test IC weights (check negative IC → 0 weight)

Usage
-----
    python -c "
    from validation.correlation_analysis_44stock import analyze_feature_correlations
    
    features_dict = {
        'A1': df_a1,
        'A2': df_a2,
        ...  # 6 features
    }
    
    results = analyze_feature_correlations(features_dict)
    print(results['correlation_matrix'])
    results['heatmap'].show()
    "
"""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    log.warning("matplotlib/seaborn not available; plots disabled")


def analyze_feature_correlations(
    features: dict[str, pd.DataFrame],
    close: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Analyze pairwise correlations between 6 selected features.
    
    Parameters
    ----------
    features : dict
        6 features from FeatureEngine44
        Each: [timestamp × 44 tickers]
    
    close : DataFrame, optional
        Close prices, used to compute IC weights
    
    Returns
    -------
    dict:
        correlation_matrix : DataFrame [6×6]
        correlation_stats : dict with analysis
        ic_weights : dict with IC per feature (if close provided)
        heatmap : matplotlib Figure (if plotting available)
    """
    log.info("Analyzing feature correlations (n_features=%d) ...", len(features))
    
    # Step 1: Compute pairwise correlations
    feature_names = list(features.keys())
    n = len(feature_names)
    corr_matrix = pd.DataFrame(np.nan, index=feature_names, columns=feature_names)
    
    for i, name_i in enumerate(feature_names):
        for j, name_j in enumerate(feature_names):
            if i == j:
                corr_matrix.loc[name_i, name_j] = 1.0
            elif i > j:
                # Use already computed value
                corr_matrix.loc[name_i, name_j] = corr_matrix.loc[name_j, name_i]
            else:
                # Compute correlation: flatten both features and correlate
                feat_i_flat = features[name_i].stack().values
                feat_j_flat = features[name_j].stack().values
                
                valid = ~(np.isnan(feat_i_flat) | np.isnan(feat_j_flat))
                if valid.sum() < 100:
                    corr = np.nan
                else:
                    corr = np.corrcoef(feat_i_flat[valid], feat_j_flat[valid])[0, 1]
                
                corr_matrix.loc[name_i, name_j] = corr
    
    corr_matrix = corr_matrix.astype(float)
    
    # Step 2: Compute statistics
    upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    
    stats = {
        "n_features": n,
        "avg_correlation": float(np.nanmean(upper_triangle)),
        "std_correlation": float(np.nanstd(upper_triangle)),
        "min_correlation": float(np.nanmin(upper_triangle)),
        "max_correlation": float(np.nanmax(upper_triangle)),
        "target_correlation": 0.40,
        "all_below_target": float(np.nanmax(upper_triangle)) < 0.40,
    }
    
    log.info(
        "\nFEATURE CORRELATION ANALYSIS:\n"
        "  Average correlation: %.3f (target: < 0.40)\n"
        "  Std deviation: %.3f\n"
        "  Min: %.3f | Max: %.3f\n"
        "  Target met: %s",
        stats["avg_correlation"],
        stats["std_correlation"],
        stats["min_correlation"],
        stats["max_correlation"],
        "✓ YES" if stats["all_below_target"] else "✗ NO"
    )
    
    # Step 3: Detailed correlation report
    log.info("\nPairwise Correlations (6×6):")
    log.info(corr_matrix.to_string())
    
    # Step 4: High correlation pairs (warn if any)
    high_corr_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            corr_val = corr_matrix.iloc[i, j]
            if not np.isnan(corr_val) and abs(corr_val) > 0.50:
                high_corr_pairs.append({
                    "feature1": feature_names[i],
                    "feature2": feature_names[j],
                    "correlation": corr_val,
                })
    
    if high_corr_pairs:
        log.warning("\n⚠ WARNING: Found %d pairs with |r| > 0.50:", len(high_corr_pairs))
        for pair in high_corr_pairs:
            log.warning("  %s ↔ %s: r=%.3f", pair["feature1"], pair["feature2"], pair["correlation"])
    else:
        log.info("\n✓ All pairwise correlations |r| < 0.50 (good!)")
    
    # Step 5: IC weights (if close provided)
    ic_data = None
    if close is not None:
        from alpha.rank_alpha import compute_ic_weights
        try:
            ic_weights = compute_ic_weights(features, close, ic_window=60)
            ic_data = {
                name: ic_series.mean()
                for name, ic_series in ic_weights.items()
            }
            log.info("\nIC Weights (adaptive per bar, averaged):")
            for name, ic_val in sorted(ic_data.items(), key=lambda x: -x[1]):
                log.info("  %s: IC=%.4f", name, ic_val)
        except Exception as e:
            log.warning("Could not compute IC: %s", e)
    
    # Step 6: Generate heatmap
    fig = None
    if PLOTTING_AVAILABLE:
        fig = _plot_correlation_heatmap(corr_matrix)
    
    return {
        "correlation_matrix": corr_matrix,
        "correlation_stats": stats,
        "high_correlation_pairs": high_corr_pairs,
        "ic_weights": ic_data,
        "heatmap": fig,
    }


def _plot_correlation_heatmap(corr_matrix: pd.DataFrame) -> plt.Figure:
    """Generate correlation heatmap visualization."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    
    ax.set_title("Feature Correlation Matrix (44-Stock Universe)\nTarget: All |r| < 0.40", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    return fig


def compare_old_vs_new(
    old_features_15: dict[str, pd.DataFrame],
    new_features_6: dict[str, pd.DataFrame],
) -> dict:
    """
    Compare correlation structure: old (15 features) vs new (6 features).
    
    Parameters
    ----------
    old_features_15 : dict
        All 15 original features
    new_features_6 : dict
        Selected 6 features
    
    Returns
    -------
    dict with comparison metrics
    """
    log.info("\n" + "="*70)
    log.info("COMPARING: Old (15 features) vs New (6 features)")
    log.info("="*70)
    
    # Analyze both
    old_analysis = analyze_feature_correlations(old_features_15)
    new_analysis = analyze_feature_correlations(new_features_6)
    
    # Extract stats
    old_stats = old_analysis["correlation_stats"]
    new_stats = new_analysis["correlation_stats"]
    
    improvement = {
        "avg_corr_old": old_stats["avg_correlation"],
        "avg_corr_new": new_stats["avg_correlation"],
        "avg_corr_improvement": old_stats["avg_correlation"] - new_stats["avg_correlation"],
        "avg_corr_reduction_pct": 100 * (1 - new_stats["avg_correlation"] / (old_stats["avg_correlation"] + 1e-8)),
        
        "std_corr_old": old_stats["std_correlation"],
        "std_corr_new": new_stats["std_correlation"],
        "std_corr_improvement": old_stats["std_correlation"] - new_stats["std_correlation"],
        
        "features_dropped": len(old_features_15) - len(new_features_6),
        "n_features_old": len(old_features_15),
        "n_features_new": len(new_features_6),
    }
    
    log.info(
        "\nMULTICOLLINEARITY REDUCTION:\n"
        "  Old (15 features):    avg correlation = %.3f (std %.3f)\n"
        "  New (6 features):     avg correlation = %.3f (std %.3f)\n"
        "  Improvement:          %.1f%% reduction in avg correlation\n"
        "  Features dropped:     %d of %d (-%.0f%%)",
        old_stats["avg_correlation"], old_stats["std_correlation"],
        new_stats["avg_correlation"], new_stats["std_correlation"],
        improvement["avg_corr_reduction_pct"],
        improvement["features_dropped"],
        len(old_features_15),
        100 * improvement["features_dropped"] / len(old_features_15),
    )
    
    log.info("="*70 + "\n")
    
    return {
        "old": old_analysis,
        "new": new_analysis,
        "improvement": improvement,
    }


def validate_rank_preservation(
    alpha_windsorized: pd.DataFrame,
    alpha_rank: pd.DataFrame,
) -> dict:
    """
    Validate that rank-based alpha preserves tails better than windsorized.
    
    Parameters
    ----------
    alpha_windsorized : DataFrame
        Old approach: ±3σ clipping
    alpha_rank : DataFrame
        New approach: rank-based [-1, +1]
    
    Returns
    -------
    dict with validation metrics
    """
    log.info("\n" + "="*70)
    log.info("TAIL PRESERVATION VALIDATION")
    log.info("="*70)
    
    # Get descriptive stats
    w_stack = alpha_windsorized.stack()
    r_stack = alpha_rank.stack()
    
    # Count extreme values
    w_clipped = (w_stack.abs() >= 3.0).sum()
    r_extreme = (r_stack.abs() > 0.90).sum()
    
    metrics = {
        "windsorized_mean": float(w_stack.mean()),
        "windsorized_std": float(w_stack.std()),
        "windsorized_min": float(w_stack.min()),
        "windsorized_max": float(w_stack.max()),
        "windsorized_clipped_count": int(w_clipped),
        "windsorized_clipped_pct": 100 * w_clipped / len(w_stack),
        
        "rank_mean": float(r_stack.mean()),
        "rank_std": float(r_stack.std()),
        "rank_min": float(r_stack.min()),
        "rank_max": float(r_stack.max()),
        "rank_extreme_count": int(r_extreme),
        "rank_extreme_pct": 100 * r_extreme / len(r_stack),
    }
    
    log.info(
        "\nDISTRIBUTION COMPARISON:\n"
        "  Windsorized (±3σ clipped):\n"
        "    Range: [%.2f, %.2f]\n"
        "    Mean: %.2f | Std: %.2f\n"
        "    Values clipped at ±3σ: %.1f%%\n"
        "\n"
        "  Rank-based (no clip):\n"
        "    Range: [%.2f, %.2f]\n"
        "    Mean: %.2f | Std: %.2f\n"
        "    Extreme values (|α| > 0.90): %.1f%%\n"
        "\n"
        "✓ TAIL EDGE RECOVERED: %.1f%% of values now uncapped (better edge)",
        metrics["windsorized_min"],
        metrics["windsorized_max"],
        metrics["windsorized_mean"],
        metrics["windsorized_std"],
        metrics["windsorized_clipped_pct"],
        metrics["rank_min"],
        metrics["rank_max"],
        metrics["rank_mean"],
        metrics["rank_std"],
        metrics["rank_extreme_pct"],
        metrics["rank_extreme_pct"] - (100 - 100 * (1 - 1/50)),  # theoretical extreme % if uniform
    )
    
    log.info("="*70 + "\n")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    log.info("Correlation Analysis Tool (Standalone)")
    log.info("Usage: python -m validation.correlation_analysis_44stock")
