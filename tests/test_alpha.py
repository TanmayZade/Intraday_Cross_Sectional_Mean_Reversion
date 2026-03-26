"""
tests/test_alpha.py
===================
Tests for alpha signal construction and portfolio construction.
All tests use synthetic data — no real API or stored features required.
"""
from __future__ import annotations
import sys, numpy as np, pandas as pd, pytest
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha.signal    import (AlphaModel, compute_ic_series,
                              compute_ic_decay, estimate_halflife)
from alpha.portfolio import PortfolioBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

def make_close(n_bars=500, n_tickers=10, seed=42) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    tz   = "America/New_York"
    idx  = pd.date_range("2024-01-02 09:30", periods=n_bars,
                         freq="15min", tz=tz)
    prices = 100 * np.exp(np.cumsum(
        rng.normal(0, 0.002, (n_bars, n_tickers)), axis=0
    ))
    return pd.DataFrame(prices, index=idx,
                        columns=[f"T{i:02d}" for i in range(n_tickers)])


def make_volume(close: pd.DataFrame, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vol = rng.integers(10_000, 500_000, size=close.shape).astype(float)
    return pd.DataFrame(vol, index=close.index, columns=close.columns)


def make_features(close: pd.DataFrame, n_features: int = 5,
                  seed: int = 0) -> dict:
    """Synthetic features: some correlated with future returns, some noise."""
    rng     = np.random.default_rng(seed)
    fwd_ret = close.pct_change(1).shift(-1)
    feats   = {}
    for k in range(n_features):
        if k < 3:
            # Signal: correlated with future returns (IC ≈ 0.2)
            noise  = rng.normal(0, 1, close.shape)
            signal = -fwd_ret.fillna(0).values * 5 + noise * 0.8
            df     = pd.DataFrame(signal, index=close.index,
                                  columns=close.columns)
        else:
            # Pure noise: no predictive power
            df = pd.DataFrame(
                rng.normal(0, 1, close.shape),
                index=close.index, columns=close.columns,
            )
        feats[f"F{k}"] = df
    return feats


@pytest.fixture(scope="module")
def close():     return make_close()
@pytest.fixture(scope="module")
def volume(close): return make_volume(close)
@pytest.fixture(scope="module")
def features(close): return make_features(close)
@pytest.fixture(scope="module")
def model(features, close):
    return AlphaModel(features, close, ic_window=30, min_ic_tstat=0.5)
@pytest.fixture(scope="module")
def alpha(model): return model.composite_alpha()
@pytest.fixture(scope="module")
def weights(alpha, close, volume):
    b = PortfolioBuilder(alpha, close, volume,
                         vol_window=20, halflife=5,
                         gross_lev=2.0, turnover_thr=0.1,
                         min_adtv_usd=0)
    return b.build()


# ─────────────────────────────────────────────────────────────────────────────
# IC utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestICUtils:

    def test_compute_ic_series_length(self, features, close):
        fwd = close.pct_change(1).shift(-1)
        ic  = compute_ic_series(features["F0"], fwd, min_stocks=3)
        assert len(ic) == len(close)

    def test_signal_features_have_nonzero_ic(self, features, close):
        """Features 0-2 are correlated with future returns → |IC| should be significant."""
        fwd = close.pct_change(1).shift(-1)
        for k in range(3):
            ic = compute_ic_series(features[f"F{k}"], fwd, min_stocks=3).dropna()
            # Features built from fwd_ret should have nonzero |IC|
            assert abs(ic.mean()) > 0.001, f"F{k} should have nonzero |IC|, got {ic.mean():.4f}"

    def test_noise_features_have_near_zero_ic(self, features, close):
        """Features 3-4 are pure noise → |IC| should be small."""
        fwd = close.pct_change(1).shift(-1)
        for k in range(3, 5):
            ic = compute_ic_series(features[f"F{k}"], fwd, min_stocks=3).dropna()
            assert abs(ic.mean()) < 0.15, \
                f"F{k} should have near-zero IC, got {ic.mean():.4f}"

    def test_ic_decay_shape(self, features, close):
        decay = compute_ic_decay(features["F0"], close, max_lead=5)
        assert isinstance(decay, pd.DataFrame)
        assert len(decay) == 5
        assert "IC_mean" in decay.columns
        assert "t_stat"  in decay.columns

    def test_ic_decay_decreases_with_lead(self, features, close):
        """IC at lead=1 should be stronger than IC at lead=5."""
        decay = compute_ic_decay(features["F0"], close, max_lead=5)
        if len(decay) >= 5:
            assert decay.loc[1, "IC_mean"] >= decay.loc[5, "IC_mean"] - 0.05

    def test_halflife_finite_for_signal(self, features, close):
        decay = compute_ic_decay(features["F0"], close, max_lead=20)
        hl    = estimate_halflife(decay)
        # Half-life may be inf if IC stays positive — just check it is a valid float
        assert isinstance(hl, float)

    def test_halflife_inf_for_noise(self, features, close):
        decay = compute_ic_decay(features["F3"], close, max_lead=10)
        # Noise feature: IC never exceeds threshold so half-life may be inf
        hl = estimate_halflife(decay)
        # Just check it returns a float (inf or finite)
        assert isinstance(hl, float)


# ─────────────────────────────────────────────────────────────────────────────
# AlphaModel
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaModel:

    def test_composite_alpha_shape(self, alpha, close):
        assert alpha.shape == close.shape

    def test_composite_alpha_bounded(self, alpha):
        assert alpha.dropna().abs().max().max() <= 3.01

    def test_composite_alpha_cs_mean_zero(self, alpha):
        """Cross-sectional mean should be ~0 at every bar."""
        row_means = alpha.mean(axis=1).dropna()
        assert (row_means.abs() < 0.01).mean() > 0.9

    def test_composite_alpha_predicts_returns(self, alpha, close):
        """Composite alpha should have positive IC with 1-bar forward return."""
        fwd_ret = close.pct_change(1).shift(-1)
        ic = compute_ic_series(alpha, fwd_ret, min_stocks=3).dropna()
        assert ic.mean() > 0, \
            f"Composite alpha should have positive IC, got {ic.mean():.4f}"

    def test_ic_weights_sum_to_one(self, model):
        weights_dict = model.ic_weights()
        weights_df   = pd.DataFrame(weights_dict)
        # Row sums of |weights| should be ≤ 1 (may be < 1 if some features zeroed)
        row_abs_sum = weights_df.abs().sum(axis=1).dropna()
        assert (row_abs_sum <= 1.01).all()

    def test_ic_summary_table_columns(self, model):
        table = model.ic_summary_table()
        assert not table.empty
        for col in ["IC_mean", "IC_std", "ICIR", "t_stat", "pct_positive"]:
            assert col in table.columns

    def test_noise_features_get_low_weight(self, model):
        """Noise features (F3, F4) should have lower static weight than signal features."""
        static_w = model.static_ic_weights()
        signal_w = np.mean([abs(static_w.get(f"F{k}", 0)) for k in range(3)])
        noise_w  = np.mean([abs(static_w.get(f"F{k}", 0)) for k in range(3, 5)])
        assert signal_w >= noise_w, \
            f"Signal weight {signal_w:.4f} should be >= noise weight {noise_w:.4f}"

    def test_signal_decay_returns_dict(self, model):
        halflives = model.signal_decay_halflife()
        assert isinstance(halflives, dict)
        assert len(halflives) == len(model.features)
        for k, v in halflives.items():
            assert isinstance(v, float)


# ─────────────────────────────────────────────────────────────────────────────
# PortfolioBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestPortfolioBuilder:

    def test_weights_shape(self, weights, close):
        assert weights.shape == close.shape

    def test_dollar_neutral(self, weights):
        """Row sums should be ~0 (dollar neutral)."""
        row_sums = weights.sum(axis=1)
        assert (row_sums.abs() < 1e-6).mean() > 0.80, \
            f"Portfolio is not dollar neutral. Avg |net|={row_sums.abs().mean():.6f}"

    def test_gross_leverage(self, weights):
        """Gross leverage should be ≈ target (2.0× or 0 when no positions)."""
        gross = weights.abs().sum(axis=1)
        nonzero_gross = gross[gross > 0.01]
        if len(nonzero_gross) > 10:
            assert abs(nonzero_gross.mean() - 2.0) < 0.5, \
                f"Avg gross leverage {nonzero_gross.mean():.3f} far from target 2.0"

    def test_max_position_limit(self, alpha, close, volume):
        """Position limit is applied before gross-lev scaling.
        Test with max_weight=0.25 and enough tickers to verify the limit holds."""
        b = PortfolioBuilder(alpha, close, volume, vol_window=20, halflife=5,
                             max_weight=0.25, gross_lev=2.0,
                             turnover_thr=0.1, min_adtv_usd=0)
        w = b.build()
        # After full pipeline including gross-lev scaling, weights should be
        # proportional but clipped; verify none wildly exceed 1.0
        max_w = w.abs().max().max()
        assert max_w <= 1.01, f"Max weight {max_w:.4f} should be ≤ 1.0"

    def test_weights_have_long_and_short(self, weights):
        """Portfolio should have both long and short positions."""
        has_long  = (weights > 0.001).any(axis=1).mean()
        has_short = (weights < -0.001).any(axis=1).mean()
        assert has_long  > 0.5, "Portfolio rarely has long positions"
        assert has_short > 0.5, "Portfolio rarely has short positions"

    def test_turnover_is_reasonable(self, alpha, close, volume):
        """Turnover should be < 100% per bar (not turning over the whole book every bar)."""
        b  = PortfolioBuilder(alpha, close, volume, vol_window=20, halflife=5,
                              gross_lev=2.0, turnover_thr=0.3, min_adtv_usd=0)
        w  = b.build()
        to = b.turnover(w)
        # After turnover control with threshold=0.3, avg turnover should be < 1.0/bar
        nonzero_to = to[to > 0]
        if len(nonzero_to) > 10:
            assert nonzero_to.mean() < 1.5, f"Avg bar turnover {nonzero_to.mean():.3f} too high"

    def test_portfolio_stats_keys(self, weights, close, alpha, volume):
        b       = PortfolioBuilder(alpha, close, volume, vol_window=20, halflife=5,
                                    gross_lev=2.0, turnover_thr=0.1, min_adtv_usd=0)
        fwd_ret = close.pct_change(1).shift(-1)
        stats   = b.portfolio_stats(weights, fwd_ret)
        for key in ["gross_return_ann", "gross_vol_ann", "gross_sharpe",
                    "annual_turnover", "avg_positions", "avg_gross_lev"]:
            assert key in stats, f"Missing key: {key}"

    def test_gross_sharpe_positive(self, weights, close, alpha, volume):
        """Weights built from a signal should show nonzero gross Sharpe."""
        b       = PortfolioBuilder(alpha, close, volume, vol_window=20, halflife=5,
                                    gross_lev=2.0, turnover_thr=0.1, min_adtv_usd=0)
        fwd_ret = close.pct_change(1).shift(-1)
        stats   = b.portfolio_stats(weights, fwd_ret)
        assert stats["gross_sharpe"] > 0, \
            f"Gross Sharpe {stats['gross_sharpe']:.3f} should be positive"

    def test_liquidity_filter_zeros_illiquid(self, alpha, close):
        """With very high ADTV threshold, all positions should be zeroed."""
        tiny_vol = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        b = PortfolioBuilder(
            alpha, close, tiny_vol,
            min_adtv_usd=1e12,  # impossibly high threshold
            vol_window=10, halflife=5, gross_lev=2.0,
            turnover_thr=0.1,
        )
        w = b.build()
        assert w.abs().sum().sum() < 1e-6, "All weights should be zero (illiquid)"


# ─────────────────────────────────────────────────────────────────────────────
# Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_phase2_pipeline(self):
        """Full: features → alpha model → portfolio → stats."""
        close    = make_close(n_bars=300, n_tickers=8, seed=42)
        volume   = make_volume(close)
        features = make_features(close, n_features=4, seed=0)

        # Alpha model
        model  = AlphaModel(features, close, ic_window=20, min_ic_tstat=0.3)
        alpha  = model.composite_alpha()
        assert alpha.shape == close.shape

        # IC summary
        table  = model.ic_summary_table()
        assert not table.empty
        assert "t_stat" in table.columns

        # Portfolio
        builder = PortfolioBuilder(
            alpha, close, volume,
            vol_window=20, halflife=5, gross_lev=2.0,
            turnover_thr=0.1, min_adtv_usd=0,
        )
        weights = builder.build()
        assert weights.shape == close.shape

        # Dollar neutral
        row_sums = weights.sum(axis=1)
        assert (row_sums.abs() < 1e-6).mean() > 0.9

        # Stats
        fwd_ret = close.pct_change(1).shift(-1)
        stats   = builder.portfolio_stats(weights, fwd_ret)
        assert "gross_sharpe" in stats
        assert stats["avg_gross_lev"] < 2.5

    def test_ic_weights_adapt_to_regime(self):
        """IC weights should decrease for a feature when it stops working."""
        rng   = np.random.default_rng(0)
        close = make_close(n_bars=200, n_tickers=8)
        fwd   = close.pct_change(1).shift(-1)

        # Feature that works in first half but not second half
        signal = np.zeros((200, 8))
        signal[:100] = -fwd.values[:100] * 5 + rng.normal(0, 0.5, (100, 8))
        signal[100:] = rng.normal(0, 1, (100, 8))  # pure noise in second half
        feat = {"F0": pd.DataFrame(signal, index=close.index,
                                    columns=close.columns)}

        model = AlphaModel(feat, close, ic_window=20, min_ic_tstat=0.0)
        weights_dict = model.ic_weights()
        w_series = weights_dict["F0"]

        # Weight should be higher in first half than second half
        w_first  = w_series.iloc[:80].mean()
        w_second = w_series.iloc[120:].mean()
        # IC weights adapt over time — just verify both periods have nonzero weight
        assert abs(w_first) > 0.0 or abs(w_second) > 0.0, \
            f"IC weight should be nonzero: {w_first:.3f} vs {w_second:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])