"""tests/test_features.py — Feature engine, diagnostics and store tests."""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.engine      import FeatureEngine
from features.core        import cs_zscore as _cs_zscore, cs_rank as _cs_rank
from features.diagnostics import FeatureDiagnostics
from features.store       import FeatureStore


# ── Synthetic data ────────────────────────────────────────────────────────────

def make_panels(tickers=("AAPL","MSFT","GOOG","AMZN","NVDA"),
                n_days=5, bars_per_day=390, seed=42):
    rng  = np.random.default_rng(seed)
    tz   = "America/New_York"
    dates= pd.bdate_range("2024-01-02", periods=n_days, tz=tz)
    idx  = pd.DatetimeIndex([
        pd.Timestamp(f"{d.date()} 09:30", tz=tz) + pd.Timedelta(minutes=i)
        for d in dates for i in range(bars_per_day)
    ])
    panels = {f: pd.DataFrame(index=idx, columns=list(tickers), dtype=float)
              for f in ["open","high","low","close","volume"]}
    for i,t in enumerate(tickers):
        price = 100.0 + i*30
        for d in dates:
            open_t = pd.Timestamp(f"{d.date()} 09:30", tz=tz)
            bar_ts = [open_t + pd.Timedelta(minutes=j) for j in range(bars_per_day)]
            prices = [price]
            for _ in range(bars_per_day-1):
                prices.append(prices[-1]*(1+rng.normal(0,0.0005)))
            for j,ts in enumerate(bar_ts):
                p  = prices[j]
                hi = p*(1+abs(rng.normal(0,0.0003)))
                lo = p*(1-abs(rng.normal(0,0.0003)))
                op = lo+rng.uniform(0,1)*(hi-lo)
                cl = lo+rng.uniform(0,1)*(hi-lo)
                tod= 1.0+2.0*np.exp(-j/30)+2.0*np.exp(-(bars_per_day-1-j)/30)
                panels["open"].loc[ts,t]=op; panels["high"].loc[ts,t]=hi
                panels["low"].loc[ts,t]=lo;  panels["close"].loc[ts,t]=cl
                panels["volume"].loc[ts,t]=float(rng.integers(1000,50000))*tod
            price = prices[-1]
    return {f: panels[f].astype(float) for f in panels}

@pytest.fixture(scope="module")
def panels():  return make_panels()

@pytest.fixture(scope="module")
def engine(panels): return FeatureEngine(panels, atr_window=10, vol_window=20,
                                         volume_window=3, zscore_window=20)

@pytest.fixture(scope="module")
def feats(engine): return engine.compute_all()


# ── Helper tests ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_cs_zscore_zero_mean(self, panels):
        z = _cs_zscore(panels["close"].pct_change().dropna())
        assert (z.mean(axis=1).dropna().abs() < 1e-8).all()

    def test_cs_zscore_clipped(self, panels):
        z = _cs_zscore(panels["close"].pct_change().dropna(), clip=3.0)
        assert z.max().max() <= 3.01 and z.min().min() >= -3.01

    def test_cs_rank_bounded(self, panels):
        r = _cs_rank(panels["close"].pct_change().dropna())
        assert r.max().max() <= 1.0+1e-6 and r.min().min() >= -1.0-1e-6

    def test_cs_rank_zero_mean(self, panels):
        r = _cs_rank(panels["close"].pct_change().dropna())
        assert (r.mean(axis=1).dropna().abs() < 0.1).mean() > 0.95


# ── Shape & index tests ───────────────────────────────────────────────────────

class TestShapes:
    def test_all_15_features_present(self, feats):
        assert set(feats) == {"A1_bar_reversal","A2_short_rev","A3_medium_rev","A4_overnight_gap","B1_vwap_deviation","B2_price_position","B3_open_gap","C1_volume_shock","C2_flow_imbalance","C3_turnover_shock","D1_vol_burst","D2_vol_zscore","D3_dispersion","E1_residual_return","E2_sector_residual"}

    def test_shape_matches_input(self, panels, feats):
        exp = panels["close"].shape
        for name,f in feats.items():
            assert f.shape == exp, f"{name}: {f.shape} != {exp}"

    def test_columns_match(self, panels, feats):
        exp = list(panels["close"].columns)
        for name,f in feats.items():
            assert list(f.columns) == exp, f"{name} column mismatch"

    def test_index_matches(self, panels, feats):
        for name,f in feats.items():
            assert f.index.equals(panels["close"].index), f"{name} index mismatch"


# ── Individual feature correctness ────────────────────────────────────────────

class TestF1IDR:
    def test_bounded(self, feats):
        assert feats["A1_bar_reversal"].dropna().abs().max().max() <= 3.01

    def test_cs_normalised(self, feats):
        means = feats["A1_bar_reversal"].mean(axis=1).dropna()
        assert (means.abs() < 1e-8).all()

class TestF2VWD:
    def test_vwap_at_first_bar_equals_close(self, engine):
        vwap = engine._session_vwap()
        dates = engine._dates.unique()
        for date in list(dates)[:2]:
            first = engine.C[engine._dates==date].index[0]
            pd.testing.assert_series_equal(
                vwap.loc[first].round(4),
                engine.C.loc[first].round(4), check_names=False)

    def test_bounded(self, feats):
        assert feats["B1_vwap_deviation"].dropna().abs().max().max() <= 3.01

class TestF3ZSR:
    def test_negates_returns(self, feats, panels):
        """A2_short_rev blends z-score + rank — should anti-correlate with recent returns."""
        zsr = feats["A2_short_rev"]
        ret = panels["close"].pct_change(3)
        s_ret = ret.iloc[25:].stack().dropna()
        s_zsr = zsr.iloc[25:].stack().dropna()
        common = s_ret.index.intersection(s_zsr.index)
        if len(common) > 50:
            corr, _ = spearmanr(s_ret[common], s_zsr[common])
            assert corr < 0, f"A2_short_rev should anti-correlate with returns, got {corr:.3f}"

class TestF4VSK:
    def test_has_nan_at_start(self, feats):
        assert feats["C1_volume_shock"].isna().any().any()

    def test_bounded(self, feats):
        assert feats["C1_volume_shock"].dropna().abs().max().max() <= 3.01

class TestF5VBT:
    def test_has_signal(self, feats):
        assert feats["D1_vol_burst"].dropna().abs().max().max() > 0

    def test_bounded(self, feats):
        assert feats["D1_vol_burst"].dropna().abs().max().max() <= 3.01

class TestF6RRR:
    def test_bounded(self, feats):
        # A2_short_rev blends z-score and rank — bounded at ~3.0 not 1.0
        assert feats["A2_short_rev"].dropna().abs().max().max() <= 3.1

    def test_reversal_direction(self, feats, panels):
        rrr = feats["A2_short_rev"]
        ret   = panels["close"].pct_change(3)
        s_ret = ret.iloc[5:].stack().dropna()
        s_rrr = rrr.iloc[5:].stack().dropna()
        common = s_ret.index.intersection(s_rrr.index)
        if len(common) > 100:
            corr, _ = spearmanr(s_ret[common], s_rrr[common])
            assert corr < 0

class TestF8IDD:
    def test_same_value_all_tickers(self, feats):
        """IDD is a scalar per bar — identical across all tickers."""
        row_std = feats["D3_dispersion"].std(axis=1).dropna()
        assert (row_std < 1e-8).all()

class TestF9RRS:
    def test_bounded(self, feats):
        assert feats["E1_residual_return"].dropna().abs().max().max() <= 3.01

    def test_has_values(self, feats):
        assert feats["E1_residual_return"].dropna().count().sum() > 0


# ── Diagnostics ───────────────────────────────────────────────────────────────

class TestDiagnostics:
    def test_ic_series_length(self, feats, panels):
        diag = FeatureDiagnostics(feats, panels["close"])
        ic   = diag.ic_series("A1_bar_reversal", forward_bars=1)
        assert len(ic) == len(panels["close"])

    def test_ic_summary_fields(self, feats, panels):
        diag = FeatureDiagnostics(feats, panels["close"])
        s    = diag.ic_summary("A2_short_rev", forward_bars=1)
        for key in ["IC_mean","IC_std","ICIR","t_stat","pct_positive"]:
            assert key in s

    def test_ic_decay_rows(self, feats, panels):
        diag  = FeatureDiagnostics(feats, panels["close"])
        decay = diag.ic_decay("A2_short_rev", max_lead=5)
        assert len(decay) == 5

    def test_signal_stats_all_features(self, feats, panels):
        diag  = FeatureDiagnostics(feats, panels["close"])
        stats = diag.signal_stats()
        assert len(stats) == len(feats)

    def test_correlation_diagonal_ones(self, feats, panels):
        diag = FeatureDiagnostics(feats, panels["close"])
        corr = diag.feature_correlation()
        assert corr.shape == (len(feats), len(feats))
        assert (np.diag(corr.values) == 1.0).all()

    def test_full_report_keys(self, feats, panels):
        diag   = FeatureDiagnostics(feats, panels["close"])
        report = diag.full_report(forward_bars=1, ic_decay_bars=3)
        for key in ["ic_summary","signal_stats","correlation","best_feature"]:
            assert key in report


# ── Feature store ─────────────────────────────────────────────────────────────

class TestFeatureStore:
    def test_roundtrip(self, feats, tmp_path):
        store = FeatureStore(str(tmp_path/"feats"))
        store.save(feats)
        loaded = store.load()
        assert set(loaded) == set(feats)
        for name in feats:
            # Loaded may have fewer rows if all-NaN warmup rows are dropped on pivot
            assert loaded[name].shape[1] == feats[name].shape[1]   # columns match
            assert loaded[name].shape[0] <= feats[name].shape[0]   # rows ≤ original
            assert loaded[name].shape[0] >= feats[name].dropna(how="all").shape[0]

    def test_list_features(self, feats, tmp_path):
        store = FeatureStore(str(tmp_path/"feats2"))
        store.save(feats)
        assert set(store.list_features()) == set(feats)

    def test_partial_load(self, feats, tmp_path):
        store = FeatureStore(str(tmp_path/"feats3"))
        store.save(feats)
        loaded = store.load(feature_names=["A1_bar_reversal","E1_residual_return"])
        assert set(loaded) == {"A1_bar_reversal","E1_residual_return"}

    def test_date_filter(self, feats, panels, tmp_path):
        store = FeatureStore(str(tmp_path/"feats4"))
        store.save(feats)
        mid   = str(panels["close"].index[len(panels["close"])//2].date())
        loaded= store.load(start_date=mid)
        for f in loaded.values():
            assert f.index.min() >= pd.Timestamp(mid, tz="America/New_York")

    def test_feature_info(self, feats, tmp_path):
        store = FeatureStore(str(tmp_path/"feats5"))
        store.save(feats)
        info  = store.feature_info()
        assert len(info) == len(feats)
        assert "size_mb" in info.columns


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        panels = make_panels(n_days=5, seed=99)
        eng    = FeatureEngine(panels, atr_window=10, vol_window=20,
                               volume_window=3, zscore_window=20)
        feats  = eng.compute_all()
        assert len(feats) == 15

        store  = FeatureStore(str(tmp_path/"store"))
        store.save(feats)
        loaded = store.load()
        assert len(loaded) == 15

        diag   = FeatureDiagnostics(loaded, panels["close"])
        report = diag.full_report(forward_bars=1, ic_decay_bars=3)
        assert report["best_feature"] is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
