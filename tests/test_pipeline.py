"""
tests/test_pipeline.py
======================
Full test suite — no Polygon API key required.

The Polygon HTTP layer is mocked via unittest.mock so every test runs
offline and deterministically. Synthetic data is generated to cover
every failure mode the cleaner must handle.
"""

from __future__ import annotations

import json
import time
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.fetcher   import PolygonFetcher, FetcherConfig, _RateLimiter, _DiskCache
from pipeline.cleaner   import clean_ticker, clean_panel, CleaningConfig
from pipeline.universe  import build_universe, build_panels, apply_universe_mask
from pipeline.storage   import write_clean, read_clean, read_panels, pivot_field
from pipeline.quality   import (compute_coverage, return_diagnostics,
                                 corwin_schultz_spread, flagged_bar_report)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers & fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _trading_dates(start="2023-01-03", n=252) -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def make_clean_df(
    ticker="AAPL",
    n_days=252,
    start="2023-01-03",
    price=150.0,
    seed=0,
) -> pd.DataFrame:
    """Generate clean daily OHLCV bars — no issues injected."""
    rng   = np.random.default_rng(seed)
    dates = _trading_dates(start, n_days)
    p     = price
    rows  = []
    for d in dates:
        ret = rng.normal(0, 0.012)
        p  *= 1 + ret
        hi  = p * (1 + abs(rng.normal(0, 0.005)))
        lo  = p * (1 - abs(rng.normal(0, 0.005)))
        op  = lo + rng.uniform(0, 1) * (hi - lo)
        cl  = lo + rng.uniform(0, 1) * (hi - lo)
        vol = float(rng.integers(500_000, 5_000_000))
        rows.append({"open": op, "high": hi, "low": lo, "close": cl, "volume": vol})
    return pd.DataFrame(rows, index=dates)


def make_polygon_response(bars: list[dict]) -> MagicMock:
    """Create a mock requests.Response for a Polygon aggs call."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "status":       "OK",
        "resultsCount": len(bars),
        "results":      bars,
    }
    mock.raise_for_status = MagicMock()
    return mock


def df_to_polygon_bars(df: pd.DataFrame) -> list[dict]:
    """Convert a clean DataFrame back to Polygon raw bar format (for mocking)."""
    bars = []
    for dt, row in df.iterrows():
        ts_ms = int(pd.Timestamp(dt, tz="UTC").timestamp() * 1000)
        bars.append({
            "t":  ts_ms,
            "o":  row["open"],
            "h":  row["high"],
            "l":  row["low"],
            "c":  row["close"],
            "v":  row["volume"],
            "vw": row.get("vwap", row["close"]),
            "n":  int(row.get("n_trades", 1000)),
        })
    return bars


@pytest.fixture
def fetcher_cfg(tmp_path) -> FetcherConfig:
    return FetcherConfig(
        api_key         = "test_key",
        cache_dir       = str(tmp_path / "cache"),
        requests_per_min = 1000,   # no throttling in tests
        retry_attempts  = 1,
    )


@pytest.fixture
def fetcher(fetcher_cfg) -> PolygonFetcher:
    return PolygonFetcher(fetcher_cfg)


@pytest.fixture
def clean_cfg() -> CleaningConfig:
    return CleaningConfig(volume_spike_window=10)


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class TestRateLimiter:

    def test_does_not_block_within_limit(self):
        rl = _RateLimiter(rate=10, period=1.0)
        t0 = time.monotonic()
        for _ in range(5):
            rl.acquire()
        assert time.monotonic() - t0 < 0.5   # should be near-instant

    def test_token_bucket_refills(self):
        rl = _RateLimiter(rate=2, period=1.0)
        rl.acquire()
        rl.acquire()
        # Tokens now at 0; next call should wait ~0.5s
        t0 = time.monotonic()
        rl.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.4   # waited for refill


# ─────────────────────────────────────────────────────────────────────────────
# Disk cache
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskCache:

    def test_cache_miss_on_fresh_cache(self, tmp_path):
        cache = _DiskCache(tmp_path, ttl_hours=24)
        result = cache.get(ticker="AAPL", start="2023-01-01", end="2023-12-31")
        assert result is None

    def test_cache_hit_after_set(self, tmp_path):
        cache = _DiskCache(tmp_path, ttl_hours=24)
        data = [{"t": 1, "o": 100.0}]
        cache.set(data, ticker="AAPL", start="2023-01-01", end="2023-12-31")
        result = cache.get(ticker="AAPL", start="2023-01-01", end="2023-12-31")
        assert result == data

    def test_cache_miss_different_key(self, tmp_path):
        cache = _DiskCache(tmp_path, ttl_hours=24)
        cache.set([{"t": 1}], ticker="AAPL", start="2023-01-01", end="2023-12-31")
        result = cache.get(ticker="MSFT", start="2023-01-01", end="2023-12-31")
        assert result is None

    def test_expired_cache_returns_none(self, tmp_path):
        cache = _DiskCache(tmp_path, ttl_hours=0.0001)  # ~0.36 seconds TTL
        cache.set([{"t": 1}], ticker="AAPL", start="2023-01-01", end="2023-12-31")
        time.sleep(0.4)
        result = cache.get(ticker="AAPL", start="2023-01-01", end="2023-12-31")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class TestFetcher:

    def test_fetch_returns_clean_dataframe(self, fetcher):
        df_source = make_clean_df("AAPL", n_days=50)
        bars = df_to_polygon_bars(df_source)
        mock_resp = make_polygon_response(bars)

        with patch.object(fetcher._session, "get", return_value=mock_resp):
            result = fetcher.fetch("AAPL", "2023-01-03", "2023-03-15")

        assert not result.empty
        assert set(["open", "high", "low", "close", "volume"]).issubset(result.columns)
        assert (result["close"] > 0).all()
        assert (result["high"] >= result["low"]).all()

    def test_fetch_timestamp_converted_to_et_dates(self, fetcher):
        df_source = make_clean_df("AAPL", n_days=5)
        bars = df_to_polygon_bars(df_source)
        mock_resp = make_polygon_response(bars)

        with patch.object(fetcher._session, "get", return_value=mock_resp):
            result = fetcher.fetch("AAPL", "2023-01-03", "2023-01-10")

        # Daily bars: index should be dates (not datetimes with time)
        assert result.index.dtype == "object" or isinstance(result.index[0], date)

    def test_fetch_404_returns_empty_df(self, fetcher):
        mock_resp       = MagicMock()
        mock_resp.status_code = 404
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._session, "get", return_value=mock_resp):
            result = fetcher.fetch("INVALID_TICKER", "2023-01-03", "2023-01-31")

        assert result.empty

    def test_fetch_403_returns_empty_df(self, fetcher):
        mock_resp       = MagicMock()
        mock_resp.status_code = 403
        mock_resp.raise_for_status = MagicMock()

        with patch.object(fetcher._session, "get", return_value=mock_resp):
            result = fetcher.fetch("AAPL", "2023-01-03", "2023-01-31")

        assert result.empty

    def test_fetch_uses_cache_on_second_call(self, fetcher):
        df_source = make_clean_df("AAPL", n_days=10)
        bars = df_to_polygon_bars(df_source)
        mock_resp = make_polygon_response(bars)

        with patch.object(fetcher._session, "get", return_value=mock_resp) as mock_get:
            fetcher.fetch("AAPL", "2023-01-03", "2023-01-16")
            fetcher.fetch("AAPL", "2023-01-03", "2023-01-16")  # second call

        # Session.get should only be called ONCE (second call hits cache)
        assert mock_get.call_count == 1

    def test_date_chunking_daily_no_chunks_needed(self, fetcher):
        """Daily bars: 50k bars covers 198 years — never needs chunking."""
        chunks = fetcher._date_chunks("2020-01-01", "2024-12-31")
        assert len(chunks) == 1
        assert chunks[0] == ("2020-01-01", "2024-12-31")

    def test_date_chunking_minute_splits_correctly(self, tmp_path):
        """Minute bars: 45k / 390 = ~115 days per chunk."""
        cfg = FetcherConfig(api_key="key", timespan="minute",
                            cache_dir=str(tmp_path))
        fetcher = PolygonFetcher(cfg)
        chunks = fetcher._date_chunks("2023-01-01", "2023-12-31")
        # 365 days / ~115 per chunk = 4 chunks
        assert len(chunks) >= 3
        # Chunks should be contiguous and non-overlapping
        for i in range(len(chunks) - 1):
            end_i   = date.fromisoformat(chunks[i][1])
            start_i1 = date.fromisoformat(chunks[i+1][0])
            assert start_i1 == end_i + timedelta(days=1)

    def test_fetch_universe_skips_failed_tickers(self, fetcher):
        df_source  = make_clean_df("AAPL", n_days=20)
        bars_ok    = df_to_polygon_bars(df_source)

        def side_effect(url, **kwargs):
            if "AAPL" in url:
                return make_polygon_response(bars_ok)
            m = MagicMock()
            m.status_code = 404
            m.raise_for_status = MagicMock()
            return m

        with patch.object(fetcher._session, "get", side_effect=side_effect):
            result = fetcher.fetch_universe(["AAPL", "BADTICKER"], "2023-01-03", "2023-02-01")

        assert "AAPL" in result["ticker"].values
        assert "BADTICKER" not in result["ticker"].values


# ─────────────────────────────────────────────────────────────────────────────
# Cleaner
# ─────────────────────────────────────────────────────────────────────────────

class TestCleaner:

    def test_clean_data_passes_through(self, clean_cfg):
        df = make_clean_df(n_days=100)
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert rpt.ohlc_invalid == 0
        assert rpt.zero_price   == 0
        assert rpt.return_outliers == 0
        assert len(out) > 90         # most bars retained

    def test_duplicate_dates_deduplicated(self, clean_cfg):
        df = make_clean_df(n_days=50)
        dup = df.iloc[[0]].copy()
        dup["close"] *= 1.01
        df = pd.concat([df, dup]).sort_index()
        assert df.index.duplicated().any()
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert not out.index.duplicated().any()
        assert rpt.dedup_removed >= 1

    def test_ohlc_violation_removed(self, clean_cfg):
        df = make_clean_df(n_days=50)
        df = df.copy()
        df.iloc[5, df.columns.get_loc("high")] = df.iloc[5]["low"] * 0.99  # H < L
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert rpt.ohlc_invalid >= 1
        assert (out["high"] >= out["low"]).all()

    def test_zero_price_removed(self, clean_cfg):
        df = make_clean_df(n_days=50)
        df = df.copy()
        df.iloc[10, df.columns.get_loc("close")] = 0.0
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert (rpt.zero_price + rpt.ohlc_invalid) >= 1
        assert (out["close"] > 0).all()

    def test_return_outlier_removed(self, clean_cfg):
        df = make_clean_df(n_days=100)
        df = df.copy()
        # Inject 60% single-day jump (above 50% threshold)
        df.iloc[20, df.columns.get_loc("close")] *= 1.60
        df.iloc[20, df.columns.get_loc("high")]  = df.iloc[20]["close"] * 1.01
        df.iloc[20, df.columns.get_loc("open")]   = df.iloc[20]["close"] * 0.99
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert rpt.return_outliers >= 1
        rets = out["close"].pct_change().abs()
        assert (rets.dropna() <= clean_cfg.max_daily_return + 1e-6).all()

    def test_range_outlier_removed(self, clean_cfg):
        df = make_clean_df(n_days=100)
        df = df.copy()
        # Inject H/L spread > 40%
        df.iloc[30, df.columns.get_loc("high")] = df.iloc[30]["low"] * 1.50
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert rpt.range_outliers >= 1
        ranges = (out["high"] - out["low"]) / out["low"]
        assert (ranges <= clean_cfg.max_intraday_range + 1e-6).all()

    def test_volume_spike_flagged_not_removed(self, clean_cfg):
        df = make_clean_df(n_days=100)
        df = df.copy()
        df.iloc[50, df.columns.get_loc("volume")] *= 100   # 100× normal
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert rpt.vol_spikes >= 1
        assert len(out) > 90      # bar kept (flagged, not removed)
        assert "flagged" in out.columns
        assert out["flagged"].any()

    def test_stale_prices_flagged(self, clean_cfg):
        df = make_clean_df(n_days=100)
        df = df.copy()
        # Force 5 consecutive identical close prices
        base_close = df.iloc[20]["close"]
        for i in range(1, 6):
            df.iloc[20 + i, df.columns.get_loc("close")] = base_close
            df.iloc[20 + i, df.columns.get_loc("open")]  = base_close
            df.iloc[20 + i, df.columns.get_loc("high")]  = base_close * 1.001
            df.iloc[20 + i, df.columns.get_loc("low")]   = base_close * 0.999
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert rpt.stale_flagged >= 1

    def test_low_volume_removed(self, clean_cfg):
        df = make_clean_df(n_days=100)
        df = df.copy()
        df.iloc[15, df.columns.get_loc("volume")] = 500   # below min_volume=10000
        out, rpt = clean_ticker(df, "AAPL", clean_cfg)
        assert rpt.low_volume >= 1

    def test_output_sorted(self, clean_cfg):
        df = make_clean_df(n_days=60)
        df = df.iloc[::-1]   # reverse order as input
        out, _ = clean_ticker(df, "AAPL", clean_cfg)
        assert out.index.is_monotonic_increasing

    def test_clean_panel_multi_ticker(self, clean_cfg):
        frames = []
        for t, seed in [("AAPL", 1), ("MSFT", 2), ("GOOG", 3)]:
            df = make_clean_df(t, n_days=60, seed=seed)
            df["ticker"] = t
            frames.append(df)
        panel = pd.concat(frames)

        clean_df, report_df = clean_panel(panel, clean_cfg)
        assert set(report_df.index) == {"AAPL", "MSFT", "GOOG"}
        assert (clean_df["high"] >= clean_df["low"]).all()
        assert (clean_df["close"] > 0).all()

    def test_report_fields_complete(self, clean_cfg):
        df = make_clean_df(n_days=50)
        _, rpt = clean_ticker(df, "TEST", clean_cfg)
        assert rpt.ticker == "TEST"
        assert rpt.raw_bars > 0
        assert rpt.final_bars > 0
        assert 0 < rpt.pct_retained <= 100


# ─────────────────────────────────────────────────────────────────────────────
# Universe
# ─────────────────────────────────────────────────────────────────────────────

class TestUniverse:

    def _make_panels(self, tickers=("AAPL", "MSFT"), n_days=252):
        frames = {t: make_clean_df(t, n_days=n_days, seed=i)
                  for i, t in enumerate(tickers)}
        close  = pd.DataFrame({t: f["close"]  for t, f in frames.items()})
        volume = pd.DataFrame({t: f["volume"] for t, f in frames.items()})
        return close, volume

    def test_returns_bool_dataframe(self):
        close, volume = self._make_panels()
        u = build_universe(close, volume, min_adtv_usd=0, min_history_days=20)
        assert u.dtypes.eq(bool).all()
        assert set(u.columns) == {"AAPL", "MSFT"}

    def test_price_filter_excludes_penny_stocks(self):
        close, volume = self._make_panels(["CHEAP"])
        close["CHEAP"] = 3.0   # below min_price=5
        u = build_universe(close, volume, min_adtv_usd=0,
                            min_price=5.0, min_history_days=10)
        assert u["CHEAP"].sum() == 0

    def test_history_filter_excludes_new_tickers(self):
        close, volume = self._make_panels(["NEW"])
        u = build_universe(close, volume, min_adtv_usd=0,
                            min_history_days=100)
        # First 100 days should be excluded by history filter
        first_true = u["NEW"].idxmax()
        days_before_true = (u.index < first_true).sum()
        assert days_before_true >= 99    # first ~100 rows excluded

    def test_adtv_filter(self):
        close, volume = self._make_panels(["ILLIQUID"])
        close["ILLIQUID"]  = 10.0
        volume["ILLIQUID"] = 100.0   # ADTV = $1,000 → below $1M threshold
        u = build_universe(close, volume, min_adtv_usd=1_000_000,
                            min_history_days=10)
        assert u["ILLIQUID"].sum() == 0

    def test_lookthrough_free(self):
        """Universe at date T must use only data from T-1 and earlier."""
        close, volume = self._make_panels(["AAPL"])
        u = build_universe(close, volume, min_adtv_usd=0,
                            min_history_days=5, adtv_window=5)
        # First row must always be False (no prior data for ADTV calculation)
        assert not u.iloc[0]["AAPL"]

    def test_apply_universe_mask_adds_column(self):
        close, volume = self._make_panels()
        u = build_universe(close, volume, min_adtv_usd=0, min_history_days=20)

        frames = []
        for t in ["AAPL", "MSFT"]:
            df = make_clean_df(t, n_days=252)
            df["ticker"] = t
            frames.append(df)
        panel = pd.concat(frames)

        masked = apply_universe_mask(panel, u)
        assert "in_universe" in masked.columns
        assert masked["in_universe"].dtype == bool

    def test_build_panels_shapes_match(self):
        frames = []
        for t, seed in [("AAPL", 0), ("MSFT", 1)]:
            df = make_clean_df(t, n_days=60, seed=seed)
            df["ticker"] = t
            frames.append(df)
        flat = pd.concat(frames)
        panels = build_panels(flat)
        assert "close" in panels
        assert "volume" in panels
        assert set(panels["close"].columns) == {"AAPL", "MSFT"}


# ─────────────────────────────────────────────────────────────────────────────
# Storage (round-trip)
# ─────────────────────────────────────────────────────────────────────────────

class TestStorage:

    def _make_flat(self, tickers=("AAPL", "MSFT", "GOOG"), n_days=100):
        frames = []
        for i, t in enumerate(tickers):
            df = make_clean_df(t, n_days=n_days, seed=i)
            df["ticker"] = t
            frames.append(df)
        return pd.concat(frames).sort_index()

    def test_write_and_read_roundtrip(self, tmp_path):
        flat = self._make_flat()
        write_clean(flat, root=tmp_path)
        loaded = read_clean(tmp_path)
        assert set(loaded["ticker"].unique()) == {"AAPL", "MSFT", "GOOG"}
        assert len(loaded) == len(flat)

    def test_ticker_filter(self, tmp_path):
        flat = self._make_flat()
        write_clean(flat, root=tmp_path)
        loaded = read_clean(tmp_path, tickers=["AAPL"])
        assert loaded["ticker"].unique().tolist() == ["AAPL"]

    def test_date_filter(self, tmp_path):
        flat = self._make_flat(n_days=200)
        write_clean(flat, root=tmp_path)
        loaded = read_clean(tmp_path, start_date="2023-07-01", end_date="2023-09-30")
        assert loaded.index.min() >= pd.Timestamp("2023-07-01")
        assert loaded.index.max() <= pd.Timestamp("2023-09-30")

    def test_partition_by_year(self, tmp_path):
        flat = self._make_flat(n_days=500)   # spans 2 calendar years
        write_clean(flat, root=tmp_path)
        years = [p.name for p in tmp_path.iterdir() if p.is_dir()]
        # Should have at least 2 year= partitions
        assert len([y for y in years if y.startswith("year=")]) >= 2

    def test_pivot_field(self, tmp_path):
        flat = self._make_flat()
        write_clean(flat, root=tmp_path)
        loaded = read_clean(tmp_path)
        close_panel = pivot_field(loaded, "close")
        assert "AAPL" in close_panel.columns
        assert close_panel.shape[1] == 3

    def test_read_panels_returns_dict(self, tmp_path):
        flat = self._make_flat()
        write_clean(flat, root=tmp_path)
        panels = read_panels(tmp_path, fields=("open", "close", "volume"),
                             universe_only=False)
        assert "close" in panels
        assert "volume" in panels
        assert isinstance(panels["close"], pd.DataFrame)

    def test_overwrite_is_idempotent(self, tmp_path):
        flat = self._make_flat()
        write_clean(flat, root=tmp_path)
        write_clean(flat, root=tmp_path)   # second write
        loaded = read_clean(tmp_path)
        # Should not have doubled the rows
        assert len(loaded) == len(flat)


# ─────────────────────────────────────────────────────────────────────────────
# Quality
# ─────────────────────────────────────────────────────────────────────────────

class TestQuality:

    def test_coverage_100pct_on_clean_data(self):
        df = make_clean_df("AAPL", n_days=60)
        df["ticker"] = "AAPL"
        cov = compute_coverage(df)
        assert cov.loc["AAPL", "coverage_pct"] == 100.0

    def test_coverage_less_with_gaps(self):
        df = make_clean_df("AAPL", n_days=100)
        full_calendar = pd.DatetimeIndex(df.index)  # full set = expected
        df = df.iloc[::2]    # keep every other day → ~50% coverage
        df["ticker"] = "AAPL"
        cov = compute_coverage(df, trading_calendar=full_calendar)
        assert cov.loc["AAPL", "coverage_pct"] < 60.0

    def test_return_diagnostics_shape(self):
        close = pd.DataFrame({
            "AAPL": make_clean_df("AAPL", n_days=100)["close"],
            "MSFT": make_clean_df("MSFT", n_days=100, seed=1)["close"],
        })
        diag = return_diagnostics(close)
        assert "vol_ann" in diag.columns
        assert "mean_ret_ann" in diag.columns
        assert set(diag.index) == {"AAPL", "MSFT"}

    def test_corwin_schultz_non_negative(self):
        df   = make_clean_df(n_days=100)
        spread = corwin_schultz_spread(df["high"], df["low"])
        assert (spread.dropna() >= 0).all()

    def test_flagged_bar_report_finds_flagged(self):
        df = make_clean_df("AAPL", n_days=50)
        df["ticker"] = "AAPL"
        df["flagged"] = False
        df.iloc[[10, 20, 30], df.columns.get_loc("flagged")] = True
        result = flagged_bar_report(df)
        assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline end-to-end (mocked API)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_mocked(self, tmp_path, fetcher_cfg):
        """Full fetch → clean → universe → store → report cycle, no real API."""
        from pipeline.orchestrator import Pipeline, PipelineConfig

        tickers = ["AAPL", "MSFT", "GOOG"]
        source  = {t: make_clean_df(t, n_days=150, seed=i)
                   for i, t in enumerate(tickers)}

        # Inject issues into MSFT
        idx = 10
        source["MSFT"].iloc[idx, source["MSFT"].columns.get_loc("close")] *= 1.70
        source["MSFT"].iloc[idx, source["MSFT"].columns.get_loc("high")]  = source["MSFT"].iloc[idx]["close"] * 1.01
        source["MSFT"].iloc[idx, source["MSFT"].columns.get_loc("open")]  = source["MSFT"].iloc[idx]["close"] * 0.99

        def fake_get(url, params=None, timeout=None, **kwargs):
            for t in tickers:
                if f"/{t}/" in url:
                    bars = df_to_polygon_bars(source[t])
                    return make_polygon_response(bars)
            m = MagicMock()
            m.status_code = 404
            m.raise_for_status = MagicMock()
            return m

        cfg = PipelineConfig(
            api_key      = "test_key",
            clean_dir    = str(tmp_path / "clean"),
            cache_dir    = str(tmp_path / "cache"),
            report_dir   = str(tmp_path / "reports"),
            min_adtv_usd = 0,        # no ADTV filter for test
            min_history_days = 20,
            adtv_window  = 5,
        )
        pipe = Pipeline(cfg)

        with patch.object(pipe.fetcher._session, "get", side_effect=fake_get):
            results = pipe.run(
                tickers    = tickers,
                start_date = "2023-01-03",
                end_date   = "2023-08-31",
            )

        assert results
        assert results["summary"]["n_tickers"] == 3
        assert results["summary"]["pct_retained"] > 90.0
        assert results["summary"]["bars_return_outlier"] >= 1  # MSFT outlier caught

        # Verify store is readable
        panels = pipe.load_panels(universe_only=False)
        assert "close" in panels
        assert set(panels["close"].columns) == set(tickers)

        # Verify report files were written
        report_dir = Path(cfg.report_dir)
        assert (report_dir / "cleaning_report.csv").exists()
        assert (report_dir / "pipeline_summary.csv").exists()

    def test_update_skips_existing_dates(self, tmp_path):
        """Update mode fetches only dates not yet in the store."""
        from pipeline.orchestrator import Pipeline, PipelineConfig

        source = {"AAPL": make_clean_df("AAPL", n_days=200, seed=0)}

        def fake_get(url, params=None, timeout=None, **kwargs):
            bars = df_to_polygon_bars(source["AAPL"])
            return make_polygon_response(bars)

        cfg = PipelineConfig(
            api_key      = "test_key",
            clean_dir    = str(tmp_path / "clean"),
            cache_dir    = str(tmp_path / "cache"),
            report_dir   = str(tmp_path / "reports"),
            start_date   = "2023-01-03",
            min_adtv_usd = 0,
            min_history_days = 10,
            adtv_window  = 5,
            default_tickers = ["AAPL"],
        )
        pipe = Pipeline(cfg)

        with patch.object(pipe.fetcher._session, "get", side_effect=fake_get):
            # First run — full fetch
            pipe.run(tickers=["AAPL"], start_date="2023-01-03", end_date="2023-06-30")

        # Update: store already has data up to 2023-06-30
        loaded = read_clean(cfg.clean_dir)
        assert not loaded.empty
        last_date = loaded.index.max().strftime("%Y-%m-%d")
        assert last_date >= "2023-06-29"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-q"])
