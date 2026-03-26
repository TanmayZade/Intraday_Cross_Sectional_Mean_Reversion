"""
pipeline/fetcher.py
===================
Polygon.io REST API fetcher for OHLCV aggregate bars.

Handles every free-tier constraint:
  ┌─────────────────────────────────────────────────────────┐
  │  Free tier limits (as of 2025)                         │
  │  • 5 requests / minute  (hard cap → 429 on breach)     │
  │  • 50,000 bars max per request                         │
  │  • Minute data (1, 5, 15, 30, 60 min multiplier)       │
  │  • No websocket / real-time                            │
  │  • Prices split-adjusted by default                    │
  │  • Timestamps: UTC milliseconds                        │
  └─────────────────────────────────────────────────────────┘

CURRENT CONFIGURATION: 1-minute bars
  - Fetches 1-min OHLCV directly from Polygon API
  - ~390 bars/trading day (vs 96 bars for 5-min, 26 bars for 15-min)
  - Fine granularity for capturing mean reversion at minute scale
  - Use "multiplier: 1, timespan: minute" in config.yaml
  - API quota: ~220 calls/month for full universe (44 tickers × 5 date ranges)

Key design decisions:
  - Token-bucket rate limiter (thread-safe, not just sleep-based)
  - Automatic date chunking: each request ≤ 50k bars (~128 trading days for 1-min)
  - Disk cache: skip API if data already fetched today
  - Retry with exponential back-off on 429 / 5xx
  - Returns clean DataFrame with proper tz-aware DatetimeIndex
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# ── Polygon API response field map ───────────────────────────────────────────
_FIELD_MAP = {
    "t": "timestamp_ms",   # Unix ms UTC
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",          # volume-weighted avg price (free on daily)
    "n": "n_trades",       # number of trades
}


# ─────────────────────────────────────────────────────────────────────────────
# Token-bucket rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class _RateLimiter:
    """
    Thread-safe token bucket: allows at most `rate` calls per `period` seconds.
    """
    def __init__(self, rate: int = 5, period: float = 60.0):
        self._rate   = rate
        self._period = period
        self._tokens = rate
        self._last   = time.monotonic()
        self._lock   = Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            # Refill tokens proportionally
            self._tokens = min(
                self._rate,
                self._tokens + elapsed * (self._rate / self._period)
            )
            self._last = now
            if self._tokens >= 1:
                self._tokens -= 1
                return
            # Need to wait
            wait = (1 - self._tokens) / (self._rate / self._period)
        log.debug("Rate limit: sleeping %.2fs", wait)
        time.sleep(wait)
        with self._lock:
            self._tokens = max(0, self._tokens - 1)
            self._last = time.monotonic()


# ─────────────────────────────────────────────────────────────────────────────
# Disk response cache
# ─────────────────────────────────────────────────────────────────────────────

class _DiskCache:
    """
    Simple JSON cache keyed by a hash of (ticker, start, end, params).
    Entries expire after `ttl_hours` hours (default 24 — data doesn't change).
    """
    def __init__(self, cache_dir: str | Path, ttl_hours: float = 24.0):
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_hours * 3600

    def _key(self, **kwargs) -> str:
        raw = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, **kwargs) -> Optional[list[dict]]:
        path = self._dir / (self._key(**kwargs) + ".json")
        if not path.exists():
            return None
        age = time.time() - path.stat().st_mtime
        if age > self._ttl:
            path.unlink(missing_ok=True)
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def set(self, data: list[dict], **kwargs) -> None:
        path = self._dir / (self._key(**kwargs) + ".json")
        path.write_text(json.dumps(data))


# ─────────────────────────────────────────────────────────────────────────────
# Main Fetcher class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FetcherConfig:
    api_key:           str
    base_url:          str   = "https://api.polygon.io"
    timespan:          str   = "day"
    multiplier:        int   = 1
    adjusted:          bool  = True
    limit:             int   = 50_000
    requests_per_min:  int   = 5
    retry_attempts:    int   = 3
    retry_backoff_s:   float = 13.0
    timeout_s:         int   = 30
    cache_dir:         str   = "data/cache/"


class PolygonFetcher:
    """
    Fetches split-adjusted OHLCV bars from Polygon.io for a list of tickers
    over an arbitrary date range, respecting all free-tier constraints.

    Usage
    -----
        fetcher = PolygonFetcher.from_config("configs/config.yaml")
        df = fetcher.fetch("AAPL", "2020-01-01", "2024-12-31")
        panel = fetcher.fetch_universe(["AAPL","MSFT","GOOG"], "2020-01-01", "2024-12-31")
    """

    def __init__(self, cfg: FetcherConfig):
        self.cfg     = cfg
        self._rl     = _RateLimiter(rate=cfg.requests_per_min, period=60.0)
        self._cache  = _DiskCache(cfg.cache_dir)
        self._session = self._build_session()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "PolygonFetcher":
        import yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f)
        poly = raw["polygon"]
        api_key = os.environ.get("POLYGON_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "Set POLYGON_API_KEY environment variable.\n"
                "  export POLYGON_API_KEY=your_key_here"
            )
        return cls(FetcherConfig(
            api_key          = api_key,
            base_url         = poly.get("base_url", "https://api.polygon.io"),
            timespan         = poly.get("timespan", "day"),
            multiplier       = poly.get("multiplier", 1),
            adjusted         = poly.get("adjusted", True),
            limit            = poly.get("limit", 50_000),
            requests_per_min = poly.get("requests_per_min", 5),
            retry_attempts   = poly.get("retry_attempts", 3),
            retry_backoff_s  = poly.get("retry_backoff_s", 13.0),
            timeout_s        = poly.get("timeout_s", 30),
            cache_dir        = raw.get("storage", {}).get("cache_dir", "data/cache/"),
        ))

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker: str,
        start_date: str,          # "YYYY-MM-DD"
        end_date: str,            # "YYYY-MM-DD"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a single ticker.

        Automatically chunks the date range if it would exceed 50k bars.
        Returns DataFrame with:
            index   : date (daily) or tz-aware timestamp (intraday)
            columns : open, high, low, close, volume, vwap, n_trades, ticker
        """
        ticker = ticker.upper()
        chunks = self._date_chunks(start_date, end_date)
        frames = []

        for chunk_start, chunk_end in chunks:
            bars = self._fetch_chunk(ticker, chunk_start, chunk_end)
            if bars:
                frames.append(self._to_dataframe(bars, ticker))

        if not frames:
            log.warning("%s: no data returned for %s → %s", ticker, start_date, end_date)
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        log.info("%s: fetched %d bars  (%s → %s)",
                 ticker, len(df), start_date, end_date)
        return df

    def fetch_universe(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch all tickers in the universe.

        Returns a flat DataFrame with a 'ticker' column.
        Rate-limited: at 5 req/min with chunking this will take time for
        large universes — progress is logged every 10 tickers.
        """
        all_frames = []
        n = len(tickers)
        t0 = time.monotonic()

        for i, ticker in enumerate(tickers, 1):
            try:
                df = self.fetch(ticker, start_date, end_date)
                if not df.empty:
                    df["ticker"] = ticker
                    all_frames.append(df)
            except Exception as e:
                log.error("Failed to fetch %s: %s", ticker, e)

            if show_progress and i % 10 == 0:
                elapsed = time.monotonic() - t0
                eta = elapsed / i * (n - i)
                log.info("Progress: %d/%d tickers  (%.0fs elapsed, ~%.0fs remaining)",
                         i, n, elapsed, eta)

        if not all_frames:
            log.error("No data fetched for any ticker")
            return pd.DataFrame()

        combined = pd.concat(all_frames).sort_index()
        log.info("Universe fetch complete: %d tickers, %d total bars",
                 combined["ticker"].nunique(), len(combined))
        return combined

    # ── Private: date chunking ────────────────────────────────────────────────

    def _date_chunks(
        self,
        start_date: str,
        end_date: str,
    ) -> list[tuple[str, str]]:
        """
        Split date range into chunks that stay within the 50k bar limit.

        Daily bars: 50k / 252 trading days ≈ 198 years → never needs chunking.
        Minute bars: 50k / (390 bars/day × 252 days/yr) ≈ 0.51 years → chunk monthly.
        Hour bars: 50k / (7 bars/day × 252) ≈ 28 years → rarely needs chunking.
        """
        # Polygon returns ALL bars including pre/post market extended hours.
        # A "390 bar" regular session day can have 480+ bars with extended hours.
        # Use a conservative estimate of 500 bars/day to avoid hitting the limit.
        # Formula: chunk_days = limit * 0.80 / bars_per_day_conservative
        BARS_PER_DAY = {
            "minute": 500,    # 390 regular + ~110 extended hours (4am-8pm)
            "hour":   16,     # full extended hours day
            "day":    1,
            "week":   1,
            "month":  1,
        }
        bars_per_day = BARS_PER_DAY.get(self.cfg.timespan, 1)
        # Use 80% of limit as safe ceiling
        safe_limit   = int(self.cfg.limit * 0.80)
        chunk_days   = max(1, int(safe_limit / bars_per_day))

        start = date.fromisoformat(start_date)
        end   = date.fromisoformat(end_date)
        chunks = []
        cur = start
        while cur <= end:
            chunk_end = min(cur + timedelta(days=chunk_days - 1), end)
            chunks.append((cur.isoformat(), chunk_end.isoformat()))
            cur = chunk_end + timedelta(days=1)

        log.debug("Date range %s→%s split into %d chunks", start_date, end_date, len(chunks))
        return chunks

    # ── Private: single chunk fetch ───────────────────────────────────────────

    def _fetch_chunk(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> list[dict]:
        """Fetch one date chunk, using cache if available."""
        cache_key = dict(ticker=ticker, start=start, end=end,
                         timespan=self.cfg.timespan,
                         multiplier=self.cfg.multiplier,
                         adjusted=self.cfg.adjusted)
        cached = self._cache.get(**cache_key)
        if cached is not None:
            log.debug("%s [%s→%s]: cache hit (%d bars)", ticker, start, end, len(cached))
            return cached

        url = (
            f"{self.cfg.base_url}/v2/aggs/ticker/{ticker}"
            f"/range/{self.cfg.multiplier}/{self.cfg.timespan}/{start}/{end}"
        )
        params = {
            "adjusted": str(self.cfg.adjusted).lower(),
            "sort":     "asc",
            "limit":    self.cfg.limit,
            "apiKey":   self.cfg.api_key,
        }

        for attempt in range(1, self.cfg.retry_attempts + 1):
            self._rl.acquire()
            try:
                resp = self._session.get(url, params=params,
                                         timeout=self.cfg.timeout_s)

                if resp.status_code == 429:
                    wait = self.cfg.retry_backoff_s * attempt
                    log.warning("429 rate limit hit on %s (attempt %d) — waiting %.0fs",
                                ticker, attempt, wait)
                    time.sleep(wait)
                    continue

                if resp.status_code == 403:
                    # On free tier, 403 is often transient — Polygon throttles
                    # the key temporarily when bursting too fast.
                    # Retry with backoff before giving up.
                    if attempt < self.cfg.retry_attempts:
                        wait = self.cfg.retry_backoff_s * attempt
                        log.warning(
                            "%s: 403 on chunk [%s->%s] (attempt %d/%d) — "
                            "waiting %.0fs then retrying",
                            ticker, start, end, attempt, self.cfg.retry_attempts, wait
                        )
                        time.sleep(wait)
                        continue
                    else:
                        log.error(
                            "%s: 403 Forbidden after %d attempts — "
                            "check API key is valid and not blocked.",
                            ticker, self.cfg.retry_attempts
                        )
                        return []

                if resp.status_code == 404:
                    log.warning("%s: 404 — ticker not found or no data in range", ticker)
                    return []

                resp.raise_for_status()
                data = resp.json()

                if data.get("status") in ("ERROR", "NOT_FOUND"):
                    log.warning("%s: API error: %s", ticker, data.get("error", "unknown"))
                    return []

                results = data.get("results", [])
                if not results:
                    log.debug("%s [%s→%s]: empty results", ticker, start, end)
                    # Cache empty results too (ticker has no data in this range)
                    self._cache.set([], **cache_key)
                    return []

                # Warn if results were truncated at limit
                if len(results) == self.cfg.limit:
                    log.warning(
                        "%s [%s→%s]: result count hit limit (%d) — "
                        "consider narrowing date range",
                        ticker, start, end, self.cfg.limit
                    )

                self._cache.set(results, **cache_key)
                return results

            except requests.exceptions.Timeout:
                log.warning("%s: request timeout (attempt %d/%d)",
                            ticker, attempt, self.cfg.retry_attempts)
                time.sleep(self.cfg.retry_backoff_s)

            except requests.exceptions.RequestException as e:
                log.error("%s: request error (attempt %d/%d): %s",
                          ticker, attempt, self.cfg.retry_attempts, e)
                time.sleep(self.cfg.retry_backoff_s)

        log.error("%s: all %d fetch attempts failed for [%s→%s]",
                  ticker, self.cfg.retry_attempts, start, end)
        return []

    # ── Private: raw JSON → DataFrame ────────────────────────────────────────

    def _to_dataframe(self, bars: list[dict], ticker: str) -> pd.DataFrame:
        """
        Convert Polygon raw bar dicts to a clean DataFrame.

        Polygon timestamps are Unix milliseconds in UTC.
        For daily bars: convert to US/Eastern date (market closes ET).
        For intraday bars: convert to tz-aware ET DatetimeIndex.
        """
        df = pd.DataFrame(bars)

        # Rename raw Polygon fields to canonical names
        df = df.rename(columns={k: v for k, v in _FIELD_MAP.items() if k in df.columns})

        # Drop any extra fields we don't need
        keep = ["timestamp_ms", "open", "high", "low", "close",
                "volume", "vwap", "n_trades"]
        df = df[[c for c in keep if c in df.columns]]

        # ── Timestamp handling ────────────────────────────────────────────────
        # Polygon gives UTC milliseconds. The 't' value for a daily bar is the
        # Unix ms of midnight UTC at the start of that trading day.
        ts_utc = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

        if self.cfg.timespan == "day":
            # For daily bars: just use the date in ET (market date)
            ts_et = ts_utc.dt.tz_convert("America/New_York")
            df.index = ts_et.dt.normalize()          # midnight ET (date only)
            df.index = pd.DatetimeIndex(
                [t.date() for t in ts_et], name="date"
            )
        else:
            # For intraday bars: keep full tz-aware timestamp in ET
            df.index = ts_utc.dt.tz_convert("America/New_York")
            df.index.name = "timestamp"

        df = df.drop(columns=["timestamp_ms"])

        # ── Dtype enforcement ─────────────────────────────────────────────────
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ── Private: session with connection pooling ──────────────────────────────

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        # Retry on connection errors only (not 429 — we handle that manually)
        retry = Retry(
            total=0,                   # we handle retries ourselves
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=1,        # single connection — sequential fetch
            pool_maxsize=1,
        )
        session.mount("https://", adapter)
        return session

    # ── Utility: available tickers from Polygon reference ────────────────────

    def fetch_ticker_list(
        self,
        market: str = "stocks",
        exchange: str = "XNAS,XNYS",  # NASDAQ + NYSE
        active: bool = True,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetch all active tickers from Polygon reference endpoint.
        Free tier: this endpoint IS available.
        Returns DataFrame with columns: ticker, name, market, exchange, type.
        """
        url = f"{self.cfg.base_url}/v3/reference/tickers"
        params = {
            "market":   market,
            "exchange": exchange,
            "active":   str(active).lower(),
            "limit":    limit,
            "apiKey":   self.cfg.api_key,
        }
        results = []
        while url:
            self._rl.acquire()
            resp = self._session.get(url, params=params, timeout=self.cfg.timeout_s)
            resp.raise_for_status()
            data = resp.json()
            results.extend(data.get("results", []))
            # Pagination: Polygon returns next_url
            url = data.get("next_url")
            if url:
                params = {"apiKey": self.cfg.api_key}  # key already in next_url? no
                url = url + f"&apiKey={self.cfg.api_key}"

        log.info("Fetched %d tickers from Polygon reference", len(results))
        return pd.DataFrame(results)[["ticker", "name", "market", "primary_exchange", "type"]] \
                 if results else pd.DataFrame()

    def fetch_splits(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        Fetch corporate split events for one ticker.
        Free tier endpoint.
        Returns DataFrame: execution_date, split_from, split_to, ratio.
        """
        url = f"{self.cfg.base_url}/v3/reference/splits"
        params = {
            "ticker":                  ticker,
            "execution_date.gte":      start_date,
            "execution_date.lte":      end_date,
            "limit":                   1000,
            "apiKey":                  self.cfg.api_key,
        }
        self._rl.acquire()
        resp = self._session.get(url, params=params, timeout=self.cfg.timeout_s)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        df["ratio"] = df["split_to"] / df["split_from"]
        return df[["execution_date", "split_from", "split_to", "ratio"]]