"""
nse_pipeline/universe.py
========================
Dynamic universe construction for NSE intraday mean reversion.

Selects 50 volatile + 50 non-volatile stocks from a seed pool
based on trailing ATR% (Average True Range as % of price).

Algorithm:
  1. Start with a seed pool (NIFTY 200 / F&O eligible stocks)
  2. Fetch 1 year of daily OHLCV for all
  3. Apply filters: ADTV ≥ ₹50L, Price ≥ ₹50, History ≥ 60 days
  4. Compute ATR% = ATR(20) / Close for each stock
  5. Rank by ATR%
  6. Volatile 50 = Top-50 by ATR% (highest volatility)
  7. Non-Volatile 50 = Bottom-50 by ATR% (lowest volatility)

Usage
-----
    from nse_pipeline.universe import UniverseBuilder
    
    builder = UniverseBuilder()
    volatile, nonvolatile = builder.select()
    all_tickers = volatile + nonvolatile  # 100 stocks
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seed pool: Large liquid NSE stocks (NIFTY 200 representative set)
# These are screened dynamically — the final 100 depend on ATR% ranking.
# ─────────────────────────────────────────────────────────────────────────────

SEED_POOL = [
    # ── Large-Cap / NIFTY 50 ──────────────────────────────────
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "KOTAKBANK", "BHARTIARTL", "ITC", "WIPRO", "HCLTECH",
    "AXISBANK", "LT", "SUNPHARMA", "BAJFINANCE", "TITAN",
    "ASIANPAINT", "MARUTI", "ULTRACEMCO", "NESTLEIND", "HINDUNILVR",
    "DRREDDY", "CIPLA", "DIVISLAB", "BAJAJ-AUTO", "HEROMOTOCO",
    "TECHM", "POWERGRID", "NTPC", "COALINDIA", "ONGC",
    "BPCL", "GRASIM", "BEL", "EICHERMOT", "SBILIFE",
    "BAJAJFINSV", "HDFCLIFE", "APOLLOHOSP", "TATACONSUM", "M&M",
    "SHRIRAMFIN", "TRENT", "SBIN", "TMCV", "TATASTEEL",
    "JSWSTEEL", "HINDALCO", "ADANIENT", "ADANIPORTS", "INDUSINDBK",
    
    # ── Mid-Cap / NIFTY Next 50 ──────────────────────────────
    "PIDILITIND", "DABUR", "GODREJCP", "MARICO", "COLPAL",
    "BRITANNIA", "HAVELLS", "VOLTAS", "PAGEIND", "MUTHOOTFIN",
    "DALBHARAT", "IDFCFIRSTB", "FEDERALBNK", "BANDHANBNK", "PNB",
    "CANBK", "BANKBARODA", "UNIONBANK", "RBLBANK", "AUBANK",
    "PFC", "RECLTD", "IRFC", "IRCTC", "RVNL",
    "NHPC", "SJVN", "TATAPOWER", "ADANIGREEN", "ADANIPOWER",
    "SUZLON", "SAIL", "BHEL", "NBCC", "NMDC",
    "VEDL", "JINDALSTEL", "NATIONALUM", "POLYCAB", "DIXON",
    "KAYNES", "ETERNAL", "NYKAA", "DELHIVERY", "POLICYBZR",
    
    # ── Additional Mid-Cap (IT, Pharma, Auto, etc.) ──────────
    "MPHASIS", "COFORGE", "PERSISTENT", "LTTS", "LTIM",
    "LUPIN", "BIOCON", "IPCALAB", "AUROPHARMA", "TORNTPHARM",
    "BALKRISIND", "MRF", "TVSMOTOR", "ESCORTS", "ASHOKLEY",
    "DLF", "OBEROIRLTY", "GODREJPROP", "PRESTIGE", "BRIGADE",
    "CUMMINSIND", "SIEMENS", "ABB", "HONAUT", "CROMPTON",
    "MCX", "BSE", "ANGELONE", "CDSL", "KPITTECH",
    "DEEPAKNTR", "ATUL", "PIIND", "CLEAN", "SOLARINDS",
    "OFSS", "MFSL", "MAXHEALTH", "YESBANK", "IDEA",
]


class UniverseBuilder:
    """
    Dynamically select 50 volatile + 50 non-volatile NSE stocks.
    
    Parameters
    ----------
    seed_pool : list[str]
        Starting pool of NSE tickers to screen
    volatile_count : int
        Number of volatile stocks to select (default 50)
    nonvolatile_count : int
        Number of non-volatile stocks to select (default 50)
    atr_window : int
        ATR lookback period in days (default 20)
    min_adtv_inr : float
        Minimum average daily turnover in INR (default ₹50L)
    min_price : float
        Minimum stock price (default ₹50)
    min_history_days : int
        Minimum trading days of history required (default 60)
    """
    
    def __init__(
        self,
        seed_pool: Optional[list[str]] = None,
        volatile_count: int = 50,
        nonvolatile_count: int = 50,
        atr_window: int = 20,
        min_adtv_inr: float = 5_000_000,
        min_price: float = 50.0,
        min_history_days: int = 60,
    ):
        self.seed_pool = seed_pool or SEED_POOL
        self.volatile_count = volatile_count
        self.nonvolatile_count = nonvolatile_count
        self.atr_window = atr_window
        self.min_adtv = min_adtv_inr
        self.min_price = min_price
        self.min_history = min_history_days
    
    def select(
        self,
        daily_panels: dict[str, pd.DataFrame],
    ) -> tuple[list[str], list[str]]:
        """
        Select volatile and non-volatile stocks from daily OHLCV data.
        
        Parameters
        ----------
        daily_panels : dict
            Daily OHLCV panels: {"open": df, "high": df, ...}
            Each DataFrame: [date × ticker]
        
        Returns
        -------
        volatile : list[str] — Top N by ATR% (highest volatility)
        nonvolatile : list[str] — Bottom N by ATR% (lowest volatility)
        """
        close = daily_panels.get("close")
        high = daily_panels.get("high")
        low = daily_panels.get("low")
        volume = daily_panels.get("volume")
        
        if close is None or close.empty:
            log.error("No close data for universe screening")
            return [], []
        
        log.info("Universe screening: %d candidate tickers", len(close.columns))
        
        # ── Filter 1: Minimum history ────────────────────────────────────
        obs_count = close.notna().sum()
        has_history = obs_count[obs_count >= self.min_history].index.tolist()
        log.info("  After history filter (≥%d days): %d tickers",
                 self.min_history, len(has_history))
        
        # ── Filter 2: Minimum price ──────────────────────────────────────
        last_price = close.iloc[-1]
        above_price = last_price[last_price >= self.min_price].index.tolist()
        valid = list(set(has_history) & set(above_price))
        log.info("  After price filter (≥₹%.0f): %d tickers",
                 self.min_price, len(valid))
        
        # ── Filter 3: Minimum ADTV ──────────────────────────────────────
        if volume is not None:
            dollar_vol = close * volume
            adtv = dollar_vol.rolling(20, min_periods=10).mean().iloc[-1]
            above_adtv = adtv[adtv >= self.min_adtv].index.tolist()
            valid = list(set(valid) & set(above_adtv))
            log.info("  After ADTV filter (≥₹%.0fL): %d tickers",
                     self.min_adtv / 100_000, len(valid))
        
        if len(valid) < self.volatile_count + self.nonvolatile_count:
            log.warning(
                "  Only %d tickers pass filters (need %d). Relaxing criteria.",
                len(valid), self.volatile_count + self.nonvolatile_count
            )
        
        # ── Compute ATR% for ranking ─────────────────────────────────────
        atr_pct = self._compute_atr_pct(
            high[valid] if high is not None else None,
            low[valid] if low is not None else None,
            close[valid],
        )
        
        # Sort by ATR% descending (most volatile first)
        atr_ranked = atr_pct.sort_values(ascending=False)
        
        # ── Select top N volatile + bottom N non-volatile ────────────────
        n_vol = min(self.volatile_count, len(atr_ranked) // 2)
        n_nonvol = min(self.nonvolatile_count, len(atr_ranked) // 2)
        
        volatile = atr_ranked.head(n_vol).index.tolist()
        nonvolatile = atr_ranked.tail(n_nonvol).index.tolist()
        
        log.info("  Volatile 50 (highest ATR%%): %s ... (ATR%% range: %.2f%% - %.2f%%)",
                 volatile[:5], atr_ranked.iloc[0] * 100, atr_ranked.iloc[n_vol - 1] * 100)
        log.info("  Non-volatile 50 (lowest ATR%%): %s ... (ATR%% range: %.2f%% - %.2f%%)",
                 nonvolatile[:5], atr_ranked.iloc[-n_nonvol] * 100, atr_ranked.iloc[-1] * 100)
        
        return volatile, nonvolatile
    
    def _compute_atr_pct(
        self,
        high: Optional[pd.DataFrame],
        low: Optional[pd.DataFrame],
        close: pd.DataFrame,
    ) -> pd.Series:
        """
        Compute ATR as percentage of close (trailing average).
        
        ATR% = ATR(window) / Close
        Higher ATR% = more volatile stock.
        
        Returns pd.Series indexed by ticker, values = ATR%.
        """
        if high is not None and low is not None:
            # True Range = max(H-L, |H-Cprev|, |L-Cprev|)
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
        else:
            # Fallback: use absolute return as volatility proxy
            tr = close.pct_change(1).abs() * close
        
        atr = tr.rolling(self.atr_window, min_periods=10).mean()
        atr_pct = (atr / close.replace(0, np.nan)).iloc[-1]
        
        return atr_pct.dropna().sort_values(ascending=False)
    
    def get_full_universe(
        self,
        daily_panels: dict[str, pd.DataFrame],
    ) -> tuple[list[str], list[str], pd.Series]:
        """
        Get universe with ATR% scores for analysis.
        
        Returns
        -------
        volatile, nonvolatile, atr_scores
        """
        volatile, nonvolatile = self.select(daily_panels)
        
        close = daily_panels.get("close")
        high = daily_panels.get("high")
        low = daily_panels.get("low")
        
        valid = volatile + nonvolatile
        atr_scores = self._compute_atr_pct(
            high[valid] if high is not None else None,
            low[valid] if low is not None else None,
            close[valid],
        )
        
        return volatile, nonvolatile, atr_scores
