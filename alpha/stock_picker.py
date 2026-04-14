"""
alpha/stock_picker.py
=====================
Concentrated Stock Picker for Single-Day Max Profit (NSE)

Instead of spreading alpha across 100 stocks, this module picks the
TOP 8-10 stocks with the highest conviction signal for a long-only
intraday trade.

Architecture
------------
  1. Score all tickers using DailySignalEngine composite score
  2. Filter: skip stocks with insufficient data or low liquidity
  3. Rank: pick top N by composite z-score
  4. Allocate: equal-weight capital among selected stocks

Usage
-----
    from alpha.stock_picker import StockPicker
    
    picker = StockPicker(panels, nifty_close, capital=1_000_000)
    picks = picker.pick(target_date)
    # picks = [
    #   {"ticker": "RELIANCE", "score": 2.8, "allocation": 100000, "shares": 76},
    #   {"ticker": "TATAPOWER", "score": 2.3, "allocation": 100000, "shares": 258},
    #   ...
    # ]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from features.daily_signals import DailySignalEngine

log = logging.getLogger(__name__)


class StockPicker:
    """
    Selects top N stocks for long-only intraday trades.
    
    Parameters
    ----------
    panels : dict
        OHLCV panels
    nifty_close : Series
        NIFTY 50 close prices
    capital : float
        Total trading capital (default ₹10L)
    n_picks : int
        Number of stocks to pick (default 10)
    min_score : float
        Minimum z-score to qualify (default 0.5)
    min_price : float
        Minimum stock price to trade (default ₹50)
    min_avg_volume : float
        Minimum 20-day avg daily volume (default 500,000 shares)
    preopen_weight : float
        Weight for pre-open signals in composite (default 0.6)
    confirm_weight : float
        Weight for confirmation signals in composite (default 0.4)
    """
    
    def __init__(
        self,
        panels: dict,
        nifty_close: pd.Series = None,
        capital: float = 1_000_000,
        n_picks: int = 10,
        min_score: float = 0.5,
        min_price: float = 50.0,
        min_avg_volume: float = 500_000,
        preopen_weight: float = 0.6,
        confirm_weight: float = 0.4,
    ):
        self.panels = panels
        self.nifty_close = nifty_close
        self.capital = capital
        self.n_picks = n_picks
        self.min_score = min_score
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.preopen_weight = preopen_weight
        self.confirm_weight = confirm_weight
        
        self.signal_engine = DailySignalEngine(panels, nifty_close)
        
        self._dates = panels["close"].index.normalize()
        self._unique_dates = sorted(self._dates.unique())
    
    def pick(self, target_date: pd.Timestamp) -> list[dict]:
        """
        Pick top N stocks for a given trading day.
        
        Parameters
        ----------
        target_date : the trading date
        
        Returns
        -------
        list of dicts, each with:
            ticker: str
            score: float (composite z-score)
            allocation: float (capital allocated in INR)
            entry_price: float (opening price at 9:30)
            shares: int (number of shares to buy)
            preopen_signals: dict (individual pre-open feature scores)
            confirm_signals: dict (individual confirmation scores)
        """
        target_date = pd.Timestamp(target_date).normalize()
        
        log.info("=" * 60)
        log.info("  Stock Picker: %s", target_date.date())
        log.info("=" * 60)
        
        # 1. Compute composite scores
        scores = self.signal_engine.composite_score(
            target_date,
            self.preopen_weight,
            self.confirm_weight,
        )
        
        if scores.empty:
            log.warning("  No scores computed for %s", target_date.date())
            return []
        
        # 2. Get individual signal DataFrames for reporting
        preopen_df = self.signal_engine.compute_preopen_signals(target_date)
        confirm_df = self.signal_engine.compute_confirmation(target_date)
        
        # 3. Apply filters
        filtered = self._apply_filters(scores, target_date)
        
        if filtered.empty:
            log.warning("  No stocks passed filters for %s", target_date.date())
            return []
        
        # 4. Pick top N above minimum score threshold
        qualified = filtered[filtered >= self.min_score]
        
        if qualified.empty:
            log.info("  No stocks above min_score=%.1f, taking top %d anyway",
                    self.min_score, self.n_picks)
            qualified = filtered
        
        top_n = qualified.head(self.n_picks)
        
        # 5. Allocate capital equally
        n_stocks = len(top_n)
        per_stock_capital = self.capital / n_stocks
        
        # 6. Get entry prices (bar 4 open = 9:30 AM, after confirmation window)
        day_mask = self._dates == target_date
        day_open = self.panels["open"][day_mask]
        
        if len(day_open) < 4:
            entry_prices = day_open.iloc[0]  # Fallback to first bar
            entry_bar = 0
        else:
            entry_prices = day_open.iloc[3]  # 9:30 AM bar open
            entry_bar = 3
        
        # 7. Build picks list
        picks = []
        for ticker in top_n.index:
            price = entry_prices.get(ticker, np.nan)
            if pd.isna(price) or price <= 0:
                continue
            
            shares = int(per_stock_capital / price)
            if shares <= 0:
                continue
            
            pick = {
                "ticker": ticker,
                "score": float(top_n[ticker]),
                "allocation": per_stock_capital,
                "entry_price": float(price),
                "shares": shares,
                "entry_bar_idx": entry_bar,
                "preopen_signals": (
                    preopen_df.loc[ticker].to_dict()
                    if ticker in preopen_df.index else {}
                ),
                "confirm_signals": (
                    confirm_df.loc[ticker].to_dict()
                    if not confirm_df.empty and ticker in confirm_df.index else {}
                ),
            }
            picks.append(pick)
        
        # Log picks
        log.info("  Selected %d stocks (capital: ₹%.0f each):", len(picks),
                per_stock_capital)
        for p in picks:
            log.info("    %-12s  score=%+.2fσ  entry=₹%.1f  shares=%d",
                    p["ticker"], p["score"], p["entry_price"], p["shares"])
        
        return picks
    
    def _apply_filters(
        self,
        scores: pd.Series,
        target_date: pd.Timestamp,
    ) -> pd.Series:
        """
        Filter out stocks that shouldn't be traded.
        
        Removes:
          - Stocks below min_price
          - Stocks with insufficient volume
          - Stocks with missing data
        """
        dates = self._unique_dates
        tgt_idx = dates.index(target_date)
        
        # Price filter
        day_mask = self._dates == target_date
        day_close = self.panels["close"][day_mask]
        if len(day_close) > 0:
            last_price = day_close.iloc[0]
        else:
            prev_date = dates[tgt_idx - 1]
            prev_mask = self._dates == prev_date
            last_price = self.panels["close"][prev_mask].iloc[-1]
        
        price_ok = last_price >= self.min_price
        
        # Volume filter (20-day avg)
        lookback_dates = dates[max(0, tgt_idx - 20):tgt_idx]
        daily_vol = self.panels["volume"].groupby(self._dates).sum()
        if len(lookback_dates) > 0 and len(daily_vol) > 0:
            avg_vol = daily_vol.reindex(lookback_dates).mean()
            vol_ok = avg_vol >= self.min_avg_volume
        else:
            vol_ok = pd.Series(True, index=scores.index)
        
        # Data completeness (must have data for today)
        has_data = scores.notna()
        
        # Combine filters
        mask = price_ok.reindex(scores.index).fillna(False) & \
               vol_ok.reindex(scores.index).fillna(False) & \
               has_data
        
        n_filtered = (~mask).sum()
        if n_filtered > 0:
            log.info("  Filtered out %d stocks (price/volume/data)", n_filtered)
        
        return scores[mask].sort_values(ascending=False)
    
    def get_trading_dates(self) -> list:
        """Return all dates available for backtesting."""
        # Need at least lookback_days of history
        lookback = self.signal_engine.lookback
        if len(self._unique_dates) <= lookback:
            return []
        return self._unique_dates[lookback:]
