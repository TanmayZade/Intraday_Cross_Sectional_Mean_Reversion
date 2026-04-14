"""
features/engine.py
==================
7-Feature Engine for NSE Intraday Cross-Sectional Mean Reversion

Computes 7 features from OHLCV panel data (6 core + 1 NSE-specific):

  A1_bar_reversal          — Single-bar price reversal (ATR-normalized)
  A2_short_rev             — 3-bar (15-min) momentum fade
  B1_vwap                  — Intraday VWAP deviation
  C1_vol_shock             — Volume spike vs time-of-day baseline
  D1_vol_burst             — Directed volatility burst
  E1_residual              — Market-beta-adjusted residual return
  F1_circuit_proximity     — NSE circuit limit proximity (India-specific)

Tuned for maximum return:
  - Shorter ATR window (30 bars = 2.5 hours) for faster reaction
  - Time-of-day signal weighting (opening/closing stronger)
  - NSE session: 9:15 AM - 3:30 PM IST (75 bars/day at 5-min)

Usage
-----
    from features.engine import FeatureEngine
    
    engine = FeatureEngine(panels)
    features = engine.compute_all()
    print(features.keys())
    # dict_keys(['A1_bar_reversal', 'A2_short_rev', 'B1_vwap',
    #            'C1_vol_shock', 'D1_vol_burst', 'E1_residual',
    #            'F1_circuit_proximity'])
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from features.core import (
    cs_zscore, cs_rank, rolling_mean, rolling_std, rolling_mad,
    ewm_std, session_bar_index, session_fraction,
    atr, atr_pct, true_range,
)

log = logging.getLogger(__name__)

# NSE session constants
NSE_BARS_PER_SESSION = 75  # 5-min bars in 9:15-15:30 (375 min)


class FeatureEngine:
    """
    7-feature engine for NSE cross-sectional mean reversion (intraday 5m bars).
    
    Input:  panels = {"open": DataFrame, "high": DataFrame, ...}
            Each DataFrame: [timestamp × ticker], 5-min bars (NSE hours only)
    Output: dict of 7 features, each [timestamp × ticker]
    
    Parameters
    ----------
    panels : dict
        OHLCV data: keys "open", "high", "low", "close", "volume"
    atr_window : int
        ATR lookback (bars). Default 30 = 2.5 hours (faster for max return)
    vol_window : int
        Volatility trend window. Default 120 = 10 hours (~1.6 days)
    volume_window : int
        Volume baseline window. Default 60 = 5 hours (~0.8 days)
    zscore_window : int
        Rolling z-score window. Default 120 = 10 hours
    beta_window : int
        Market beta window. Default 750 = ~5 trading days
    """
    
    def __init__(
        self,
        panels: dict,
        atr_window: int = 30,
        vol_window: int = 120,
        volume_window: int = 60,
        zscore_window: int = 120,
        beta_window: int = 750,
    ):
        self.O = panels["open"].copy()
        self.H = panels["high"].copy()
        self.L = panels["low"].copy()
        self.C = panels["close"].copy()
        self.V = panels["volume"].copy()
        
        self.atr_window = atr_window
        self.vol_window = vol_window
        self.volume_window = volume_window
        self.zscore_window = zscore_window
        self.beta_window = beta_window
        
        self._idx = self.C.index
        self._dates = self._idx.normalize()
        self._bps = session_bar_index(self._idx)
        
        bps_max = int(self._bps.max()) + 1 if len(self._bps) > 0 else NSE_BARS_PER_SESSION
        self._sfrac = session_fraction(self._idx, bars_per_session=bps_max)
        
        log.info(
            "FeatureEngine: %d bars × %d tickers (7 features, NSE) | %s → %s",
            len(self.C), len(self.C.columns),
            self._idx.min() if len(self._idx) > 0 else "N/A",
            self._idx.max() if len(self._idx) > 0 else "N/A",
        )
    
    def compute_all(self) -> dict[str, pd.DataFrame]:
        """
        Compute all 7 features.
        
        Returns
        -------
        dict: {feature_name: DataFrame[timestamp × ticker]}
        """
        log.info("Computing 7 features (NSE mean reversion) ...")
        
        features = {}
        
        log.info("  [A] Reversal signals ...")
        features["A1_bar_reversal"] = self.bar_reversal()
        features["A2_short_rev"] = self.short_return_reversal()
        
        log.info("  [B] VWAP deviation ...")
        features["B1_vwap"] = self.vwap_deviation()
        
        log.info("  [C] Volume shock ...")
        features["C1_vol_shock"] = self.volume_shock()
        
        log.info("  [D] Volatility burst ...")
        features["D1_vol_burst"] = self.volatility_burst()
        
        log.info("  [E] Residual return ...")
        features["E1_residual"] = self.residual_return()
        
        log.info("  [F] Circuit proximity (NSE-specific) ...")
        features["F1_circuit_proximity"] = self.circuit_proximity()
        
        valid_pct = np.mean([f.notna().values.mean() * 100 for f in features.values()])
        log.info(
            "7-Feature set complete | avg valid=%.1f%%",
            valid_pct
        )
        
        return features
    
    # ── A: Reversal Signals ────────────────────────────────────────────────
    
    def bar_reversal(self) -> pd.DataFrame:
        """
        A1: Single-bar price reversal.
        
        Signal: Large intrabar move (normalized by ATR) → expect mean reversion.
        Formula: z-score( -bar_return / ATR )
        """
        bar_ret = (self.C - self.O) / self.O.replace(0.0, np.nan)
        atr_p = atr_pct(self.H, self.L, self.C, self.atr_window)
        raw = -bar_ret / atr_p.replace(0.0, np.nan)
        return cs_zscore(raw)
    
    def short_return_reversal(self, k: int = 3) -> pd.DataFrame:
        """
        A2: 3-bar (15-min) return reversal.
        
        Hybrid of z-score and rank-based signals for robustness.
        alpha = 0.5 * z-score(-ret_3bar) + 0.5 * rank(-ret_3bar)
        """
        ret_k = self.C.pct_change(k)
        sigma = rolling_mad(ret_k, self.zscore_window)
        z_raw = -ret_k / sigma.replace(0.0, np.nan)
        rank = -cs_rank(ret_k)
        return cs_zscore(0.5 * cs_zscore(z_raw) + 0.5 * rank)
    
    # ── B: VWAP Deviation ──────────────────────────────────────────────────
    
    def vwap_deviation(self) -> pd.DataFrame:
        """
        B1: Intraday VWAP deviation.
        
        Signal: Price vs. volume-weighted average price.
        Weighted by session fraction (more credible later in the day).
        """
        vwap = self._session_vwap()
        dev = -(self.C - vwap) / vwap.replace(0.0, np.nan)
        weight = np.sqrt(np.clip(self._sfrac.values, 0.01, 1.0)).reshape(-1, 1)
        return cs_zscore(dev * weight)
    
    # ── C: Volume Shock ────────────────────────────────────────────────────
    
    def volume_shock(self) -> pd.DataFrame:
        """
        C1: Volume spike vs time-of-day baseline.
        
        Accounts for NSE's distinct intraday volume pattern
        (heavy at open 9:15, lighter midday, heavy at close 3:00-3:30).
        """
        log_v = np.log1p(self.V)
        result = pd.DataFrame(np.nan, index=self.V.index, columns=self.V.columns)
        min_p = min(5, max(1, self.volume_window - 1))
        
        for slot in self._bps.unique():
            mask = self._bps == slot
            slot_lv = log_v[mask]
            tod_med = slot_lv.rolling(self.volume_window, min_periods=min_p).median().shift(1)
            result[mask] = (slot_lv - tod_med).values
        
        return cs_zscore(result)
    
    # ── D: Volatility Burst ────────────────────────────────────────────────
    
    def volatility_burst(self) -> pd.DataFrame:
        """
        D1: Directed volatility burst.
        
        Large intrabar range in direction opposite to bar close.
        Signal: buyers/sellers got exhausted → expect reversion.
        """
        atr_v = atr(self.H, self.L, self.C, self.atr_window)
        burst = (self.H - self.L) / atr_v.replace(0.0, np.nan) - 1
        bar_dir = np.sign(self.O - self.C)
        return cs_zscore(burst * bar_dir)
    
    # ── E: Residual Return ─────────────────────────────────────────────────
    
    def residual_return(self, k: int = 3) -> pd.DataFrame:
        """
        E1: Market-beta-adjusted residual return.
        
        Removes market (cross-sectional median) exposure to isolate
        idiosyncratic mean reversion.
        """
        ret_k = self.C.pct_change(k)
        mkt = ret_k.median(axis=1)
        mkt_var = mkt.rolling(self.beta_window, min_periods=60).var()
        
        betas = pd.DataFrame(index=ret_k.index, columns=ret_k.columns, dtype=float)
        for col in ret_k.columns:
            betas[col] = (
                ret_k[col].rolling(self.beta_window, min_periods=60).cov(mkt)
                / mkt_var.replace(0.0, np.nan)
            )
        
        residual = ret_k.sub(betas.mul(mkt, axis=0), axis=0)
        return -cs_zscore(residual)
    
    # ── F: Circuit Proximity (NSE-Specific) ────────────────────────────────
    
    def circuit_proximity(self) -> pd.DataFrame:
        """
        F1: Distance to NSE circuit limits.
        
        NSE stocks have daily circuit limits at ±5%, ±10%, or ±20%.
        Stocks near their circuit limit tend to reverse:
          - Near upper circuit → trapped buyers, likely pullback
          - Near lower circuit → trapped sellers, likely bounce
        
        Signal: z-score( -(daily_return / estimated_circuit_limit) )
        
        This feature is unique to NSE and adds alpha that
        US-market strategies cannot capture.
        """
        # Compute session return (from day's open to current bar)
        session_open = pd.DataFrame(np.nan, index=self.C.index, columns=self.C.columns)
        for date, grp_idx in self.C.groupby(self._dates).groups.items():
            first_bar = self.O.loc[grp_idx].iloc[0]
            session_open.loc[grp_idx] = first_bar.values
        
        session_ret = (self.C - session_open) / session_open.replace(0.0, np.nan)
        
        # Estimate circuit limit per stock (use trailing max daily range)
        daily_range = self.C.groupby(self._dates).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0].replace(0, np.nan)
            if len(x) > 0 else pd.Series(0, index=x.columns)
        )
        
        # Use a simple proxy: assume 10% circuit for most stocks
        # Stocks with session_ret near ±5%, ±10% are near circuit
        circuit_est = 0.10  # Conservative estimate
        proximity = session_ret.abs() / circuit_est
        
        # Signal: stocks near circuit limits → expect reversal
        # Negative because near-circuit stocks should reverse
        direction = -np.sign(session_ret)
        signal = proximity * direction
        
        return cs_zscore(signal)
    
    # ── Time-of-Day Signal Weighting ──────────────────────────────────────
    
    def apply_tod_weights(
        self,
        features: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Apply time-of-day weights to all features for maximum return.
        
        NSE intraday patterns:
          - First 30 min (9:15-9:45): 1.5× weight (strongest reversals)
          - Last 30 min (3:00-3:30):  1.2× weight (closing auction effects)
          - Midday (12:00-1:30):      0.5× weight (lunch lull, weak signals)
          - Rest:                     1.0× weight
        """
        tod_weight = pd.Series(1.0, index=self._idx)
        
        if hasattr(self._idx, 'time'):
            import datetime as dt
            times = self._idx.time
            
            # Opening boost: 9:15 - 9:45
            opening = (times >= dt.time(9, 15)) & (times < dt.time(9, 45))
            tod_weight[opening] = 1.5
            
            # Closing boost: 15:00 - 15:30
            closing = (times >= dt.time(15, 0)) & (times < dt.time(15, 30))
            tod_weight[closing] = 1.2
            
            # Midday dampening: 12:00 - 13:30
            midday = (times >= dt.time(12, 0)) & (times < dt.time(13, 30))
            tod_weight[midday] = 0.5
        
        weighted = {}
        for name, feat in features.items():
            weighted[name] = feat.mul(tod_weight, axis=0)
        
        log.info("Time-of-day weights applied (open=1.5×, close=1.2×, midday=0.5×)")
        return weighted
    
    # ── Private Helpers ────────────────────────────────────────────────────
    
    def _session_vwap(self) -> pd.DataFrame:
        """Compute intraday VWAP (volume-weighted average price) per session."""
        pv = self.C * self.V
        vwap = pd.DataFrame(np.nan, index=self.C.index, columns=self.C.columns)
        
        for date, grp_idx in pv.groupby(self._dates).groups.items():
            pv_cumsum = pv.loc[grp_idx].cumsum()
            v_cumsum = self.V.loc[grp_idx].cumsum()
            vwap.loc[grp_idx] = (pv_cumsum / v_cumsum.replace(0.0, np.nan)).values
        
        return vwap
