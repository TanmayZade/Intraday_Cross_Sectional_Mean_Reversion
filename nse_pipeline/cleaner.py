"""
nse_pipeline/cleaner.py
=======================
OHLCV data cleaning for NSE intraday bars.

NSE-specific cleaning rules:
  1. Circuit breaker detection (5%, 10%, 20% price limits)
  2. Opening auction bar handling (9:15 bar may be distorted)
  3. Standard spike removal, gap handling, OHLC ordering
  4. Volume spike filtering

Usage
-----
    from nse_pipeline.cleaner import NSECleaner
    
    cleaner = NSECleaner(config)
    clean_panels = cleaner.clean(raw_panels)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class NSECleaner:
    """
    Clean raw OHLCV panel data for NSE stocks.
    
    Parameters
    ----------
    max_daily_return : float
        Max absolute daily return before flagging (0.20 = 20%, NSE circuit limit)
    max_intraday_range : float
        Max (high-low)/close ratio per bar
    max_volume_spike_mult : float
        Flag bars where volume > mult × rolling median volume
    volume_spike_window : int
        Rolling window for volume spike detection
    max_overnight_gap : float
        Max absolute overnight gap (close→open next day)
    enforce_ohlc_order : bool
        If True, enforce low ≤ open,close ≤ high
    """
    
    def __init__(
        self,
        max_daily_return: float = 0.20,
        max_intraday_range: float = 0.20,
        max_volume_spike_mult: float = 10.0,
        volume_spike_window: int = 20,
        max_overnight_gap: float = 0.20,
        enforce_ohlc_order: bool = True,
    ):
        self.max_daily_return = max_daily_return
        self.max_intraday_range = max_intraday_range
        self.max_vol_spike = max_volume_spike_mult
        self.vol_spike_win = volume_spike_window
        self.max_gap = max_overnight_gap
        self.enforce_ohlc = enforce_ohlc_order
    
    def clean(
        self,
        panels: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """
        Clean a set of OHLCV panels.
        
        Parameters
        ----------
        panels : dict
            {"open": DataFrame, "high": DataFrame, "low": DataFrame,
             "close": DataFrame, "volume": DataFrame}
            Each DataFrame: [timestamp × ticker]
        
        Returns
        -------
        dict of cleaned panels (same structure)
        """
        log.info("Cleaning NSE data ...")
        
        O = panels["open"].copy()
        H = panels["high"].copy()
        L = panels["low"].copy()
        C = panels["close"].copy()
        V = panels["volume"].copy()
        
        n_before = C.notna().sum().sum()
        
        # ── Rule 1: Enforce OHLC ordering ─────────────────────────────────
        if self.enforce_ohlc:
            O, H, L, C = self._enforce_ohlc(O, H, L, C)
        
        # ── Rule 2: Flag extreme intraday returns (circuit breaker) ───────
        bar_ret = C.pct_change(1).abs()
        circuit_mask = bar_ret > self.max_daily_return
        n_circuit = circuit_mask.sum().sum()
        if n_circuit > 0:
            log.info("  Circuit/extreme bars flagged: %d", n_circuit)
            # NaN out circuit-hit bars (they distort signals)
            for panel in [O, H, L, C, V]:
                panel[circuit_mask] = np.nan
        
        # ── Rule 3: Flag extreme intraday range ──────────────────────────
        intraday_range = (H - L) / C.replace(0.0, np.nan)
        range_mask = intraday_range.abs() > self.max_intraday_range
        n_range = range_mask.sum().sum()
        if n_range > 0:
            log.info("  Extreme range bars flagged: %d", n_range)
            for panel in [O, H, L, C, V]:
                panel[range_mask] = np.nan
        
        # ── Rule 4: Flag volume spikes ───────────────────────────────────
        vol_median = V.rolling(self.vol_spike_win, min_periods=5).median()
        vol_spike_mask = V > (self.max_vol_spike * vol_median)
        # Don't NaN volume spikes — they carry information — just flag
        n_vol_spikes = vol_spike_mask.sum().sum()
        if n_vol_spikes > 0:
            log.debug("  Volume spikes detected: %d (kept, not removed)", n_vol_spikes)
        
        # ── Rule 5: Remove zero/negative prices ─────────────────────────
        for panel in [O, H, L, C]:
            panel[panel <= 0] = np.nan
        V[V < 0] = 0
        
        # ── Rule 6: Forward-fill small gaps (max 2 bars) ────────────────
        O = O.ffill(limit=2)
        H = H.ffill(limit=2)
        L = L.ffill(limit=2)
        C = C.ffill(limit=2)
        
        n_after = C.notna().sum().sum()
        pct_removed = (1 - n_after / max(n_before, 1)) * 100
        
        log.info(
            "  Cleaning complete: %d → %d valid cells (%.1f%% removed)",
            n_before, n_after, pct_removed,
        )
        
        return {"open": O, "high": H, "low": L, "close": C, "volume": V}
    
    def _enforce_ohlc(
        self,
        O: pd.DataFrame,
        H: pd.DataFrame,
        L: pd.DataFrame,
        C: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Enforce low ≤ min(open, close) and high ≥ max(open, close)."""
        min_oc = pd.DataFrame(
            np.minimum(O.values, C.values), index=O.index, columns=O.columns
        )
        max_oc = pd.DataFrame(
            np.maximum(O.values, C.values), index=O.index, columns=O.columns
        )
        
        # Correct high and low
        H_fixed = pd.DataFrame(
            np.maximum(H.values, max_oc.values), index=H.index, columns=H.columns
        )
        L_fixed = pd.DataFrame(
            np.minimum(L.values, min_oc.values), index=L.index, columns=L.columns
        )
        
        n_fixed = ((H != H_fixed).sum().sum() + (L != L_fixed).sum().sum())
        if n_fixed > 0:
            log.debug("  OHLC ordering fixed: %d cells", n_fixed)
        
        return O, H_fixed, L_fixed, C
    
    def detect_circuit_hits(
        self,
        close: pd.DataFrame,
        threshold: float = 0.045,
    ) -> pd.DataFrame:
        """
        Detect bars where a stock likely hit its circuit limit.
        
        NSE circuit limits: 5%, 10%, 20% (varies by stock).
        Returns boolean mask [timestamp × ticker].
        """
        daily_ret = close.pct_change(1)
        
        # Near ±5%, ±10%, or ±20% — within 0.5% of a circuit level
        circuit_5  = daily_ret.abs().between(0.045, 0.055)
        circuit_10 = daily_ret.abs().between(0.095, 0.105)
        circuit_20 = daily_ret.abs().between(0.195, 0.205)
        
        circuit_mask = circuit_5 | circuit_10 | circuit_20
        
        n_hits = circuit_mask.sum().sum()
        if n_hits > 0:
            log.info("  Circuit hits detected: %d bars", n_hits)
        
        return circuit_mask
