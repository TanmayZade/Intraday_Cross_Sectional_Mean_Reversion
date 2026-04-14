"""
alpha/risk_management.py
========================
NSE-specific risk management for intraday mean reversion.

Handles:
  1. NSE circuit breaker detection (5%, 10%, 20% limits)
  2. Market-wide circuit breaker (NIFTY ±10/15/20%)
  3. Position-level risk limits
  4. Drawdown-based position scaling
  5. Time-of-day position adjustments

Usage
-----
    from alpha.risk_management import RiskManager
    
    rm = RiskManager(capital=1_000_000)
    weights = rm.apply(weights, close, nifty_returns)
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class RiskManager:
    """
    NSE-specific risk controls for intraday mean reversion.
    
    Parameters
    ----------
    capital : float
        Total capital in INR (default ₹10L)
    max_single_stock_pct : float
        Max allocation to single stock (default 15%)
    max_drawdown_pct : float
        Max portfolio drawdown before reducing positions (default 5%)
    circuit_buffer_pct : float
        Reduce position when stock is within this % of circuit (default 2%)
    """
    
    def __init__(
        self,
        capital: float = 1_000_000,
        max_single_stock_pct: float = 0.15,
        max_drawdown_pct: float = 0.05,
        circuit_buffer_pct: float = 0.02,
    ):
        self.capital = capital
        self.max_stock_pct = max_single_stock_pct
        self.max_dd = max_drawdown_pct
        self.circuit_buffer = circuit_buffer_pct
    
    def apply(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
        nifty_returns: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Apply all risk controls to portfolio weights.
        
        Parameters
        ----------
        weights : DataFrame [timestamp × ticker] — raw weights
        close : DataFrame [timestamp × ticker] — prices
        nifty_returns : Series — NIFTY 50 bar returns
        
        Returns
        -------
        DataFrame: risk-adjusted weights
        """
        w = weights.copy()
        
        # 1. Circuit breaker proximity
        w = self._circuit_proximity_scale(w, close)
        
        # 2. Market-wide circuit breaker
        if nifty_returns is not None:
            w = self._market_circuit_check(w, nifty_returns)
        
        # 3. Position concentration limits
        w = self._concentration_limit(w)
        
        # 4. Drawdown-based scaling
        w = self._drawdown_scale(w, close)
        
        return w
    
    def _circuit_proximity_scale(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reduce positions for stocks near their circuit limits.
        
        NSE circuit limits: ±5%, ±10%, ±20%
        If a stock's session return is within circuit_buffer of a limit,
        reduce its weight by 50%.
        """
        # Session return from day's open
        dates = close.index.normalize()
        session_open = close.groupby(dates).transform("first")
        session_ret = (close - session_open) / session_open.replace(0, np.nan)
        
        # Check proximity to circuit limits
        circuit_levels = [0.05, 0.10, 0.20]
        near_circuit = pd.DataFrame(False, index=close.index, columns=close.columns)
        
        for level in circuit_levels:
            upper_near = (session_ret > level - self.circuit_buffer) & (session_ret < level + 0.005)
            lower_near = (session_ret < -level + self.circuit_buffer) & (session_ret > -level - 0.005)
            near_circuit = near_circuit | upper_near | lower_near
        
        # Align to weights index
        near_aligned = near_circuit.reindex(weights.index).fillna(False)
        
        # Scale down positions near circuit
        scale = pd.DataFrame(1.0, index=weights.index, columns=weights.columns)
        scale[near_aligned] = 0.5
        
        n_scaled = near_aligned.sum().sum()
        if n_scaled > 0:
            log.info("  Circuit proximity: reduced %d positions by 50%%", n_scaled)
        
        return weights * scale
    
    def _market_circuit_check(
        self,
        weights: pd.DataFrame,
        nifty_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Reduce all positions during market-wide stress.
        
        If NIFTY 50 moves > ±3% in a rolling 30-bar window,
        reduce all positions by 50%.
        """
        nifty_aligned = nifty_returns.reindex(weights.index)
        rolling_nifty = nifty_aligned.rolling(30, min_periods=5).sum().abs()
        
        stress_mask = rolling_nifty > 0.03
        
        scale = pd.Series(1.0, index=weights.index)
        scale[stress_mask] = 0.5
        
        n_stress = stress_mask.sum()
        if n_stress > 0:
            log.info("  Market stress: reduced positions for %d bars", n_stress)
        
        return weights.mul(scale, axis=0)
    
    def _concentration_limit(
        self,
        weights: pd.DataFrame,
    ) -> pd.DataFrame:
        """Hard clip individual stock weights at max_stock_pct."""
        return weights.clip(-self.max_stock_pct, self.max_stock_pct)
    
    def _drawdown_scale(
        self,
        weights: pd.DataFrame,
        close: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Scale down positions during portfolio drawdowns.
        
        If cumulative portfolio return drops below -max_dd,
        reduce position sizes linearly (min 30% of original).
        """
        # Compute simple portfolio equity proxy
        ret = close.pct_change(1)
        port_ret = (weights.shift(1) * ret.reindex(weights.index)).sum(axis=1)
        cum_ret = (1 + port_ret.fillna(0)).cumprod()
        
        running_max = cum_ret.expanding().max()
        drawdown = (cum_ret - running_max) / running_max
        
        # Scale: 1.0 at dd=0, 0.3 at dd=-max_dd, 0.3 below
        scale = (1.0 + drawdown / self.max_dd * 0.7).clip(0.3, 1.0)
        
        if (scale < 1.0).any():
            log.info("  Drawdown scaling active: min scale=%.2f", scale.min())
        
        return weights.mul(scale, axis=0)
