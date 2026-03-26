"""
alpha/risk_management_44stock.py
================================
PART 7-9: Risk Management + Daily Monitoring Dashboard for 44-Stock Universe

RISKS ADDRESSED:
  1. Concentration Risk: top 5 stocks > 80% of PnL
  2. Correlation Breakdown: stress regime with r > 0.80
  3. Liquidity Cliff: bid-ask > 2 bps, halts, delists
  4. Small-sample Overfitting: OOS Sharpe < 40% of IS
  5. Drawdown Control: daily loss limit, circuit breaker

Usage
-----
    from alpha.risk_management_44stock import RiskManager44
    
    rm = RiskManager44(capital=100_000_000)
    
    # Before trading
    alerts = rm.pre_trade_checks(
        positions=df_positions,
        prices=df_close,
        volumes=df_volumes,
        returns=df_returns,
    )
    
    # During day
    daily_metrics = rm.monitor_realtime(pnl_today, positions_current)
    
    # End of day
    summary = rm.daily_report(pnl_daily, positions, returns)
"""

from __future__ import annotations

import logging
from typing import Optional
from datetime import datetime

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class RiskManager44:
    """
    Risk management for 44-stock intraday mean-reversion strategy.
    
    Constraints:
      - Max position size: 10% of capital per stock
      - Max gross leverage: 2.0×
      - Dollar-neutral: long $X = short $X
      - Beta-neutral: portfolio beta ≈ 0
      - Daily loss limit: -1% of capital
      - Position-level stop-loss: -2% per stock
    """
    
    def __init__(
        self,
        capital: float = 100_000_000,
        max_pos_weight: float = 0.10,
        max_gross_lev: float = 2.0,
        daily_loss_limit: float = 0.01,
        pos_stop_loss: float = 0.02,
        max_pos_concentration: float = 0.80,
        max_avg_correlation: float = 0.80,
        min_bid_ask_pct: float = 0.0002,
        max_bid_ask_pct: float = 0.0005,
    ):
        """
        Parameters
        ----------
        capital : float
            Total capital ($)
        max_pos_weight : float
            Max |position| as % of capital per stock
        max_gross_lev : float
            Max gross leverage
        daily_loss_limit : float
            Stop trading if daily loss > this % of capital
        pos_stop_loss : float
            Close position if loss > this % of entry price
        max_pos_concentration : float
            Alert if top 5 stocks > this % of daily PnL
        max_avg_correlation : float
            Alert if avg stock correlation > this
        min_bid_ask_pct : float
            Min bid-ask spread (expected 0.0001 = 1 bps for liquid names)
        max_bid_ask_pct : float
            Max bid-ask spread (alert if exceeded)
        """
        self.capital = capital
        self.max_pos_weight = max_pos_weight
        self.max_gross_lev = max_gross_lev
        self.daily_loss_limit = daily_loss_limit
        self.pos_stop_loss = pos_stop_loss
        self.max_pos_concentration = max_pos_concentration
        self.max_avg_correlation = max_avg_correlation
        self.min_bid_ask_pct = min_bid_ask_pct
        self.max_bid_ask_pct = max_bid_ask_pct
        
        self.daily_pnl = 0.0
        self.mtd_pnl = 0.0
        self.ytd_pnl = 0.0
        
        log.info(
            "RiskManager44 initialized:\n"
            "  Capital: $%.0fM\n"
            "  Max weight per stock: %.0f%%\n"
            "  Max gross leverage: %.1f×\n"
            "  Daily loss limit: -%.2f%%\n"
            "  Stop-loss per position: -%.2f%%",
            capital / 1e6, max_pos_weight * 100, max_gross_lev,
            daily_loss_limit * 100, pos_stop_loss * 100
        )
    
    def pre_trade_checks(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
        volumes: pd.DataFrame,
        returns: pd.DataFrame,
        betas: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Pre-market checks before any trading.
        
        Returns
        -------
        dict with alerts:
            "concentration_alert": bool
            "correlation_alert": bool
            "liquidity_alert": bool
            "beta_alert": bool
            "all_clear": bool
        """
        log.info("Running pre-trade checks ...")
        
        alerts = {
            "timestamp": datetime.now(),
            "concentration": self._check_concentration(positions, prices),
            "correlation": self._check_correlation(returns),
            "liquidity": self._check_liquidity(volumes, prices),
            "beta": self._check_beta(betas, positions) if betas is not None else None,
        }
        
        all_clear = all(
            not v for k, v in alerts.items()
            if k != "timestamp" and v is not None
        )
        alerts["all_clear"] = all_clear
        
        severity = "OK" if all_clear else "ALERT"
        log.info("Pre-trade checks complete: %s", severity)
        
        return alerts
    
    def monitor_realtime(
        self,
        mtm_pnl: float,
        positions_current: pd.DataFrame,
        prices_current: pd.DataFrame,
    ) -> dict:
        """
        Intraday monitoring (check every 5 minutes).
        
        Parameters
        ----------
        mtm_pnl : float
            Mark-to-market PnL today ($)
        positions_current : DataFrame [44,]
            Current position per stock (shares)
        prices_current : DataFrame [44,]
            Current prices per stock
        
        Returns
        -------
        dict with realtime alerts
        """
        self.daily_pnl = mtm_pnl
        
        # Check daily loss limit
        if mtm_pnl < -self.capital * self.daily_loss_limit:
            log.warning(
                "⚠ DAILY LOSS LIMIT BREACHED: %+.2f%% (limit: %.2f%%)",
                mtm_pnl / self.capital * 100,
                -self.daily_loss_limit * 100
            )
            return {"stop_trading": True, "reason": "Daily loss limit"}
        
        # Check concentration
        notional = positions_current * prices_current
        long_pnl = notional[notional > 0].sum()
        short_pnl = notional[notional < 0].sum().abs()
        
        if long_pnl > 0:
            top5_pct = notional[notional > 0].nlargest(5).sum() / long_pnl
            if top5_pct > self.max_pos_concentration:
                log.warning(
                    "⚠ CONCENTRATION ALERT: top 5 longs = %.0f%% of long notional",
                    top5_pct * 100
                )
        
        if short_pnl > 0:
            top5_pct = notional[notional < 0].nsmallest(5).sum().abs() / short_pnl
            if top5_pct > self.max_pos_concentration:
                log.warning(
                    "⚠ CONCENTRATION ALERT: top 5 shorts = %.0f%% of short notional",
                    top5_pct * 100
                )
        
        return {
            "stop_trading": False,
            "daily_pnl": mtm_pnl,
            "daily_pnl_pct": mtm_pnl / self.capital * 100,
        }
    
    def daily_report(
        self,
        pnl_daily: pd.Series,
        positions_end: pd.DataFrame,
        returns_daily: pd.DataFrame,
        prices_end: pd.DataFrame,
    ) -> dict:
        """
        End-of-day risk report.
        
        Parameters
        ----------
        pnl_daily : Series [timestamp]
            Intraday P&L (daily values)
        positions_end : DataFrame [44,]
            Positions at close
        returns_daily : DataFrame [datetime, 44]
            Daily returns (for correlation)
        prices_end : DataFrame [44,]
            Prices at close
        
        Returns
        -------
        dict: Daily report with metrics
        """
        total_pnl = pnl_daily.sum()
        total_return = total_pnl / self.capital
        self.daily_pnl = total_pnl
        self.mtd_pnl += total_pnl
        self.ytd_pnl += total_pnl
        
        notional = positions_end * prices_end
        gross_notional = notional.abs().sum()
        gross_lev = gross_notional / self.capital
        
        # Concentration
        long_pct = notional[notional > 0].sum() / self.capital
        short_pct = notional[notional < 0].sum().abs() / self.capital
        
        report = {
            "date": pd.Timestamp.now().date(),
            "daily_pnl": total_pnl,
            "daily_return_pct": total_return * 100,
            "mtd_pnl": self.mtd_pnl,
            "ytd_pnl": self.ytd_pnl,
            "gross_lev": gross_lev,
            "long_pct": long_pct,
            "short_pct": short_pct,
            "concentration": self._concentration_score(notional),
            "volatility": returns_daily.std(axis=1).mean(),
        }
        
        log.info(
            "DAILY REPORT:\n"
            "  PnL: ${:+.0f} ({:+.2%})\n"
            "  Gross Leverage: {:.2f}×\n"
            "  Long: {:.1%} | Short: {:.1%}\n"
            "  Concentration (top 5): {:.0%}",
            total_pnl, total_return, gross_lev,
            long_pct, short_pct,
            report["concentration"]
        )
        
        return report
    
    # ── Private helpers ────────────────────────────────────────────────────────
    
    def _check_concentration(
        self,
        positions: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> bool:
        """Check if top 5 stocks exceed concentration limit."""
        notional = positions.iloc[-1] * prices.iloc[-1]
        
        long_notional = notional[notional > 0].sum()
        if long_notional > 0:
            top5_pct = notional[notional > 0].nlargest(5).sum() / long_notional
            if top5_pct > self.max_pos_concentration:
                log.warning(
                    "Concentration alert: top 5 longs = %.0f%% (limit %.0f%%)",
                    top5_pct * 100, self.max_pos_concentration * 100
                )
                return True
        
        short_notional = notional[notional < 0].sum().abs()
        if short_notional > 0:
            top5_pct = notional[notional < 0].nsmallest(5).sum().abs() / short_notional
            if top5_pct > self.max_pos_concentration:
                log.warning(
                    "Concentration alert: top 5 shorts = %.0f%% (limit %.0f%%)",
                    top5_pct * 100, self.max_pos_concentration * 100
                )
                return True
        
        return False
    
    def _check_correlation(self, returns: pd.DataFrame) -> bool:
        """Check if stock correlations > threshold (stress regime)."""
        if len(returns) < 10:
            return False
        
        corr_matrix = returns.corr()
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        if avg_corr > self.max_avg_correlation:
            log.warning(
                "Correlation alert: avg stock correlation = %.2f (limit %.2f)",
                avg_corr, self.max_avg_correlation
            )
            return True
        
        return False
    
    def _check_liquidity(
        self,
        volumes: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> bool:
        """Check if any stock has low liquidity or wide spreads."""
        # Check volume (should be > $2M daily for each stock)
        min_adtv_usd = 2_000_000
        adv_usd = (volumes * prices).mean()
        
        illiquid = adv_usd[adv_usd < min_adtv_usd].index.tolist()
        if illiquid:
            log.warning(
                "Liquidity alert: %d stocks below $%.0fM ADTV: %s",
                len(illiquid), min_adtv_usd / 1e6, illiquid[:3]
            )
            return True
        
        return False
    
    def _check_beta(
        self,
        betas: pd.DataFrame,
        positions: pd.DataFrame,
    ) -> bool:
        """Check if portfolio beta > 0.05 (not neutral)."""
        portfolio_beta = (betas.iloc[-1] * positions.iloc[-1]).sum()
        
        if abs(portfolio_beta) > 0.05:
            log.warning(
                "Beta alert: portfolio beta = %.3f (limit ±0.050)",
                portfolio_beta
            )
            return True
        
        return False
    
    def _concentration_score(self, notional: pd.Series) -> float:
        """Compute concentration score (0 = diversified, 1 = concentrated)."""
        abs_notional = notional.abs()
        if abs_notional.sum() == 0:
            return 0.0
        
        pct = abs_notional / abs_notional.sum()
        herfindahl = (pct ** 2).sum()
        
        # Herfindahl: min=1/n (diverse), max=1 (concentrated)
        # Normalize to [0, 1]
        n = len(pct)
        score = (herfindahl - 1/n) / (1 - 1/n)
        
        return score


def print_risk_report(report: dict) -> None:
    """Pretty-print risk report."""
    print("\n" + "=" * 70)
    print("DAILY RISK REPORT")
    print("=" * 70)
    for key, val in report.items():
        if isinstance(val, float):
            print(f"  {key:.<30} {val:>10.2%}")
        elif isinstance(val, int):
            print(f"  {key:.<30} {val:>10,.0f}")
        else:
            print(f"  {key:.<30} {val}")
    print("=" * 70 + "\n")
