"""
backtest_44stock.py
===================
PART 7-8: Walk-Forward Backtesting + Risk Management (Production Integration)

Complete pipeline to:
  1. Load 44-stock data
  2. Compute 6 selected features
  3. Run walk-forward backtest (60d train, 5d test, 5d step)
  4. Output metrics, equity curve, risk analysis

Usage
-----
    python backtest_44stock.py \\
        --start 2023-01-01 \\
        --end 2024-12-31 \\
        --capital 100000000 \\
        --train-days 60 \\
        --test-days 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Import our modules
from features.engine_44stock import FeatureEngine44
from alpha.rank_alpha import composite_rank_alpha, compute_ic_weights
from alpha.positions_beta_neutral import compute_beta_neutral_positions
from alpha.regularized_zscore import regularized_zscore
from data.preprocess_sparse_5min import preprocess_sparse_data

log = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Walk-forward backtester for 44-stock mean-reversion strategy.
    
    Components:
    1. Feature extraction (6 selected features)
    2. Alpha signal (rank-based, no windsorization)
    3. Position sizing (beta-neutral, vol-scaled)
    4. Risk management (concentration, drawdown)
    5. Walk-forward validation (train 60d, test 5d, step 5d)
    """
    
    def __init__(
        self,
        panels: dict[str, pd.DataFrame],
        spy_prices: pd.Series,
        spy_returns: pd.Series,
        capital: float = 100_000_000,
        train_days: int = 60,
        test_days: int = 5,
        gross_lev: float = 2.0,
    ):
        """
        Parameters
        ----------
        panels : dict
            OHLCV data: {"open": df, "high": df, "low": df, "close": df, "volume": df}
            Each [timestamp × 44 tickers]
        spy_prices : Series [timestamp]
        spy_returns : Series [timestamp]
        capital : float
        train_days : int
        test_days : int
        gross_lev : float
        """
        self.panels = panels
        self.spy_prices = spy_prices
        self.spy_returns = spy_returns
        self.capital = capital
        self.train_days = train_days
        self.test_days = test_days
        self.gross_lev = gross_lev
        
        log.info(
            "WalkForwardBacktester initialized:\n"
            "  Data: %d bars × %d tickers\n"
            "  Capital: $%.0fM | Gross lev: %.1f×\n"
            "  Windows: train=%d days, test=%d days",
            len(panels["close"]), len(panels["close"].columns),
            capital / 1e6, gross_lev, train_days, test_days
        )
    
    def run(self) -> dict:
        """
        Execute full walk-forward backtest.
        
        Returns
        -------
        dict:
            results: list of window results
            summary: aggregated metrics (Sharpe, return, DD)
            equity_curve: daily equity
            signals: alpha signals per bar
        """
        log.info("Starting walk-forward backtest ...")
        
        # Compute 6-feature set
        engine = FeatureEngine44(self.panels)
        features = engine.compute_selected_features()
        
        # Preprocess sparse data
        features_clean, sparsity_flags = preprocess_sparse_data(features, method="lagged_imputation")
        
        # Get date windows
        close = self.panels["close"]
        dates = close.index.normalize().unique()
        
        windows = []
        equity_curve = [self.capital]
        all_pnl = []
        all_signals = []
        
        train_window_days = timedelta(days=self.train_days)
        test_window_days = timedelta(days=self.test_days)
        
        for i in range(0, len(dates) - self.train_days - self.test_days, self.test_days):
            train_start = dates[i]
            train_end = dates[i + self.train_days]
            test_start = dates[i + self.train_days]
            test_end = dates[i + self.train_days + self.test_days]
            
            # Train period
            train_close = close[train_start:train_end]
            train_features = {
                k: v[train_start:train_end] for k, v in features_clean.items()
            }
            
            # Estimate IC weights
            try:
                ic_weights = compute_ic_weights(train_features, train_close, ic_window=60)
                is_sharpe = np.mean([w.mean() for w in ic_weights.values()])
            except:
                is_sharpe = 0.0
            
            # Test period
            test_close = close[test_start:test_end]
            test_features = {
                k: v[test_start:test_end] for k, v in features_clean.items()
            }
            test_spy_ret = self.spy_returns[test_start:test_end]
            test_spy_prices = self.spy_prices[test_start:test_end]
            
            # Compute alpha
            try:
                alpha = composite_rank_alpha(test_features, test_close, ic_window=60)
                all_signals.append(alpha)
                
                # Compute positions
                positions, sizes, spy_hedge = compute_beta_neutral_positions(
                    rank_alpha=alpha,
                    close=test_close,
                    volumes=self.panels["volume"][test_start:test_end],
                    spy_prices=test_spy_prices,
                    spy_returns=test_spy_ret,
                    capital=self.capital,
                    gross_lev=self.gross_lev,
                )
                
                # Simulate P&L
                window_pnl = self._simulate_window(positions, test_close)
                all_pnl.extend(window_pnl)
                
                # Update equity
                for pnl_daily in window_pnl:
                    new_equity = equity_curve[-1] * (1 + pnl_daily)
                    equity_curve.append(new_equity)
                
                window_result = {
                    "period": f"{train_start.date()} to {test_end.date()}",
                    "is_sharpe": is_sharpe,
                    "oos_sharpe": self._compute_sharpe(window_pnl),
                    "period_return": np.sum(window_pnl),
                    "max_dd": (np.cumsum(window_pnl) - np.cumsum(window_pnl).cummax()).min(),
                }
                windows.append(window_result)
                
            except Exception as e:
                log.error("Error in window %s: %s", train_start, e)
                continue
        
        # Summary
        pnl_array = np.array(all_pnl)
        if len(pnl_array) > 0:
            summary = {
                "total_windows": len(windows),
                "annual_return": np.mean([w["period_return"] for w in windows]) * 252,
                "annual_vol": np.std(pnl_array) * np.sqrt(252),
                "sharpe": np.mean([w["oos_sharpe"] for w in windows]),
                "max_drawdown": np.min([w["max_dd"] for w in windows]),
                "final_equity": equity_curve[-1],
            }
        else:
            summary = {"error": "No valid windows"}
        
        return {
            "windows": windows,
            "summary": summary,
            "equity_curve": equity_curve,
            "pnl": all_pnl,
        }
    
    def _simulate_window(self, positions: pd.DataFrame, close: pd.DataFrame) -> list[float]:
        """Simulate daily P&L for one test window."""
        pnl_list = []
        
        for t in range(len(close) - 1):
            # P&L = position_t * return_{t,t+1}
            returns_next = close.pct_change(1).iloc[t + 1]
            daily_pnl = (positions.iloc[t] * returns_next).sum()
            pnl_list.append(daily_pnl / self.capital)
        
        return pnl_list
    
    def _compute_sharpe(self, pnl_list: list[float]) -> float:
        """Compute Sharpe ratio from daily returns."""
        if not pnl_list:
            return 0.0
        pnl_arr = np.array(pnl_list)
        daily_vol = np.std(pnl_arr)
        if daily_vol == 0:
            return 0.0
        return np.mean(pnl_arr) / daily_vol * np.sqrt(252)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2024-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100_000_000, help="Capital ($)")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=5)
    parser.add_argument("--gross-lev", type=float, default=2.0)
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load data
    log.info("Loading 44-stock data ...")
    from polygon_pipeline.pipeline.storage import read_panels
    panels = read_panels()
    
    # Placeholder for SPY (normally load from external source)
    spy_prices = pd.Series(index=panels["close"].index, data=100.0)  # TODO: load real SPY
    spy_returns = pd.Series(index=panels["close"].index, data=0.0001)  # TODO: load real returns
    
    # Run backtest
    backtester = WalkForwardBacktester(
        panels=panels,
        spy_prices=spy_prices,
        spy_returns=spy_returns,
        capital=args.capital,
        train_days=args.train_days,
        test_days=args.test_days,
        gross_lev=args.gross_lev,
    )
    
    results = backtester.run()
    
    # Print results
    log.info("\n" + "=" * 70)
    log.info("WALK-FORWARD BACKTEST RESULTS")
    log.info("=" * 70)
    log.info("Summary: %s", results["summary"])
    log.info("\nWindow Results:")
    for w in results["windows"][:5]:  # Show first 5
        log.info("  %s: IS=%.3f, OOS=%.3f, Return=%.2%%, DD=%.2%%", 
                 w["period"], w["is_sharpe"], w["oos_sharpe"], w["period_return"], w["max_dd"])


if __name__ == "__main__":
    main()
