"""
features/engine.py — Institutional Feature Engine (1-minute bars)
15 signals across 5 categories for cross-sectional mean reversion.

Updated 2026-03-25: Now operates on 1-minute bars from Polygon API.
Frequency-dependent parameters have been scaled appropriately.
All windows maintain calendar time equivalence (300 bars ≈ 1 trading day at 1-min).
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

class FeatureEngine:
    """
    15 alpha features for intraday cross-sectional mean reversion.

    Category A — Reversal (core hypothesis)
    Category B — VWAP & Microstructure
    Category C — Volume & Flow
    Category D — Volatility Regime
    Category E — Residual / Market-stripped

    Input : {field: DataFrame[timestamp × ticker]}  5-min bars
    Output: {name:  DataFrame[timestamp × ticker]}  normalised signals
    
    Default parameters are scaled for 1-minute bars:
      atr_window=300      → ~300 min = 1 trading day volatility lookback
      vol_window=900      → ~900 min = 1.5 trading days realized vol
      beta_window=5850    → ~6 trading days market factor exposure
      zscore_window=900   → ~900 min = 1.5 trading days
      halflife=195        → EWM halflife ~195 min (5× from 39 bars at 5-min)
    """

    def __init__(self, panels, atr_window=300, vol_window=900,
                 volume_window=300, zscore_window=900, beta_window=5850, halflife=195):
        self.O = panels["open"].copy()
        self.H = panels["high"].copy()
        self.L = panels["low"].copy()
        self.C = panels["close"].copy()
        self.V = panels["volume"].copy()
        self.atr_window    = atr_window
        self.vol_window    = vol_window
        self.volume_window = volume_window
        self.zscore_window = zscore_window
        self.beta_window   = beta_window
        self.halflife      = halflife
        self._idx   = self.C.index
        self._dates = self._idx.normalize()
        self._bps   = session_bar_index(self._idx)
        bps_max = int(self._bps.max()) + 1 if len(self._bps) > 0 else 26
        self._sfrac = session_fraction(self._idx, bars_per_session=bps_max)
        log.info("FeatureEngine: %d bars × %d tickers | %s → %s",
                 len(self.C), len(self.C.columns), self._idx.min(), self._idx.max())

    def compute_all(self) -> dict[str, pd.DataFrame]:
        log.info("Computing 15 features across 5 categories ...")
        features = {}
        log.info("  [A] Reversal signals ...")
        features["A1_bar_reversal"]      = self.bar_reversal()
        features["A2_short_rev"]         = self.short_return_reversal()
        features["A3_medium_rev"]        = self.medium_return_reversal()
        features["A4_overnight_gap"]     = self.overnight_gap_fade()
        log.info("  [B] VWAP & microstructure ...")
        features["B1_vwap_deviation"]    = self.vwap_deviation()
        features["B2_price_position"]    = self.price_position_in_range()
        features["B3_open_gap"]          = self.intraday_open_gap()
        log.info("  [C] Volume & flow ...")
        features["C1_volume_shock"]      = self.volume_shock()
        features["C2_flow_imbalance"]    = self.order_flow_imbalance()
        features["C3_turnover_shock"]    = self.turnover_shock()
        log.info("  [D] Volatility regime ...")
        features["D1_vol_burst"]         = self.volatility_burst()
        features["D2_vol_zscore"]        = self.volatility_zscore()
        features["D3_dispersion"]        = self.cross_sectional_dispersion()
        log.info("  [E] Residual / factor ...")
        features["E1_residual_return"]   = self.residual_return()
        features["E2_sector_residual"]   = self.sector_relative_return()
        valid_pct = np.mean([f.notna().values.mean()*100 for f in features.values()])
        log.info("Features complete: 15 signals | avg valid=%.1f%%", valid_pct)
        return features

    # ── A: Reversal ──────────────────────────────────────────────────────────

    def bar_reversal(self) -> pd.DataFrame:
        """A1: Single-bar price reversal. Large intrabar move → expect reversion next bar."""
        bar_ret = (self.C - self.O) / self.O.replace(0.0, np.nan)
        atr_p   = atr_pct(self.H, self.L, self.C, self.atr_window)
        raw     = -bar_ret / atr_p.replace(0.0, np.nan)
        return cs_zscore(raw)

    def short_return_reversal(self, k: int = 3) -> pd.DataFrame:
        """A2: 3-bar (15-min) return reversal. Highest IC signal in the model."""
        ret_k  = self.C.pct_change(k)
        sigma  = rolling_mad(ret_k, self.zscore_window)
        z_raw  = -ret_k / sigma.replace(0.0, np.nan)
        rank   = -cs_rank(ret_k)
        return cs_zscore(0.5 * cs_zscore(z_raw) + 0.5 * rank)

    def medium_return_reversal(self, k: int = 13) -> pd.DataFrame:
        """A3: 13-bar (~65-min) return reversal. Captures half-session momentum fade."""
        ret_k  = self.C.pct_change(k)
        sigma  = rolling_mad(ret_k, self.zscore_window)
        z_raw  = -ret_k / sigma.replace(0.0, np.nan)
        return cs_zscore(z_raw.clip(-5, 5))

    def overnight_gap_fade(self) -> pd.DataFrame:
        """A4: Overnight gap fade. Active only in first 4 bars of session (first 20-min)."""
        dates_unique = self._dates.unique()
        prior_close_df = pd.DataFrame(np.nan, index=self.C.index, columns=self.C.columns)
        for i, d in enumerate(dates_unique):
            if i == 0:
                continue
            prev_d    = dates_unique[i - 1]
            prev_mask = self._dates == prev_d
            day_mask  = self._dates == d
            if prev_mask.any():
                pc = self.C[prev_mask].iloc[-1]
                prior_close_df.loc[day_mask] = pc.values
        gap    = self.O / prior_close_df.replace(0.0, np.nan) - 1
        raw    = -cs_zscore(gap)
        active = self._bps <= 3
        gate   = pd.DataFrame(
            np.outer(active.values, np.ones(len(self.C.columns))),
            index=self.C.index, columns=self.C.columns,
        ).astype(bool)
        return raw.where(gate)

    # ── B: VWAP & Microstructure ──────────────────────────────────────────────

    def vwap_deviation(self) -> pd.DataFrame:
        """B1: Intraday VWAP deviation. Weighted by session credibility (sqrt ramp)."""
        vwap   = self._session_vwap()
        dev    = -(self.C - vwap) / vwap.replace(0.0, np.nan)
        weight = np.sqrt(np.clip(self._sfrac.values, 0.01, 1.0)).reshape(-1, 1)
        return cs_zscore(dev * weight)

    def price_position_in_range(self, window: int = 20) -> pd.DataFrame:
        """B2: Price position within 20-bar high-low range. Williams %R style."""
        rng_hi   = self.H.rolling(window, min_periods=5).max()
        rng_lo   = self.L.rolling(window, min_periods=5).min()
        rng_span = (rng_hi - rng_lo).replace(0.0, np.nan)
        pos      = (self.C - rng_lo) / rng_span
        return -cs_zscore(pos)

    def intraday_open_gap(self) -> pd.DataFrame:
        """B3: Current price vs session open. Captures intraday drift reversal."""
        session_open = pd.DataFrame(np.nan, index=self.C.index, columns=self.C.columns)
        for date, grp_idx in self.C.groupby(self._dates).groups.items():
            first_close = self.C.loc[grp_idx].iloc[0]
            session_open.loc[grp_idx] = first_close.values
        drift = (self.C - session_open) / session_open.replace(0.0, np.nan)
        return -cs_zscore(drift)

    # ── C: Volume & Flow ──────────────────────────────────────────────────────

    def volume_shock(self) -> pd.DataFrame:
        """C1: Volume spike vs same-time-of-day baseline (20-day rolling median per slot)."""
        log_v  = np.log1p(self.V)
        result = pd.DataFrame(np.nan, index=self.V.index, columns=self.V.columns)
        min_p  = min(5, max(1, self.volume_window - 1))
        for slot in self._bps.unique():
            mask    = self._bps == slot
            slot_lv = log_v[mask]
            tod_med = slot_lv.rolling(self.volume_window, min_periods=min_p).median().shift(1)
            result[mask] = (slot_lv - tod_med).values
        return cs_zscore(result)

    def order_flow_imbalance(self) -> pd.DataFrame:
        """C2: Lee-Ready OFI approximation. Close near High → sellers overwhelmed → revert."""
        hl     = (self.H - self.L).replace(0.0, np.nan)
        ofi    = (self.C - self.L) / hl - 0.5
        return -cs_zscore(rolling_mean(ofi, 3))

    def turnover_shock(self) -> pd.DataFrame:
        """C3: Dollar turnover shock. Better than pure volume for cross-stock comparisons."""
        log_to = np.log1p(self.V * self.C)
        result = pd.DataFrame(np.nan, index=log_to.index, columns=log_to.columns)
        min_p  = min(5, max(1, self.volume_window - 1))
        for slot in self._bps.unique():
            mask    = self._bps == slot
            slot_lt = log_to[mask]
            tod_med = slot_lt.rolling(self.volume_window, min_periods=min_p).median().shift(1)
            result[mask] = (slot_lt - tod_med).values
        return cs_zscore(result)

    # ── D: Volatility Regime ─────────────────────────────────────────────────

    def volatility_burst(self) -> pd.DataFrame:
        """D1: Directed volatility burst. Down-bar burst → buy; up-bar burst → sell."""
        atr_v   = atr(self.H, self.L, self.C, self.atr_window)
        burst   = (self.H - self.L) / atr_v.replace(0.0, np.nan) - 1
        bar_dir = np.sign(self.O - self.C)
        return cs_zscore(burst * bar_dir)

    def volatility_zscore(self) -> pd.DataFrame:
        """D2: Realised vol z-score. High vol = news regime = reduce reversal exposure."""
        ret    = self.C.pct_change(1)
        rv     = ewm_std(ret, self.halflife)
        rv_mu  = rolling_mean(rv, self.vol_window)
        rv_sig = rolling_std(rv, self.vol_window)
        volz   = (rv - rv_mu) / rv_sig.replace(0.0, np.nan)
        return -cs_zscore(volz)

    def cross_sectional_dispersion(self) -> pd.DataFrame:
        """D3: Cross-sectional dispersion. High = idiosyncratic regime = stronger signals."""
        ret   = self.C.pct_change(1)
        disp  = ret.std(axis=1) / (ret.abs().mean(axis=1) + 1e-8)
        d_ser = disp.to_frame("d")
        disp_z = ((d_ser - rolling_mean(d_ser, 60)) /
                  rolling_std(d_ser, 60).replace(0.0, np.nan)).squeeze().clip(-3, 3)
        return pd.DataFrame(
            np.outer(disp_z.values, np.ones(len(self.C.columns))),
            index=self.C.index, columns=self.C.columns,
        )

    # ── E: Residual / Factor ──────────────────────────────────────────────────

    def residual_return(self, k: int = 3) -> pd.DataFrame:
        """E1: Market-beta-adjusted residual return. Strongest signal in the model."""
        ret_k   = self.C.pct_change(k)
        mkt     = ret_k.median(axis=1)
        mkt_var = mkt.rolling(self.beta_window, min_periods=60).var()
        betas   = pd.DataFrame(index=ret_k.index, columns=ret_k.columns, dtype=float)
        for col in ret_k.columns:
            betas[col] = ret_k[col].rolling(self.beta_window, min_periods=60).cov(mkt) / \
                         mkt_var.replace(0.0, np.nan)
        residual = ret_k.sub(betas.mul(mkt, axis=0), axis=0)
        return -cs_zscore(residual)

    def sector_relative_return(self, k: int = 3) -> pd.DataFrame:
        """E2: Cross-sectionally double-demeaned return. Approximates sector neutralisation."""
        ret_k  = self.C.pct_change(k)
        mkt    = ret_k.median(axis=1)
        resid  = ret_k.sub(mkt, axis=0)
        resid  = resid.sub(resid.mean(axis=1), axis=0)
        return -cs_zscore(resid.reindex(self.C.index))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _session_vwap(self) -> pd.DataFrame:
        pv   = self.C * self.V
        vwap = pd.DataFrame(np.nan, index=self.C.index, columns=self.C.columns)
        for date, grp_idx in pv.groupby(self._dates).groups.items():
            cum_pv = pv.loc[grp_idx].cumsum()
            cum_v  = self.V.loc[grp_idx].cumsum().replace(0.0, np.nan)
            vwap.loc[grp_idx] = (cum_pv / cum_v).values
        return vwap.astype(float)


def load_and_compute(clean_dir, freq="5min", tickers=None,
                     start_date=None, end_date=None, **kwargs):
    """One-shot: load 5-min panels → optionally resample → compute all features."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from polygon_pipeline.pipeline.storage import read_panels
    from features.resampler import Resampler
    panels_5m = read_panels(clean_dir, tickers=tickers,
                             start_date=start_date, end_date=end_date,
                             universe_only=False)
    if freq != "5min":
        panels = Resampler(panels_5m, freq=freq).resample()
    else:
        panels = panels_5m
    features = FeatureEngine(panels, **kwargs).compute_all()
    return panels, features
