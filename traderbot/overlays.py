"""Position-sizing and filter overlays applied on top of a strategy's 0/1 target series.

Like strategies, overlays only use data up to the current bar. Their parameters are fixed up front
(not optimised), so they do not add extra overfitting on top of the walk-forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import periods_per_year
from .data import timeframe_seconds


def realized_vol(df: pd.DataFrame, timeframe: str, lookback_days: int = 30,
                 estimator: str = "rogers_satchell") -> pd.Series:
    """Annualised volatility over the last `lookback_days`.

    Defaults to the drift-independent Rogers-Satchell OHLC estimator so a strong upside trend does
    not inflate estimated volatility and shrink the next breakout entry; `estimator="close"` uses
    standard close-to-close log returns.
    """
    bars = int(lookback_days * 86400 / timeframe_seconds(timeframe))
    ppy = periods_per_year(timeframe)
    if estimator == "close":
        return np.log(df.close).diff().rolling(bars).std() * np.sqrt(ppy)
    rs = (np.log(df.high / df.close) * np.log(df.high / df.open)
          + np.log(df.low / df.close) * np.log(df.low / df.open))
    return np.sqrt(rs.clip(lower=0.0).rolling(bars).mean() * ppy)


def size_at_entry(target: pd.Series, weight: pd.Series) -> pd.Series:
    """Scale each trade by the weight known at its entry bar and keep that size until the exit
    (no rebalancing while in a trade, which would cost fees)."""
    target = target.fillna(0.0)
    entries = (target > 0) & (target.shift(1, fill_value=0.0) <= 0)
    w = weight.reindex(target.index).where(entries).ffill().fillna(0.0)
    return target * w


def vol_target(df: pd.DataFrame, target: pd.Series, timeframe: str, annual_vol: float = 0.5,
               lookback_days: int = 30, estimator: str = "rogers_satchell") -> pd.Series:
    """Invest less in coins that are currently wild: weight = annual_vol / realised vol, capped at 1."""
    weight = (annual_vol / realized_vol(df, timeframe, lookback_days, estimator)).clip(upper=1.0).fillna(0.0)
    return size_at_entry(target, weight)


def trend_regime(df: pd.DataFrame, sma_days: int = 50, timeframe: str = "4h") -> pd.Series:
    """True while the close is above its `sma_days` moving average (an uptrend)."""
    bars = int(sma_days * 86400 / timeframe_seconds(timeframe))
    return df.close > df.close.rolling(bars).mean()


def regime_filter(target: pd.Series, regime: pd.Series) -> pd.Series:
    """Only be in a position while the regime is on (e.g. BTC in an uptrend); exit when it turns off."""
    return target * regime.reindex(target.index).ffill().fillna(False).astype(float)


def blended_costs(maker_share: float, taker_fee: float = 0.0025, maker_fee: float = 0.0015,
                  taker_slippage: float = 0.0005) -> tuple[float, float]:
    """(fee, slippage) per side when `maker_share` of orders fill as post-only limit orders (maker fee,
    no slippage) and the rest fall back to market orders (taker fee + slippage)."""
    return (maker_share * maker_fee + (1 - maker_share) * taker_fee, (1 - maker_share) * taker_slippage)
