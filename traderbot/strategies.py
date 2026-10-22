"""Long-only spot strategies.

Each strategy takes an OHLCV DataFrame and returns a target position series (0.0 = flat,
1.0 = long) computed from the CLOSE of each bar. The backtester and live runner act on it at
the next bar, so a strategy must only use data up to and including the current bar.
"""
from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def _hold(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """Go long on entry, stay long until exit."""
    sig = pd.Series(np.nan, index=entry.index)
    sig[exit_] = 0.0
    sig[entry] = 1.0
    return sig.ffill().fillna(0.0)


def rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + gain / loss)


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Trend-following: long while the fast moving average is above the slow one."""
    return (df.close.rolling(fast).mean() > df.close.rolling(slow).mean()).astype(float)


def donchian_breakout(df: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
    """Trend-following ("turtle"): buy a new N-bar high, sell a new M-bar low."""
    upper = df.high.rolling(entry).max().shift(1)
    lower = df.low.rolling(exit).min().shift(1)
    return _hold(df.close > upper, df.close < lower)


def rsi_reversion(df: pd.DataFrame, period: int = 14, buy: float = 30, sell: float = 55,
                  trend: int = 0) -> pd.Series:
    """Mean reversion: buy oversold RSI, sell when it recovers. Optional SMA trend filter."""
    r = rsi(df.close, period)
    entry = r < buy
    if trend:
        entry &= df.close > df.close.rolling(trend).mean()
    return _hold(entry, r > sell)


def _grid(**axes) -> list[dict]:
    return [dict(zip(axes, values)) for values in product(*axes.values())]


STRATEGIES = {
    "sma_cross": (sma_cross, [p for p in _grid(fast=[10, 20, 50], slow=[50, 100, 200])
                              if p["fast"] < p["slow"]]),
    "donchian_breakout": (donchian_breakout, [p for p in _grid(entry=[20, 55, 100], exit=[10, 20, 50])
                                              if p["exit"] < p["entry"]]),
    "rsi_reversion": (rsi_reversion, _grid(period=[7, 14], buy=[25, 30, 35], sell=[50, 60, 70],
                                           trend=[0, 200])),
}


def get_strategy(name: str):
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}")
    return STRATEGIES[name][0]
