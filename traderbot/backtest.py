from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import timeframe_seconds
from .strategies import STRATEGIES


@dataclass
class Result:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    stats: dict


def periods_per_year(timeframe: str) -> float:
    return 365 * 86400 / timeframe_seconds(timeframe)


def run_backtest(df: pd.DataFrame, target: pd.Series, timeframe: str, fee: float, slippage: float,
                 exposure: float = 1.0, initial: float = 1000.0) -> Result:
    """Signal on bar t's close is executed at bar t+1's open; position is held open-to-open."""
    pos = target.reindex(df.index).shift(1).fillna(0.0).clip(0, 1) * exposure
    bar_ret = (df.open.shift(-1) / df.open - 1).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    rets = pos * bar_ret - turnover * (fee + slippage)
    equity = initial * (1 + rets).cumprod()
    return Result(equity, rets, pos, compute_stats(equity, rets, pos, periods_per_year(timeframe), initial))


def _round_trips(equity: pd.Series, pos: pd.Series, initial: float) -> list[float]:
    eq_before = equity.shift(1).fillna(initial)
    in_pos = pos > 0
    starts = np.flatnonzero(in_pos & ~in_pos.shift(1, fill_value=False))
    ends = np.flatnonzero(~in_pos & in_pos.shift(1, fill_value=False))
    return [equity.iloc[ends[i] if i < len(ends) else -1] / eq_before.iloc[s] - 1 for i, s in enumerate(starts)]


def compute_stats(equity: pd.Series, rets: pd.Series, pos: pd.Series, ppy: float, initial: float) -> dict:
    total = equity.iloc[-1] / initial - 1
    years = len(rets) / ppy
    std = rets.std()
    trades = _round_trips(equity, pos, initial)
    return {
        "total_return": total,
        "cagr": (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0,
        "max_drawdown": (equity / equity.cummax() - 1).min(),
        "sharpe": rets.mean() / std * math.sqrt(ppy) if std > 0 else 0.0,
        "trades": len(trades),
        "win_rate": float(np.mean([t > 0 for t in trades])) if trades else 0.0,
        "exposure": float((pos > 0).mean()),
        "final_equity": equity.iloc[-1],
    }


def backtest_strategy(df: pd.DataFrame, name: str, params: dict, timeframe: str, fee: float,
                      slippage: float, signal_df: pd.DataFrame | None = None) -> Result:
    """`signal_df` lets indicators warm up on earlier data while only `df` is traded."""
    func = STRATEGIES[name][0]
    target = func(signal_df if signal_df is not None else df, **params)
    return run_backtest(df, target, timeframe, fee, slippage)


def buy_and_hold(df: pd.DataFrame, timeframe: str, fee: float, slippage: float) -> Result:
    return run_backtest(df, pd.Series(1.0, index=df.index), timeframe, fee, slippage)


def compare(df: pd.DataFrame, timeframe: str, fee: float, slippage: float, split: float = 0.7,
            min_trades: int = 5) -> pd.DataFrame:
    """Optimise each strategy's params on the in-sample part (by Sharpe), then report how the
    chosen params do on the unseen out-of-sample part. Only the OOS columns are evidence."""
    cut = int(len(df) * split)
    is_df, oos_df = df.iloc[:cut], df.iloc[cut:]
    rows = []
    for name, (_, grid) in STRATEGIES.items():
        best, oos_sharpes = None, []
        for params in grid:
            is_res = backtest_strategy(is_df, name, params, timeframe, fee, slippage)
            oos_res = backtest_strategy(oos_df, name, params, timeframe, fee, slippage, signal_df=df)
            oos_sharpes.append(oos_res.stats["sharpe"])
            if is_res.stats["trades"] >= min_trades and (best is None or is_res.stats["sharpe"] > best[1].stats["sharpe"]):
                best = (params, is_res, oos_res)
        if best is None:
            continue
        params, is_res, oos_res = best
        rows.append(_row(name, params, is_res, oos_res, float(np.median(oos_sharpes)),
                         float(np.mean([s > 0 for s in oos_sharpes]))))
    rows.append(_row("buy_and_hold", {}, buy_and_hold(is_df, timeframe, fee, slippage),
                     buy_and_hold(oos_df, timeframe, fee, slippage), None, None))
    out = pd.DataFrame(rows).set_index("strategy")
    out.attrs["in_sample"] = (is_df.index[0], is_df.index[-1])
    out.attrs["out_of_sample"] = (oos_df.index[0], oos_df.index[-1])
    return out


@dataclass
class WalkForward:
    summary: pd.DataFrame       # one row per strategy, stitched out-of-sample performance
    windows: pd.DataFrame       # one row per (window, strategy): chosen params and OOS return
    equity: dict[str, pd.Series]
    targets: dict[str, pd.Series]  # stitched out-of-sample target positions per strategy


def walk_forward(df: pd.DataFrame, timeframe: str, fee: float, slippage: float, train_days: int = 730,
                 test_days: int = 180, min_trades: int = 5) -> WalkForward:
    """Rolling re-optimisation: tune on `train_days`, trade the next `test_days` with those params,
    roll forward by `test_days`. The stitched test periods form one fully out-of-sample track record."""
    bars_per_day = 86400 / timeframe_seconds(timeframe)
    train, test = int(train_days * bars_per_day), int(test_days * bars_per_day)
    if len(df) < train + test:
        raise ValueError(f"need at least {train_days + test_days} days of data, have {len(df) / bars_per_day:.0f}")
    bounds = [(s, s + train, s + train + test) for s in range(0, len(df) - train - test + 1, test)]
    bounds[-1] = (bounds[-1][0], bounds[-1][1], len(df))  # last window absorbs the leftover bars
    oos_index = df.index[bounds[0][1]:bounds[-1][2]]
    oos_df = df.loc[oos_index]

    rows, summary, equity, targets = [], [], {}, {}
    for name, (func, grid) in STRATEGIES.items():
        signals = [func(df, **params) for params in grid]  # rolling indicators: no look-ahead
        stitched = []
        for w, (s, m, e) in enumerate(bounds):
            train_df, test_df = df.iloc[s:m], df.iloc[m:e]
            scored = [(run_backtest(train_df, sig.iloc[s:m], timeframe, fee, slippage).stats, i)
                      for i, sig in enumerate(signals)]
            valid = [(st["sharpe"], i) for st, i in scored if st["trades"] >= min_trades]
            best = max(valid)[1] if valid else max((st["sharpe"], i) for st, i in scored)[1]
            stitched.append(signals[best].iloc[m:e])
            test_res = run_backtest(test_df, signals[best].iloc[m:e], timeframe, fee, slippage)
            rows.append({"window": w, "test_start": test_df.index[0], "test_end": test_df.index[-1],
                         "strategy": name, "params": grid[best],
                         "test_return": test_res.stats["total_return"]})
        targets[name] = pd.concat(stitched)
        res = run_backtest(oos_df, targets[name], timeframe, fee, slippage)
        equity[name] = res.equity
        summary.append({"strategy": name, **res.stats})

    bh = buy_and_hold(oos_df, timeframe, fee, slippage)
    equity["buy_and_hold"] = bh.equity
    summary.append({"strategy": "buy_and_hold", **bh.stats})
    for w, (s, m, e) in enumerate(bounds):
        test_df = df.iloc[m:e]
        rows.append({"window": w, "test_start": test_df.index[0], "test_end": test_df.index[-1],
                     "strategy": "buy_and_hold", "params": {},
                     "test_return": test_df.open.iloc[-1] / test_df.open.iloc[0] - 1})

    windows = pd.DataFrame(rows)
    out = pd.DataFrame(summary).set_index("strategy")
    out["pct_windows_positive"] = windows.groupby("strategy")["test_return"].apply(lambda r: (r > 0).mean())
    out.attrs["out_of_sample"] = (oos_index[0], oos_index[-1])
    out.attrs["n_windows"] = len(bounds)
    return WalkForward(out, windows, equity, targets)


def _row(name, params, is_res, oos_res, grid_median_sharpe, grid_pct_positive) -> dict:
    return {
        "strategy": name,
        "params": params,
        "is_return": is_res.stats["total_return"],
        "is_sharpe": is_res.stats["sharpe"],
        "oos_return": oos_res.stats["total_return"],
        "oos_cagr": oos_res.stats["cagr"],
        "oos_max_dd": oos_res.stats["max_drawdown"],
        "oos_sharpe": oos_res.stats["sharpe"],
        "oos_trades": oos_res.stats["trades"],
        "oos_win_rate": oos_res.stats["win_rate"],
        "grid_oos_median_sharpe": grid_median_sharpe,
        "grid_oos_pct_sharpe_pos": grid_pct_positive,
    }
