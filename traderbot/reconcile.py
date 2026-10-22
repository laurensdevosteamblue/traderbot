"""Compare what the bot actually did (journal + equity) with what the backtest says it should have done
over the same period. Large differences mean a bug, or backtest cost assumptions that are too optimistic."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Config
from .data import load_history, timeframe_seconds
from .live import state_paths
from .overlays import blended_costs, vol_target
from .strategies import get_strategy

MATCH_EARLY = pd.Timedelta(minutes=10)


@dataclass
class Reconciliation:
    trades: pd.DataFrame
    summary: dict
    expected_equity: pd.Series
    actual_equity: pd.Series


def expected_trades(candles: pd.DataFrame, target: pd.Series, start: pd.Timestamp, timeframe: str) -> pd.DataFrame:
    """Trades the strategy implies for a bot that starts flat at `start` and acts at each candle close."""
    tf = pd.Timedelta(seconds=timeframe_seconds(timeframe))
    decisions = pd.Series(target.values, index=candles.index + tf)
    before, after = decisions[decisions.index < start], decisions[decisions.index >= start]
    rows, state = [], 0.0
    if len(before) and before.iloc[-1] > 0:
        rows.append({"time": start, "side": "buy", "price": candles.close.iloc[len(before) - 1]})
        state = before.iloc[-1]
    for time, value in after.items():
        if (value > 0) != (state > 0):  # entries and exits only (sizes may be fractional)
            price = candles.open.get(time, candles.close.get(time - tf))
            rows.append({"time": time, "side": "buy" if value > 0 else "sell", "price": price})
            state = value
    return pd.DataFrame(rows, columns=["time", "side", "price"])


def match_trades(expected: pd.DataFrame, actual: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Pair expected and actual trades (same side, actual within [-10 min, +1 candle] of expected)."""
    tf = pd.Timedelta(seconds=timeframe_seconds(timeframe))
    used, rows = set(), []
    for _, e in expected.iterrows():
        hit = next((i for i, a in actual.iterrows() if i not in used and a.side == e.side
                    and e.time - MATCH_EARLY <= a.time <= e.time + tf), None)
        if hit is None:
            rows.append({"status": "missed", "side": e.side, "expected_time": e.time, "expected_price": e.price})
            continue
        used.add(hit)
        a = actual.loc[hit]
        sign = 1 if e.side == "buy" else -1
        rows.append({"status": "matched", "side": e.side, "expected_time": e.time, "expected_price": e.price,
                     "actual_time": a.time, "actual_price": a.price,
                     "cost_bps": sign * (a.price - e.price) / e.price * 1e4})
    for i, a in actual.iterrows():
        if i not in used:
            rows.append({"status": "extra", "side": a.side, "actual_time": a.time, "actual_price": a.price})
    cols = ["status", "side", "expected_time", "expected_price", "actual_time", "actual_price", "cost_bps"]
    return pd.DataFrame(rows, columns=cols)


def reconcile(candles: pd.DataFrame, target: pd.Series, journal: pd.DataFrame, equity_log: pd.DataFrame,
              start: pd.Timestamp, start_equity: float, timeframe: str, fee: float, slippage: float,
              exposure: float, max_position_quote: float = 0.0) -> Reconciliation:
    tf = pd.Timedelta(seconds=timeframe_seconds(timeframe))
    actual = journal[journal.time >= start - MATCH_EARLY].reset_index(drop=True)
    trades = match_trades(expected_trades(candles, target, start, timeframe), actual, timeframe)

    decided_before = int((candles.index + tf <= start).sum())  # include the candle whose close drove the first step
    session = candles.iloc[max(decided_before - 1, 0):]
    sig = target.loc[session.index]
    bt = run_backtest(session, sig, timeframe, fee, slippage, exposure, initial=start_equity)
    if max_position_quote > 0 and bt.equity.max() * exposure > max_position_quote:
        entries = (sig > 0) & (sig.shift(1, fill_value=0.0) <= 0)
        cap_scale = (max_position_quote / (bt.equity * exposure)).clip(upper=1.0).where(entries).ffill().fillna(1.0)
        bt = run_backtest(session, sig * cap_scale, timeframe, fee, slippage, exposure, initial=start_equity)
    expected_equity = bt.equity.set_axis(bt.equity.index + tf)  # equity known at each candle close
    actual_equity = equity_log.set_index("time")["equity"] if len(equity_log) else pd.Series(dtype=float)
    end_equity = actual_equity.iloc[-1] if len(actual_equity) else start_equity
    matched = trades[trades.status == "matched"]
    summary = {
        "start": start,
        "days": (pd.Timestamp.now(tz="UTC") - start).total_seconds() / 86400,
        "expected_trades": int((trades.status != "extra").sum()),
        "actual_trades": len(actual),
        "matched": len(matched),
        "missed": int((trades.status == "missed").sum()),
        "extra": int((trades.status == "extra").sum()),
        "avg_cost_vs_expected_bps": float(matched.cost_bps.mean()) if len(matched) else np.nan,
        "assumed_cost_bps": slippage * 1e4,
        "limit_fill_rate": float(pd.to_numeric(actual.get("maker_share"), errors="coerce").mean())
        if len(actual) and "maker_share" in actual else np.nan,
        "expected_return": bt.stats["total_return"],
        "actual_return": end_equity / start_equity - 1,
    }
    return Reconciliation(trades, summary, expected_equity, actual_equity)


def reconcile_session(cfg: Config, exchange) -> Reconciliation:
    """Load the session files written by the runner and reconcile them against the configured strategy."""
    journal_path, session_path, _, equity_path = state_paths(cfg)
    journal = pd.read_csv(journal_path, parse_dates=["time"]) if journal_path.exists() else \
        pd.DataFrame(columns=["time", "side", "price", "cost", "equity_before"])
    equity_log = pd.read_csv(equity_path, parse_dates=["time"]) if equity_path.exists() else pd.DataFrame(
        columns=["time", "equity"])
    session = json.loads(session_path.read_text()) if session_path.exists() else {}
    starts = [pd.Timestamp(session["started"])] if session else []
    if len(journal):
        starts.append(journal.time.min())
    if not starts:
        raise FileNotFoundError(f"no session found in {cfg.state_dir}; run `python -m traderbot run` first")
    start = min(starts)
    if len(journal) and journal.time.min() < pd.Timestamp(session.get("started", start)):
        start_equity = float(journal.equity_before.iloc[0])
    else:
        start_equity = float(session.get("start_equity", cfg.budget_quote))

    exposure = (min(cfg.position_fraction, cfg.max_position_quote / start_equity)
                if cfg.max_position_quote > 0 else cfg.position_fraction)

    tf = pd.Timedelta(seconds=timeframe_seconds(cfg.timeframe))
    since = (start - 1100 * tf).strftime("%Y-%m-%d %H:%M")
    candles = load_history(exchange, cfg.symbol, cfg.timeframe, cfg.data_dir, since)
    target = get_strategy(cfg.strategy)(candles, **cfg.params)
    if cfg.vol_target > 0:
        target = vol_target(candles, target, cfg.timeframe, cfg.vol_target, cfg.vol_lookback_days)
    # Same cost assumption as the research: half of the limit orders fill, the rest falls back to market.
    fee, slippage = blended_costs(0.5 if cfg.order_type == "limit" else 0.0, cfg.fee, cfg.maker_fee, cfg.slippage)
    return reconcile(candles, target, journal, equity_log, start, start_equity, cfg.timeframe, fee, slippage,
                     exposure, cfg.max_position_quote)
