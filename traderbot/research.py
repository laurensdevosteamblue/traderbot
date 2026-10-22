"""Compare improvement ideas on top of the walk-forward (fully out-of-sample) track record of every bot."""
from __future__ import annotations

import pandas as pd

from .backtest import compute_stats, periods_per_year, run_backtest, walk_forward
from .config import Config
from .data import load_history, make_exchange
from .overlays import blended_costs, regime_filter, trend_regime, vol_target

# name -> (share of orders filled as maker, vol estimator or None, BTC trend filter for altcoins)
VARIANTS = {
    "base (market orders)": (0.0, None, False),
    "limit orders, 50% filled": (0.5, None, False),
    "limit orders, 100% filled": (1.0, None, False),
    "vol sizing (close-to-close)": (0.0, "close", False),
    "vol sizing (OHLC Rogers-Satchell)": (0.0, "rogers_satchell", False),
    "vol (OHLC) + limit 50% (live)": (0.5, "rogers_satchell", False),
    "BTC trend filter": (0.0, None, True),
    "vol + BTC + limit 50%": (0.5, "rogers_satchell", True),
}


def _stats(equity: pd.Series, timeframe: str) -> dict:
    rets = equity.pct_change().fillna(0.0)
    s = compute_stats(equity, rets, pd.Series(1.0, index=equity.index), periods_per_year(timeframe), equity.iloc[0])
    s["calmar"] = s["cagr"] / abs(s["max_drawdown"]) if s["max_drawdown"] < 0 else float("nan")
    return {k: s[k] for k in ("total_return", "cagr", "max_drawdown", "sharpe", "calmar")}


def overlay_study(bots: list[Config], since: str = "2019-01-01", annual_vol: float = 0.5,
                  regime_sma_days: int = 50) -> dict[str, pd.DataFrame]:
    cfg = bots[0]
    tf, strategy = cfg.timeframe, cfg.strategy
    exchange = make_exchange(cfg.exchange)
    candles = {b.name: load_history(exchange, b.symbol, tf, cfg.data_dir, since) for b in bots}
    btc = next((b.name for b in bots if b.symbol.startswith("BTC/")), None)
    regime = trend_regime(candles[btc], regime_sma_days, tf) if btc else None

    # One walk-forward per coin at market-order costs: every variant trades the same signals, so differences
    # come only from the variant itself (costs, sizing, filter) and not from re-picked params.
    wf = {b.name: walk_forward(candles[b.name], tf, cfg.fee, cfg.slippage).targets[strategy] for b in bots}

    targets, costs = {}, {}
    for variant, (maker, vol_est, use_btc) in VARIANTS.items():
        costs[variant] = blended_costs(maker, cfg.fee, taker_slippage=cfg.slippage)
        for b in bots:
            t = wf[b.name]
            df = candles[b.name].loc[t.index]
            if vol_est:
                t = vol_target(candles[b.name], t, tf, annual_vol, estimator=vol_est).loc[t.index]
            if use_btc and regime is not None and b.name != btc:
                t = regime_filter(t, regime)
            targets[(variant, b.name)] = (df, t)

    per_coin = pd.DataFrame([
        {"variant": v, "bot": b.name,
         **{k: r.stats[k] for k in ("total_return", "max_drawdown", "sharpe", "trades")}}
        for v in VARIANTS for b in bots
        for r in [run_backtest(*targets[(v, b.name)], tf, *costs[v], initial=b.budget_quote)]])

    starts = {b.name: wf[b.name].index[0] for b in bots}
    earliest = min(starts.values())
    groups = {"all bots": [b.name for b in bots],
              "longest history": [n for n, s in starts.items() if s <= earliest + pd.Timedelta(days=180)]}
    out = {"per_coin": per_coin}
    for label, names in groups.items():
        start = max(starts[n] for n in names)
        rows = []
        for v in VARIANTS:
            curves = [run_backtest(df.loc[start:], t.loc[start:], tf, *costs[v],
                                   initial=next(b.budget_quote for b in bots if b.name == n)).equity
                      for n in names for df, t in [targets[(v, n)]]]
            equity = pd.concat(curves, axis=1).ffill().dropna().sum(axis=1)  # tolerate missing candles
            rows.append({"variant": v, **_stats(equity, tf)})
        table = pd.DataFrame(rows).set_index("variant")
        table.attrs.update(bots=names, start=start, end=max(df.index[-1] for df, _ in targets.values()))
        out[label] = table
    return out
