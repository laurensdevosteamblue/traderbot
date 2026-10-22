"""Read-only dashboard. Start with `python -m traderbot dashboard` (binds to 127.0.0.1 only).

Live data comes from the files the runner writes in state/, so the dashboard never needs API keys and
cannot place orders. Backtests use public exchange candles (cached in data/).
"""
from __future__ import annotations

import inspect
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # streamlit runs this file as a script, not as part of the package

from traderbot.backtest import backtest_strategy, buy_and_hold, compare, walk_forward  # noqa: E402
from traderbot.config import MODES, Config, load_bots, load_config  # noqa: E402
from traderbot.data import load_history, make_exchange, timeframe_seconds  # noqa: E402
from traderbot.live import state_paths  # noqa: E402
from traderbot.reconcile import reconcile_session  # noqa: E402
from traderbot.strategies import STRATEGIES  # noqa: E402

PCT = st.column_config.NumberColumn(format="percent")
NUM = st.column_config.NumberColumn(format="%.2f")


def config_path(config_file: str) -> Path:
    return Path(config_file) if Path(config_file).is_absolute() else ROOT / config_file


def _resolve(cfg: Config, mode: str) -> Config:
    resolve = lambda p: p if p.is_absolute() else ROOT / p  # noqa: E731
    return replace(cfg, mode=mode, state_dir=resolve(cfg.state_dir), data_dir=resolve(cfg.data_dir))


def get_config(config_file: str, mode: str, bot: str | None = None) -> Config:
    return _resolve(load_config(config_path(config_file), bot), mode)


def all_configs(config_file: str, mode: str) -> list[Config]:
    return [_resolve(c, mode) for c in load_bots(config_path(config_file))]


def fmt_price(p: float) -> str:
    return f"{p:,.2f}" if p >= 100 else f"{p:,.4f}"


@st.cache_data(ttl=3600, show_spinner="Loading candles...")
def candles(exchange: str, symbol: str, timeframe: str, since: str, data_dir: str) -> pd.DataFrame:
    return load_history(make_exchange(exchange), symbol, timeframe, Path(data_dir), since)


@st.cache_data(ttl=3600, show_spinner="Running in-sample / out-of-sample comparison...")
def cached_compare(exchange, symbol, timeframe, since, data_dir, fee, slippage, split):
    return compare(candles(exchange, symbol, timeframe, since, data_dir), timeframe, fee, slippage, split)


@st.cache_data(ttl=3600, show_spinner="Running walk-forward test...")
def cached_walk_forward(exchange, symbol, timeframe, since, data_dir, fee, slippage, train_days, test_days):
    return walk_forward(candles(exchange, symbol, timeframe, since, data_dir), timeframe, fee, slippage,
                        train_days, test_days)


@st.cache_data(ttl=300, show_spinner="Reconciling with backtest...")
def cached_reconcile(config_file: str, mode: str, bot: str, _stamp: float):
    cfg = get_config(config_file, mode, bot)
    return reconcile_session(cfg, make_exchange(cfg.exchange))


def read_csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def pct_table(df: pd.DataFrame, pct_cols) -> None:
    cfg = {c: PCT for c in pct_cols if c in df.columns} | {c: NUM for c in ("sharpe", "is_sharpe", "oos_sharpe",
                                                                           "grid_oos_median_sharpe")}
    df = df.astype({"params": str}) if "params" in df else df.copy()
    df.attrs = {}
    st.dataframe(df, column_config=cfg, width="stretch")


def equity_chart(curves: dict[str, pd.Series], log_y: bool = False, drawdown: bool = True) -> go.Figure:
    fig = make_subplots(rows=2 if drawdown else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3] if drawdown else [1], vertical_spacing=0.04)
    for name, eq in curves.items():
        dash = "dot" if name == "buy_and_hold" else None
        fig.add_scatter(x=eq.index, y=eq, name=name, line={"dash": dash}, row=1, col=1)
        if drawdown:
            fig.add_scatter(x=eq.index, y=eq / eq.cummax() - 1, name=f"{name} DD", showlegend=False,
                            line={"dash": dash, "width": 1}, row=2, col=1)
    if log_y:
        fig.update_yaxes(type="log", row=1, col=1)
    if drawdown:
        fig.update_yaxes(tickformat=".0%", title="drawdown", row=2, col=1)
    fig.update_layout(height=520 if drawdown else 380, margin={"t": 30, "b": 10}, legend={"orientation": "h"})
    return fig


# ---------------------------------------------------------------------------------------------------------------
st.set_page_config(page_title="traderbot", layout="wide")
st.sidebar.title("traderbot")
config_file = st.sidebar.text_input("Config file", "config.toml")
default_mode = load_config(config_path(config_file)).mode
mode = st.sidebar.selectbox("Mode", MODES, index=MODES.index(default_mode))
bots = all_configs(config_file, mode)
bot_name = st.sidebar.selectbox("Bot (for the Live / Paper vs backtest tabs)", [b.name for b in bots],
                                format_func=lambda n: next(f"{b.name}: {b.symbol}" for b in bots if b.name == n))
cfg = next(b for b in bots if b.name == bot_name)
st.sidebar.caption(f"{cfg.exchange} {cfg.symbol} {cfg.timeframe} | {cfg.strategy} {cfg.params}")
st.sidebar.caption(f"budget {cfg.budget_quote:,.0f} {cfg.symbol.split('/')[1]} per bot (other account funds are ignored)")
st.sidebar.caption(f"fee {cfg.fee:.3%} + slippage {cfg.slippage:.3%} per side")
st.sidebar.info("Read-only. The dashboard cannot place orders.")

overview_tab, live_tab, rec_tab, bt_tab, cmp_tab, wf_tab = st.tabs(
    ["Overview", "Live", "Paper vs backtest", "Backtest", "Compare", "Walk-forward"])


@st.fragment(run_every="60s")
def render_overview(bots: list[Config]) -> None:
    rows, curves = [], {}
    now = pd.Timestamp.now(tz="UTC")
    for b in bots:
        _, session_p, status_p, equity_p = state_paths(b)
        if not status_p.exists():
            rows.append({"bot": b.name, "symbol": b.symbol, "status": "not started"})
            continue
        s = json.loads(status_p.read_text())
        start = (json.loads(session_p.read_text()) if session_p.exists() else {}).get("start_equity") or b.budget_quote
        age = now - pd.Timestamp(s["time"])
        stale = age > pd.Timedelta(seconds=timeframe_seconds(b.timeframe)) + pd.Timedelta(minutes=10)
        rows.append({
            "bot": b.name, "symbol": b.symbol,
            "status": "KILL SWITCH" if s.get("kill_switch") else ("STALE" if stale else "ok"),
            "signal": {1.0: "LONG", 0.0: "FLAT"}.get(s["target"], "-"), "last_action": s["action"],
            "price": fmt_price(s["price"]), "position": s["base"] * s["price"], "equity": s["equity"],
            "start": start, "return": s["equity"] / start - 1,
            "updated_min_ago": round(age.total_seconds() / 60),
        })
        eq = read_csv(equity_p, parse_dates=["time"])
        if eq is not None and len(eq):
            curves[b.name] = eq.set_index("time").equity
    table = pd.DataFrame(rows)
    if "equity" not in table:
        st.info("No bots have run yet. Start them with `python -m traderbot run`.")
        return
    running = table.dropna(subset=["equity"])
    total, start = running.equity.sum(), running.start.sum()
    c = st.columns(4)
    c[0].metric("Total equity (all bots)", f"{total:,.2f}", f"{total / start - 1:+.2%} since start")
    c[1].metric("Started with", f"{start:,.2f}")
    c[2].metric("Bots in a position", f"{(running.signal == 'LONG').sum()} / {len(running)}")
    c[3].metric("Problems", int((table.status != "ok").sum()), delta_color="off")
    for _, r in table[table.status != "ok"].iterrows():
        st.warning(f"{r.bot} ({r.symbol}): {r.status}")
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"return": PCT, "equity": NUM, "position": NUM, "start": NUM})
    if curves:
        fig = go.Figure()
        for name, eq in curves.items():
            fig.add_scatter(x=eq.index, y=eq / eq.iloc[0] - 1, name=name)
        fig.update_layout(height=380, margin={"t": 30, "b": 10}, legend={"orientation": "h"},
                          yaxis={"tickformat": ".1%", "title": "return per bot"})
        st.plotly_chart(fig, width="stretch")
    st.caption("Each bot trades its own budget independently. Compare bots over months, not days: with ~1 trade per "
               "month per coin, short-term differences are mostly luck.")


with overview_tab:
    render_overview(bots)


@st.fragment(run_every="60s")
def render_live(cfg: Config) -> None:
    journal_p, session_p, status_p, equity_p = state_paths(cfg)
    if not status_p.exists():
        st.info(f"No status for bot `{cfg.name}` in mode `{cfg.mode}` yet. Start it with `python -m traderbot run`.")
        return
    status = json.loads(status_p.read_text())
    session = json.loads(session_p.read_text()) if session_p.exists() else {}
    tf = pd.Timedelta(seconds=timeframe_seconds(status["timeframe"]))
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(status["time"])
    age_txt = f"{age.total_seconds() / 60:.0f} min ago"

    if status.get("kill_switch"):
        st.error(f"KILL SWITCH ACTIVE: {status['kill_switch']}. No orders are being placed.")
    if age > tf + pd.Timedelta(minutes=10):
        st.warning(f"Last update {age_txt}: expected one every {tf}. The bot may not be running.")
    else:
        st.success(f"Bot alive, last update {age_txt}. {status['strategy']} {status['params']} on "
                   f"{status['symbol']} {status['timeframe']}")

    equity, price, base = status["equity"], status["price"], status["base"]
    start_eq = session.get("start_equity") or cfg.budget_quote
    risk = status["risk"]
    asset = status["symbol"].split("/")[0]
    c = st.columns(5)
    c[0].metric("Equity", f"{equity:,.2f}", f"{equity / start_eq - 1:+.2%} since start")
    c[1].metric("Position value", f"{base * price:,.2f}", f"{base:.6f} {asset}", delta_color="off")
    c[2].metric(f"{status['symbol']} price", fmt_price(price))
    signal = {1.0: "LONG", 0.0: "FLAT"}.get(status["target"], "-")
    c[3].metric("Signal / last action", signal, status["action"], delta_color="off")
    c[4].metric("Trades today", f"{risk['trades_today']} / {cfg.max_trades_per_day}")

    day_pnl = equity / risk["day_start_equity"] - 1 if risk["day_start_equity"] else 0.0
    dd = equity / risk["peak_equity"] - 1 if risk["peak_equity"] else 0.0
    r1, r2 = st.columns(2)
    r1.caption(f"Today's P&L {day_pnl:+.2%} (buys blocked at -{cfg.daily_loss_limit_pct:.0%})")
    r1.progress(min(max(-day_pnl, 0) / cfg.daily_loss_limit_pct, 1.0))
    r2.caption(f"Drawdown from peak {dd:.2%} (flatten + halt at -{cfg.max_drawdown_pct:.0%})")
    r2.progress(min(max(-dd, 0) / cfg.max_drawdown_pct, 1.0))

    eq = read_csv(equity_p, parse_dates=["time"])
    journal = read_csv(journal_p, parse_dates=["time"])
    if eq is not None and len(eq):
        fig = go.Figure()
        fig.add_scatter(x=eq.time, y=eq.equity, name="bot equity")
        fig.add_scatter(x=eq.time, y=start_eq * eq.price / eq.price.iloc[0], name="buy & hold",
                        line={"dash": "dot"})
        if journal is not None:
            for side, symbol, color in (("buy", "triangle-up", "green"), ("sell", "triangle-down", "red")):
                t = journal[journal.side == side]
                fig.add_scatter(x=t.time, y=t.equity_before, mode="markers", name=side,
                                marker={"symbol": symbol, "size": 12, "color": color})
        fig.update_layout(height=380, margin={"t": 30, "b": 10}, legend={"orientation": "h"})
        st.plotly_chart(fig, width="stretch")
    st.subheader("Trades")
    if journal is not None and len(journal):
        st.dataframe(journal.sort_values("time", ascending=False), width="stretch", hide_index=True)
    else:
        st.caption("No trades yet.")


with live_tab:
    render_live(cfg)

with rec_tab:
    st.caption("What the bot actually did compared with what the backtest says it should have done over the same "
               "period. Missed or extra trades mean a bug. Fill costs well above the assumed slippage mean the "
               "backtest is too optimistic.")
    journal_p = state_paths(cfg)[0]
    try:
        stamp = journal_p.stat().st_mtime if journal_p.exists() else 0.0
        rec = cached_reconcile(config_file, mode, bot_name, stamp)
    except FileNotFoundError as e:
        st.info(str(e))
    else:
        s = rec.summary
        c = st.columns(6)
        c[0].metric("Session", f"{s['days']:.1f} days", f"since {s['start']:%Y-%m-%d}", delta_color="off")
        c[1].metric("Trades matched", f"{s['matched']} / {s['expected_trades']}",
                    f"{s['missed']} missed, {s['extra']} extra", delta_color="off")
        cost = s["avg_cost_vs_expected_bps"]
        c[2].metric("Fill cost vs backtest", "-" if pd.isna(cost) else f"{cost:.1f} bps",
                    f"assumed {s['assumed_cost_bps']:.1f} bps", delta_color="off")
        rate = s["limit_fill_rate"]
        c[3].metric("Limit orders filled", "-" if pd.isna(rate) else f"{rate:.0%}",
                    "assumed 50%" if cfg.order_type == "limit" else "market orders only", delta_color="off")
        c[4].metric("Expected return", f"{s['expected_return']:+.2%}")
        c[5].metric("Actual return", f"{s['actual_return']:+.2%}",
                    f"{(s['actual_return'] - s['expected_return']) * 100:+.2f} pp vs expected")
        curves = {"backtest (expected)": rec.expected_equity}
        if len(rec.actual_equity):
            curves["bot (actual)"] = rec.actual_equity
        st.plotly_chart(equity_chart(curves, drawdown=False), width="stretch")
        if s["days"] < 21:
            st.warning("Less than 3 weeks of data. Too early to judge; keep paper trading.")
        st.dataframe(rec.trades, width="stretch", hide_index=True,
                     column_config={"cost_bps": NUM, "expected_price": NUM, "actual_price": NUM})

with bt_tab:
    c = st.columns(4)
    symbol = c[0].text_input("Symbol", cfg.symbol, key="bt_symbol")
    timeframe = c[1].selectbox("Timeframe", ["1h", "4h", "1d"], index=["1h", "4h", "1d"].index(cfg.timeframe)
                               if cfg.timeframe in ("1h", "4h", "1d") else 1, key="bt_tf")
    since = str(c[2].date_input("Since", pd.Timestamp("2019-01-01"), key="bt_since"))
    strategy = c[3].selectbox("Strategy", list(STRATEGIES), index=list(STRATEGIES).index(cfg.strategy))
    func = STRATEGIES[strategy][0]
    specs = list(inspect.signature(func).parameters.values())[1:]
    pcols = st.columns(len(specs) + 2)
    params = {}
    for col, p in zip(pcols, specs):
        default = cfg.params.get(p.name, p.default) if strategy == cfg.strategy else p.default
        params[p.name] = col.number_input(p.name, value=default, step=1 if isinstance(default, int) else 1.0,
                                          key=f"bt_{strategy}_{p.name}")
    fee = pcols[-2].number_input("fee", value=cfg.fee, format="%.4f", step=0.0005)
    slippage = pcols[-1].number_input("slippage", value=cfg.slippage, format="%.4f", step=0.0005)

    df = candles(cfg.exchange, symbol, timeframe, since, str(cfg.data_dir))
    res = backtest_strategy(df, strategy, params, timeframe, fee, slippage)
    bh = buy_and_hold(df, timeframe, fee, slippage)
    stats = pd.DataFrame({strategy: res.stats, "buy_and_hold": bh.stats}).T
    st.warning("Full-period backtests are in-sample: if you tune parameters here you are fitting the past. "
               "Use the Walk-forward tab for honest numbers.")
    pct_table(stats, ["total_return", "cagr", "max_drawdown", "win_rate", "exposure"])
    log_y = st.toggle("Log scale", value=True)
    st.plotly_chart(equity_chart({strategy: res.equity, "buy_and_hold": bh.equity}, log_y), width="stretch")

    st.subheader("Recent signals")
    recent = df.iloc[-500:]
    pos = res.positions.loc[recent.index]
    entries, exits = recent[(pos > 0) & (pos.shift(1) == 0)], recent[(pos == 0) & (pos.shift(1) > 0)]
    fig = go.Figure()
    fig.add_scatter(x=recent.index, y=recent.close, name="close", line={"width": 1})
    fig.add_scatter(x=entries.index, y=entries.open, mode="markers", name="buy",
                    marker={"symbol": "triangle-up", "size": 11, "color": "green"})
    fig.add_scatter(x=exits.index, y=exits.open, mode="markers", name="sell",
                    marker={"symbol": "triangle-down", "size": 11, "color": "red"})
    fig.update_layout(height=380, margin={"t": 30, "b": 10}, legend={"orientation": "h"})
    st.plotly_chart(fig, width="stretch")

with cmp_tab:
    st.caption("Each strategy is tuned on the first part of the data and then tested on the rest, which the tuning "
               "never saw. Only the oos_* columns count as evidence.")
    c = st.columns(3)
    split = c[0].slider("In-sample fraction", 0.5, 0.9, 0.7, 0.05)
    cmp_symbol = c[1].text_input("Symbol", cfg.symbol, key="cmp_symbol")
    cmp_since = str(c[2].date_input("Since", pd.Timestamp("2019-01-01"), key="cmp_since"))
    if st.button("Run comparison") or st.session_state.get("cmp_ran"):
        st.session_state.cmp_ran = True
        table = cached_compare(cfg.exchange, cmp_symbol, cfg.timeframe, cmp_since, str(cfg.data_dir), cfg.fee, cfg.slippage, split)
        st.caption("In-sample %s → %s | out-of-sample %s → %s" % (*[f"{t:%Y-%m-%d}" for t in table.attrs["in_sample"]],
                                                                *[f"{t:%Y-%m-%d}" for t in table.attrs["out_of_sample"]]))
        pct_table(table, ["is_return", "oos_return", "oos_cagr", "oos_max_dd", "oos_win_rate",
                          "grid_oos_pct_sharpe_pos"])

with wf_tab:
    st.caption("Re-tunes every strategy on a rolling training window and trades the following test window. The "
               "stitched test windows form one fully out-of-sample track record, the most honest test here.")
    c = st.columns(4)
    quote_ccy = cfg.symbol.split("/")[1]
    choices = list(dict.fromkeys([cfg.symbol] + [f"{b}/{quote_ccy}" for b in ("BTC", "ETH", "SOL", "XRP", "ADA")]))
    wf_symbols = c[0].multiselect("Symbols", choices, default=[cfg.symbol])
    train_days = c[1].number_input("Train days", 180, 2000, 730, 30)
    test_days = c[2].number_input("Test days", 30, 720, 180, 30)
    wf_since = str(c[3].date_input("Since", pd.Timestamp("2019-01-01"), key="wf_since"))
    if st.button("Run walk-forward") or st.session_state.get("wf_ran"):
        st.session_state.wf_ran = True
        for sym in wf_symbols:
            try:
                wf = cached_walk_forward(cfg.exchange, sym, cfg.timeframe, wf_since, str(cfg.data_dir), cfg.fee, cfg.slippage,
                                         int(train_days), int(test_days))
            except ValueError as e:
                st.warning(f"{sym}: {e}")
                continue
            start, end = wf.summary.attrs["out_of_sample"]
            st.subheader(f"{sym}: {wf.summary.attrs['n_windows']} windows, out-of-sample "
                         f"{start:%Y-%m-%d} → {end:%Y-%m-%d}")
            pct_table(wf.summary.drop(columns=["final_equity"]),
                      ["total_return", "cagr", "max_drawdown", "win_rate", "exposure", "pct_windows_positive"])
            st.plotly_chart(equity_chart(wf.equity, log_y=True), width="stretch", key=f"wf_chart_{sym}")
            with st.expander("Per-window results and chosen params"):
                w = wf.windows.assign(params=wf.windows.params.astype(str),
                                      test_start=wf.windows.test_start.dt.strftime("%Y-%m-%d"))
                st.dataframe(w.pivot(index="test_start", columns="strategy", values="test_return"),
                             column_config={s: PCT for s in w.strategy.unique()}, width="stretch")
                st.dataframe(w.pivot(index="test_start", columns="strategy", values="params"), width="stretch")
