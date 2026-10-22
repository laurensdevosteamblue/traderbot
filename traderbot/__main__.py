from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .alerts import Alerts
from .backtest import backtest_strategy, buy_and_hold, compare, walk_forward
from .broker import ExchangeBroker, PaperBroker
from .config import load_bots, load_env
from .data import load_history, make_exchange
from .live import Runner, run_forever, step_all
from .reconcile import reconcile_session
from .research import overlay_study
from .risk import RiskManager
from .strategies import STRATEGIES

PCT_COLS = ["is_return", "oos_return", "oos_cagr", "oos_max_dd", "oos_win_rate", "grid_oos_pct_sharpe_pos"]


def parse_params(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        out[key] = float(value) if "." in value else int(value)
    return out


def setup_logging(state_dir) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(state_dir / "traderbot.log")])


def fmt_stats(stats: dict) -> str:
    return (f"return {stats['total_return']:+.1%} | CAGR {stats['cagr']:+.1%} | max DD {stats['max_drawdown']:.1%} | "
            f"Sharpe {stats['sharpe']:.2f} | trades {stats['trades']} | win {stats['win_rate']:.0%} | "
            f"exposure {stats['exposure']:.0%}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="traderbot")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--bot", help="only this bot (default: all bots; analysis commands use the first)")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name in ("fetch", "backtest", "compare"):
        sp = sub.add_parser(name)
        sp.add_argument("--since", default="2019-01-01")
        sp.add_argument("--until")
    sub.choices["backtest"].add_argument("--strategy")
    sub.choices["backtest"].add_argument("--params", nargs="*", help="e.g. fast=20 slow=100")
    sub.choices["compare"].add_argument("--split", type=float, default=0.7, help="in-sample fraction")
    wf = sub.add_parser("walkforward", help="rolling re-optimisation over several symbols")
    wf.add_argument("--since", default="2019-01-01")
    wf.add_argument("--symbols", nargs="+", help="default: the symbols of the configured bots")
    wf.add_argument("--train-days", type=int, default=730)
    wf.add_argument("--test-days", type=int, default=180)
    rs = sub.add_parser("research", help="test limit orders, volatility sizing and a BTC trend filter")
    rs.add_argument("--since", default="2019-01-01")
    rs.add_argument("--annual-vol", type=float, default=0.5, help="volatility sizing target (fixed up front)")
    rs.add_argument("--regime-sma-days", type=int, default=50, help="BTC trend filter SMA (fixed up front)")
    sub.add_parser("reconcile", help="compare paper/live trades with the backtest over the same period")
    sub.add_parser("dashboard", help="read-only web dashboard on http://127.0.0.1:8501")

    run = sub.add_parser("run", help="paper / testnet / live trading loop")
    run.add_argument("--once", action="store_true", help="run a single step and exit")
    run.add_argument("--confirm-live", action="store_true", help="required when mode = live")
    sub.add_parser("status")
    sub.add_parser("kill", help="activate the kill switch")
    sub.add_parser("resume", help="remove the kill switch")

    args = p.parse_args(argv)
    if args.cmd == "dashboard":
        root = Path(__file__).resolve().parent.parent
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(root / "traderbot" / "dashboard.py"),
                                "--server.address", "127.0.0.1"], cwd=root)
    load_env()
    bots = load_bots(args.config)
    selected = [b for b in bots if args.bot in (None, b.name)]
    if not selected:
        p.error(f"no bot named {args.bot!r}; available: {[b.name for b in bots]}")
    cfg = selected[0]
    setup_logging(cfg.state_dir.parent if cfg.name != "main" else cfg.state_dir)

    if args.cmd in ("fetch", "backtest", "compare"):
        df = load_history(make_exchange(cfg.exchange), cfg.symbol, cfg.timeframe, cfg.data_dir, args.since, args.until)
        print(f"{cfg.exchange} {cfg.symbol} {cfg.timeframe}: {len(df)} candles {df.index[0]} -> {df.index[-1]}")
        print(f"costs per side: fee {cfg.fee:.3%} + slippage {cfg.slippage:.3%}\n")
        if args.cmd == "backtest":
            name = args.strategy or cfg.strategy
            params = parse_params(args.params) if args.params is not None else (cfg.params if name == cfg.strategy else {})
            res = backtest_strategy(df, name, params, cfg.timeframe, cfg.fee, cfg.slippage)
            print(f"{name} {params}\n  {fmt_stats(res.stats)}")
            print(f"buy_and_hold\n  {fmt_stats(buy_and_hold(df, cfg.timeframe, cfg.fee, cfg.slippage).stats)}")
            print("\nWARNING: a full-period backtest is in-sample. Use `compare` for out-of-sample evidence.")
        elif args.cmd == "compare":
            table = compare(df, cfg.timeframe, cfg.fee, cfg.slippage, args.split)
            print("in-sample:     %s -> %s" % table.attrs["in_sample"])
            print("out-of-sample: %s -> %s\n" % table.attrs["out_of_sample"])
            shown = table.copy()
            for col in PCT_COLS:
                shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
            with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", 60):
                print(shown.round(2).to_string())
            print("\nParams were chosen on in-sample data only. Judge strategies by the oos_* columns, and prefer "
                  "ones where\nmost of the parameter grid is also profitable OOS (grid_oos_*), not just the best one.")
        return 0

    if args.cmd == "walkforward":
        exchange = make_exchange(cfg.exchange)
        print(f"costs per side: fee {cfg.fee:.3%} + slippage {cfg.slippage:.3%}")
        for symbol in args.symbols or list(dict.fromkeys(b.symbol for b in selected)):
            df = load_history(exchange, symbol, cfg.timeframe, cfg.data_dir, args.since)
            res = walk_forward(df, cfg.timeframe, cfg.fee, cfg.slippage, args.train_days, args.test_days)
            start, end = res.summary.attrs["out_of_sample"]
            print(f"\n=== {cfg.exchange} {symbol} {cfg.timeframe} | {res.summary.attrs['n_windows']} windows "
                  f"(train {args.train_days}d, test {args.test_days}d) | out-of-sample {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
            shown = res.summary[["total_return", "cagr", "max_drawdown", "sharpe", "trades", "win_rate",
                                 "exposure", "pct_windows_positive"]].copy()
            for col in ("total_return", "cagr", "max_drawdown", "win_rate", "exposure", "pct_windows_positive"):
                shown[col] = shown[col].map(lambda v: f"{v:+.1%}")
            print(shown.round(2).to_string())
        print("\nEvery number above is out-of-sample: params were re-chosen each window using only earlier data.")
        return 0

    if args.cmd == "research":
        res = overlay_study(selected, args.since, args.annual_vol, args.regime_sma_days)
        print(f"strategy {cfg.strategy}, walk-forward out-of-sample, vol target {args.annual_vol:.0%}, "
              f"BTC filter SMA {args.regime_sma_days}d")
        pct = ("total_return", "cagr", "max_drawdown")
        for label in ("all bots", "longest history"):
            t = res[label].copy()
            print(f"\n=== portfolio: {label} {t.attrs['bots']} | {t.attrs['start']:%Y-%m-%d} -> {t.attrs['end']:%Y-%m-%d}")
            for col in pct:
                t[col] = t[col].map(lambda v: f"{v:+.1%}")
            print(t.round(2).to_string())
        pc = res["per_coin"]
        print("\n=== per coin: total return / max drawdown")
        cell = pc.assign(v=pc.total_return.map("{:+.0%}".format) + " / " + pc.max_drawdown.map("{:.0%}".format))
        print(cell.pivot(index="variant", columns="bot", values="v").loc[list(dict.fromkeys(pc.variant))].to_string())
        return 0

    if args.cmd == "reconcile":
        exchange = make_exchange(cfg.exchange)
        for b in selected:
            print(f"\n=== [{b.name}] {b.symbol} {b.strategy} {b.params}")
            try:
                rec = reconcile_session(b, exchange)
            except FileNotFoundError as e:
                print(e)
                continue
            s = rec.summary
            print(f"{b.mode} session since {s['start']:%Y-%m-%d %H:%M} UTC ({s['days']:.1f} days)")
            print(f"trades: expected {s['expected_trades']}, actual {s['actual_trades']}, matched {s['matched']}, "
                  f"missed {s['missed']}, extra {s['extra']}")
            print(f"fill cost vs backtest price: {s['avg_cost_vs_expected_bps']:.1f} bps "
                  f"(backtest assumes {s['assumed_cost_bps']:.1f} bps slippage)")
            if b.order_type == "limit":
                rate = s["limit_fill_rate"]
                print(f"limit orders filled: {'-' if pd.isna(rate) else f'{rate:.0%}'} (backtest assumes 50%)")
            print(f"return: expected {s['expected_return']:+.2%} | actual {s['actual_return']:+.2%}")
            if len(rec.trades):
                print(rec.trades.to_string(index=False))
        return 0

    if args.cmd in ("kill", "resume"):
        for b in selected:
            risk = RiskManager(b)
            if args.cmd == "kill":
                risk.trip("manual kill via CLI")
            else:
                risk.kill_file.unlink(missing_ok=True)
            print(f"[{b.name}] kill switch {'ON' if args.cmd == 'kill' else 'off'}")
        return 0

    if len({(b.exchange, b.mode) for b in selected}) > 1:
        p.error("all bots must use the same exchange and mode")
    alerts = Alerts(os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), prefix=f"traderbot-{cfg.mode}")
    if cfg.mode == "paper":
        exchange = make_exchange(cfg.exchange)
    else:
        if cfg.mode == "live" and args.cmd == "run" and not args.confirm_live:
            print("mode = live trades REAL money. Re-run with --confirm-live once paper trading matched the backtest.")
            return 2
        prefix = cfg.exchange.upper() + ("_TESTNET" if cfg.mode == "testnet" else "")
        exchange = make_exchange(cfg.exchange, os.getenv(f"{prefix}_API_KEY"), os.getenv(f"{prefix}_API_SECRET"),
                                 sandbox=cfg.mode == "testnet", operator_id=cfg.operator_id)

    runners = []
    for b in selected:
        ledger_path = b.state_dir / f"ledger_{b.mode}.json"
        if b.mode == "paper":
            broker = PaperBroker(exchange, b.symbol, b.fee, b.slippage, b.budget_quote, ledger_path,
                                 order_type=b.order_type, maker_fee=b.maker_fee, limit_timeout_s=b.limit_timeout_s)
        else:
            broker = ExchangeBroker(exchange, b.symbol, b.budget_quote, ledger_path, sandbox=b.mode == "testnet",
                                    order_type=b.order_type, limit_timeout_s=b.limit_timeout_s)
        runners.append(Runner(b, exchange, broker, RiskManager(b), alerts))

    if args.cmd == "status":
        total = 0.0
        for r in runners:
            price = r.broker.price()
            base, quote = r.broker.balances()
            total += quote + base * price
            print(f"[{r.cfg.name}] {r.cfg.exchange} {r.cfg.mode} | {r.cfg.symbol} {price:.6g} | bot base {base:.8f} | "
                  f"bot quote {quote:.2f} | bot equity {quote + base * price:.2f} | "
                  f"trades today {r.risk.state.trades_today} | kill switch {r.risk.killed() or 'off'}")
            if isinstance(r.broker, ExchangeBroker):
                acct_base, acct_quote = r.broker.account_balances()
                print(f"    whole account (free): base {acct_base:.8f} | quote {acct_quote:.2f}  <- ignored by the bot")
        print(f"total bot equity: {total:.2f}")
        return 0

    if args.once:
        for name, result in step_all(runners, {}).items():
            print(f"[{name}] {result}")
    else:
        run_forever(runners)
    return 0


if __name__ == "__main__":
    sys.exit(main())
