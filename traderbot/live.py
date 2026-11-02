from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .alerts import Alerts
from .config import Config
from .data import closed_candles, timeframe_seconds, to_frame
from .overlays import realized_vol
from .risk import RiskManager
from .strategies import get_strategy

log = logging.getLogger(__name__)
MAX_CONSECUTIVE_ERRORS = 5
JOURNAL_FIELDS = ["time", "mode", "symbol", "side", "amount", "price", "cost", "fee", "order_id", "equity_before",
                  "maker_share"]


def state_paths(cfg: Config) -> tuple[Path, Path, Path, Path]:
    """(trade journal, session info, latest status, equity log) for the configured mode."""
    d, m = cfg.state_dir, cfg.mode
    return d / f"trades_{m}.csv", d / f"session_{m}.json", d / f"status_{m}.json", d / f"equity_{m}.csv"


class Runner:
    def __init__(self, cfg: Config, exchange, broker, risk: RiskManager, alerts: Alerts):
        self.cfg, self.exchange, self.broker, self.risk, self.alerts = cfg, exchange, broker, risk, alerts
        self.strategy = get_strategy(cfg.strategy)
        self.journal, self.session, self.status, self.equity_log = state_paths(cfg)

    def signal(self, candles: pd.DataFrame | None = None) -> tuple[float, float]:
        """(target position 0/1, size weight for a new entry) from the latest closed candles."""
        c = self.cfg
        if candles is None:
            candles = closed_candles(to_frame(self.exchange.fetch_ohlcv(c.symbol, c.timeframe, limit=1000)),
                                     c.timeframe)
        if candles.empty:
            raise RuntimeError(f"no closed candles returned for {c.symbol} {c.timeframe}")
        target = float(self.strategy(candles, **c.params).iloc[-1])
        self.entry_level = float(candles.high.rolling(c.params["entry"]).max().iloc[-1]) \
            if c.strategy == "donchian_breakout" and "entry" in c.params else None
        self.exit_level = float(candles.low.rolling(c.params["exit"]).min().iloc[-1]) \
            if c.strategy == "donchian_breakout" and "exit" in c.params else None
        weight = 1.0
        if c.vol_target > 0:
            vol = float(realized_vol(candles, c.timeframe, c.vol_lookback_days).iloc[-1])
            self.realized_vol = vol
            weight = min(1.0, c.vol_target / vol) if vol > 0 else 1.0
        else:
            self.realized_vol = None
        return target, weight

    def step(self, candles: pd.DataFrame | None = None) -> str:
        killed = self.risk.killed()
        if killed:
            log.warning("[%s] kill switch active (%s); not trading", self.cfg.name, killed)
        target, self.weight = (None, None) if killed else self.signal(candles)
        try:
            price = self.broker.price()
            base, quote = self.broker.balances()
        except Exception:
            if not killed:
                raise
            prev = json.loads(self.status.read_text()) if self.status.exists() else {}
            price = float(prev.get("price") or 0.0)
            base, quote = self.broker.ledger.base, self.broker.ledger.quote
        if not self.session.exists():
            self.session.write_text(json.dumps({"started": pd.Timestamp.now(tz="UTC").isoformat(),
                                                "start_equity": quote + base * price, "strategy": self.cfg.strategy,
                                                "params": self.cfg.params}, indent=2))
        action = "killed" if killed else self._decide(target, price, base, quote)
        if action in ("buy", "sell", "halted"):
            base, quote = self.broker.balances()
        self._snapshot(action, target, price, base, quote)
        return action

    def _snapshot(self, action: str, target: float | None, price: float, base: float, quote: float) -> None:
        """Files the dashboard reads; it never needs API keys or exchange access."""
        now = pd.Timestamp.now(tz="UTC").isoformat()
        row = {"time": now, "price": price, "base": base, "quote": quote, "equity": quote + base * price,
               "target": target, "action": action}
        new = not self.equity_log.exists()
        with self.equity_log.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        c = self.cfg
        self.status.write_text(json.dumps({**row, "name": c.name, "mode": c.mode, "symbol": c.symbol,
                                           "timeframe": c.timeframe, "strategy": c.strategy, "params": c.params,
                                           "entry_size_weight": getattr(self, "weight", None),
                                           "realized_vol": getattr(self, "realized_vol", None),
                                           "entry_level": getattr(self, "entry_level", None),
                                           "exit_level": getattr(self, "exit_level", None),
                                           "order_type": c.order_type, "vol_target": c.vol_target,
                                           "kill_switch": self.risk.killed(), "risk": asdict(self.risk.state)},
                                          indent=2))

    def _decide(self, target: float, price: float, base: float, quote: float) -> str:
        c = self.cfg
        position_value = base * price
        equity = quote + position_value

        if breach := self.risk.update(equity):
            if position_value >= c.min_order_quote:
                self._execute("sell", base, price, equity, order_type="market")  # emergency: no waiting
            self.risk.trip(breach)
            self.alerts.send(f"[{c.name}] HALTED - {breach}. Position flattened. "
                             f"Run `resume --bot {c.name}` to resume.")
            return "halted"

        flat = position_value < c.min_order_quote
        cap = (min(equity * c.position_fraction, c.max_position_quote)
               if c.max_position_quote > 0 else equity * c.position_fraction)
        notional = cap * target * self.weight
        if target > 0 and flat and notional >= c.min_order_quote:
            # Size is fixed at entry (volatility sizing) and kept until the exit: no top-ups or rebalancing.
            side, amount = "buy", notional / price
        elif target == 0 and not flat:
            side, notional, amount = "sell", position_value, base
        else:
            log.info("[%s] hold: target=%.0f weight=%.2f price=%.6g position=%.2f equity=%.2f", c.name, target,
                     self.weight, price, position_value, equity)
            return "hold"

        ok, reason = self.risk.check(side, notional, equity, position_value)
        if not ok:
            self.alerts.send(f"[{c.name}] {side.upper()} blocked by risk manager: {reason}")
            return "blocked"
        self._execute(side, amount, price, equity)
        return side

    def _execute(self, side: str, amount: float, price: float, equity: float, order_type: str | None = None) -> dict:
        fill = self.broker.execute(side, amount, price, order_type=order_type)
        self.risk.record_trade()
        self._journal(fill, equity)
        self.alerts.send(f"[{self.cfg.name}] {self.cfg.mode.upper()} {side.upper()} {fill['amount']:.6f} "
                         f"{self.cfg.symbol} @ {fill['price']:.6g} (cost {fill['cost']:.2f}, "
                         f"{fill.get('maker_share', 0):.0%} as limit order, equity before {equity:.2f})")
        return fill

    def _journal(self, fill: dict, equity: float) -> None:
        if self.journal.exists():
            with self.journal.open() as f:
                header = f.readline().strip().split(",")
            if header != JOURNAL_FIELDS:  # older journal: migrate to the current columns
                pd.read_csv(self.journal).reindex(columns=JOURNAL_FIELDS).to_csv(self.journal, index=False)
        new = not self.journal.exists()
        with self.journal.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"time": pd.Timestamp.now(tz="UTC").isoformat(), "mode": self.cfg.mode,
                        "symbol": self.cfg.symbol, "order_id": fill.get("id"), "equity_before": round(equity, 2),
                        **{k: fill.get(k) for k in ("side", "amount", "price", "cost", "fee", "maker_share")}})


def step_all(runners: list[Runner], errors: dict[str, int]) -> dict[str, str]:
    """One step for every bot. A failing bot never stops the others; after MAX_CONSECUTIVE_ERRORS
    failures in a row that bot trips its own kill switch."""
    results = {}
    candle_cache: dict[tuple[int, str, str], pd.DataFrame] = {}
    for r in runners:
        name = r.cfg.name
        try:
            key = (id(r.exchange), r.cfg.symbol, r.cfg.timeframe)
            if not r.risk.killed() and key not in candle_cache:
                candle_cache[key] = closed_candles(
                    to_frame(r.exchange.fetch_ohlcv(r.cfg.symbol, r.cfg.timeframe, limit=1000)), r.cfg.timeframe)
            results[name] = r.step(candle_cache.get(key))
            errors[name] = 0
        except Exception as e:
            errors[name] = errors.get(name, 0) + 1
            results[name] = "error"
            log.exception("[%s] step failed", name)
            r.alerts.send(f"[{name}] ERROR ({errors[name]}/{MAX_CONSECUTIVE_ERRORS}): {type(e).__name__}: {e}")
            if errors[name] >= MAX_CONSECUTIVE_ERRORS:
                r.risk.trip(f"{errors[name]} consecutive errors, last: {e}")
                prev = json.loads(r.status.read_text()) if r.status.exists() else {}
                r._snapshot("killed", None, float(prev.get("price") or 0.0), r.broker.ledger.base,
                            r.broker.ledger.quote)
                r.alerts.send(f"[{name}] HALTED after repeated errors. Investigate, then run `resume --bot {name}`.")
                errors[name] = 0
    return results


def run_forever(runners: list[Runner]) -> None:
    tf = min(timeframe_seconds(r.cfg.timeframe) for r in runners)
    runners[0].alerts.send(f"started {len(runners)} bot(s) in {runners[0].cfg.mode} mode: " + ", ".join(
        f"{r.cfg.name}={r.cfg.symbol} {r.cfg.strategy} {r.cfg.params}" for r in runners))
    errors: dict[str, int] = {}
    while True:
        step_all(runners, errors)
        # Wake shortly after the next candle closes (or retry sooner after an error).
        now = time.time()
        time.sleep(60 if any(errors.values()) else tf - now % tf + 15)
