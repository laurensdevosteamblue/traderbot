from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    trades_today: int = 0


class RiskManager:
    """Sits between the strategy and the broker and can veto orders.

    - Kill switch: if `<state_dir>/KILL` exists, no orders are sent at all.
    - Max drawdown from peak equity: trips the kill switch (the runner flattens first).
    - Daily loss limit / max trades per day / position caps: block new buys; exits stay allowed.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = cfg.state_dir / "risk.json"
        self.kill_file = cfg.state_dir / "KILL"
        self.state = RiskState(**json.loads(self.state_path.read_text())) if self.state_path.exists() else RiskState()

    def save(self) -> None:
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def killed(self) -> str | None:
        return (self.kill_file.read_text().strip() or "kill switch file present") if self.kill_file.exists() else None

    def trip(self, reason: str) -> None:
        log.critical("KILL SWITCH: %s", reason)
        self.kill_file.write_text(reason + "\n")

    def update(self, equity: float, now: pd.Timestamp | None = None) -> str | None:
        """Roll daily counters and track peak equity. Returns a reason if max drawdown is breached."""
        day = (now or pd.Timestamp.now(tz="UTC")).strftime("%Y-%m-%d")
        if day != self.state.day:
            self.state.day, self.state.day_start_equity, self.state.trades_today = day, equity, 0
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.save()
        if equity <= self.state.peak_equity * (1 - self.cfg.max_drawdown_pct):
            return (f"max drawdown breached: equity {equity:.2f} vs peak {self.state.peak_equity:.2f} "
                    f"(limit {self.cfg.max_drawdown_pct:.0%})")
        return None

    def check(self, side: str, notional: float, equity: float, position_value: float = 0.0) -> tuple[bool, str]:
        if reason := self.killed():
            return False, f"kill switch active: {reason}"
        if side == "sell":
            return True, "ok"
        c, s = self.cfg, self.state
        if s.day_start_equity and equity <= s.day_start_equity * (1 - c.daily_loss_limit_pct):
            return False, f"daily loss limit hit ({equity:.2f} vs day start {s.day_start_equity:.2f})"
        if s.trades_today >= c.max_trades_per_day:
            return False, f"max trades per day reached ({c.max_trades_per_day})"
        if notional < c.min_order_quote:
            return False, f"order {notional:.2f} below minimum {c.min_order_quote}"
        if c.max_position_quote > 0 and position_value + notional > c.max_position_quote * 1.001:
            return False, f"position would exceed cap {c.max_position_quote}"
        if position_value + notional > equity * c.position_fraction * 1.001:
            return False, "position would exceed position_fraction of equity"
        return True, "ok"

    def record_trade(self) -> None:
        self.state.trades_today += 1
        self.save()
