from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("paper", "testnet", "live")


@dataclass
class Config:
    name: str = "main"
    exchange: str = "bitvavo"
    symbol: str = "BTC/EUR"
    timeframe: str = "4h"
    operator_id: int = 1
    fee: float = 0.0025
    slippage: float = 0.0005
    maker_fee: float = 0.0015
    order_type: str = "market"
    limit_timeout_s: int = 90
    vol_target: float = 0.0
    vol_lookback_days: int = 30
    strategy: str = "sma_cross"
    params: dict = field(default_factory=dict)
    position_fraction: float = 0.95
    max_position_quote: float = 0.0
    daily_loss_limit_pct: float = 0.05
    max_drawdown_pct: float = 0.25
    max_trades_per_day: int = 6
    min_order_quote: float = 10.0
    mode: str = "paper"
    budget_quote: float = 250.0
    state_dir: Path = Path("state")
    data_dir: Path = Path("data")

    def __post_init__(self):
        self.state_dir, self.data_dir = Path(self.state_dir), Path(self.data_dir)
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if not 0 < self.position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        if self.order_type not in ("market", "limit"):
            raise ValueError("order_type must be 'market' or 'limit'")
        if self.mode == "testnet" and self.exchange != "binance":
            raise ValueError(f"{self.exchange} has no testnet; use mode = 'paper'")


def load_env(path: str | Path = ".env") -> None:
    """Minimal .env loader; existing environment variables take precedence."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\"").replace("\\n", "\n"))


def load_bots(path: str | Path = "config.toml") -> list[Config]:
    """Shared settings come from the [sections]; every [[bots]] entry overrides them and gets its own
    state folder (state/<name>/). A config without [[bots]] is a single bot using state/ directly."""
    raw = tomllib.loads(Path(path).read_text())
    bots = raw.pop("bots", None)
    flat = {k: v for section in raw.values() for k, v in section.items()}
    if "name" in flat:
        flat["strategy"] = flat.pop("name")
    if not bots:
        return [Config(**flat)]
    names = [b["name"] for b in bots]
    if len(set(names)) != len(names):
        raise ValueError(f"bot names must be unique: {names}")
    base_state = Path(flat.get("state_dir", "state"))
    return [Config(**{**flat, **b, "state_dir": base_state / b["name"]}) for b in bots]


def load_config(path: str | Path = "config.toml", bot: str | None = None) -> Config:
    """One bot's config: the named bot, or the first one."""
    bots = load_bots(path)
    if bot is None:
        return bots[0]
    for cfg in bots:
        if cfg.name == bot:
            return cfg
    raise KeyError(f"no bot named {bot!r}; available: {[c.name for c in bots]}")
