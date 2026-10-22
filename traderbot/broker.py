from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import ccxt

log = logging.getLogger(__name__)


class Ledger:
    """The bot's own sub-balance (base + quote), persisted to JSON.

    The bot only ever trades what the ledger says it owns, so other funds on the same exchange account
    (your personal holdings) are never bought with, sold or counted in the bot's equity.
    """

    def __init__(self, path: Path, budget_quote: float):
        self.path = path
        self.state = json.loads(path.read_text()) if path.exists() else {"base": 0.0, "quote": budget_quote}

    @property
    def base(self) -> float:
        return self.state["base"]

    @property
    def quote(self) -> float:
        return self.state["quote"]

    def apply_fill(self, side: str, amount: float, cost: float, fee: float, fee_in_base: bool = False) -> None:
        base_fee, quote_fee = (fee, 0.0) if fee_in_base else (0.0, fee)
        if side == "buy":
            self.state["base"] += amount - base_fee
            self.state["quote"] -= cost + quote_fee
        else:
            self.state["base"] -= amount + base_fee
            self.state["quote"] += cost - quote_fee
        self.state = {k: max(v, 0.0) for k, v in self.state.items()}
        self.path.write_text(json.dumps(self.state, indent=2))


def combine_fills(side: str, fills: list[dict], order_id) -> dict:
    """Merge the limit part and the market fallback into one fill record."""
    amount = sum(f["amount"] for f in fills)
    cost = sum(f["cost"] for f in fills)
    maker = sum(f["amount"] for f in fills if f["maker"])
    return {"id": order_id, "side": side, "amount": amount, "price": cost / amount if amount else 0.0,
            "cost": cost, "fee": sum(f["fee"] for f in fills), "maker_share": maker / amount if amount else 0.0}


class PaperBroker:
    """Simulated fills against live exchange prices, with fees and slippage.

    Limit orders are simulated conservatively: a post-only order at the best bid (buy) / ask (sell) counts
    as filled only if real trades happened strictly through that price during the wait; otherwise the
    order falls back to a market fill (taker fee + slippage), exactly like the live broker.
    """

    def __init__(self, exchange, symbol: str, fee: float, slippage: float, start_quote: float, state_path: Path,
                 order_type: str = "market", maker_fee: float = 0.0015, limit_timeout_s: int = 90,
                 sleep=time.sleep):
        self.exchange, self.symbol, self.fee, self.slippage = exchange, symbol, fee, slippage
        self.order_type, self.maker_fee, self.limit_timeout_s, self.sleep = order_type, maker_fee, limit_timeout_s, sleep
        self.ledger = Ledger(state_path, start_quote)

    def price(self) -> float:
        return float(self.exchange.fetch_ticker(self.symbol)["last"])

    def balances(self) -> tuple[float, float]:
        return self.ledger.base, self.ledger.quote

    def _fill(self, side: str, amount: float, fill: float, fee_rate: float) -> dict:
        if side == "buy":
            amount = min(amount, self.ledger.quote / (fill * (1 + fee_rate)))
        else:
            amount = min(amount, self.ledger.base)
        cost = amount * fill
        fee = cost * fee_rate
        self.ledger.apply_fill(side, amount, cost, fee)
        return {"amount": amount, "cost": cost, "fee": fee}

    def market_order(self, side: str, amount: float, price: float) -> dict:
        fill = price * (1 + self.slippage if side == "buy" else 1 - self.slippage)
        f = self._fill(side, amount, fill, self.fee)
        return {"id": "paper", "side": side, "price": fill, "maker_share": 0.0, **f}

    def execute(self, side: str, amount: float, price: float, order_type: str | None = None) -> dict:
        if (order_type or self.order_type) != "limit":
            return self.market_order(side, amount, price)
        ticker = self.exchange.fetch_ticker(self.symbol)
        limit = float(ticker.get("bid" if side == "buy" else "ask") or price)
        placed = int(time.time() * 1000)
        self.sleep(self.limit_timeout_s)
        now = int(time.time() * 1000)
        trades = self.exchange.fetch_trades(self.symbol, since=placed, params={"until": now})
        crossed = any((t["price"] < limit) if side == "buy" else (t["price"] > limit) for t in trades)
        if crossed:
            if side == "buy":
                amount *= price / limit  # same EUR amount at the (lower) limit price
            f = self._fill(side, amount, limit, self.maker_fee)
            return {"id": "paper-limit", "side": side, "price": limit, "maker_share": 1.0, **f}
        log.info("paper limit %s %s @ %s not filled in %ss; market fallback", side, self.symbol, limit,
                 self.limit_timeout_s)
        return self.market_order(side, amount, self.price())


class ExchangeBroker:
    """Real orders via ccxt, restricted to the bot's ledger (its budget plus whatever it bought).

    Works on an account that also holds personal funds: buys are limited to the ledger's quote balance and
    sells to the coins the bot itself bought.
    """

    def __init__(self, exchange, symbol: str, budget_quote: float, ledger_path: Path, sandbox: bool = False,
                 order_type: str = "market", limit_timeout_s: int = 90, sleep=time.sleep):
        self.exchange, self.symbol = exchange, symbol
        self.order_type, self.limit_timeout_s, self.sleep = order_type, limit_timeout_s, sleep
        exchange.load_markets()
        self.market = exchange.market(symbol)
        self.ledger = Ledger(ledger_path, budget_quote)
        if exchange.id == "binance" and not sandbox:
            self.verify_no_withdrawals()
        elif not sandbox:
            log.warning("%s does not expose API key permissions; make sure the key has NO withdrawal permission "
                        "and is restricted to your server's IP.", exchange.id)

    def verify_no_withdrawals(self) -> None:
        restrictions = self.exchange.sapi_get_account_apirestrictions()
        if str(restrictions.get("enableWithdrawals")).lower() == "true":
            raise RuntimeError("Refusing to run: this API key has WITHDRAWAL permission. Create a trade-only key.")
        if str(restrictions.get("ipRestrict")).lower() != "true":
            log.warning("API key is not IP-restricted; restrict it to your server's IP in the exchange settings.")

    def price(self) -> float:
        return float(self.exchange.fetch_ticker(self.symbol)["last"])

    def account_balances(self) -> tuple[float, float]:
        free = self.exchange.fetch_balance()["free"]
        return float(free.get(self.market["base"]) or 0.0), float(free.get(self.market["quote"]) or 0.0)

    def balances(self) -> tuple[float, float]:
        """The bot's balances: its ledger, capped by what is actually free on the account."""
        base_free, quote_free = self.account_balances()
        if base_free < self.ledger.base * 0.999 or quote_free < self.ledger.quote * 0.999:
            log.warning("account has less free balance (base %.8f, quote %.2f) than the bot ledger (base %.8f, "
                        "quote %.2f); trading with what is available", base_free, quote_free,
                        self.ledger.base, self.ledger.quote)
        return min(self.ledger.base, base_free), min(self.ledger.quote, quote_free)

    def _record(self, side: str, order: dict, maker: bool, fallback_price: float) -> dict | None:
        """Apply an order's actual fill to the ledger."""
        filled = float(order.get("filled") or 0.0)
        if filled <= 0:
            return None
        avg = float(order.get("average") or order.get("price") or fallback_price)
        cost = float(order.get("cost") or filled * avg)
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0)
        self.ledger.apply_fill(side, filled, cost, fee, fee_in_base=fee_info.get("currency") == self.market["base"])
        return {"amount": filled, "cost": cost, "fee": fee, "maker": maker}

    def _final(self, order: dict) -> dict:
        if order.get("status") != "closed" and order.get("id"):
            return self.exchange.fetch_order(order["id"], self.symbol)
        return order

    def market_order(self, side: str, amount: float, price: float) -> dict:
        return self.execute(side, amount, price, order_type="market")

    def cancel_stale_orders(self) -> None:
        """Cancel any resting open orders for this symbol (e.g. left over from an interrupted run)."""
        if not hasattr(self.exchange, "fetch_open_orders"):
            return
        for o in self.exchange.fetch_open_orders(self.symbol):
            try:
                self.exchange.cancel_order(o["id"], self.symbol)
                log.warning("cancelled stale open order %s on %s", o["id"], self.symbol)
            except (ccxt.InvalidOrder, ccxt.OperationRejected, ccxt.CancelPending):
                pass

    def execute(self, side: str, amount: float, price: float, order_type: str | None = None) -> dict:
        order_type = order_type or self.order_type
        self.cancel_stale_orders()
        if side == "buy":
            amount *= 0.995  # headroom for fees and price movement
        else:
            amount = min(amount, self.ledger.base)  # never sell coins the bot did not buy
        amount = float(self.exchange.amount_to_precision(self.symbol, amount))
        fills, order_id, remaining = [], None, amount

        if order_type == "limit":
            ticker = self.exchange.fetch_ticker(self.symbol)
            limit = float(self.exchange.price_to_precision(
                self.symbol, ticker.get("bid" if side == "buy" else "ask") or price))
            try:
                order = self.exchange.create_order(self.symbol, "limit", side, amount, limit, {"postOnly": True})
                order_id = order.get("id")
                self.sleep(self.limit_timeout_s)
                try:
                    self.exchange.cancel_order(order_id, self.symbol)
                except (ccxt.InvalidOrder, ccxt.OperationRejected, ccxt.CancelPending):
                    pass  # already fully filled, cancel pending, or rejected as post-only
                order = self.exchange.fetch_order(order_id, self.symbol)
                if f := self._record(side, order, True, limit):
                    fills.append(f)
                    remaining = amount - f["amount"]
            except (ccxt.InvalidOrder, ccxt.OperationRejected) as e:
                log.warning("limit order rejected (%s); falling back to market", e)

        min_cost = (self.market.get("limits", {}).get("cost") or {}).get("min") or 0.0
        if remaining > 0 and remaining * price >= max(min_cost, 1e-12):
            remaining = float(self.exchange.amount_to_precision(self.symbol, remaining))
            if remaining > 0:
                order = self._final(self.exchange.create_order(self.symbol, "market", side, remaining))
                order_id = order_id or order.get("id")
                if f := self._record(side, order, False, price):
                    fills.append(f)
        if not fills:
            raise RuntimeError(f"{side} {amount} {self.symbol}: nothing was filled")
        return combine_fills(side, fills, order_id)
