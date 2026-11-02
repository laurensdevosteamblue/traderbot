import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traderbot.backtest import buy_and_hold, compare, run_backtest, walk_forward
from traderbot.broker import ExchangeBroker, PaperBroker
from traderbot.config import Config, load_bots, load_config, load_env
from traderbot.data import make_exchange
from traderbot.live import Runner, state_paths, step_all
from traderbot.overlays import blended_costs, realized_vol, regime_filter, size_at_entry, vol_target
from traderbot.reconcile import expected_trades, reconcile
from traderbot.alerts import Alerts
from traderbot.risk import RiskManager
from traderbot.strategies import STRATEGIES


def make_df(closes, freq="4h"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2022-01-01", periods=len(closes), freq=freq, tz="UTC")
    opens = np.r_[closes[0], closes[:-1]]
    return pd.DataFrame({"open": opens, "high": np.maximum(opens, closes) * 1.01,
                         "low": np.minimum(opens, closes) * 0.99, "close": closes, "volume": 1.0}, index=idx)


def random_walk(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return make_df(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n))))


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_have_no_lookahead(name):
    df = random_walk()
    func, grid = STRATEGIES[name]
    for params in grid[:4]:
        full = func(df, **params)
        cut = 2000
        tampered = df.copy()
        tampered.iloc[cut:] *= 3  # change the future
        assert full.iloc[:cut].equals(func(tampered, **params).iloc[:cut])
        assert set(full.unique()) <= {0.0, 1.0}


def test_backtest_charges_costs_on_round_trip():
    df = make_df([100.0] * 10)
    target = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=df.index, dtype=float)
    res = run_backtest(df, target, "4h", fee=0.001, slippage=0.0005, initial=1000)
    assert res.equity.iloc[-1] == pytest.approx(1000 * (1 - 0.0015) ** 2)
    assert res.stats["trades"] == 1 and res.stats["win_rate"] == 0


def test_signal_executes_next_bar():
    df = make_df([100, 100, 200, 200])
    target = pd.Series([0, 1, 1, 1], index=df.index, dtype=float)  # decided on bar 1's close
    res = run_backtest(df, target, "4h", 0, 0, initial=1000)
    # long from bar 2's open (=100) onwards, captures the move to 200
    assert res.equity.iloc[-1] == pytest.approx(2000)


def test_buy_and_hold_matches_price():
    df = random_walk(500)
    res = buy_and_hold(df, "4h", 0, 0)
    assert res.equity.iloc[-1] / 1000 == pytest.approx(df.open.iloc[-1] / df.open.iloc[0])


def test_compare_runs():
    table = compare(random_walk(), "4h", 0.001, 0.0005)
    assert "buy_and_hold" in table.index and len(table) >= 2


def cfg_for(tmp_path, **kw):
    return Config(state_dir=tmp_path, data_dir=tmp_path, **kw)


def test_risk_limits(tmp_path):
    risk = RiskManager(cfg_for(tmp_path, max_position_quote=100, min_order_quote=10, max_trades_per_day=1))
    risk.update(1000)
    assert risk.check("buy", 50, 1000) == (True, "ok")
    assert not risk.check("buy", 5, 1000)[0]
    assert not risk.check("buy", 150, 1000)[0]
    risk.record_trade()
    assert not risk.check("buy", 50, 1000)[0]
    assert risk.check("sell", 50, 1000)[0]
    risk.trip("test")
    assert not risk.check("sell", 50, 1000)[0]


def test_daily_loss_and_drawdown(tmp_path):
    risk = RiskManager(cfg_for(tmp_path, daily_loss_limit_pct=0.05, max_drawdown_pct=0.2))
    t = pd.Timestamp("2024-01-01 01:00", tz="UTC")
    assert risk.update(1000, t) is None
    assert not risk.check("buy", 50, 940)[0]
    assert risk.update(790, t) is not None


def test_paper_broker_accounting(tmp_path):
    broker = PaperBroker(None, "BTC/USDT", fee=0.001, slippage=0.0, start_quote=300, state_path=tmp_path / "p.json")
    fill = broker.market_order("buy", 10, 100.0)  # wants 1000 USDT, only has 300
    base, quote = broker.balances()
    assert quote == pytest.approx(0) and base == pytest.approx(300 / 100.1)
    broker.market_order("sell", base, 100.0)
    assert broker.balances()[1] == pytest.approx(300 * 0.999 / 1.001)
    assert json.loads((tmp_path / "p.json").read_text())["base"] == pytest.approx(0)


class FakeExchange:
    def __init__(self, closes):
        self.df = make_df(closes)
        self.df.index = self.df.index - (self.df.index[-1] - pd.Timestamp.now(tz="UTC")) - pd.Timedelta(hours=8)

    def fetch_ohlcv(self, symbol, timeframe, limit=1000):
        return [[int(ts.timestamp() * 1000), *row] for ts, row in zip(self.df.index, self.df.values.tolist())]

    def fetch_ticker(self, symbol):
        return {"last": float(self.df.close.iloc[-1])}


def test_runner_buys_on_signal_then_holds(tmp_path):
    cfg = cfg_for(tmp_path, strategy="sma_cross", params={"fast": 2, "slow": 5}, max_position_quote=250,
                  budget_quote=300)
    ex = FakeExchange(np.linspace(100, 200, 50))  # uptrend -> long
    broker = PaperBroker(ex, cfg.symbol, cfg.fee, cfg.slippage, cfg.budget_quote, tmp_path / "p.json")
    runner = Runner(cfg, ex, broker, RiskManager(cfg), Alerts())
    assert runner.step() == "buy"
    base, quote = broker.balances()
    assert base * 200 == pytest.approx(250, rel=0.01)  # capped by max_position_quote
    assert runner.step() == "hold"
    assert (tmp_path / "trades_paper.csv").exists()

    ex.df = make_df(np.linspace(200, 180, 50)).set_axis(ex.df.index)  # mild downtrend -> exit
    assert runner.step() == "sell"
    assert broker.balances()[0] == pytest.approx(0)


def test_runner_flattens_and_halts_on_max_drawdown(tmp_path):
    cfg = cfg_for(tmp_path, params={"fast": 2, "slow": 5}, max_position_quote=1000, position_fraction=1.0)
    ex = FakeExchange(np.linspace(100, 200, 50))
    broker = PaperBroker(ex, cfg.symbol, 0, 0, 300, tmp_path / "p.json")
    risk = RiskManager(cfg)
    runner = Runner(cfg, ex, broker, risk, Alerts())
    assert runner.step() == "buy"
    ex.df = make_df(np.linspace(100, 100, 50)).set_axis(ex.df.index)  # price halves
    assert runner.step() == "halted"
    assert broker.balances()[0] == 0 and risk.killed()
    assert runner.step() == "killed"


def test_walk_forward_is_out_of_sample():
    df = random_walk(4000)
    wf = walk_forward(df, "4h", 0.001, 0.0005, train_days=180, test_days=60)
    train_bars, test_bars = 180 * 6, 60 * 6
    n = wf.summary.attrs["n_windows"]
    assert n == (len(df) - train_bars - test_bars) // test_bars + 1
    assert wf.summary.attrs["out_of_sample"][0] == df.index[train_bars]
    assert wf.equity["sma_cross"].index[-1] == df.index[-1]
    assert set(wf.summary.index) == set(STRATEGIES) | {"buy_and_hold"}
    # changing data after the first test window must not change the first window's chosen params
    tampered = df.copy()
    tampered.iloc[train_bars + test_bars:] *= 2
    wf2 = walk_forward(tampered, "4h", 0.001, 0.0005, train_days=180, test_days=60)
    first = lambda w: w.windows[w.windows.window == 0].set_index("strategy").params.to_dict()  # noqa: E731
    assert first(wf) == first(wf2)


def test_runner_writes_status_session_and_equity_log(tmp_path):
    cfg = cfg_for(tmp_path, params={"fast": 2, "slow": 5})
    ex = FakeExchange(np.linspace(100, 200, 50))
    runner = Runner(cfg, ex, PaperBroker(ex, cfg.symbol, 0, 0, 300, tmp_path / "p.json"), RiskManager(cfg), Alerts())
    runner.step()
    runner.step()
    journal, session, status, equity = state_paths(cfg)
    assert json.loads(session.read_text())["start_equity"] == 300
    st = json.loads(status.read_text())
    assert st["action"] == "hold" and st["target"] == 1.0 and st["base"] > 0 and st["kill_switch"] is None
    assert list(pd.read_csv(equity).action) == ["buy", "hold"]


def test_reconcile_matches_paper_session(tmp_path):
    closes = np.r_[np.linspace(100, 120, 30), np.linspace(120, 100, 30), np.linspace(100, 130, 30)]
    df = make_df(closes)
    target = STRATEGIES["sma_cross"][0](df, fast=2, slow=5)
    start = df.index[20] + pd.Timedelta(hours=4, seconds=15)  # bot started right after candle 20 closed
    expected = expected_trades(df, target, start, "4h")
    assert list(expected.side[:3]) == ["buy", "sell", "buy"] and expected.time.iloc[0] == start

    # actual trades: same as expected, 15s late, 10 bps worse; plus one spurious extra trade
    actual = expected.assign(time=expected.time + pd.Timedelta(seconds=15),
                             price=expected.price * np.where(expected.side == "buy", 1.001, 0.999))
    actual = pd.concat([actual, pd.DataFrame([{"time": df.index[-1], "side": "sell", "price": 130.0}])])
    actual = actual.assign(cost=100.0, equity_before=300.0)
    rec = reconcile(df, target, actual, pd.DataFrame(columns=["time", "equity"]), start, 300, "4h", 0.001, 0.0005, 1.0)
    s = rec.summary
    assert s["matched"] == len(expected) and s["missed"] == 0 and s["extra"] == 1
    assert s["avg_cost_vs_expected_bps"] == pytest.approx(10)

    rec = reconcile(df, target, actual.iloc[1:], pd.DataFrame(columns=["time", "equity"]), start, 300, "4h", 0.001,
                    0.0005, 1.0)
    assert rec.summary["missed"] >= 1


class FakeAccountExchange(FakeExchange):
    """A Bitvavo-like account that also holds the user's personal funds."""
    id = "bitvavo"

    def __init__(self, closes, btc, eur, fee=0.0025):
        super().__init__(closes)
        self.free, self.fee, self.orders = {"BTC": btc, "EUR": eur}, fee, []

    def load_markets(self):
        pass

    def market(self, symbol):
        return {"base": "BTC", "quote": "EUR"}

    def fetch_balance(self):
        return {"free": dict(self.free)}

    def amount_to_precision(self, symbol, amount):
        return f"{amount:.8f}"

    def create_order(self, symbol, type_, side, amount):
        price = float(self.df.close.iloc[-1])
        cost, fee = amount * price, amount * price * self.fee
        sign = 1 if side == "buy" else -1
        self.free["BTC"] += sign * amount
        self.free["EUR"] -= sign * cost + fee
        assert self.free["BTC"] >= 0 and self.free["EUR"] >= 0
        self.orders.append((side, amount))
        return {"id": str(len(self.orders)), "status": "closed", "filled": amount, "average": price, "cost": cost,
                "fee": {"cost": fee, "currency": "EUR"}}


def test_bot_never_touches_personal_funds(tmp_path):
    cfg = cfg_for(tmp_path, params={"fast": 2, "slow": 5}, budget_quote=250, max_position_quote=1000)
    ex = FakeAccountExchange(np.linspace(100, 200, 50), btc=1.0, eur=1000.0)  # personal: 1 BTC + 1000 EUR
    broker = ExchangeBroker(ex, cfg.symbol, cfg.budget_quote, tmp_path / "ledger.json")
    runner = Runner(cfg, ex, broker, RiskManager(cfg), Alerts())

    assert runner.step() == "buy"
    bought = ex.orders[0][1]
    assert bought * 200 == pytest.approx(250 * 0.95 * 0.995, rel=1e-6)  # only the bot budget is used
    assert broker.ledger.base == pytest.approx(bought)
    assert ex.free["EUR"] == pytest.approx(1000 - bought * 200 * 1.0025)

    ex.df = make_df(np.linspace(200, 180, 50)).set_axis(ex.df.index)
    assert runner.step() == "sell"
    assert ex.orders[1] == ("sell", pytest.approx(bought))  # sells only what the bot bought
    assert ex.free["BTC"] == pytest.approx(1.0)  # personal BTC untouched
    assert broker.ledger.base == 0 and broker.ledger.quote < 250


def test_bot_caps_to_available_account_balance(tmp_path):
    ex = FakeAccountExchange(np.linspace(100, 200, 50), btc=0.0, eur=100.0)  # less EUR than the budget
    broker = ExchangeBroker(ex, "BTC/EUR", 250, tmp_path / "ledger.json")
    assert broker.balances() == (0.0, 100.0)


def test_load_bots_shares_defaults_and_separates_state(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("""
[costs]
fee = 0.0025
[live]
budget_quote = 250.0
state_dir = "st"
[[bots]]
name = "btc"
symbol = "BTC/EUR"
strategy = "donchian_breakout"
params = { entry = 100, exit = 50 }
[[bots]]
name = "eth"
symbol = "ETH/EUR"
budget_quote = 100.0
""")
    btc, eth = load_bots(path)
    assert (btc.name, btc.symbol, btc.fee, btc.budget_quote, btc.state_dir) == ("btc", "BTC/EUR", 0.0025, 250, Path("st/btc"))
    assert (eth.symbol, eth.budget_quote, eth.state_dir, eth.strategy) == ("ETH/EUR", 100, Path("st/eth"), "sma_cross")
    assert load_config(path, "eth").symbol == "ETH/EUR"
    with pytest.raises(KeyError):
        load_config(path, "doge")
    single = tmp_path / "single.toml"
    single.write_text('[market]\nsymbol = "BTC/EUR"\n[live]\nstate_dir = "st"\n')
    assert [(c.name, c.state_dir) for c in load_bots(single)] == [("main", Path("st"))]


def test_step_all_isolates_failing_bot(tmp_path):
    class Broken(FakeExchange):
        def fetch_ohlcv(self, *a, **kw):
            raise ConnectionError("exchange down")

    runners = []
    for name, ex in (("good", FakeExchange(np.linspace(100, 200, 50))), ("bad", Broken(np.linspace(100, 200, 50)))):
        cfg = cfg_for(tmp_path / name, name=name, params={"fast": 2, "slow": 5})
        runners.append(Runner(cfg, ex, PaperBroker(ex, cfg.symbol, 0, 0, 250, tmp_path / f"{name}.json"),
                              RiskManager(cfg), Alerts()))
    errors = {}
    assert step_all(runners, errors) == {"good": "buy", "bad": "error"}
    for _ in range(4):
        step_all(runners, errors)
    assert runners[1].risk.killed() and not runners[0].risk.killed()  # only the failing bot halts
    bad_status = json.loads(runners[1].status.read_text())
    assert bad_status["action"] == "killed" and "exchange down" in bad_status["kill_switch"]
    assert step_all(runners, errors)["good"] == "hold"


def test_size_at_entry_keeps_size_until_exit():
    idx = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
    target = pd.Series([0, 1, 1, 1, 0, 1], index=idx, dtype=float)
    weight = pd.Series([0.9, 0.5, 0.2, 0.8, 0.3, 0.7], index=idx)
    assert list(size_at_entry(target, weight)) == [0, 0.5, 0.5, 0.5, 0, 0.7]


def test_vol_target_has_no_lookahead_and_caps_at_one():
    df = random_walk(3000)
    target = STRATEGIES["donchian_breakout"][0](df, entry=20, exit=10)
    sized = vol_target(df, target, "4h", annual_vol=0.5)
    tampered = df.copy()
    tampered.iloc[2000:] *= np.linspace(1, 3, len(df) - 2000)[:, None]
    assert sized.iloc[:2000].equals(vol_target(tampered, target, "4h", annual_vol=0.5).iloc[:2000])
    assert sized.max() <= 1.0 and (sized[target == 0] == 0).all()


def test_regime_filter_and_blended_costs():
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    target = pd.Series([1, 1, 1, 0], index=idx, dtype=float)
    regime = pd.Series([True, False, True, True], index=idx)
    assert list(regime_filter(target, regime)) == [1, 0, 1, 0]
    assert blended_costs(0.0) == (0.0025, 0.0005)
    assert blended_costs(1.0) == (0.0015, 0.0)
    assert blended_costs(0.5) == pytest.approx((0.002, 0.00025))


class FakeBookExchange(FakeExchange):
    """Adds bid/ask and recent trades for limit-order simulation."""

    def __init__(self, closes, bid, ask, trade_prices):
        super().__init__(closes)
        self.bid, self.ask, self.trade_prices = bid, ask, trade_prices

    def fetch_ticker(self, symbol):
        return {"last": float(self.df.close.iloc[-1]), "bid": self.bid, "ask": self.ask}

    def fetch_trades(self, symbol, since=None, params=None):
        return [{"price": p} for p in self.trade_prices]


def test_paper_limit_order_fills_only_if_price_trades_through(tmp_path):
    ex = FakeBookExchange([100.0] * 10, bid=99.9, ask=100.1, trade_prices=[100.0, 99.8])
    broker = PaperBroker(ex, "BTC/EUR", fee=0.0025, slippage=0.0005, start_quote=250, state_path=tmp_path / "p.json",
                         order_type="limit", maker_fee=0.0015, limit_timeout_s=90, sleep=lambda s: None)
    fill = broker.execute("buy", 2.0, 100.0)  # EUR 200
    assert fill["maker_share"] == 1.0 and fill["price"] == 99.9
    assert fill["cost"] == pytest.approx(200) and fill["fee"] == pytest.approx(200 * 0.0015)

    ex.trade_prices = [100.1, 100.0]  # nobody traded above the ask: sell falls back to market
    fill = broker.execute("sell", broker.ledger.base, 100.0)
    assert fill["maker_share"] == 0.0 and fill["price"] == pytest.approx(100.0 * (1 - 0.0005))
    assert broker.ledger.base == 0


class FakeLimitAccount(FakeAccountExchange):
    """Live-like account where a resting limit order fills only `fill_fraction` before it is cancelled."""

    def __init__(self, closes, btc, eur, fill_fraction):
        super().__init__(closes, btc, eur)
        self.fill_fraction, self.limit_orders = fill_fraction, {}

    def price_to_precision(self, symbol, price):
        return f"{price:.2f}"

    def fetch_ticker(self, symbol):
        last = float(self.df.close.iloc[-1])
        return {"last": last, "bid": last - 0.1, "ask": last + 0.1}

    def create_order(self, symbol, type_, side, amount, price=None, params=None):
        if type_ == "market":
            return super().create_order(symbol, type_, side, amount)
        assert params == {"postOnly": True}
        oid = f"L{len(self.limit_orders)}"
        filled = amount * self.fill_fraction
        cost, fee = filled * price, filled * price * 0.0015
        sign = 1 if side == "buy" else -1
        self.free["BTC"] += sign * filled
        self.free["EUR"] -= sign * cost + fee
        self.limit_orders[oid] = {"id": oid, "status": "canceled", "filled": filled, "average": price, "cost": cost,
                                  "fee": {"cost": fee, "currency": "EUR"}}
        return {"id": oid, "status": "open"}

    def cancel_order(self, oid, symbol):
        pass

    def fetch_order(self, oid, symbol):
        return self.limit_orders[oid]


def test_live_limit_order_partial_fill_then_market_rest(tmp_path):
    ex = FakeLimitAccount([200.0] * 10, btc=1.0, eur=1000.0, fill_fraction=0.4)
    broker = ExchangeBroker(ex, "BTC/EUR", 250, tmp_path / "ledger.json", order_type="limit", sleep=lambda s: None)
    fill = broker.execute("buy", 1.0, 200.0)
    assert fill["amount"] == pytest.approx(0.995)  # 0.4 as limit + 0.6 at market
    assert fill["maker_share"] == pytest.approx(0.4)
    assert ex.orders == [("buy", pytest.approx(0.995 * 0.6))]  # only the remainder went to market
    assert broker.ledger.base == pytest.approx(0.995)
    assert ex.free["BTC"] == pytest.approx(1.995)


def test_volatility_sizing_at_entry_without_top_ups(tmp_path):
    rng = np.random.default_rng(1)
    closes = np.r_[np.full(300, 100.0) * np.exp(np.cumsum(rng.normal(0, 0.04, 300))), np.linspace(1, 1.5, 60)]
    closes[300:] *= closes[299]  # wild history, then a steady breakout
    cfg = cfg_for(tmp_path, params={"fast": 2, "slow": 5}, vol_target=0.5, max_position_quote=1000,
                  position_fraction=1.0)
    ex = FakeExchange(closes)
    broker = PaperBroker(ex, cfg.symbol, 0, 0, 1000, tmp_path / "p.json")
    runner = Runner(cfg, ex, broker, RiskManager(cfg), Alerts())
    assert runner.step() == "buy"
    assert 0 < runner.weight < 1
    assert broker.ledger.base * closes[-1] == pytest.approx(1000 * runner.weight, rel=1e-6)
    assert runner.step() == "hold"  # no top-up even though the position is below a full allocation


def test_journal_migrates_old_columns(tmp_path):
    cfg = cfg_for(tmp_path, params={"fast": 2, "slow": 5})
    ex = FakeExchange(np.linspace(100, 200, 50))
    journal = state_paths(cfg)[0]
    journal.write_text("time,mode,symbol,side,amount,price,cost,fee,order_id,equity_before\n"
                       "2026-01-01T00:00:00+00:00,paper,BTC/EUR,buy,1,100,100,0.1,paper,250\n")
    Runner(cfg, ex, PaperBroker(ex, cfg.symbol, 0, 0, 300, tmp_path / "p.json"), RiskManager(cfg), Alerts()).step()
    df = pd.read_csv(journal)
    assert list(df.columns)[-1] == "maker_share" and len(df) == 2 and df.maker_share.iloc[1] == 0


def test_runner_respects_kill_switch(tmp_path):
    cfg = cfg_for(tmp_path, params={"fast": 2, "slow": 5})
    ex = FakeExchange(np.linspace(100, 200, 50))
    risk = RiskManager(cfg)
    risk.trip("manual")
    runner = Runner(cfg, ex, PaperBroker(ex, cfg.symbol, 0, 0, 300, tmp_path / "p.json"), risk, Alerts())
    assert runner.step() == "killed"


def test_rogers_satchell_vol_is_drift_independent():
    # A steady uptrend where each bar closes at its high and opens at its low has positive close-to-close
    # variance (because of the drift), whereas Rogers-Satchell OHLC volatility does not penalise pure drift.
    idx = pd.date_range("2024-01-01", periods=300, freq="4h", tz="UTC")
    closes = 100 * np.exp(np.linspace(0, 1.5, 300))
    opens = np.r_[closes[0], closes[:-1]]
    df = pd.DataFrame({"open": opens, "high": closes, "low": opens, "close": closes, "volume": 1.0}, index=idx)
    rs = realized_vol(df, "4h", 30, estimator="rogers_satchell").iloc[-1]
    cc = realized_vol(df, "4h", 30, estimator="close").iloc[-1]
    assert rs == pytest.approx(0.0, abs=1e-9) and cc >= 0


def test_uncapped_position_compounds_and_reconcile_respects_cap(tmp_path):
    cfg = cfg_for(tmp_path, strategy="sma_cross", params={"fast": 2, "slow": 5}, max_position_quote=0.0,
                  position_fraction=0.95, budget_quote=1000)
    ex = FakeExchange(np.linspace(100, 200, 50))
    broker = PaperBroker(ex, cfg.symbol, 0, 0, cfg.budget_quote, tmp_path / "p.json")
    runner = Runner(cfg, ex, broker, RiskManager(cfg), Alerts())
    assert runner.step() == "buy"
    base, _ = broker.balances()
    assert base * 200 == pytest.approx(1000 * 0.95, rel=0.01)  # not capped at 250 EUR

    # Reconcile with a hard cap scales down expected returns once equity grows past max_position_quote
    closes = np.r_[np.linspace(100, 200, 30), np.linspace(200, 180, 20), np.linspace(180, 360, 30)]
    df = make_df(closes)
    target = STRATEGIES["sma_cross"][0](df, fast=2, slow=5)
    start = df.index[10] + pd.Timedelta(hours=4)
    empty = pd.DataFrame(columns=["time", "side", "price", "cost", "equity_before"])
    empty_eq = pd.DataFrame(columns=["time", "equity"])
    uncapped = reconcile(df, target, empty, empty_eq, start, 250, "4h", 0, 0, 0.95, max_position_quote=0.0)
    capped = reconcile(df, target, empty, empty_eq, start, 250, "4h", 0, 0, 0.95, max_position_quote=250.0)
    assert uncapped.summary["expected_return"] > capped.summary["expected_return"]


def test_live_broker_cancels_stale_orders_and_handles_cancel_pending(tmp_path):
    import ccxt

    class StaleOrderAccount(FakeLimitAccount):
        def __init__(self):
            super().__init__([200.0] * 10, btc=1.0, eur=1000.0, fill_fraction=1.0)
            self.cancelled = []

        def fetch_open_orders(self, symbol):
            return [{"id": "stale-1"}]

        def cancel_order(self, oid, symbol):
            self.cancelled.append(oid)
            if oid != "stale-1":
                raise ccxt.CancelPending("already filling")

    ex = StaleOrderAccount()
    broker = ExchangeBroker(ex, "BTC/EUR", 250, tmp_path / "ledger.json", order_type="limit", sleep=lambda s: None)
    fill = broker.execute("buy", 1.0, 200.0)
    assert "stale-1" in ex.cancelled
    assert fill["maker_share"] == 1.0


def test_step_all_deduplicates_candle_fetches_and_guards_empty_candles(tmp_path):
    class CountingExchange(FakeExchange):
        def __init__(self, closes):
            super().__init__(closes)
            self.fetches = 0

        def fetch_ohlcv(self, symbol, timeframe, limit=1000):
            self.fetches += 1
            return super().fetch_ohlcv(symbol, timeframe, limit=limit)

    ex = CountingExchange(np.linspace(100, 200, 50))
    runners = [
        Runner(cfg_for(tmp_path / n, name=n, params={"fast": f, "slow": 5}), ex,
               PaperBroker(ex, "BTC/EUR", 0, 0, 250, tmp_path / f"{n}.json"),
               RiskManager(cfg_for(tmp_path / n, name=n)), Alerts())
        for n, f in (("b1", 2), ("b2", 3))
    ]
    assert step_all(runners, {}) == {"b1": "buy", "b2": "buy"}
    assert ex.fetches == 1  # shared (exchange, symbol, timeframe) fetched only once

    class EmptyExchange(FakeExchange):
        def fetch_ohlcv(self, symbol, timeframe, limit=1000):
            return []

    empty_runner = Runner(cfg_for(tmp_path / "e", name="e"), EmptyExchange([100.0] * 10),
                          PaperBroker(ex, "BTC/EUR", 0, 0, 250, tmp_path / "e.json"),
                          RiskManager(cfg_for(tmp_path / "e", name="e")), Alerts())
    with pytest.raises(RuntimeError, match="no closed candles"):
        empty_runner.step()


def test_load_env_and_make_exchange_support_pem_private_keys(tmp_path, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    priv = ed25519.Ed25519PrivateKey.generate()
    pem = priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                             serialization.NoEncryption()).decode()
    escaped = pem.strip().replace("\n", "\\n")
    env_file = tmp_path / ".env"
    env_file.write_text(f"TEST_PEM_SECRET='{escaped}'\n")
    monkeypatch.delenv("TEST_PEM_SECRET", raising=False)
    load_env(env_file)
    ex = make_exchange("binance", api_key="test-key", secret=escaped)
    signed = ex.sign("account", "private", "GET", {})
    assert "signature=" in signed["url"]

