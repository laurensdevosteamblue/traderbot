# traderbot

Long-only crypto spot trading bot, configured for **Bitvavo (BTC/EUR)** by default. Any ccxt exchange works;
Binance is also tested. You work through it in stages: **backtest → paper → live small**. No leverage, no shorting.
The bot only trades its own budget (see "Sharing an account with your own crypto" below).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # only needed for testnet/live keys and Telegram alerts
```

## 1. Backtest (fees + slippage included, no look-ahead)

```bash
.venv/bin/python -m traderbot compare --since 2019-01-01      # all strategies, in-sample vs out-of-sample
.venv/bin/python -m traderbot backtest --strategy donchian_breakout --params entry=55 exit=20
```

`compare` optimises each strategy's parameters on the first 70% of the data, then tests them on the
remaining 30%, which the optimiser never saw. **Only the `oos_*` columns count as evidence.** The `grid_oos_*`
columns show whether the whole parameter grid holds up out-of-sample or only one lucky setting.

Strategies (`traderbot/strategies.py`):
- `sma_cross`: trend-following moving-average crossover
- `donchian_breakout`: trend-following channel breakout ("turtle")
- `rsi_reversion`: buy oversold RSI, sell on recovery, optional trend filter

Signals are computed on closed candles and executed at the next candle's open.

### Walk-forward (the most honest test)

```bash
.venv/bin/python -m traderbot walkforward --symbols BTC/EUR ETH/EUR --train-days 730 --test-days 180
```

Re-tunes every strategy on a rolling 2-year window, trades the next 6 months with those settings, then rolls
forward. The stitched 6-month test windows form one fully out-of-sample track record. `pct_windows_positive`
shows how consistent the strategy is from window to window. For live trading, use the params the
walk-forward picked for the **latest** window (that's how its results were produced).

Fees matter a lot: Bitvavo charges 0.25% per market order below EUR 100k monthly volume, 2.5× Binance's
0.1%. Always backtest with the fee of the exchange you actually trade on.

## Multiple bots

`config.toml` holds shared settings plus one `[[bots]]` entry per coin. Each entry can override any key,
e.g. `symbol`, `strategy`, `params` or `budget_quote`. Every bot has its own budget, ledger, risk limits, kill switch
and state folder (`state/<name>/`). All bots run in one process; if one fails, the others carry on.

```bash
.venv/bin/python -m traderbot status                  # all bots + total equity
.venv/bin/python -m traderbot --bot eth reconcile     # one bot
.venv/bin/python -m traderbot --bot sol kill          # stop one bot; `kill` without --bot stops all
```

Current setup: 5 paper bots (BTC, ETH, SOL, XRP, ADA), EUR 250 each, all running `donchian_breakout`.

## Order execution and position sizing

Both were validated with `python -m traderbot research` (walk-forward, settings fixed up front):

- **Limit orders** (`order_type = "limit"`): a post-only order at the best bid/ask pays the 0.15% maker fee instead
  of 0.25%. If it hasn't filled after `limit_timeout_s`, the rest goes out as a market order. Paper mode counts a limit
  order as filled only if real trades went strictly through its price. `reconcile` and the dashboard show the
  actual fill rate (the backtest assumes 50%). Max-drawdown exits always use market orders.
- **Volatility sizing** (`vol_target = 0.5`): each new trade is sized at `min(1, 0.5 / the coin's 30-day annualised
  Rogers-Satchell OHLC volatility)` and keeps that size until the exit. Using drift-independent OHLC volatility avoids
  shrinking position sizes after a strong upside trend, cutting the portfolio's max drawdown from ~31% to ~20% in the
  walk-forward with higher Sharpe (~1.60) and Calmar across the 40/50/70% settings tested.
- A BTC trend filter for altcoins was tested and **not** adopted: its results depended too much on the setting.

## 2. Paper trade (several weeks)

Set `mode = "paper"` in `config.toml`, then:

```bash
.venv/bin/python -m traderbot run          # loop: acts shortly after each candle closes
.venv/bin/python -m traderbot status
```

Fills are simulated against live exchange prices. Every trade is written to `state/trades_<mode>.csv`.
Each step also writes `state/status_<mode>.json`, `state/equity_<mode>.csv` and `state/session_<mode>.json`.

Check regularly whether the bot is doing what the backtest predicts:

```bash
.venv/bin/python -m traderbot reconcile
```

This reports matched, missed and extra trades, how much worse your fills were than the backtest price (in bps),
and the expected vs actual return for the session. Missed or extra trades point to a bug. Fill costs well above
the assumed `slippage` mean the backtest is too optimistic. The very first trade of a session is usually off
because the bot starts in the middle of a candle.

## Dashboard

```bash
.venv/bin/python -m traderbot dashboard     # http://127.0.0.1:8501
```

Read-only Streamlit app with five tabs:
- **Live:** equity, position, signal, risk-limit gauges, a stale-bot warning, the equity curve vs buy & hold, and trades.
- **Paper vs backtest:** the reconcile results.
- **Backtest:** interactive single backtest.
- **Compare:** the in-sample vs out-of-sample comparison.
- **Walk-forward:** the rolling test.

The dashboard only reads `state/` files and public candles. It needs no API keys and cannot place orders.
It binds to localhost only (`.streamlit/config.toml`). On a VPS, reach it with
`ssh -L 8501:localhost:8501 user@vps`; never expose it publicly.

## 3. Live, small

Bitvavo has no testnet, so paper trading is the last step before real money. (With `exchange = "binance"`,
`mode = "testnet"` against https://testnet.binance.vision is also available.)

1. In the Bitvavo app/website, create an API key with **View + Trade only. Never enable Withdraw.**
   Whitelist your server's IP. Put the key in `.env` as `BITVAVO_API_KEY` / `BITVAVO_API_SECRET`.
2. Make sure at least `budget_quote` EUR is free on the account (deposit it, or sell some crypto).
3. Set `mode = "live"`, then:

```bash
.venv/bin/python -m traderbot status                 # shows bot balances AND whole-account balances
.venv/bin/python -m traderbot run --confirm-live
```

Bitvavo's API can't report a key's permissions, so the bot can't verify that withdrawals are disabled. Check it yourself.

### Sharing an account with your own crypto

The bot keeps its own ledger (`state/ledger_live.json`). It starts at `budget_quote` EUR and 0 coins, and is
updated with every fill. That means:

- **Buys** only use the bot's EUR budget plus any profits.
- **Sells** only sell coins the bot bought itself. Your personal BTC is never sold, even when the bot exits.
- **Equity, drawdown and loss limits** count only the bot's money.
- If you move or spend funds so the account holds less than the ledger, the bot trades only what is actually there and logs a warning.

Limitations:
- **This separation is enforced by the bot's code, not by Bitvavo.** The API key itself can trade
  everything on the account. For isolation enforced by the exchange, open the bot's account at a different exchange.
- **Don't trade BTC/EUR manually on the same account while the bot is running** if you can avoid it.
  It still works, but your personal trades and the bot's will be mixed together in Bitvavo's history.
- **To add or withdraw bot money:** stop the bot, edit `quote` in the ledger file, then restart.

## Risk controls (`[risk]` in config.toml)

| Control | Effect |
|---|---|
| `position_fraction`, `max_position_quote` | cap on position size (`max_position_quote = 0` disables the fixed EUR cap so profits compound) |
| `daily_loss_limit_pct` | blocks new buys for the rest of the UTC day (exits still allowed) |
| `max_trades_per_day` | guards against runaway loops |
| `max_drawdown_pct` | flattens the position and activates the kill switch |
| Kill switch | `python -m traderbot kill` / `resume` (or create/delete `state/KILL`); blocks all orders |
| 5 consecutive errors | activates the kill switch |

Telegram alerts are sent for trades, blocked orders, errors and halts when `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set.

## Deploy

### Docker (current setup: Raspberry Pi at 192.168.2.200, `~/traderbot`)

```bash
rsync -az --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache --exclude .env \
      --exclude state --exclude data ./ laurens@192.168.2.200:~/traderbot/      # push code changes
ssh laurens@192.168.2.200 'cd ~/traderbot && DOCKER_BUILDKIT=0 docker build -t traderbot:latest . && docker compose up -d'
ssh laurens@192.168.2.200 'cd ~/traderbot && docker compose logs -f bot'
```

- `bot`: the trading loop. It is the only container that gets `.env` secrets. For live trading, add `--confirm-live` to its command.
- `dashboard`: reachable on the LAN at http://192.168.2.200:8501, with `state/` mounted read-only and no secrets.
  Never publish it to the internet (e.g. via nginx-proxy-manager).
- Both use `restart: unless-stopped`, so they come back after a reboot. The **Pi's** `state/` folder is the
  source of truth; don't overwrite it from the laptop (the rsync command above excludes it).

### systemd (alternative)

See `deploy/traderbot.service` (runs as an unprivileged user from `/opt/traderbot`).

## Taxes (Belgium)

Frequent trading can be taxed as speculative income or professional income instead of being tax-free
"normal management of private assets". Keep the trade journal CSVs and ask a tax adviser.
