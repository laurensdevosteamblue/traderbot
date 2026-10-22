# AGENTS.md

- Python venv at `.venv` (Python 3.14). Install: `.venv/bin/pip install -r requirements.txt`
- Tests: `.venv/bin/python -m pytest -q` (offline, no exchange access needed)
- CLI: `.venv/bin/python -m traderbot [--config FILE] {fetch,backtest,compare,walkforward,reconcile,dashboard,run,status,kill,resume}`
- Dashboard (`traderbot/dashboard.py`, Streamlit) must stay read-only and file-based: it reads `state/*_<mode>` files
  written by `Runner._snapshot` and never gets API keys. Smoke test: `streamlit.testing.v1.AppTest.from_file("traderbot/dashboard.py")`.
- Default exchange is Bitvavo (BTC/EUR, 0.25% taker fee, orders need `operatorId`, no testnet, candles from 2019-03).
  Live trading goes through `ExchangeBroker` + `Ledger` (`state/ledger_<mode>.json`): the bot must only trade its own
  budget and never sell coins it did not buy, because the account also holds the user's personal crypto.
  `test_bot_never_touches_personal_funds` covers this.
- Production (paper) runs on a Raspberry Pi 4: `ssh laurens@192.168.2.200`, project in `~/traderbot`, Docker Compose
  services `bot` and `dashboard` (LAN port 8501). Deploy with rsync excluding `state`/`data`/`.venv`/`.env`, then
  `DOCKER_BUILDKIT=0 docker build -t traderbot:latest . && docker compose up -d` (BuildKit `buildx` on the Pi's
  Cortex-A72 hits `SIGILL` on ARMv8.2 SHA3 instructions). The Pi's `state/` is authoritative; never run the bot locally
  at the same time.
  Run tests on the Pi: `docker run --rm -v ./tests:/app/tests:ro traderbot:latest python -m pytest -q tests`.
- Multi-bot: `config.toml` = shared sections + `[[bots]]` (name, symbol, strategy, params, overrides).
  `load_bots()` gives one `Config` per bot with `state_dir = state/<name>`; `run` steps all bots in one process via
  `step_all()` (per-bot error isolation). Global CLI flag `--bot NAME` selects one bot.
- `python -m traderbot research` (traderbot/research.py + overlays.py) compares limit-order costs, volatility sizing
  (drift-independent Rogers-Satchell OHLC vs close-to-close) and a BTC trend filter on top of the walk-forward targets.
  Overlay params are fixed up front; never tune them on these results. Volatility sizing (`vol_target`, Rogers-Satchell)
  and limit orders (`order_type = "limit"`, post-only + market fallback, `broker.execute()`) are live; the BTC trend
  filter was rejected (parameter-sensitive).
- Runner sizes a trade only when flat (size fixed at entry, no top-ups). `max_position_quote = 0` disables the fixed EUR
  cap so winning bots compound via `position_fraction`. Journal CSVs gained `maker_share`; older journals are migrated
  automatically on the next trade.
- Candle caches are named `data/<exchange>_<SYMBOL>_<tf>.csv`.
- New deps must be >= 7 days old: `pip install --uploaded-prior-to <date> ...`, then pin in requirements.txt.
  (`--config` must come before the subcommand)
- Strategies live in `traderbot/strategies.py` and are registered in `STRATEGIES` with a param grid.
  They must return 0/1 target series using only data up to the current bar. `test_strategies_have_no_lookahead` enforces this.
- Runtime state (paper account, risk state, KILL file, trade journal, logs) is in `state/`; candle cache in `data/`. Both are gitignored.
- Never commit `.env`. Never weaken the withdrawal-permission check or the `--confirm-live` guard.
