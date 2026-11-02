from __future__ import annotations

import logging
from pathlib import Path

import ccxt
import pandas as pd

log = logging.getLogger(__name__)
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(exchange_id: str = "bitvavo", api_key: str | None = None, secret: str | None = None,
                  sandbox: bool = False, operator_id: int | None = None) -> ccxt.Exchange:
    options = {"defaultType": "spot"}
    if operator_id is not None:
        options["operatorId"] = operator_id  # Bitvavo requires this on every order
    if secret:
        secret = secret.replace("\\n", "\n")
    ex = getattr(ccxt, exchange_id)({"apiKey": api_key, "secret": secret, "enableRateLimit": True,
                                     "options": options})
    if sandbox:
        ex.set_sandbox_mode(True)  # raises NotSupported if the exchange has no testnet
    return ex


def timeframe_seconds(timeframe: str) -> int:
    return ccxt.Exchange.parse_timeframe(timeframe)


def to_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.drop_duplicates("timestamp").set_index("timestamp").sort_index().astype(float)


def closed_candles(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop the still-forming candle so signals never use incomplete data."""
    now = now or pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(seconds=timeframe_seconds(timeframe)) <= now]


def fetch_ohlcv(exchange, symbol: str, timeframe: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    step = timeframe_seconds(timeframe) * 1000
    cursor, end_ms, rows = int(start.timestamp() * 1000), int(end.timestamp() * 1000), []
    while cursor < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            if rows:
                break
            cursor += 1000 * step  # before the market existed: skip ahead
            continue
        rows += batch
        cursor = batch[-1][0] + step
        log.info("fetched %s %s candles up to %s", len(rows), timeframe, pd.Timestamp(batch[-1][0], unit="ms"))
    df = to_frame(rows) if rows else pd.DataFrame(columns=COLUMNS[1:], index=pd.DatetimeIndex([], tz="UTC"))
    return closed_candles(df[df.index < end], timeframe)


def load_history(exchange, symbol: str, timeframe: str, data_dir: Path, since: str,
                 until: str | None = None) -> pd.DataFrame:
    """Load candles from a CSV cache, fetching missing history/new candles from the exchange."""
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"{exchange.id}_{symbol.replace('/', '')}_{timeframe}.csv"
    covered = cache.with_suffix(".since")  # earliest date already requested (market may start later)
    start, now = pd.Timestamp(since, tz="UTC"), pd.Timestamp.now(tz="UTC")
    end = pd.Timestamp(until, tz="UTC") if until else now
    step = pd.Timedelta(seconds=timeframe_seconds(timeframe))

    df = pd.read_csv(cache, index_col="timestamp", parse_dates=True) if cache.exists() else None
    known_start = pd.Timestamp(covered.read_text().strip()) if covered.exists() else None
    if df is None or df.empty or (df.index[0] > start + step and (known_start is None or start < known_start)):
        df = fetch_ohlcv(exchange, symbol, timeframe, start, now)
        covered.write_text(start.isoformat())
    elif df.index[-1] + 2 * step <= now:
        new = fetch_ohlcv(exchange, symbol, timeframe, df.index[-1] + step, now)
        df = pd.concat([df, new])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(cache)
    return df[(df.index >= start) & (df.index < end)]
