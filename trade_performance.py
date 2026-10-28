"""Persistent, recommendations-only intraday performance tracking."""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import requests

import option_positions


SCHEMA_VERSION = 5
ACTIVE_STATUSES = ("tracking", "t1_hit")
WIN_STATUSES = ("t1_hit", "t2_hit", "t1_then_breakeven")
RESOLVED_STATUSES = (*WIN_STATUSES, "stop_before_t1")


def connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 15000")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY,
            strategy TEXT NOT NULL CHECK (strategy IN ('breakout', 'trend_pullback')),
            session_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL CHECK (side IN ('CE', 'PE')),
            security_id INTEGER NOT NULL CHECK (security_id > 0),
            signal_time TEXT NOT NULL,
            entry_at TEXT,
            reference_entry REAL NOT NULL,
            stop_price REAL NOT NULL,
            target_1 REAL NOT NULL,
            target_2 REAL NOT NULL,
            last_price REAL,
            option_security_id INTEGER,
            option_expiry TEXT,
            option_instrument TEXT NOT NULL DEFAULT 'OPTSTK',
            option_segment TEXT NOT NULL DEFAULT 'NSE_FNO',
            option_entry_ltp REAL,
            option_entry_bid REAL,
            option_entry_ask REAL,
            option_lot_size INTEGER,
            signal_json TEXT,
            data_capture_error TEXT,
            status TEXT NOT NULL CHECK (
                status IN ('tracking', 'stop_before_t1', 't1_hit', 't2_hit',
                           't1_then_breakeven', 'unresolved')
            ),
            target_1_at TEXT,
            target_2_at TEXT,
            stop_at TEXT,
            last_candle_at TEXT,
            tracking_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (strategy, session_date, symbol, side)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_trades_day_status
            ON paper_trades(session_date, status);
        CREATE INDEX IF NOT EXISTS idx_paper_trades_day_strategy
            ON paper_trades(session_date, strategy);
        CREATE TABLE IF NOT EXISTS paper_trade_events (
            id INTEGER PRIMARY KEY,
            trade_id INTEGER NOT NULL REFERENCES paper_trades(id) ON DELETE CASCADE,
            candle_time TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            event TEXT NOT NULL CHECK (
                event IN ('none', 'stop_before_t1', 'target_1', 'target_2',
                          'breakeven_after_t1')
            ),
            recorded_at TEXT NOT NULL,
            UNIQUE (trade_id, candle_time)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_trade_events_trade_time
            ON paper_trade_events(trade_id, candle_time);
        CREATE TABLE IF NOT EXISTS paper_instrument_candles (
            id INTEGER PRIMARY KEY,
            trade_id INTEGER NOT NULL REFERENCES paper_trades(id) ON DELETE CASCADE,
            instrument_type TEXT NOT NULL CHECK (instrument_type IN ('underlying', 'option')),
            exchange_segment TEXT NOT NULL,
            security_id INTEGER NOT NULL,
            interval_minutes INTEGER NOT NULL CHECK (interval_minutes = 1),
            candle_time TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL,
            open_interest REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE (trade_id, instrument_type, candle_time)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_candles_trade_time
            ON paper_instrument_candles(trade_id, instrument_type, candle_time);
        CREATE TABLE IF NOT EXISTS paper_market_ticks (
            instrument_segment TEXT NOT NULL,
            security_id INTEGER NOT NULL,
            session_date TEXT NOT NULL,
            sampled_at TEXT NOT NULL,
            last_price REAL,
            last_trade_at TEXT,
            best_bid REAL,
            best_ask REAL,
            bid_quantity INTEGER,
            ask_quantity INTEGER,
            volume INTEGER,
            open_interest INTEGER,
            source TEXT NOT NULL CHECK (source = 'dhan_websocket_full'),
            PRIMARY KEY (instrument_segment, security_id, session_date, sampled_at)
        );
        CREATE INDEX IF NOT EXISTS idx_paper_ticks_instrument_time
            ON paper_market_ticks(instrument_segment, security_id, sampled_at);
        CREATE TABLE IF NOT EXISTS performance_reports (
            session_date TEXT PRIMARY KEY,
            status TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed')),
            html_path TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            delivered_at TEXT,
            telegram_message_id TEXT,
            last_error TEXT
        );
        """
    )
    connection.executescript(option_positions.SCHEMA_SQL)
    columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(paper_trades)").fetchall()
    }
    tick_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(paper_market_ticks)").fetchall()
    }
    if "session_date" not in tick_columns:
        connection.execute(
            "ALTER TABLE paper_market_ticks ADD COLUMN session_date TEXT NOT NULL DEFAULT ''"
        )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_paper_ticks_session_instrument_time
        ON paper_market_ticks(session_date, instrument_segment, security_id, sampled_at)
        """
    )
    if "last_price" not in columns:
        connection.execute("ALTER TABLE paper_trades ADD COLUMN last_price REAL")
    new_columns = {
        "option_security_id": "INTEGER",
        "option_expiry": "TEXT",
        "option_instrument": "TEXT NOT NULL DEFAULT 'OPTSTK'",
        "option_segment": "TEXT NOT NULL DEFAULT 'NSE_FNO'",
        "option_entry_ltp": "REAL",
        "option_entry_bid": "REAL",
        "option_entry_ask": "REAL",
        "option_lot_size": "INTEGER",
        "signal_json": "TEXT",
        "data_capture_error": "TEXT",
        "entry_at": "TEXT",
    }
    for column, definition in new_columns.items():
        if column not in columns:
            connection.execute(
                f"ALTER TABLE paper_trades ADD COLUMN {column} {definition}"
            )
            if column == "entry_at":
                connection.execute(
                    "UPDATE paper_trades SET entry_at = signal_time WHERE entry_at IS NULL"
                )
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    connection.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (1, ?)",
        (now,),
    )
    connection.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, now),
    )
    connection.commit()
    if os.name == "posix":
        os.chmod(database, 0o600)
    return connection


def _as_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _timestamp(session_date: dt.date, value: str | None) -> dt.datetime:
    if value:
        parsed = dt.datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
        return parsed
    return dt.datetime.combine(
        session_date,
        dt.time(0),
        tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)),
    )


def record_buy_recommendations(
    connection: sqlite3.Connection,
    strategy: str,
    snapshot: dict,
    universe: dict[str, int],
    session_date: dt.date,
) -> int:
    """Persist each strategy/symbol/side BUY once per session."""
    scan_time = snapshot.get("scan_time")
    inserted = 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    for result in snapshot.get("results", []):
        side = result.get("side")
        if not str(result.get("verdict", "")).startswith("BUY_") or side not in ("CE", "PE"):
            continue
        symbol = result.get("symbol")
        plan = result.get("plan") or {}
        security_id = universe.get(symbol)
        entry = _as_number(plan.get("entry", result.get("ltp")))
        stop = _as_number(plan.get("stop"))
        target_1 = _as_number(plan.get("t1"))
        target_2 = _as_number(plan.get("t2"))
        if not symbol or not security_id or not all(
            value is not None for value in (entry, stop, target_1, target_2)
        ):
            print(
                f"Paper tracker skipped malformed {strategy} BUY record "
                f"for {symbol or 'unknown symbol'}.",
                file=sys.stderr,
            )
            continue
        ordered_levels = (
            stop < entry < target_1 <= target_2
            if side == "CE"
            else stop > entry > target_1 >= target_2
        )
        if not ordered_levels:
            print(
                f"Paper tracker skipped {strategy} BUY for {symbol}: "
                "entry, stop, and targets are not ordered for its side.",
                file=sys.stderr,
            )
            continue
        signal_price = _as_number(result.get("ltp"))
        entry_at = (
            _timestamp(session_date, scan_time).isoformat()
            if signal_price is not None
            and (signal_price >= entry if side == "CE" else signal_price <= entry)
            else None
        )

        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO paper_trades (
                strategy, session_date, symbol, side, security_id, signal_time, entry_at,
                    reference_entry, stop_price, target_1, target_2, option_security_id,
                    option_expiry, option_entry_ltp, option_entry_bid, option_entry_ask,
                    option_lot_size, signal_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'tracking', ?, ?)
            """,
            (
                strategy,
                session_date.isoformat(),
                symbol,
                side,
                int(security_id),
                _timestamp(session_date, scan_time).isoformat(),
                entry_at,
                entry,
                stop,
                target_1,
                target_2,
                (result.get("option") or {}).get("security_id"),
                (result.get("option") or {}).get("expiry"),
                _as_number((result.get("option") or {}).get("premium")),
                _as_number((result.get("option") or {}).get("bid")),
                _as_number((result.get("option") or {}).get("ask")),
                (result.get("option") or {}).get("lot_size"),
                json.dumps(
                    {"strategy": strategy, "scan_time": scan_time, "result": result},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                now,
                now,
            ),
        )
        inserted += cursor.rowcount
    connection.commit()
    return inserted


def import_saved_signals(
    connection: sqlite3.Connection,
    strategy: str,
    path: Path,
    universe: dict[str, int],
    session_date: dt.date,
) -> int:
    """Recover same-day BUYs saved by the scanners before a runner restart."""
    if not path.exists():
        return 0
    inserted = 0
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            signal = json.loads(line)
            side = signal.get("side")
            symbol = signal.get("symbol")
            plan = signal.get("plan") or {}
            security_id = universe.get(symbol)
            entry = _as_number(plan.get("entry", signal.get("ltp")))
            stop = _as_number(plan.get("stop"))
            target_1 = _as_number(plan.get("t1"))
            target_2 = _as_number(plan.get("t2"))
            if (side not in ("CE", "PE") or not symbol or not security_id
                    or not all(value is not None for value in (entry, stop, target_1, target_2))):
                print(
                    f"Paper tracker skipped malformed saved {strategy} signal "
                    f"for {symbol or 'unknown symbol'}.",
                    file=sys.stderr,
                )
                continue
            ordered_levels = (
                stop < entry < target_1 <= target_2
                if side == "CE"
                else stop > entry > target_1 >= target_2
            )
            if not ordered_levels:
                print(
                    f"Paper tracker skipped saved {strategy} signal for {symbol}: "
                    "entry, stop, and targets are not ordered for its side.",
                    file=sys.stderr,
                )
                continue
            signal_time = signal.get("scan_time") or signal.get("time")
            if signal_time and len(signal_time) <= 8:
                signal_time = f"{session_date.isoformat()} {signal_time}"
            signal_price = _as_number(signal.get("ltp"))
            entry_at = (
                _timestamp(session_date, signal_time).isoformat()
                if signal_price is not None
                and (signal_price >= entry if side == "CE" else signal_price <= entry)
                else None
            )
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO paper_trades (
                    strategy, session_date, symbol, side, security_id, signal_time, entry_at,
                    reference_entry, stop_price, target_1, target_2, option_security_id,
                    option_expiry, option_entry_ltp, option_entry_bid, option_entry_ask,
                    option_lot_size, signal_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'tracking', ?, ?)
                """,
                (
                    strategy, session_date.isoformat(), symbol, side, int(security_id),
                    _timestamp(session_date, signal_time).isoformat(),
                    entry_at,
                    entry, stop, target_1, target_2,
                    (signal.get("option") or {}).get("security_id"),
                    (signal.get("option") or {}).get("expiry"),
                    _as_number((signal.get("option") or {}).get("premium")),
                    _as_number((signal.get("option") or {}).get("bid")),
                    _as_number((signal.get("option") or {}).get("ask")),
                    (signal.get("option") or {}).get("lot_size"),
                    json.dumps(signal, ensure_ascii=False, separators=(",", ":")),
                    now, now,
                ),
            )
            inserted += cursor.rowcount
    connection.commit()
    return inserted


def _crossed(side: str, price: float, level: float, target: bool) -> bool:
    if side == "CE":
        return price >= level if target else price <= level
    return price <= level if target else price >= level


def _entry_touched(entry: float, candle: pd.Series) -> bool:
    high, low = float(candle["high"]), float(candle["low"])
    return low <= entry <= high


def _process_candle(trade: sqlite3.Row, candle: pd.Series) -> tuple[str, str | None]:
    """Return next status and event; a stop wins ambiguous OHLC candles."""
    side = trade["side"]
    status = trade["status"]
    opening, high, low = (float(candle[name]) for name in ("open", "high", "low"))
    if status == "tracking":
        if (_crossed(side, opening, trade["stop_price"], False)
                or _crossed(side, low if side == "CE" else high, trade["stop_price"], False)):
            return "stop_before_t1", "stop_before_t1"
        target_1_hit = _crossed(side, high if side == "CE" else low, trade["target_1"], True)
        if not target_1_hit:
            return status, "none"
        target_2_hit = _crossed(side, high if side == "CE" else low, trade["target_2"], True)
        if target_2_hit:
            return "t2_hit", "target_2"
        return "t1_hit", "target_1"

    breakeven = trade["reference_entry"]
    if (_crossed(side, opening, breakeven, False)
            or _crossed(side, low if side == "CE" else high, breakeven, False)):
        return "t1_then_breakeven", "breakeven_after_t1"
    target_2_hit = _crossed(side, high if side == "CE" else low, trade["target_2"], True)
    if target_2_hit:
        return "t2_hit", "target_2"
    return status, "none"


def update_open_trades(
    connection: sqlite3.Connection,
    dhan,
    session_date: dt.date,
    now: dt.datetime,
) -> int:
    """Fetch completed post-signal 1-minute candles and persist outcomes idempotently."""
    placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
    trades = connection.execute(
        f"""
        SELECT * FROM paper_trades
        WHERE session_date = ? AND status IN ({placeholders})
        ORDER BY symbol, signal_time
        """,
        (session_date.isoformat(), *ACTIVE_STATUSES),
    ).fetchall()
    if not trades:
        return 0

    grouped: dict[tuple[str, int], list[sqlite3.Row]] = {}
    for trade in trades:
        grouped.setdefault((trade["symbol"], trade["security_id"]), []).append(trade)

    complete_through = pd.Timestamp(now).floor("min") - dt.timedelta(minutes=1)
    processed = 0
    for (symbol, security_id), symbol_trades in grouped.items():
        try:
            candles = dhan.intraday(
                security_id,
                dt.datetime.combine(session_date, dt.time(9, 15), now.tzinfo),
                now,
                interval=1,
            )
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            connection.executemany(
                "UPDATE paper_trades SET tracking_error = ?, updated_at = ? WHERE id = ?",
                [
                    (message, dt.datetime.now(dt.timezone.utc).isoformat(), row["id"])
                    for row in symbol_trades
                ],
            )
            connection.commit()
            print(f"Paper tracking failed for {symbol}: {message}", file=sys.stderr)
            continue

        if candles.empty:
            message = "Dhan returned no intraday candles"
            connection.executemany(
                "UPDATE paper_trades SET tracking_error = ?, updated_at = ? WHERE id = ?",
                [
                    (message, dt.datetime.now(dt.timezone.utc).isoformat(), row["id"])
                    for row in symbol_trades
                ],
            )
            connection.commit()
            print(f"Paper tracking has no candle data for {symbol}.", file=sys.stderr)
            continue
        if candles.index.tz is None:
            candles.index = candles.index.tz_localize(now.tzinfo)
        else:
            candles.index = candles.index.tz_convert(now.tzinfo)

        for trade in symbol_trades:
            signal_time = pd.Timestamp(trade["signal_time"]).tz_convert(now.tzinfo)
            first_following_bar = signal_time.ceil("min")
            last_candle = (
                pd.Timestamp(trade["last_candle_at"]).tz_convert(now.tzinfo)
                if trade["last_candle_at"] else None
            )
            for candle_time, candle in candles.iterrows():
                if candle_time < first_following_bar or candle_time > complete_through:
                    continue
                if last_candle is not None and candle_time <= last_candle:
                    continue
                current_trade = connection.execute(
                    "SELECT * FROM paper_trades WHERE id = ?", (trade["id"],)
                ).fetchone()
                if current_trade["status"] not in ACTIVE_STATUSES:
                    break
                candle_iso = candle_time.isoformat()
                entry_at = current_trade["entry_at"]
                if entry_at is None:
                    if _entry_touched(current_trade["reference_entry"], candle):
                        entry_at = candle_iso
                    status, event = current_trade["status"], "none"
                else:
                    status, event = _process_candle(current_trade, candle)
                recorded_at = dt.datetime.now(dt.timezone.utc).isoformat()
                connection.execute(
                    """
                    INSERT OR REPLACE INTO paper_instrument_candles (
                        trade_id, instrument_type, exchange_segment, security_id,
                        interval_minutes, candle_time, open, high, low, close,
                        volume, open_interest, fetched_at
                    ) VALUES (?, 'underlying', 'NSE_EQ', ?, 1, ?, ?, ?, ?, ?, ?, NULL, ?)
                    """,
                    (
                        trade["id"], security_id, candle_iso,
                        float(candle["open"]), float(candle["high"]),
                        float(candle["low"]), float(candle["close"]),
                        float(candle.get("volume", 0)), recorded_at,
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO paper_trade_events (
                        trade_id, candle_time, open, high, low, close, event, recorded_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade["id"], candle_iso,
                        float(candle["open"]), float(candle["high"]),
                        float(candle["low"]), float(candle["close"]),
                        event or "none", recorded_at,
                    ),
                )
                t1_at = candle_iso if event in ("target_1", "target_2") else current_trade["target_1_at"]
                t2_at = candle_iso if event == "target_2" else current_trade["target_2_at"]
                stop_at = candle_iso if event in ("stop_before_t1", "breakeven_after_t1") else current_trade["stop_at"]
                connection.execute(
                    """
                    UPDATE paper_trades
                    SET status = ?, entry_at = ?, target_1_at = ?, target_2_at = ?, stop_at = ?,
                        last_candle_at = ?, last_price = ?, tracking_error = NULL,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        status, entry_at, t1_at, t2_at, stop_at, candle_iso,
                        float(candle["close"]), recorded_at, trade["id"],
                    ),
                )
                connection.commit()
                processed += 1
                last_candle = candle_time
                if status not in ACTIVE_STATUSES:
                    break
    return processed


def archive_one_minute_candles(
    connection: sqlite3.Connection,
    dhan,
    session_date: dt.date,
    now: dt.datetime,
) -> int:
    """Archive post-signal 1-minute underlying and option candles from Dhan."""
    trades = connection.execute(
        "SELECT * FROM paper_trades WHERE session_date = ? ORDER BY id",
        (session_date.isoformat(),),
    ).fetchall()
    if not trades:
        return 0
    connection.execute(
        """
        UPDATE paper_trades
        SET data_capture_error = CASE
            WHEN option_security_id IS NULL
            THEN 'Dhan option security ID unavailable; option candles cannot be archived'
            ELSE NULL
        END
        WHERE session_date = ?
        """,
        (session_date.isoformat(),),
    )
    connection.commit()

    start = dt.datetime.combine(session_date, dt.time(9, 15), now.tzinfo)
    complete_through = pd.Timestamp(now).floor("min") - dt.timedelta(minutes=1)
    instruments: dict[tuple[str, int, str], list[sqlite3.Row]] = {}
    for trade in trades:
        instruments.setdefault(
            ("NSE_EQ", int(trade["security_id"]), "EQUITY"), []
        ).append(trade)
        if trade["option_security_id"]:
            instruments.setdefault(
                (
                    trade["option_segment"] or "NSE_FNO",
                    int(trade["option_security_id"]),
                    trade["option_instrument"] or "OPTSTK",
                ),
                [],
            ).append(trade)

    saved = 0
    for (segment, security_id, instrument), instrument_trades in instruments.items():
        try:
            candles = dhan.intraday(
                security_id,
                start,
                now,
                seg=segment,
                inst=instrument,
                interval=1,
            )
        except Exception as exc:  # noqa: BLE001
            error = f"{segment}/{security_id}: {type(exc).__name__}: {exc}"
            connection.executemany(
                "UPDATE paper_trades SET data_capture_error = ?, updated_at = ? WHERE id = ?",
                [
                    (error, dt.datetime.now(dt.timezone.utc).isoformat(), row["id"])
                    for row in instrument_trades
                ],
            )
            connection.commit()
            print(f"One-minute candle archive failed for {error}", file=sys.stderr)
            continue

        if candles.empty:
            error = f"{segment}/{security_id}: Dhan returned no one-minute candles"
            connection.executemany(
                "UPDATE paper_trades SET data_capture_error = ?, updated_at = ? WHERE id = ?",
                [
                    (error, dt.datetime.now(dt.timezone.utc).isoformat(), row["id"])
                    for row in instrument_trades
                ],
            )
            connection.commit()
            print(error, file=sys.stderr)
            continue
        if candles.index.tz is None:
            candles.index = candles.index.tz_localize(now.tzinfo)
        else:
            candles.index = candles.index.tz_convert(now.tzinfo)

        instrument_type = "option" if segment == "NSE_FNO" else "underlying"
        for trade in instrument_trades:
            signal_time = pd.Timestamp(trade["signal_time"]).tz_convert(now.tzinfo)
            first_complete_bar = signal_time.ceil("min")
            rows = candles.loc[
                (candles.index >= first_complete_bar)
                & (candles.index <= complete_through)
            ]
            for candle_time, candle in rows.iterrows():
                fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()
                connection.execute(
                    """
                    INSERT OR REPLACE INTO paper_instrument_candles (
                        trade_id, instrument_type, exchange_segment, security_id,
                        interval_minutes, candle_time, open, high, low, close,
                        volume, open_interest, fetched_at
                    ) VALUES (?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trade["id"], instrument_type, segment, security_id,
                        candle_time.isoformat(), float(candle["open"]),
                        float(candle["high"]), float(candle["low"]),
                        float(candle["close"]),
                        float(candle.get("volume", 0)),
                        (
                            float(candle["open_interest"])
                            if "open_interest" in candle and pd.notna(candle["open_interest"])
                            else None
                        ),
                        fetched_at,
                    ),
                )
                saved += 1
        connection.commit()
    return saved


def _report_rows(connection: sqlite3.Connection, session_date: dt.date) -> list[sqlite3.Row]:
    return connection.execute(
        """
        SELECT t.*,
            CASE t.status WHEN 't2_hit' THEN t.target_2_at
                WHEN 't1_hit' THEN t.target_1_at
                WHEN 'stop_before_t1' THEN t.stop_at
                WHEN 't1_then_breakeven' THEN t.stop_at
                ELSE t.last_candle_at END AS premium_exit_ref_at,
            (SELECT c.close FROM paper_instrument_candles c
             WHERE c.trade_id = t.id AND c.instrument_type = 'option'
               AND c.candle_time <= CASE t.status WHEN 't2_hit' THEN t.target_2_at
                    WHEN 't1_hit' THEN t.target_1_at
                    WHEN 'stop_before_t1' THEN t.stop_at
                    WHEN 't1_then_breakeven' THEN t.stop_at
                    ELSE t.last_candle_at END
             ORDER BY c.candle_time DESC LIMIT 1) AS option_exit_close,
            (SELECT COUNT(*) FROM paper_market_ticks q
             WHERE q.instrument_segment = t.option_segment
               AND q.security_id = t.option_security_id
               AND q.session_date = t.session_date) AS option_quote_samples,
            (SELECT COUNT(*) FROM paper_market_ticks q
             WHERE q.instrument_segment = 'NSE_EQ'
               AND q.security_id = t.security_id
               AND q.session_date = t.session_date) AS underlying_quote_samples
        FROM paper_trades t
        WHERE t.session_date = ?
        ORDER BY CASE t.strategy WHEN 'breakout' THEN 0 ELSE 1 END, t.symbol, t.side
        """,
        (session_date.isoformat(),),
    ).fetchall()


def _metrics(rows: list[sqlite3.Row]) -> dict:
    total = len(rows)
    winners = sum(row["status"] in WIN_STATUSES for row in rows)
    resolved = sum(row["status"] in RESOLVED_STATUSES for row in rows)
    return {
        "total": total,
        "t1": sum(row["target_1_at"] is not None for row in rows),
        "t2": sum(row["target_2_at"] is not None for row in rows),
        "stopped": sum(row["status"] == "stop_before_t1" for row in rows),
        "breakeven": sum(row["status"] == "t1_then_breakeven" for row in rows),
        "unresolved": sum(row["status"] == "unresolved" for row in rows),
        "resolved": resolved,
        "wins": winners,
        "win_rate": round(winners / resolved * 100, 2) if resolved else None,
    }


def _load_multiframe_signals(report_dir: Path, session_date: dt.date) -> tuple[list[dict], list[str]]:
    path = report_dir / f"daily_hourly_15m_signals_{session_date.isoformat()}.jsonl"
    if not path.exists():
        return [], []
    signals = []
    errors = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                signal = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"Line {line_number}: {exc}")
                continue
            if not isinstance(signal, dict):
                errors.append(f"Line {line_number}: expected a JSON object")
                continue
            signals.append(signal)
    return signals, errors


def _multiframe_outcome(signal: dict) -> str:
    if signal.get("status") != "paper_exit_signal_sent":
        return "Open at report cutoff"
    reason = signal.get("exit_reason")
    if reason == "underlying 2R target":
        return "Underlying 2R target"
    if reason == "underlying stop":
        return "Underlying stop"
    if reason == "mandatory same-day exit":
        return "Time exit"
    return str(reason or "Exit recorded")


def _multiframe_premium_estimate(signal: dict) -> float | None:
    option = signal.get("option") or {}
    exit_quote = signal.get("option_exit_quote") or {}
    entry_ask = _as_number(option.get("ask"))
    exit_bid = _as_number(exit_quote.get("bid"))
    lot_size = _as_number(option.get("lot_size"))
    if entry_ask is None or exit_bid is None or lot_size is None:
        return None
    if entry_ask <= 0 or exit_bid <= 0 or lot_size <= 0:
        return None
    return round((exit_bid - entry_ask) * lot_size, 2)


def _trade_premium_pnl(row: sqlite3.Row) -> float | None:
    """Option 1-minute candle close at the underlying outcome vs the signal-time ask, per lot."""
    keys = row.keys()
    if "option_exit_close" not in keys or row["entry_at"] is None:
        return None
    pnl, _basis = option_positions._pnl(
        row["option_entry_ask"], None, row["option_exit_close"], row["option_lot_size"]
    )
    return pnl


def render_html(
    rows: list[sqlite3.Row],
    session_date: dt.date,
    generated_at: dt.datetime,
    multiframe_signals: list[dict] | None = None,
    multiframe_errors: list[str] | None = None,
    option_rows: list[sqlite3.Row] | None = None,
    swing_funnel: dict[str, int] | None = None,
    market_tick_count: int | None = None,
) -> str:
    option_section = option_positions.render_section(option_rows or [], swing_funnel)
    multiframe_signals = multiframe_signals or []
    multiframe_errors = multiframe_errors or []
    summaries = {
        name: _metrics([row for row in rows if row["strategy"] == name])
        for name in ("breakout", "trend_pullback")
    }
    labels = {
        "tracking": "No target/stop yet",
        "stop_before_t1": "Stopped before T1",
        "t1_hit": "T1 reached; T2 not reached",
        "t2_hit": "T2 reached",
        "t1_then_breakeven": "T1 reached; breakeven exit",
        "unresolved": "Unresolved at report cutoff",
    }
    body_rows = []
    for row in rows:
        last_price = f"{row['last_price']:.2f}" if row["last_price"] is not None else "N/A"
        option_contract = (
            f"{row['symbol']} {row['option_expiry'] or ''} "
            f"{row['side']} #{row['option_security_id']}"
            if row["option_security_id"] else "Unavailable"
        )
        entry_quote = (
            f"{row['option_entry_bid'] or 0:.2f} / {row['option_entry_ask'] or 0:.2f}"
            if row["option_entry_bid"] is not None or row["option_entry_ask"] is not None
            else "Unavailable"
        )
        entry_status = (
            row["entry_at"]
            if row["entry_at"]
            else "Not reached by report cutoff" if row["status"] == "unresolved"
            else "Pending"
        )
        pnl = _trade_premium_pnl(row)
        pnl_text = f"{pnl:.2f} / lot" if pnl is not None else "N/A"
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row['strategy'].replace('_', ' ').title())}</td>"
            f"<td>{html.escape(row['symbol'])}</td>"
            f"<td>{html.escape(row['side'])}</td>"
            f"<td>{html.escape(option_contract)}</td>"
            f"<td>{entry_quote}</td>"
            f"<td>{html.escape(entry_status)}</td>"
            f"<td>{row['reference_entry']:.2f}</td>"
            f"<td>{row['stop_price']:.2f}</td>"
            f"<td>{row['target_1']:.2f}</td>"
            f"<td>{row['target_2']:.2f}</td>"
            f"<td>{last_price}</td>"
            f"<td>{html.escape(row['last_candle_at'] or '')}</td>"
            f"<td class='status {html.escape(row['status'])}'>{html.escape(labels[row['status']])}</td>"
            f"<td>{'Yes' if row['target_1_at'] else 'No'}</td>"
            f"<td>{'Yes' if row['target_2_at'] else 'No'}</td>"
            f"<td>{pnl_text}</td>"
            f"<td>{html.escape(row['tracking_error'] or '')}</td>"
            f"<td>{row['option_quote_samples']}</td>"
            f"<td>{row['underlying_quote_samples']}</td>"
            f"<td>{html.escape(row['data_capture_error'] or '')}</td>"
            "</tr>"
        )
    multiframe_estimates = [
        estimate for signal in multiframe_signals
        if (estimate := _multiframe_premium_estimate(signal)) is not None
    ]
    multiframe_closed = [
        signal for signal in multiframe_signals
        if signal.get("status") == "paper_exit_signal_sent"
    ]
    multiframe_wins = sum(estimate > 0 for estimate in multiframe_estimates)
    multiframe_summary = (
        "<section class='summary daily_hourly_15m'>"
        "<h2>15-minute Long Option (same-day, paper)</h2>"
        "<div class='cards'>"
        f"<div><b>{len(multiframe_signals)}</b><span>Signals</span></div>"
        f"<div><b>{len(multiframe_closed)}</b><span>Exit signals recorded</span></div>"
        f"<div><b>{sum(signal.get('exit_reason') == 'underlying 2R target' for signal in multiframe_signals)}</b>"
        "<span>Underlying 2R targets</span></div>"
        f"<div><b>{sum(signal.get('exit_reason') == 'underlying stop' for signal in multiframe_signals)}</b>"
        "<span>Underlying stops</span></div>"
        f"<div><b>{sum(signal.get('exit_reason') == 'mandatory same-day exit' for signal in multiframe_signals)}</b>"
        "<span>Time exits</span></div>"
        f"<div><b>{multiframe_wins}/{len(multiframe_estimates)}</b>"
        "<span>Positive indicative option quote P&amp;L (priced exits)</span></div>"
        "</div></section>"
    )
    if multiframe_signals:
        multiframe_rows = []
        for signal in multiframe_signals:
            option = signal.get("option") or {}
            exit_quote = signal.get("option_exit_quote") or {}
            estimate = _multiframe_premium_estimate(signal)
            estimate_text = (
                f"{estimate:.2f} / lot" if estimate is not None else "N/A"
            )
            status = _multiframe_outcome(signal)
            multiframe_rows.append(
                "<tr>"
                f"<td>{html.escape(str(signal.get('signal_time') or ''))}</td>"
                f"<td>{html.escape(str(signal.get('symbol') or ''))}</td>"
                f"<td>{html.escape(str(signal.get('side') or ''))}</td>"
                f"<td>{html.escape(str(signal.get('expiry') or ''))} "
                f"{html.escape(str(option.get('strike') or ''))}</td>"
                f"<td>{_as_number(option.get('bid')) or 0:.2f} / "
                f"{_as_number(option.get('ask')) or 0:.2f}</td>"
                f"<td>{_as_number(signal.get('underlying_entry')) or 0:.2f}</td>"
                f"<td>{_as_number(signal.get('underlying_stop')) or 0:.2f}</td>"
                f"<td>{_as_number(signal.get('underlying_target')) or 0:.2f}</td>"
                f"<td>{_as_number(signal.get('underlying_exit_reference')) or 0:.2f}</td>"
                f"<td>{_as_number(exit_quote.get('bid')) or 0:.2f} / "
                f"{_as_number(exit_quote.get('ask')) or 0:.2f}</td>"
                f"<td>{estimate_text}</td>"
                f"<td>{html.escape(status)}</td>"
                f"<td>{html.escape(str(signal.get('exit_time') or ''))}</td>"
                "</tr>"
            )
        multiframe_table = (
            "<div class='table-wrap'><table><thead><tr>"
            "<th>Signal time (IST)</th><th>Symbol</th><th>Side</th><th>Option</th>"
            "<th>Entry bid/ask</th><th>Underlying entry</th><th>Stop</th>"
            "<th>2R target</th><th>Underlying exit</th><th>Exit bid/ask</th>"
            "<th>Indicative option P&amp;L / lot</th><th>Outcome</th><th>Exit time (IST)</th>"
            f"</tr></thead><tbody>{''.join(multiframe_rows)}</tbody></table></div>"
        )
    else:
        multiframe_table = "<p>No 15-minute Long Option signals were recorded for this session.</p>"
    multiframe_error_note = (
        "<p class='note'>15-minute signal-file issues: "
        + html.escape("; ".join(multiframe_errors))
        + "</p>"
        if multiframe_errors else ""
    )
    quote_sample_count = sum(
        int(row["option_quote_samples"] or 0)
        + int(row["underlying_quote_samples"] or 0)
        for row in rows
    )
    quote_feed_dead = (
        rows
        and quote_sample_count == 0
        and market_tick_count is not None
        and market_tick_count == 0
    )
    quote_capture_note = (
        "<section class='summary' style='background:#fff5f5;border-color:#f1b1a8;'>"
        "<h2 style='color:#b42318;'>Live quote feed DOWN this session</h2>"
        "<p class='note'><strong>Zero WebSocket packets were captured for the entire "
        "session.</strong> No ticks reached paper_market_ticks. The Dhan live feed is "
        "either not connecting, failing authentication, or silently decoding to garbage. "
        "Scanner suggestions during this session relied on polled quotes only; option "
        "live quote columns in this report are empty. Fix the quote recorder before the "
        "next session (check client_id/access_token, run `python -m paper_quote_feed "
        "--probe`, review runner logs).</p></section>"
        if quote_feed_dead else ""
    )
    sections = []
    for name, metrics in summaries.items():
        label = name.replace("_", " ").title()
        rate = f"{metrics['win_rate']:.2f}%" if metrics["win_rate"] is not None else "N/A"
        sections.append(
            f"<section class='summary {name}'>"
            f"<h2>{html.escape(label)}</h2>"
            f"<div class='cards'>"
            f"<div><b>{metrics['total']}</b><span>BUY recommendations</span></div>"
            f"<div><b>{metrics['t1']}</b><span>T1 reached</span></div>"
            f"<div><b>{metrics['t2']}</b><span>T2 reached</span></div>"
            f"<div><b>{rate}</b><span>T1 hit rate ({metrics['wins']}/{metrics['resolved']} resolved)</span></div>"
            f"<div><b>{metrics['stopped']}</b><span>Stopped before T1</span></div>"
            f"<div><b>{metrics['unresolved']}</b><span>Unresolved at cutoff</span></div>"
            f"</div></section>"
        )
    table = (
        "<p>No BUY recommendations were recorded today.</p>"
        if not rows else
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Strategy</th><th>Symbol</th><th>Side</th><th>Option contract</th>"
        "<th>Signal option bid/ask</th><th>Entry reached (IST)</th><th>Reference</th>"
        "<th>Stop</th><th>T1</th><th>T2</th><th>Last price</th><th>Price candle (IST)</th><th>Outcome</th>"
        "<th>T1 hit</th><th>T2 hit</th><th>Indicative option P&amp;L (candle close)</th><th>Tracking notes</th>"
        "<th>Option live quote samples</th><th>Underlying live quote samples</th>"
        "<th>Data capture notes</th>"
        f"</tr></thead><tbody>{''.join(body_rows)}</tbody></table></div>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper Trading Performance - {session_date.isoformat()}</title>
<style>
:root{{color-scheme:light;--ink:#152238;--muted:#5b6b82;--line:#dce4ef;--bg:#f3f6fb;}}
body{{font-family:Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--ink);margin:0;padding:24px;}}
main{{max-width:1280px;margin:auto;}}
header{{background:#17345c;color:#fff;padding:24px;border-radius:14px;margin-bottom:20px;}}
h1{{margin:0 0 8px;font-size:26px;}} header p{{margin:4px 0;color:#d6e4f5;}}
.summary{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px;margin:16px 0;}}
.summary h2{{margin:0 0 14px;}} .cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;}}
.cards div{{background:#f4f7fb;border-radius:9px;padding:14px;}} .cards b{{display:block;font-size:23px;color:#17345c;}}
.cards span{{font-size:12px;color:var(--muted);}} .table-wrap{{overflow-x:auto;background:white;border-radius:12px;}}
table{{border-collapse:collapse;width:100%;font-size:13px;}} th,td{{padding:10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap;}}
th{{background:#eaf0f7;color:#263b58;}} .status{{font-weight:700;}}
.status.stop_before_t1{{color:#b42318;}} .status.t1_hit,.status.t1_then_breakeven{{color:#946200;}}
.status.t2_hit{{color:#167647;}} .status.unresolved,.status.tracking{{color:#52657d;}}
.note{{color:var(--muted);font-size:13px;line-height:1.5;margin-top:18px;}}
@media(max-width:600px){{body{{padding:12px;}}header{{padding:18px;}}}}
</style></head><body><main>
<header><h1>Intraday Paper-Trade Performance</h1>
<p>Session: {session_date.isoformat()} IST</p>
<p>Generated: {generated_at.astimezone(dt.timezone(dt.timedelta(hours=5, minutes=30))).strftime('%H:%M:%S')} IST</p>
<p>Recommendations only — no orders were placed.</p></header>
{''.join(sections)}<h2>Recommendation outcomes</h2>{table}
{multiframe_summary}<h2>15-minute Long Option outcomes</h2>{multiframe_table}
{multiframe_error_note}{option_section}{quote_capture_note}
<p class="note">Underlying outcomes use completed 1-minute candles after the signal; a candle starting before a signal is excluded. If one candle crosses both a stop and target, the stop is counted first. This is still not an actual fill or an option P&amp;L simulation. Dhan's live full-feed samples store option/underlying LTP and best bid/ask during the runner session. Live quote samples can be missing before deployment, during feed outages, or for contracts that Dhan does not return. Historical one-minute candles are archived in the local database after the Telegram report; they contain OHLC, not historical bid/ask or proof of order fill. Capture errors are shown per trade.</p>
<p class="note">The 15-minute strategy tracks underlying stop/target/time exits separately. Indicative option P&amp;L uses the entry ask and exit bid snapshots multiplied by lot size; it excludes brokerage, taxes, slippage, and fills, and is not a verified result. An absent signal file means no saved signals were available for this report.</p>
</main></body></html>"""


def send_html_report(token: str, chat_id: str, report_path: Path, caption: str) -> str | None:
    """Upload the HTML report to Telegram and return the Telegram message id."""
    try:
        with report_path.open("rb") as report:
            response = requests.post(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": chat_id, "caption": caption},
                files={"document": (report_path.name, report, "text/html")},
                timeout=(10, 30),
            )
            response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Telegram report upload failed ({type(exc).__name__})") from exc
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("Telegram report upload returned invalid JSON") from exc
    if not isinstance(result, dict) or not result.get("ok"):
        description = result.get("description", "unknown Telegram API error") if isinstance(result, dict) else "unexpected response"
        raise RuntimeError(f"Telegram report upload failed: {description}")
    message = result.get("result")
    return str(message["message_id"]) if isinstance(message, dict) and message.get("message_id") else None


def finalize_day(
    connection: sqlite3.Connection,
    session_date: dt.date,
    report_dir: Path,
    telegram_token: str | None,
    telegram_chat_id: str | None,
    generated_at: dt.datetime,
) -> Path:
    """Finalize unresolved rows, atomically write HTML, and deliver it once to Telegram."""
    existing = connection.execute(
        "SELECT * FROM performance_reports WHERE session_date = ?",
        (session_date.isoformat(),),
    ).fetchone()
    if existing and existing["status"] == "sent":
        return Path(existing["html_path"])

    connection.execute(
        """
        UPDATE paper_trades SET status = 'unresolved', updated_at = ?
        WHERE session_date = ? AND status = 'tracking'
        """,
        (generated_at.astimezone(dt.timezone.utc).isoformat(), session_date.isoformat()),
    )
    connection.commit()
    rows = _report_rows(connection, session_date)
    multiframe_signals, multiframe_errors = _load_multiframe_signals(
        report_dir, session_date
    )
    option_rows = option_positions.report_rows(connection, session_date)
    funnel = option_positions.swing_funnel(report_dir, session_date)
    market_tick_count = connection.execute(
        "SELECT COUNT(*) FROM paper_market_ticks WHERE session_date = ?",
        (session_date.isoformat(),),
    ).fetchone()[0]
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"paper_performance_{session_date.isoformat()}.html"
    temporary_path = report_path.with_suffix(".html.tmp")
    temporary_path.write_text(
        render_html(
            rows,
            session_date,
            generated_at,
            multiframe_signals,
            multiframe_errors,
            option_rows,
            funnel,
            market_tick_count=market_tick_count,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_path, report_path)
    if rows and market_tick_count == 0:
        print(
            f"LIVE QUOTE FEED DOWN for session {session_date.isoformat()}: "
            "paper_market_ticks captured 0 packets. "
            "Report shipped with a red banner. Fix paper_quote_feed before next session.",
            file=sys.stderr,
        )
    connection.execute(
        """
        INSERT INTO performance_reports(session_date, status, html_path, generated_at)
        VALUES (?, 'pending', ?, ?)
        ON CONFLICT(session_date) DO UPDATE SET
            status = 'pending', html_path = excluded.html_path,
            generated_at = excluded.generated_at, last_error = NULL
        """,
        (
            session_date.isoformat(),
            str(report_path),
            generated_at.astimezone(dt.timezone.utc).isoformat(),
        ),
    )
    connection.commit()

    if not telegram_token or not telegram_chat_id:
        error = "Telegram report not sent: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing"
        connection.execute(
            "UPDATE performance_reports SET status = 'failed', last_error = ? WHERE session_date = ?",
            (error, session_date.isoformat()),
        )
        connection.commit()
        print(error, file=sys.stderr)
        return report_path

    by_strategy = {
        name: _metrics([row for row in rows if row["strategy"] == name])
        for name in ("breakout", "trend_pullback")
    }
    caption = (
        f"Paper performance {session_date.isoformat()} IST | "
        f"Breakout {by_strategy['breakout']['wins']}/{by_strategy['breakout']['resolved']} resolved T1 wins; "
        f"Trend Pullback {by_strategy['trend_pullback']['wins']}/"
        f"{by_strategy['trend_pullback']['resolved']} resolved T1 wins. "
        "Underlying-based tracking; no orders placed."
    )
    try:
        message_id = send_html_report(telegram_token, telegram_chat_id, report_path, caption)
    except RuntimeError as exc:
        connection.execute(
            "UPDATE performance_reports SET status = 'failed', last_error = ? WHERE session_date = ?",
            (str(exc), session_date.isoformat()),
        )
        connection.commit()
        raise
    connection.execute(
        """
        UPDATE performance_reports
        SET status = 'sent', delivered_at = ?, telegram_message_id = ?, last_error = NULL
        WHERE session_date = ?
        """,
        (
            generated_at.astimezone(dt.timezone.utc).isoformat(),
            message_id,
            session_date.isoformat(),
        ),
    )
    connection.commit()
    return report_path
