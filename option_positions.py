"""Paper lifecycle tracking for the 15-minute and swing long-option strategies.

Recommendations only: nothing here places or modifies an order.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
STRATEGIES = ("multiframe_15m", "swing_option")
LABELS = {"multiframe_15m": "15-minute same-day", "swing_option": "Swing (multi-day)"}
FORCED_EXIT = dt.time(15, 15)
STALE_AFTER_DAYS = 10

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_option_positions (
    id INTEGER PRIMARY KEY,
    strategy TEXT NOT NULL CHECK (strategy IN ('multiframe_15m', 'swing_option')),
    entry_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('CE', 'PE')),
    underlying_security_id INTEGER NOT NULL CHECK (underlying_security_id > 0),
    option_security_id INTEGER,
    option_segment TEXT NOT NULL DEFAULT 'NSE_FNO',
    option_instrument TEXT NOT NULL DEFAULT 'OPTSTK',
    option_expiry TEXT,
    option_strike REAL,
    option_lot_size INTEGER,
    entry_at TEXT NOT NULL,
    entry_underlying REAL NOT NULL,
    stop_price REAL NOT NULL,
    target_price REAL NOT NULL,
    entry_bid REAL,
    entry_ask REAL,
    entry_ltp REAL,
    max_hold_sessions INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('open', 'target_hit', 'stop_hit', 'time_exit', 'unresolved')
    ),
    sessions_held INTEGER NOT NULL DEFAULT 0,
    last_underlying REAL,
    last_checked_at TEXT,
    exit_at TEXT,
    exit_reason TEXT,
    exit_underlying REAL,
    exit_bid REAL,
    exit_ask REAL,
    exit_option_close REAL,
    premium_pnl_per_lot REAL,
    pnl_basis TEXT,
    signal_json TEXT,
    tracking_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (strategy, entry_date, symbol, side)
);
CREATE INDEX IF NOT EXISTS idx_option_positions_status
    ON paper_option_positions(status, entry_date);
"""


def _num(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _pnl(entry_ask, exit_bid, exit_close, lot) -> tuple[float | None, str | None]:
    """Entry ask to exit bid when a bid was captured, else to the option candle close."""
    entry, size = _num(entry_ask), _num(lot)
    if not entry or entry <= 0 or not size or size <= 0:
        return None, None
    bid = _num(exit_bid)
    if bid and bid > 0:
        return round((bid - entry) * size, 2), "entry ask to exit bid snapshot"
    close = _num(exit_close)
    if close and close > 0:
        return round((close - entry) * size, 2), "entry ask to option candle close"
    return None, None


def _insert(connection: sqlite3.Connection, values: dict) -> int:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    cursor = connection.execute(
        f"INSERT OR IGNORE INTO paper_option_positions ({columns}) VALUES ({marks})",
        tuple(values.values()),
    )
    return cursor.rowcount


def _valid_levels(side: str, entry, stop, target) -> bool:
    if entry is None or stop is None or target is None:
        return False
    return stop < entry < target if side == "CE" else stop > entry > target


def sync_multiframe_signals(connection: sqlite3.Connection, path: Path, day: dt.date) -> int:
    """Mirror the 15-minute JSONL signals (entries and recorded exits) into the ledger."""
    if not path.exists():
        return 0
    changed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            signal = json.loads(line)
        except json.JSONDecodeError:
            continue
        side, symbol = signal.get("side"), signal.get("symbol")
        option = signal.get("option") or {}
        entry = _num(signal.get("underlying_entry"))
        stop = _num(signal.get("underlying_stop"))
        target = _num(signal.get("underlying_target"))
        sid = signal.get("underlying_security_id")
        if (side not in ("CE", "PE") or not symbol or not sid
                or not _valid_levels(side, entry, stop, target)):
            print(f"Option-position sync skipped malformed 15m signal for {symbol}.", file=sys.stderr)
            continue
        changed += _insert(connection, {
            "strategy": "multiframe_15m", "entry_date": day.isoformat(),
            "symbol": symbol, "side": side, "underlying_security_id": int(sid),
            "option_security_id": option.get("security_id"),
            "option_expiry": signal.get("expiry"), "option_strike": _num(option.get("strike")),
            "option_lot_size": option.get("lot_size"),
            "entry_at": str(signal.get("signal_time") or day.isoformat()),
            "entry_underlying": entry, "stop_price": stop, "target_price": target,
            "entry_bid": _num(option.get("bid")), "entry_ask": _num(option.get("ask")),
            "entry_ltp": _num(option.get("premium")), "max_hold_sessions": 1,
            "status": "open", "signal_json": json.dumps(signal, separators=(",", ":")),
            "created_at": _now(), "updated_at": _now(),
        })
        if signal.get("status") != "paper_exit_signal_sent":
            continue
        reason = signal.get("exit_reason")
        status = {"underlying 2R target": "target_hit", "underlying stop": "stop_hit"}.get(
            reason, "time_exit")
        quote = signal.get("option_exit_quote") or {}
        pnl, basis = _pnl(option.get("ask"), quote.get("bid"), None, option.get("lot_size"))
        cursor = connection.execute(
            """
            UPDATE paper_option_positions
            SET status = ?, exit_at = ?, exit_reason = ?, exit_underlying = ?,
                exit_bid = ?, exit_ask = ?, premium_pnl_per_lot = ?, pnl_basis = ?,
                updated_at = ?
            WHERE strategy = 'multiframe_15m' AND entry_date = ? AND symbol = ?
              AND side = ? AND status = 'open'
            """,
            (status, signal.get("exit_time"), reason,
             _num(signal.get("underlying_exit_reference")),
             _num(quote.get("bid")), _num(quote.get("ask")), pnl, basis, _now(),
             day.isoformat(), symbol, side),
        )
        changed += cursor.rowcount
    connection.commit()
    return changed


def record_swing_confirmations(connection: sqlite3.Connection, output_dir: Path) -> int:
    """Create a position for each swing candidate whose next-session trigger was confirmed."""
    inserted = 0
    for path in sorted(output_dir.glob("swing_option_candidates_*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"Could not read {path.name}: {exc}", file=sys.stderr)
            continue
        for candidate in report.get("candidates", []):
            if candidate.get("next_session_status") != "trigger_confirmed":
                continue
            option = candidate.get("refreshed_option") or candidate.get("selected_option") or {}
            plan = candidate.get("underlying") or {}
            side, confirmed = candidate.get("side"), candidate.get("confirmed_at")
            stop, target = _num(plan.get("stop")), _num(plan.get("target_2r"))
            entry = _num(candidate.get("confirmed_underlying_close")) or _num(
                plan.get("entry_trigger"))
            sid = candidate.get("underlying_security_id")
            if (side not in ("CE", "PE") or not confirmed or not sid
                    or not _valid_levels(side, entry, stop, target)):
                print(f"Swing position skipped malformed confirmation for "
                      f"{candidate.get('symbol')}.", file=sys.stderr)
                continue
            confirmed_at = pd.Timestamp(confirmed)
            if confirmed_at.tzinfo is None:
                confirmed_at = confirmed_at.tz_localize(IST)
            # The confirming 15m candle is only known once it has closed.
            entry_at = confirmed_at + pd.Timedelta(15, unit="min")
            inserted += _insert(connection, {
                "strategy": "swing_option",
                "entry_date": confirmed_at.date().isoformat(),
                "symbol": candidate["symbol"], "side": side,
                "underlying_security_id": int(sid),
                "option_security_id": option.get("security_id"),
                "option_expiry": candidate.get("expiry"),
                "option_strike": _num(option.get("strike")),
                "option_lot_size": option.get("lot_size"),
                "entry_at": entry_at.isoformat(), "entry_underlying": entry,
                "stop_price": stop, "target_price": target,
                "entry_bid": _num(option.get("bid")), "entry_ask": _num(option.get("ask")),
                "entry_ltp": _num(option.get("premium")),
                "max_hold_sessions": int(
                    (candidate.get("holding") or {}).get("max_trading_sessions") or 3),
                "status": "open",
                "signal_json": json.dumps(candidate, separators=(",", ":")),
                "created_at": _now(), "updated_at": _now(),
            })
    connection.commit()
    return inserted


def _deadline_date(dates: list[dt.date], max_sessions: int) -> dt.date | None:
    """Last permitted session: the max-hold session or the first Friday, whichever is first."""
    for index, date in enumerate(dates):
        if index + 1 >= max_sessions or date.weekday() == 4:
            return date
    return None


def evaluate_position(position: sqlite3.Row, candles: pd.DataFrame, now: dt.datetime) -> dict:
    """Walk completed 15m candles from entry; a stop wins an ambiguous candle."""
    side = position["side"]
    stop, target = position["stop_price"], position["target_price"]
    entry_at = pd.Timestamp(position["entry_at"])
    window = candles.loc[candles.index >= entry_at]
    if window.empty:
        return {"status": "open", "sessions_held": 0}
    dates = sorted({stamp.date() for stamp in window.index})
    deadline = _deadline_date(dates, position["max_hold_sessions"])
    last_price = float(window.iloc[-1]["close"])
    for stamp, candle in window.iterrows():
        if deadline is not None and stamp.date() > deadline:
            break
        opening, high, low = float(candle["open"]), float(candle["high"]), float(candle["low"])
        stop_hit = low <= stop if side == "CE" else high >= stop
        target_hit = high >= target if side == "CE" else low <= target
        if stop_hit:
            gap = min(stop, opening) if side == "CE" else max(stop, opening)
            return {"status": "stop_hit", "reason": "underlying stop", "at": stamp,
                    "price": gap, "sessions_held": dates.index(stamp.date()) + 1}
        if target_hit:
            return {"status": "target_hit", "reason": "underlying 2R target", "at": stamp,
                    "price": target, "sessions_held": dates.index(stamp.date()) + 1}
    local_now = now.astimezone(IST)
    if deadline is not None:
        due = dates[-1] > deadline or (
            dates[-1] == deadline and local_now.date() == deadline
            and local_now.time() >= FORCED_EXIT
        ) or local_now.date() > deadline
        if due:
            final = window.loc[[stamp.date() == deadline for stamp in window.index]]
            return {"status": "time_exit", "reason": "max hold / no weekend carry",
                    "at": final.index[-1], "price": float(final.iloc[-1]["close"]),
                    "sessions_held": dates.index(deadline) + 1}
    return {"status": "open", "sessions_held": len(dates), "last_price": last_price,
            "last_at": window.index[-1]}


def _option_close(dhan, position: sqlite3.Row, at: pd.Timestamp) -> float | None:
    if not position["option_security_id"]:
        return None
    try:
        frame = dhan.intraday(
            int(position["option_security_id"]),
            dt.datetime.combine(at.date(), dt.time(9, 15), IST),
            (at + pd.Timedelta(15, unit="min")).to_pydatetime(),
            seg=position["option_segment"], inst=position["option_instrument"],
            interval=15,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Option exit candle unavailable ({type(exc).__name__}): {exc}", file=sys.stderr)
        return None
    if frame.empty:
        return None
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize(IST)
    eligible = frame.loc[frame.index <= at]
    return float(eligible.iloc[-1]["close"]) if not eligible.empty else None


def update_swing_positions(connection: sqlite3.Connection, dhan, now: dt.datetime,
                           quote_fetcher=None) -> int:
    """Advance each open swing position and record its exit and indicative premium P&L."""
    rows = connection.execute(
        "SELECT * FROM paper_option_positions WHERE strategy = 'swing_option' AND status = 'open'"
    ).fetchall()
    closed = 0
    for position in rows:
        entry_at = pd.Timestamp(position["entry_at"])
        try:
            candles = dhan.intraday(
                int(position["underlying_security_id"]),
                dt.datetime.combine(entry_at.date(), dt.time(9, 15), IST),
                now, interval=15,
            )
        except Exception as exc:  # noqa: BLE001
            if exc.__class__.__name__ == "DhanAuthError":
                raise
            connection.execute(
                "UPDATE paper_option_positions SET tracking_error = ?, updated_at = ? WHERE id = ?",
                (f"{type(exc).__name__}: {exc}", _now(), position["id"]))
            connection.commit()
            continue
        if candles.empty:
            continue
        if candles.index.tz is None:
            candles.index = candles.index.tz_localize(IST)
        complete_through = pd.Timestamp(now).floor("15min") - pd.Timedelta(15, unit="min")
        candles = candles.loc[candles.index <= complete_through]
        result = evaluate_position(position, candles, now)
        if result["status"] == "open":
            connection.execute(
                """UPDATE paper_option_positions SET sessions_held = ?, last_underlying = ?,
                   last_checked_at = ?, tracking_error = NULL, updated_at = ? WHERE id = ?""",
                (result["sessions_held"], result.get("last_price"),
                 result["last_at"].isoformat() if "last_at" in result else None,
                 _now(), position["id"]))
            connection.commit()
            continue
        quote = None
        if quote_fetcher and position["option_security_id"]:
            try:
                quote = quote_fetcher(position)
            except Exception as exc:  # noqa: BLE001
                if exc.__class__.__name__ == "DhanAuthError":
                    raise
                print(f"Swing exit quote unavailable ({type(exc).__name__}): {exc}", file=sys.stderr)
        close = _option_close(dhan, position, result["at"])
        pnl, basis = _pnl(position["entry_ask"], (quote or {}).get("bid"), close,
                          position["option_lot_size"])
        connection.execute(
            """UPDATE paper_option_positions SET status = ?, exit_at = ?, exit_reason = ?,
               exit_underlying = ?, exit_bid = ?, exit_ask = ?, exit_option_close = ?,
               premium_pnl_per_lot = ?, pnl_basis = ?, sessions_held = ?, last_underlying = ?,
               last_checked_at = ?, tracking_error = NULL, updated_at = ? WHERE id = ?""",
            (result["status"], result["at"].isoformat(), result["reason"], result["price"],
             _num((quote or {}).get("bid")), _num((quote or {}).get("ask")), close, pnl, basis,
             result["sessions_held"], result["price"], result["at"].isoformat(), _now(),
             position["id"]))
        connection.commit()
        closed += 1
    return closed


def mark_stale_unresolved(connection: sqlite3.Connection, today: dt.date) -> int:
    cutoff = (today - dt.timedelta(days=STALE_AFTER_DAYS)).isoformat()
    cursor = connection.execute(
        """UPDATE paper_option_positions SET status = 'unresolved', updated_at = ?,
           tracking_error = 'No exit could be evaluated within the stale window'
           WHERE status = 'open' AND entry_date < ?""",
        (_now(), cutoff))
    connection.commit()
    return cursor.rowcount


def report_rows(connection: sqlite3.Connection, session_date: dt.date) -> list[sqlite3.Row]:
    """Positions entered on the session date, plus any still open from earlier sessions."""
    return connection.execute(
        """
        SELECT p.*,
            (SELECT COUNT(*) FROM paper_market_ticks q
             WHERE q.instrument_segment = p.option_segment
               AND q.security_id = p.option_security_id
               AND q.session_date = ?) AS option_quote_samples
        FROM paper_option_positions p
        WHERE p.entry_date = ? OR p.status = 'open'
           OR substr(p.exit_at, 1, 10) = ?
        ORDER BY p.strategy, p.entry_date, p.symbol, p.side
        """,
        (session_date.isoformat(),) * 3,
    ).fetchall()


def swing_funnel(output_dir: Path, session_date: dt.date) -> dict[str, int]:
    """Count swing candidates by next-session status from recent candidate files."""
    counts: dict[str, int] = {}
    for path in sorted(output_dir.glob("swing_option_candidates_*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            signal_day = dt.date.fromisoformat(report["as_of"])
        except (OSError, ValueError, KeyError):
            continue
        if not 0 <= (session_date - signal_day).days <= STALE_AFTER_DAYS:
            continue
        for candidate in report.get("candidates", []):
            key = candidate.get("next_session_status", "pending")
            counts[key] = counts.get(key, 0) + 1
    return counts


def render_section(rows: list[sqlite3.Row], funnel: dict[str, int] | None = None) -> str:
    outcome = {"open": "Open", "target_hit": "Underlying 2R target", "stop_hit": "Underlying stop",
               "time_exit": "Time exit", "unresolved": "Unresolved"}
    body = []
    for row in rows:
        pnl = row["premium_pnl_per_lot"]
        pnl_text = f"{pnl:.2f} / lot" if pnl is not None else "N/A"
        exit_quote = (
            f"{row['exit_bid'] or 0:.2f} / {row['exit_ask'] or 0:.2f}"
            if row["exit_bid"] is not None or row["exit_ask"] is not None else "N/A"
        )
        last = row["exit_underlying"] if row["exit_underlying"] is not None else row["last_underlying"]
        body.append(
            "<tr>"
            f"<td>{html.escape(LABELS[row['strategy']])}</td>"
            f"<td>{html.escape(row['entry_at'])}</td>"
            f"<td>{html.escape(row['symbol'])}</td><td>{row['side']}</td>"
            f"<td>{html.escape(str(row['option_expiry'] or ''))} {row['option_strike'] or ''}</td>"
            f"<td>{row['entry_bid'] or 0:.2f} / {row['entry_ask'] or 0:.2f}</td>"
            f"<td>{row['entry_underlying']:.2f}</td><td>{row['stop_price']:.2f}</td>"
            f"<td>{row['target_price']:.2f}</td>"
            f"<td>{f'{last:.2f}' if last is not None else 'N/A'}</td>"
            f"<td>{html.escape(outcome[row['status']])}</td>"
            f"<td>{row['sessions_held']}/{row['max_hold_sessions']}</td>"
            f"<td>{html.escape(row['exit_at'] or '')}</td>"
            f"<td>{exit_quote}</td><td>{pnl_text}</td>"
            f"<td>{html.escape(row['pnl_basis'] or '')}</td>"
            f"<td>{row['option_quote_samples']}</td>"
            f"<td>{html.escape(row['tracking_error'] or '')}</td>"
            "</tr>"
        )
    priced = [row["premium_pnl_per_lot"] for row in rows if row["premium_pnl_per_lot"] is not None]
    cards = "".join(
        f"<div><b>{value}</b><span>{label}</span></div>" for value, label in (
            (sum(row["status"] == "open" for row in rows), "Open positions"),
            (sum(row["status"] == "target_hit" for row in rows), "Underlying 2R targets"),
            (sum(row["status"] == "stop_hit" for row in rows), "Underlying stops"),
            (sum(row["status"] == "time_exit" for row in rows), "Time exits"),
            (f"{sum(value > 0 for value in priced)}/{len(priced)}",
             "Positive indicative option P&amp;L (priced exits)"),
            (f"{sum(priced):.2f}", "Net indicative option P&amp;L / lot"),
        )
    )
    funnel_text = (
        "<p class='note'>Swing candidate funnel (last 10 days): "
        + html.escape(", ".join(f"{key}: {count}" for key, count in sorted(funnel.items())))
        + "</p>" if funnel else ""
    )
    table = (
        "<p>No 15-minute or swing option positions are tracked for this session.</p>" if not rows else
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Strategy</th><th>Entry (IST)</th><th>Symbol</th><th>Side</th><th>Option</th>"
        "<th>Entry bid/ask</th><th>Underlying entry</th><th>Stop</th><th>2R target</th>"
        "<th>Last / exit underlying</th><th>Outcome</th><th>Sessions held</th><th>Exit (IST)</th>"
        "<th>Exit bid/ask</th><th>Indicative option P&amp;L</th><th>P&amp;L basis</th>"
        "<th>Option quote samples</th><th>Tracking notes</th>"
        f"</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )
    return (
        "<section class='summary option_positions'><h2>Option positions: lifecycle and premium P&amp;L "
        "(15-minute and swing)</h2>"
        f"<div class='cards'>{cards}</div></section>{table}{funnel_text}"
        "<p class='note'>Swing positions are created only after the next-session 15m trigger is "
        "confirmed, then tracked on completed 15m candles to the underlying stop, 2R target, the "
        "3-session limit, or the Friday exit. Option P&amp;L is indicative: entry ask to an exit bid "
        "snapshot taken when the exit was detected (or the option candle close when no bid was "
        "captured); it excludes costs, slippage and fills. Holiday-closed Fridays are not modelled.</p>"
    )
