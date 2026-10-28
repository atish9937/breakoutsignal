"""Capture Dhan full-packet option and underlying quote snapshots during a runner session."""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
import struct
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import websocket


EXCHANGE_CODES = {
    0: "IDX_I",
    1: "NSE_EQ",
    2: "NSE_FNO",
    4: "BSE_EQ",
    5: "MCX_COMM",
    7: "BSE_CURRENCY",
    8: "BSE_FNO",
}
EXCHANGE_SEGMENTS = {value: key for key, value in EXCHANGE_CODES.items()}
FULL_PACKET_CODE = 8
FULL_PACKET_SIZE = 162
DEPTH_LEVEL_OFFSET = 62
SAMPLE_INTERVAL_SECONDS = 1.0


def parse_full_packet(packet: bytes) -> dict | None:
    """Decode Dhan v2's 162-byte little-endian Full packet (response code 8).

    Byte layout (per DhanHQ v2 annexure):
      0        uint8    feed response code
      1-2      int16    message length
      3        uint8    exchange segment
      4-7      int32    security id
      8-11     float32  last traded price
     12-13     int16    last traded quantity
     14-17     int32    last trade time (epoch)
     18-21     float32  average trade price
     22-25     int32    volume
     26-29     int32    total sell qty
     30-33     int32    total buy qty
     34-37     int32    open interest
     38-41     int32    highest oi (NSE_FNO)
     42-45     int32    lowest oi  (NSE_FNO)
     46-49     float32  day open
     50-53     float32  day close
     54-57     float32  day high
     58-61     float32  day low
     62-161    5 x 20   market depth (level 1-5)

    Depth level (20 bytes): bid_qty(i4), ask_qty(i4), bid_orders(i2),
    ask_orders(i2), bid_price(f4), ask_price(f4).
    """
    if len(packet) < FULL_PACKET_SIZE:
        return None
    response_code = packet[0]
    if response_code != FULL_PACKET_CODE:
        return None
    exchange_code = packet[3]
    segment = EXCHANGE_CODES.get(exchange_code)
    if segment is None:
        return None
    security_id = struct.unpack_from("<i", packet, 4)[0]
    last_price = struct.unpack_from("<f", packet, 8)[0]
    last_trade_epoch = struct.unpack_from("<i", packet, 14)[0]
    volume = struct.unpack_from("<i", packet, 22)[0]
    open_interest = struct.unpack_from("<i", packet, 34)[0]
    bid_quantity, ask_quantity = struct.unpack_from("<ii", packet, DEPTH_LEVEL_OFFSET)
    bid, ask = struct.unpack_from("<ff", packet, DEPTH_LEVEL_OFFSET + 12)
    last_trade_at = (
        dt.datetime.fromtimestamp(last_trade_epoch, dt.timezone.utc).isoformat()
        if last_trade_epoch > 0
        else None
    )
    return {
        "segment": segment,
        "security_id": security_id,
        "last_price": last_price if last_price > 0 else None,
        "last_trade_at": last_trade_at,
        "best_bid": bid if bid > 0 else None,
        "best_ask": ask if ask > 0 else None,
        "bid_quantity": bid_quantity if bid_quantity > 0 else 0,
        "ask_quantity": ask_quantity if ask_quantity > 0 else 0,
        "volume": volume if volume >= 0 else None,
        "open_interest": open_interest if open_interest >= 0 else None,
    }


class DhanQuoteRecorder:
    """Run one reconnecting WebSocket and sample each instrument at most once per second."""

    def __init__(
        self,
        database: Path,
        client_id: str,
        access_token: str,
        session_date: dt.date,
    ):
        self.database = database
        self.client_id = str(client_id)
        self.access_token = access_token
        self.session_date = session_date
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="dhan-paper-quote-recorder",
            daemon=True,
        )
        self._last_sample: dict[tuple[str, int], float] = {}
        self._subscribed: set[tuple[str, int]] = set()

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
            if self._thread.is_alive():
                print("Dhan quote recorder did not stop within the timeout.", file=sys.stderr)

    def _instrument_list(self, connection: sqlite3.Connection) -> set[tuple[str, int]]:
        rows = connection.execute(
            """
            SELECT DISTINCT exchange_segment, security_id FROM (
                SELECT 'NSE_EQ' AS exchange_segment, security_id
                FROM paper_trades
                WHERE session_date = ?
                UNION ALL
                SELECT option_segment AS exchange_segment, option_security_id AS security_id
                FROM paper_trades
                WHERE session_date = ? AND option_security_id IS NOT NULL
                UNION ALL
                SELECT 'NSE_EQ' AS exchange_segment, underlying_security_id AS security_id
                FROM paper_option_positions
                WHERE strategy = 'swing_option' AND status = 'open'
                UNION ALL
                SELECT option_segment AS exchange_segment, option_security_id AS security_id
                FROM paper_option_positions
                WHERE strategy = 'swing_option' AND status = 'open'
                  AND option_security_id IS NOT NULL
            )
            """,
            (self.session_date.isoformat(), self.session_date.isoformat()),
        ).fetchall()
        result = set()
        for segment, security_id in rows:
            if segment in EXCHANGE_SEGMENTS and security_id:
                result.add((segment, int(security_id)))
        signal_path = self.database.parent / (
            f"daily_hourly_15m_signals_{self.session_date.isoformat()}.jsonl"
        )
        if signal_path.exists():
            try:
                with signal_path.open(encoding="utf-8") as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        try:
                            signal = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if signal.get("status") != "paper_entry_signal_sent":
                            continue
                        underlying_id = signal.get("underlying_security_id")
                        option_id = (signal.get("option") or {}).get("security_id")
                        if underlying_id:
                            result.add(("NSE_EQ", int(underlying_id)))
                        if option_id:
                            result.add(("NSE_FNO", int(option_id)))
            except OSError as exc:
                print(
                    f"Dhan quote recorder could not read 15-minute signals "
                    f"({type(exc).__name__}).",
                    file=sys.stderr,
                )
        return result

    @staticmethod
    def _send_subscription(ws, instruments: list[dict], request_code: int) -> None:
        for offset in range(0, len(instruments), 100):
            batch = instruments[offset:offset + 100]
            ws.send(json.dumps({
                "RequestCode": request_code,
                "InstrumentCount": len(batch),
                "InstrumentList": batch,
            }))

    def _sync_subscriptions(self, ws, connection: sqlite3.Connection) -> None:
        current = self._instrument_list(connection)
        to_subscribe = current - self._subscribed
        to_unsubscribe = self._subscribed - current
        for request_code, instruments in (
            (21, to_subscribe),
            (22, to_unsubscribe),
        ):
            payload = [
                {"ExchangeSegment": segment, "SecurityId": str(security_id)}
                for segment, security_id in sorted(instruments)
            ]
            self._send_subscription(ws, payload, request_code)
        self._subscribed = current

    def _save_packet(self, connection: sqlite3.Connection, packet: bytes) -> None:
        quote = parse_full_packet(packet)
        if quote is None:
            return
        key = (quote["segment"], quote["security_id"])
        monotonic_now = time.monotonic()
        if monotonic_now - self._last_sample.get(key, 0.0) < SAMPLE_INTERVAL_SECONDS:
            return
        self._last_sample[key] = monotonic_now
        sampled_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        connection.execute(
            """
            INSERT OR IGNORE INTO paper_market_ticks (
                instrument_segment, security_id, session_date, sampled_at,
                last_price, last_trade_at, best_bid, best_ask, bid_quantity,
                ask_quantity, volume, open_interest, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'dhan_websocket_full')
            """,
            (
                quote["segment"], quote["security_id"],
                self.session_date.isoformat(), sampled_at,
                quote["last_price"], quote["last_trade_at"],
                quote["best_bid"], quote["best_ask"],
                quote["bid_quantity"], quote["ask_quantity"],
                quote["volume"], quote["open_interest"],
            ),
        )
        connection.commit()

    def _run(self) -> None:
        connection = None
        try:
            connection = sqlite3.connect(str(self.database), timeout=15)
            connection.execute("PRAGMA busy_timeout = 15000")
            connection.execute("PRAGMA journal_mode = WAL")
            base_url = "wss://api-feed.dhan.co"
            query = urlencode({
                "version": "2",
                "token": self.access_token,
                "clientId": self.client_id,
                "authType": "2",
            })
            url = f"{base_url}?{query}"
            while not self._stop.is_set():
                ws = None
                try:
                    ws = websocket.create_connection(
                        url,
                        timeout=1,
                        enable_multithread=True,
                        suppress_origin=True,
                    )
                    self._subscribed.clear()
                    self._sync_subscriptions(ws, connection)
                    print("Dhan live quote recorder connected.")
                    last_refresh = time.monotonic()
                    while not self._stop.is_set():
                        if time.monotonic() - last_refresh >= 5:
                            self._sync_subscriptions(ws, connection)
                            last_refresh = time.monotonic()
                        try:
                            opcode, payload = ws.recv_data(control_frame=True)
                        except websocket.WebSocketTimeoutException:
                            continue
                        if opcode == websocket.ABNF.OPCODE_PING:
                            ws.pong(payload)
                        elif opcode == websocket.ABNF.OPCODE_BINARY:
                            self._save_packet(connection, payload)
                except Exception as exc:  # noqa: BLE001
                    if not self._stop.is_set():
                        print(
                            f"Dhan live quote recorder disconnected "
                            f"({type(exc).__name__}); retrying in 5 seconds.",
                            file=sys.stderr,
                        )
                finally:
                    if ws is not None:
                        try:
                            ws.close()
                        except websocket.WebSocketException:
                            pass
                self._stop.wait(5)
        except sqlite3.Error as exc:
            print(
                f"Dhan live quote recorder could not access its database "
                f"({type(exc).__name__}).",
                file=sys.stderr,
            )
        finally:
            if connection is not None:
                connection.close()
