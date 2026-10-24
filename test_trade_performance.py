import datetime as dt
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import trade_performance as performance


IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION = dt.date(2026, 10, 21)


def buy_snapshot():
    return {
        "scan_time": "2026-10-21 09:20:20",
        "results": [{
            "verdict": "BUY_CE",
            "symbol": "AAA",
            "side": "CE",
            "ltp": 100,
            "plan": {"entry": 100, "stop": 95, "t1": 105, "t2": 110},
            "option": {
                "security_id": 456,
                "expiry": "2026-10-29",
                "premium": 12,
                "bid": 11.9,
                "ask": 12.1,
                "lot_size": 10,
            },
        }],
    }


class TradePerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.connection = performance.connect(self.root / "ledger.sqlite3")

    def tearDown(self):
        self.connection.close()
        self.temp.cleanup()

    def add_buy(self, symbol="AAA", strategy="breakout", side="CE"):
        snapshot = buy_snapshot()
        snapshot["results"][0]["symbol"] = symbol
        snapshot["results"][0]["side"] = side
        if side == "PE":
            snapshot["results"][0]["plan"] = {
                "entry": 100, "stop": 105, "t1": 95, "t2": 90,
            }
        return performance.record_buy_recommendations(
            self.connection, strategy, snapshot, {symbol: 123}, SESSION
        )

    def test_connect_migrates_existing_quote_table_with_session_date(self):
        self.connection.close()
        database = self.root / "legacy.sqlite3"
        legacy = sqlite3.connect(database)
        legacy.execute(
            """
            CREATE TABLE paper_market_ticks (
                instrument_segment TEXT NOT NULL,
                security_id INTEGER NOT NULL,
                sampled_at TEXT NOT NULL,
                last_price REAL,
                last_trade_at TEXT,
                best_bid REAL,
                best_ask REAL,
                bid_quantity INTEGER,
                ask_quantity INTEGER,
                volume INTEGER,
                open_interest INTEGER,
                source TEXT NOT NULL,
                PRIMARY KEY (instrument_segment, security_id, sampled_at)
            )
            """
        )
        legacy.close()

        migrated = performance.connect(database)
        try:
            columns = {
                row["name"]
                for row in migrated.execute("PRAGMA table_info(paper_market_ticks)")
            }
            self.assertIn("session_date", columns)
            migrated.execute(
                """
                INSERT INTO paper_market_ticks (
                    instrument_segment, security_id, session_date, sampled_at, source
                ) VALUES ('NSE_EQ', 123, ?, '2026-10-21T04:00:00+00:00',
                          'dhan_websocket_full')
                """,
                (SESSION.isoformat(),),
            )
            migrated.commit()
        finally:
            migrated.close()
        self.connection = performance.connect(self.root / "ledger.sqlite3")

    def test_buy_recommendation_is_persisted_once_per_strategy_day(self):
        self.assertEqual(self.add_buy(), 1)
        self.assertEqual(self.add_buy(), 0)
        row = self.connection.execute("SELECT * FROM paper_trades").fetchone()
        self.assertEqual(row["status"], "tracking")
        self.assertEqual(row["security_id"], 123)
        self.assertEqual(row["option_security_id"], 456)
        self.assertEqual(row["option_entry_bid"], 11.9)
        self.assertIn('"strategy":"breakout"', row["signal_json"])

    def test_completed_candle_with_stop_and_target_counts_stop_first(self):
        self.add_buy()
        times = pd.DatetimeIndex(["2026-10-21 09:21"], tz="Asia/Kolkata")
        candles = pd.DataFrame(
            [{"open": 100, "high": 106, "low": 94, "close": 101}],
            index=times,
        )
        dhan = Mock()
        dhan.intraday.return_value = candles
        processed = performance.update_open_trades(
            self.connection,
            dhan,
            SESSION,
            dt.datetime(2026, 10, 21, 9, 26, tzinfo=IST),
        )
        row = self.connection.execute("SELECT * FROM paper_trades").fetchone()
        self.assertEqual(processed, 1)
        self.assertEqual(row["status"], "stop_before_t1")
        self.assertIsNone(row["target_1_at"])
        event = self.connection.execute(
            "SELECT event FROM paper_trade_events"
        ).fetchone()
        self.assertEqual(event["event"], "stop_before_t1")

    def test_target_two_in_same_candle_records_both_targets(self):
        self.add_buy()
        times = pd.DatetimeIndex(["2026-10-21 09:21"], tz="Asia/Kolkata")
        candles = pd.DataFrame(
            [{"open": 100, "high": 111, "low": 99, "close": 108}],
            index=times,
        )
        dhan = Mock()
        dhan.intraday.return_value = candles
        performance.update_open_trades(
            self.connection,
            dhan,
            SESSION,
            dt.datetime(2026, 10, 21, 9, 26, tzinfo=IST),
        )
        row = self.connection.execute("SELECT * FROM paper_trades").fetchone()
        self.assertEqual(row["status"], "t2_hit")
        self.assertIsNotNone(row["target_1_at"])
        self.assertIsNotNone(row["target_2_at"])
        self.assertEqual(row["last_price"], 108)

    def test_pending_entry_ignores_pre_entry_stop_and_entry_candle(self):
        self.add_buy()
        self.connection.execute(
            "UPDATE paper_trades SET entry_at = NULL"
        )
        self.connection.commit()
        times = pd.DatetimeIndex(
            ["2026-10-21 09:21", "2026-10-21 09:22", "2026-10-21 09:23"],
            tz="Asia/Kolkata",
        )
        candles = pd.DataFrame(
            [
                {"open": 96, "high": 99, "low": 94, "close": 96},
                {"open": 100, "high": 106, "low": 94, "close": 101},
                {"open": 100, "high": 101, "low": 94, "close": 96},
            ],
            index=times,
        )
        dhan = Mock()
        dhan.intraday.return_value = candles

        performance.update_open_trades(
            self.connection,
            dhan,
            SESSION,
            dt.datetime(2026, 10, 21, 9, 27, tzinfo=IST),
        )

        row = self.connection.execute("SELECT * FROM paper_trades").fetchone()
        events = self.connection.execute(
            "SELECT event FROM paper_trade_events ORDER BY candle_time"
        ).fetchall()
        self.assertEqual(row["status"], "stop_before_t1")
        self.assertEqual(row["entry_at"], "2026-10-21T09:22:00+05:30")
        self.assertIsNone(row["target_1_at"])
        self.assertEqual([event["event"] for event in events], ["none", "none", "stop_before_t1"])

    def test_import_saved_signals_handles_full_timestamp_entry(self):
        saved = self.root / "signals.jsonl"
        saved.write_text(
            '{"symbol":"AAA","side":"CE","time":"2026-10-21T09:20:20+05:30",'
            '"ltp":100,"plan":{"entry":100,"stop":95,"t1":105,"t2":110}}\n',
            encoding="utf-8",
        )

        inserted = performance.import_saved_signals(
            self.connection, "breakout", saved, {"AAA": 123}, SESSION
        )

        row = self.connection.execute("SELECT * FROM paper_trades").fetchone()
        self.assertEqual(inserted, 1)
        self.assertEqual(row["signal_time"], "2026-10-21T09:20:20+05:30")
        self.assertEqual(row["entry_at"], row["signal_time"])

    def test_report_has_both_strategy_sections_and_escapes_symbol(self):
        self.add_buy(symbol="<AAA>")
        rows = performance._report_rows(self.connection, SESSION)
        report = performance.render_html(
            rows, SESSION, dt.datetime(2026, 10, 21, 15, 10, tzinfo=IST)
        )
        self.assertIn("Breakout", report)
        self.assertIn("Trend Pullback", report)
        self.assertIn("&lt;AAA&gt;", report)
        self.assertIn("Last price", report)
        self.assertIn("Option contract", report)
        self.assertIn("Signal option bid/ask", report)
        self.assertIn("T1 hit rate", report)
        self.assertNotIn("T1 win rate", report)
        self.assertIn("No", report)

    def test_report_includes_separate_multiframe_paper_outcomes(self):
        signals = [{
            "strategy": "daily_hourly_15m",
            "symbol": "AAA",
            "side": "CE",
            "signal_time": "2026-10-21T10:05:20+05:30",
            "expiry": "2026-10-29",
            "option": {"strike": 100, "bid": 10.0, "ask": 10.2, "lot_size": 50},
            "underlying_entry": 100,
            "underlying_stop": 95,
            "underlying_target": 110,
            "underlying_exit_reference": 110,
            "option_exit_quote": {"bid": 11.5, "ask": 11.7},
            "exit_reason": "underlying 2R target",
            "exit_time": "2026-10-21T11:05:00+05:30",
            "status": "paper_exit_signal_sent",
        }]

        report = performance.render_html(
            [],
            SESSION,
            dt.datetime(2026, 10, 21, 15, 35, tzinfo=IST),
            signals,
        )

        self.assertIn("15-minute Long Option (same-day, paper)", report)
        self.assertIn("Underlying 2R target", report)
        self.assertIn("65.00 / lot", report)
        self.assertIn("Positive indicative option quote P&amp;L", report)

    def test_archives_post_signal_underlying_and_option_one_minute_candles(self):
        self.add_buy()
        candle_times = pd.DatetimeIndex(
            ["2026-10-21 09:20", "2026-10-21 09:21"],
            tz="Asia/Kolkata",
        )
        candles = pd.DataFrame(
            [
                {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 5},
                {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 8},
            ],
            index=candle_times,
        )
        dhan = Mock()
        dhan.intraday.return_value = candles
        saved = performance.archive_one_minute_candles(
            self.connection,
            dhan,
            SESSION,
            dt.datetime(2026, 10, 21, 15, 10, tzinfo=IST),
        )
        self.assertEqual(saved, 2)
        stored = self.connection.execute(
            "SELECT instrument_type, candle_time FROM paper_instrument_candles ORDER BY instrument_type"
        ).fetchall()
        self.assertEqual([row["instrument_type"] for row in stored], ["option", "underlying"])
        self.assertTrue(all(row["candle_time"].startswith("2026-10-21T09:21") for row in stored))
        self.assertEqual(dhan.intraday.call_count, 2)
        self.assertTrue(all(call.kwargs["interval"] == 1 for call in dhan.intraday.call_args_list))

    def test_failed_telegram_delivery_is_recorded_and_can_be_retried(self):
        generated_at = dt.datetime(2026, 10, 21, 15, 10, tzinfo=IST)
        with patch.object(
            performance, "send_html_report", side_effect=RuntimeError("offline")
        ):
            with self.assertRaisesRegex(RuntimeError, "offline"):
                performance.finalize_day(
                    self.connection, SESSION, self.root, "token", "chat", generated_at
                )
        status = self.connection.execute(
            "SELECT status, last_error FROM performance_reports"
        ).fetchone()
        self.assertEqual(status["status"], "failed")
        self.assertIn("offline", status["last_error"])

        with patch.object(performance, "send_html_report", return_value="42") as send:
            path = performance.finalize_day(
                self.connection, SESSION, self.root, "token", "chat", generated_at
            )
        self.assertTrue(path.exists())
        self.assertEqual(send.call_count, 1)
        report = self.connection.execute(
            "SELECT status, telegram_message_id FROM performance_reports"
        ).fetchone()
        self.assertEqual(report["status"], "sent")
        self.assertEqual(report["telegram_message_id"], "42")

    def test_sent_report_is_not_uploaded_twice(self):
        generated_at = dt.datetime(2026, 10, 21, 15, 10, tzinfo=IST)
        with patch.object(performance, "send_html_report", return_value="42") as send:
            performance.finalize_day(
                self.connection, SESSION, self.root, "token", "chat", generated_at
            )
            performance.finalize_day(
                self.connection, SESSION, self.root, "token", "chat", generated_at
            )
        self.assertEqual(send.call_count, 1)


if __name__ == "__main__":
    unittest.main()
