import datetime as dt
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

    def test_buy_recommendation_is_persisted_once_per_strategy_day(self):
        self.assertEqual(self.add_buy(), 1)
        self.assertEqual(self.add_buy(), 0)
        row = self.connection.execute("SELECT * FROM paper_trades").fetchone()
        self.assertEqual(row["status"], "tracking")
        self.assertEqual(row["security_id"], 123)

    def test_completed_candle_with_stop_and_target_counts_stop_first(self):
        self.add_buy()
        times = pd.date_range(
            "2026-10-21 09:20", periods=1, freq="5min", tz="Asia/Kolkata"
        )
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
        times = pd.date_range(
            "2026-10-21 09:20", periods=1, freq="5min", tz="Asia/Kolkata"
        )
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
        self.assertIn("No", report)

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
