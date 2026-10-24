import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import option_positions as op
import trade_performance as performance

IST = op.IST


def candles(rows):
    index = pd.DatetimeIndex([pd.Timestamp(stamp, tz=IST) for stamp, *_ in rows])
    return pd.DataFrame(
        [{"open": o, "high": h, "low": l, "close": c} for _, o, h, l, c in rows],
        index=index,
    )


class FakeDhan:
    def __init__(self, underlying, option=None):
        self.underlying, self.option = underlying, option

    def intraday(self, security_id, start, end, seg="NSE_EQ", inst="EQUITY", interval=15):
        return (self.option if seg == "NSE_FNO" else self.underlying).copy()


def swing_file(directory, **overrides):
    candidate = {
        "symbol": "ABC", "side": "CE", "setup": "pullback", "underlying_security_id": 11,
        "expiry": "2026-11-26", "next_session_status": "trigger_confirmed",
        "confirmed_at": "2026-10-23T10:00:00+05:30", "confirmed_underlying_close": 101.0,
        "underlying": {"entry_trigger": 100.0, "stop": 95.0, "target_2r": 110.0},
        "holding": {"max_trading_sessions": 3},
        "refreshed_option": {"security_id": 777, "strike": 100.0, "bid": 4.8, "ask": 5.0,
                             "premium": 4.9, "lot_size": 500},
    }
    candidate.update(overrides)
    path = Path(directory) / "swing_option_candidates_2026-10-22.json"
    path.write_text(json.dumps({"as_of": "2026-10-22", "candidates": [candidate]}))


class OptionPositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.connection = performance.connect(Path(self.temp.name) / "ledger.sqlite3")
        self.addCleanup(self.connection.close)

    def position(self):
        return self.connection.execute("SELECT * FROM paper_option_positions").fetchone()

    def test_confirmed_swing_candidate_becomes_one_position_after_candle_close(self):
        swing_file(self.temp.name)
        self.assertEqual(op.record_swing_confirmations(self.connection, Path(self.temp.name)), 1)
        self.assertEqual(op.record_swing_confirmations(self.connection, Path(self.temp.name)), 0)
        row = self.position()
        self.assertEqual(row["entry_at"], "2026-10-23T10:15:00+05:30")
        self.assertEqual(row["entry_underlying"], 101.0)
        self.assertEqual(row["max_hold_sessions"], 3)

    def test_unconfirmed_or_misordered_candidates_are_not_recorded(self):
        swing_file(self.temp.name, next_session_status="expired_untriggered")
        self.assertEqual(op.record_swing_confirmations(self.connection, Path(self.temp.name)), 0)
        swing_file(self.temp.name, underlying={"entry_trigger": 100, "stop": 105, "target_2r": 110})
        self.assertEqual(op.record_swing_confirmations(self.connection, Path(self.temp.name)), 0)

    def test_stop_wins_when_one_candle_touches_stop_and_target(self):
        swing_file(self.temp.name)
        op.record_swing_confirmations(self.connection, Path(self.temp.name))
        underlying = candles([
            ("2026-10-23 10:15", 101, 102, 100, 101),
            ("2026-10-23 10:30", 101, 111, 94, 100),
        ])
        option = candles([("2026-10-23 10:30", 5, 5, 3, 3.5)])
        now = dt.datetime(2026, 10, 23, 11, 0, tzinfo=IST)
        closed = op.update_swing_positions(
            self.connection, FakeDhan(underlying, option), now,
            lambda position: {"bid": 3.4, "ask": 3.6},
        )
        row = self.position()
        self.assertEqual(closed, 1)
        self.assertEqual(row["status"], "stop_hit")
        self.assertEqual(row["exit_underlying"], 95.0)
        self.assertEqual(row["premium_pnl_per_lot"], round((3.4 - 5.0) * 500, 2))
        self.assertEqual(row["pnl_basis"], "entry ask to exit bid snapshot")

    def test_target_exit_falls_back_to_option_candle_close_without_a_bid(self):
        swing_file(self.temp.name)
        op.record_swing_confirmations(self.connection, Path(self.temp.name))
        underlying = candles([("2026-10-23 10:15", 101, 111, 100, 110)])
        option = candles([("2026-10-23 10:15", 5, 9, 5, 8.0)])
        op.update_swing_positions(
            self.connection, FakeDhan(underlying, option),
            dt.datetime(2026, 10, 23, 11, 0, tzinfo=IST),
        )
        row = self.position()
        self.assertEqual(row["status"], "target_hit")
        self.assertEqual(row["premium_pnl_per_lot"], (8.0 - 5.0) * 500)
        self.assertEqual(row["pnl_basis"], "entry ask to option candle close")

    def test_position_stays_open_overnight_then_exits_at_max_hold(self):
        swing_file(self.temp.name)
        op.record_swing_confirmations(self.connection, Path(self.temp.name))
        rows = [("2026-10-23 10:15", 101, 103, 100, 102)]
        dhan = FakeDhan(candles(rows))
        op.update_swing_positions(self.connection, dhan, dt.datetime(2026, 10, 23, 14, 0, tzinfo=IST))
        self.assertEqual(self.position()["status"], "open")
        # Oct 23, 2026 is a Friday, so the no-weekend-carry rule forces the exit that day.
        rows.append(("2026-10-23 15:00", 102, 103, 101, 102.5))
        op.update_swing_positions(self.connection, FakeDhan(candles(rows)),
                                  dt.datetime(2026, 10, 23, 15, 20, tzinfo=IST))
        row = self.position()
        self.assertEqual(row["status"], "time_exit")
        self.assertEqual(row["exit_underlying"], 102.5)

    def test_midweek_position_carries_until_third_session(self):
        swing_file(self.temp.name, confirmed_at="2026-10-20T10:00:00+05:30")
        op.record_swing_confirmations(self.connection, Path(self.temp.name))
        rows = [
            ("2026-10-20 10:15", 101, 103, 100, 102),
            ("2026-10-21 10:15", 102, 104, 101, 103),
            ("2026-10-22 15:00", 103, 104, 102, 103.5),
        ]
        dhan = FakeDhan(candles(rows))
        op.update_swing_positions(self.connection, dhan, dt.datetime(2026, 10, 22, 14, 0, tzinfo=IST))
        self.assertEqual(self.position()["status"], "open")
        op.update_swing_positions(self.connection, dhan, dt.datetime(2026, 10, 22, 15, 20, tzinfo=IST))
        row = self.position()
        self.assertEqual((row["status"], row["sessions_held"]), ("time_exit", 3))

    def test_multiframe_signals_sync_entry_and_exit_idempotently(self):
        signal = {
            "symbol": "XYZ", "side": "CE", "signal_time": "2026-10-23T10:20:00+05:30",
            "underlying_entry": 100.0, "underlying_stop": 98.0, "underlying_target": 104.0,
            "underlying_security_id": 5, "expiry": "2026-11-26",
            "option": {"security_id": 9, "strike": 100.0, "bid": 2.0, "ask": 2.1,
                       "premium": 2.05, "lot_size": 1000},
            "status": "paper_entry_signal_sent",
        }
        path = Path(self.temp.name) / "daily_hourly_15m_signals_2026-10-23.jsonl"
        path.write_text(json.dumps(signal) + "\n")
        day = dt.date(2026, 10, 23)
        self.assertEqual(op.sync_multiframe_signals(self.connection, path, day), 1)
        self.assertEqual(self.position()["status"], "open")
        signal.update(status="paper_exit_signal_sent", exit_reason="underlying 2R target",
                      exit_time="2026-10-23T11:00:00+05:30", underlying_exit_reference=104.0,
                      option_exit_quote={"bid": 3.1, "ask": 3.2})
        path.write_text(json.dumps(signal) + "\n")
        op.sync_multiframe_signals(self.connection, path, day)
        op.sync_multiframe_signals(self.connection, path, day)
        row = self.position()
        self.assertEqual(row["status"], "target_hit")
        self.assertEqual(row["premium_pnl_per_lot"], round((3.1 - 2.1) * 1000, 2))
        self.assertEqual(self.connection.execute(
            "SELECT COUNT(*) FROM paper_option_positions").fetchone()[0], 1)

    def test_report_section_shows_positions_and_funnel(self):
        swing_file(self.temp.name)
        op.record_swing_confirmations(self.connection, Path(self.temp.name))
        rows = op.report_rows(self.connection, dt.date(2026, 10, 23))
        funnel = op.swing_funnel(Path(self.temp.name), dt.date(2026, 10, 23))
        section = op.render_section(rows, funnel)
        self.assertIn("Swing (multi-day)", section)
        self.assertIn("trigger_confirmed: 1", section)
        self.assertIn("Open positions", section)


if __name__ == "__main__":
    unittest.main()
