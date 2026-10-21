import datetime as dt
import unittest
from unittest.mock import Mock, patch

import breakout_scanner as breakout
import strategy_runner
import trend_pullback


class StrategyRunnerTests(unittest.TestCase):
    def test_all_strategies_run_with_same_client_and_cycle_time(self):
        client = Mock()
        master = object()
        prep = {"expiry": "2026-10-29"}
        universe = {"AAA": 1}
        exclude = set()
        day = dt.date(2026, 10, 19)
        now = dt.datetime(2026, 10, 19, 10, 0, tzinfo=breakout.IST)
        fired = {"breakout": set(), "trend_pullback": set()}
        breakout_result = {"strategy": "breakout"}
        pullback_result = {"strategy": "trend_pullback"}

        with patch.object(breakout, "scan", return_value=breakout_result) as scan_breakout:
            with patch.object(trend_pullback, "scan", return_value=pullback_result) as scan_pullback:
                result = strategy_runner.run_cycle(
                    client, master, prep, universe, exclude, prep["expiry"], day, now, fired
                )

        self.assertEqual(result, {
            "breakout": breakout_result,
            "trend_pullback": pullback_result,
        })
        self.assertIs(scan_breakout.call_args.args[0], client)
        self.assertIs(scan_pullback.call_args.args[0], client)
        self.assertEqual(scan_breakout.call_args.kwargs["now"], now)
        self.assertFalse(scan_breakout.call_args.kwargs["ignore_entry_cutoff"])
        self.assertEqual(scan_pullback.call_args.args[6], now)

    def test_extended_breakout_window_does_not_extend_trend_pullback(self):
        client = Mock()
        prep = {"expiry": "2026-10-29"}
        universe = {"AAA": 1}
        day = dt.date(2026, 10, 19)
        now = dt.datetime(2026, 10, 19, 14, 0, tzinfo=breakout.IST)
        fired = {"breakout": set(), "trend_pullback": set()}

        with patch.object(breakout, "scan", return_value={}) as scan_breakout:
            with patch.object(trend_pullback, "scan") as scan_pullback:
                result = strategy_runner.run_cycle(
                    client, object(), prep, universe, set(), prep["expiry"], day, now,
                    fired, ignore_entry_cutoff=True,
                )

        self.assertEqual(result, {"breakout": {}})
        self.assertTrue(scan_breakout.call_args.kwargs["ignore_entry_cutoff"])
        scan_pullback.assert_not_called()

    def test_both_schedules_keep_runner_alive_until_the_daily_report(self):
        self.assertEqual(strategy_runner.cycle_end(True), dt.time(15, 10))
        self.assertEqual(strategy_runner.cycle_end(False), dt.time(15, 10))

    def test_report_deadline_uses_seconds_not_only_hour_and_minute(self):
        before = dt.datetime(2026, 10, 19, 15, 9, 59, tzinfo=breakout.IST)
        at_deadline = dt.datetime(2026, 10, 19, 15, 10, 0, tzinfo=breakout.IST)
        after_deadline = dt.datetime(2026, 10, 19, 15, 10, 20, tzinfo=breakout.IST)
        self.assertFalse(strategy_runner.report_due(before))
        self.assertTrue(strategy_runner.report_due(at_deadline))
        self.assertTrue(strategy_runner.report_due(after_deadline))

    def test_breakout_entry_cutoff_is_inclusive_for_the_1500_candle_cycle(self):
        client = Mock()
        now = dt.datetime(2026, 10, 19, 15, 0, 20, tzinfo=breakout.IST)
        fired = {"breakout": set(), "trend_pullback": set()}
        with patch.object(breakout, "scan", return_value={}) as scan_breakout:
            with patch.object(trend_pullback, "scan") as scan_pullback:
                strategy_runner.run_cycle(
                    client, object(), {}, {"AAA": 1}, set(), None,
                    now.date(), now, fired, ignore_entry_cutoff=True,
                )
        scan_breakout.assert_called_once()
        scan_pullback.assert_not_called()

    def test_scans_stop_after_entry_window_but_tracker_keeps_updating(self):
        client = Mock()
        ledger = object()
        now = dt.datetime(2026, 10, 19, 15, 5, 20, tzinfo=breakout.IST)
        fired = {"breakout": set(), "trend_pullback": set()}
        with patch.object(breakout, "scan") as scan_breakout:
            with patch.object(trend_pullback, "scan") as scan_pullback:
                with patch.object(
                    strategy_runner.trade_performance, "update_open_trades"
                ) as update:
                    result = strategy_runner.run_cycle(
                        client, object(), {}, {"AAA": 1}, set(), None,
                        now.date(), now, fired, ignore_entry_cutoff=True,
                        ledger=ledger,
                    )
        self.assertEqual(result, {})
        scan_breakout.assert_not_called()
        scan_pullback.assert_not_called()
        update.assert_called_once()

    def test_strategy_error_is_reported_and_next_strategy_still_runs(self):
        client = Mock()
        master = object()
        prep = {"expiry": "2026-10-29"}
        universe = {"AAA": 1}
        day = dt.date(2026, 10, 19)
        now = dt.datetime(2026, 10, 19, 10, 0, tzinfo=breakout.IST)
        fired = {"breakout": set(), "trend_pullback": set()}
        result = {"strategy": "trend_pullback"}

        with patch.object(breakout, "scan", side_effect=RuntimeError("breakout failed")):
            with patch.object(trend_pullback, "scan", return_value=result) as scan_pullback:
                with patch("sys.stderr"):
                    outputs = strategy_runner.run_cycle(
                        client, master, prep, universe, set(), prep["expiry"], day, now, fired
                    )

        self.assertEqual(outputs, {"trend_pullback": result})
        scan_pullback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
