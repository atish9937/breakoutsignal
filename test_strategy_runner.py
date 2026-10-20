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
        self.assertEqual(scan_pullback.call_args.args[6], now)

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
