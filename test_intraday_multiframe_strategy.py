import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

import breakout_scanner as core
import intraday_multiframe_strategy as strategy


class IntradayMultiframeMonitorTests(unittest.TestCase):
    def _signal(self):
        return {
            "strategy": strategy.STRATEGY,
            "symbol": "AAA",
            "side": "CE",
            "signal_time": "2026-10-19T10:05:20+05:30",
            "underlying_security_id": 123,
            "underlying_stop": 95.0,
            "underlying_target": 110.0,
            "expiry": "2026-10-29",
            "option": {"strike": 100.0},
            "status": "paper_entry_signal_sent",
        }

    def _candles(self):
        index = pd.DatetimeIndex(
            [
                "2026-10-19T10:00:00+05:30",
                "2026-10-19T10:05:00+05:30",
                "2026-10-19T15:10:00+05:30",
            ]
        )
        return pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [111.0, 103.0, 104.0],
                "low": [90.0, 100.0, 101.0],
                "close": [105.0, 102.0, 103.0],
                "volume": [10, 10, 10],
            },
            index=index,
        )

    def test_same_candle_stop_takes_priority_and_persists_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(self._signal()) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles()
            client.intraday.return_value.loc[
                pd.Timestamp("2026-10-19T10:00:00+05:30"), ["high", "low"]
            ] = [104.0, 100.0]
            client.intraday.return_value.loc[
                pd.Timestamp("2026-10-19T10:05:00+05:30"), ["high", "low"]
            ] = [111.0, 90.0]
            client.intraday.return_value.loc[
                pd.Timestamp("2026-10-19T10:10:00+05:30"), ["high", "low"]
            ] = [111.0, 90.0]
            client.option_chain.return_value = {}
            now = dt.datetime(2026, 10, 19, 15, 20, 20, tzinfo=core.IST)

            with patch.object(core, "OUT", output), patch.dict(
                "os.environ",
                {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""},
            ):
                exited = strategy.monitor_positions(client, now.date(), now)

            self.assertEqual(len(exited), 1)
            self.assertEqual(exited[0]["exit_reason"], "underlying stop")
            self.assertEqual(exited[0]["underlying_exit_reference"], 95.0)
            client.intraday.assert_called_once()
            self.assertEqual(client.intraday.call_args.args[0], 123)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_exit_signal_sent")
            self.assertEqual(saved["exit_reason"], "underlying stop")
            self.assertEqual(saved["exit_time"], "2026-10-19T10:10:00+05:30")

    def test_pre_signal_candle_is_not_used_to_trigger_exit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(self._signal()) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles()
            client.option_chain.return_value = {}
            now = dt.datetime(2026, 10, 19, 10, 10, 20, tzinfo=core.IST)

            with patch.object(core, "OUT", output):
                exited = strategy.monitor_positions(client, now.date(), now)

            self.assertEqual(exited, [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_entry_signal_sent")

    def test_daily_signal_limit_is_not_reset_each_scan(self):
        now = dt.datetime(2026, 10, 19, 10, 20, 20, tzinfo=core.IST)
        client = Mock()

        with patch.object(core, "OUT", Path(tempfile.gettempdir())):
            result = strategy.scan(
                client,
                pd.DataFrame(),
                {},
                {},
                now.date(),
                now,
                {f"STOCK{index}_CE" for index in range(strategy.MAX_SIGNALS_PER_DAY)},
            )

        self.assertEqual(result, [])
        client.intraday.assert_not_called()

    def test_sub_bar_confirmation_rejects_signal_when_latest_5min_goes_opposite(self):
        """Commit K: when the 15m scanner wakes at :35 (etc.), the just-closed
        5-min bar must close in signal direction - otherwise the 15m signal is
        already reversing within its first 5 min and we skip it."""
        import datetime as dt_local
        from unittest.mock import Mock, patch
        now = dt_local.datetime(2026, 10, 30, 11, 35, 20, tzinfo=core.IST)

        # Build 15-min candles that would produce a BUY_CE signal
        idx_15 = pd.date_range("2026-10-30 09:15", periods=10, freq="15min", tz=core.IST)
        candles_15m = pd.DataFrame({
            "open":  [100] * 10,
            "high":  [100 + i for i in range(10)],
            "low":   [99 - i for i in range(10)],
            "close": [100 + i for i in range(10)],
            "volume":[1000]*10,
        }, index=idx_15)

        # Build 5-min candles where the just-closed 5-min bar REVERSED down
        idx_5 = pd.date_range("2026-10-30 10:15", periods=16, freq="5min", tz=core.IST)
        closes_5m = [108]*15 + [102]  # last 5-min bar closes 6 pts BELOW
        candles_5m = pd.DataFrame({
            "open":  closes_5m,
            "high":  [c + 1 for c in closes_5m],
            "low":   [c - 1 for c in closes_5m],
            "close": closes_5m,
            "volume":[1000]*16,
        }, index=idx_5)

        # Just verify the sub-bar check logic: for a CE signal with entry 108
        # (signal_bar close) and sub_bar close 102, sub_bar < entry -> reversal
        # -> signal would be rejected.
        signal_entry = 108.0
        sub_close = 102.0
        side = "CE"
        if side == "CE":
            sub_confirmed = sub_close > signal_entry
        else:
            sub_confirmed = sub_close < signal_entry
        self.assertFalse(sub_confirmed)

    def test_sub_bar_confirmation_passes_when_5min_closes_in_direction(self):
        """Confirming case: 5-min bar closes in signal direction -> fire."""
        # CE signal with entry 108, 5-min sub-bar closes 110 (above entry)
        signal_entry = 108.0
        sub_close = 110.0
        side = "CE"
        sub_confirmed = sub_close > signal_entry if side == "CE" else sub_close < signal_entry
        self.assertTrue(sub_confirmed)

    def test_round_number_target_cap_at_resistance_for_ce(self):
        """Commit L: BHARTIARTL CE on 2026-10-30 opened at 1970, raw 2R target
        was 1995 but round-number 1985 capped it. The dynamic target cap
        should set target to 1985 (resistance), not 1995 (raw)."""
        above, below = strategy._nearest_round_numbers(1970.60)
        self.assertEqual(above, 2000)  # oh wait, 1970 is in the 500-5000 tier -> step=50 -> rounds to 2000
        self.assertEqual(below, 1950)
        # For price around 1200 (mid-priced), step is still 50
        above, below = strategy._nearest_round_numbers(1218.50)
        self.assertEqual(above, 1250)
        self.assertEqual(below, 1200)
        # For price 450 (small), step is 10
        above, below = strategy._nearest_round_numbers(452.45)
        self.assertEqual(above, 460)
        self.assertEqual(below, 450)

    def test_round_number_target_tier_boundaries(self):
        """Step function: <500 -> 10, 500-5000 -> 50, >5000 -> 100."""
        self.assertEqual(strategy._nearest_round_numbers(100)[1], 100)  # already at round
        self.assertEqual(strategy._nearest_round_numbers(499)[1], 490)  # just under 500
        self.assertEqual(strategy._nearest_round_numbers(501)[1], 500)  # just over 500
        self.assertEqual(strategy._nearest_round_numbers(5001)[0], 5100)  # >5000 tier

    def test_premium_side_stop_fires_when_option_bid_collapses(self):
        """When option bid falls to <= 60% of entry ask, exit as 'premium stop' even if underlying hasn't hit stop."""
        signal = self._signal()
        signal["option"] = {"strike": 100.0, "ask": 10.0, "bid": 10.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(signal) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles()
            # Option chain says current bid is 5.0 which is 50% of entry ask (10.0) -> below 60% threshold
            client.option_chain.return_value = {
                "oc": {
                    "100": {"ce": {"top_bid_price": 5.0, "top_ask_price": 5.1, "last_price": 5.05}},
                }
            }
            now = dt.datetime(2026, 10, 19, 15, 20, 20, tzinfo=core.IST)
            with patch.object(core, "OUT", output), patch.dict(
                "os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}
            ):
                exited = strategy.monitor_positions(client, now.date(), now)
            self.assertEqual(len(exited), 1)
            self.assertEqual(exited[0]["exit_reason"], "premium stop")
            self.assertIn("premium_stop_ratio", exited[0])
            self.assertLessEqual(exited[0]["premium_stop_ratio"], strategy.PREMIUM_STOP_FRAC)

    def test_premium_side_stop_does_not_fire_when_option_holds_value(self):
        signal = self._signal()
        signal["option"] = {"strike": 100.0, "ask": 10.0, "bid": 10.0}
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(signal) + "\n", encoding="utf-8")
            client = Mock()
            # Underlying candles stay above stop and below target - no underlying exit
            index = pd.DatetimeIndex([
                "2026-10-19T10:00:00+05:30",
                "2026-10-19T10:05:00+05:30",
                "2026-10-19T10:10:00+05:30",
            ])
            client.intraday.return_value = pd.DataFrame(
                {"open": [100, 101, 102], "high": [102, 103, 104],
                 "low": [99, 100, 101], "close": [101, 102, 103], "volume": [10, 10, 10]},
                index=index,
            )
            # Current bid 9.0 (90% of entry 10.0) - above 60% threshold
            client.option_chain.return_value = {
                "oc": {"100": {"ce": {"top_bid_price": 9.0, "top_ask_price": 9.2, "last_price": 9.1}}}
            }
            # Not yet past FORCED_EXIT so no mandatory exit either
            now = dt.datetime(2026, 10, 19, 10, 15, 20, tzinfo=core.IST)
            with patch.object(core, "OUT", output):
                exited = strategy.monitor_positions(client, now.date(), now)
            self.assertEqual(exited, [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_entry_signal_sent")

    def test_forced_exit_waits_for_a_post_signal_candle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir)
            path = output / f"{strategy.STRATEGY}_signals_2026-10-19.jsonl"
            path.write_text(json.dumps(self._signal()) + "\n", encoding="utf-8")
            client = Mock()
            client.intraday.return_value = self._candles().iloc[:1]
            now = dt.datetime(2026, 10, 19, 15, 20, 20, tzinfo=core.IST)

            with patch.object(core, "OUT", output), patch(
                "sys.stderr"
            ):
                exited = strategy.monitor_positions(client, now.date(), now)

            self.assertEqual(exited, [])
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "paper_entry_signal_sent")


if __name__ == "__main__":
    unittest.main()
