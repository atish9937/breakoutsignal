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
